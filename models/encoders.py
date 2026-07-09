"""
Three modal encoders:
  1. TempFieldEncoder   – spatial self-attn (160 channels) + temporal self-attn
  2. LayerDomainEncoder – layer self-attn (4 layers)       + temporal self-attn
  3. SmokeEncoder       – projection                       + temporal self-attn

All produce outputs of shape (B, T, d_model).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ──────────────── Positional Encodings ────────────────

class SinusoidalPE(nn.Module):
    """Additive sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ──────────────── 1. Temperature Field Encoder ────────────────

class TempFieldEncoder(nn.Module):
    """
    Input :  (B, T, C)   C = 160 temperature channels
    Output:  (B, T, d_model)

    Pipeline:
      1. Project each channel scalar → d_model, add spatial PE
      2. Per-timestep spatial self-attention across C tokens
      3. Mean-pool over channels → (B, T, d_model)
      4. Add temporal PE, temporal self-attention
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        n_heads: int,
        n_spatial_layers: int,
        n_temporal_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model

        self.channel_proj = nn.Linear(1, d_model)
        self.spatial_pe = nn.Embedding(n_channels, d_model)

        s_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.spatial_attn = nn.TransformerEncoder(s_layer, num_layers=n_spatial_layers)

        self.channel_pool = nn.Linear(d_model, d_model)
        self.temporal_pe = SinusoidalPE(d_model)

        t_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(t_layer, num_layers=n_temporal_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, C = x.shape

        # (B, T, C, 1) → (B, T, C, d)
        h = self.channel_proj(x.unsqueeze(-1))

        # spatial positional encoding
        ids = torch.arange(C, device=x.device)
        h = h + self.spatial_pe(ids)                # broadcast over B, T

        # spatial self-attention per timestep  → (B*T, C, d)
        h = h.reshape(B * T, C, self.d_model)
        pad_s = None
        if mask is not None:
            # mask 1=valid; True in padding mask = ignore position
            pad_s = (mask < 0.5).reshape(B * T, C)
            # If a full row is padded (all channels missing for a timestep),
            # Transformer softmax can get all -inf and produce NaNs.
            # For those rows, zero features and disable masking for stability.
            all_pad_rows = pad_s.all(dim=1)  # (B*T,)
            if all_pad_rows.any():
                h = h.clone()
                h[all_pad_rows] = 0.0
                pad_s = pad_s.clone()
                pad_s[all_pad_rows] = False
        h = self.spatial_attn(h, src_key_padding_mask=pad_s)

        # pool over channels → (B, T, d); masked mean when mask given
        h = h.view(B, T, C, self.d_model)
        if mask is not None:
            w = mask.unsqueeze(-1).clamp(0.0, 1.0)
            num = (h * w).sum(dim=2)
            den = w.sum(dim=2).clamp(min=1e-6)
            h = num / den
        else:
            h = h.mean(dim=2)
        h = self.channel_pool(h.reshape(B * T, self.d_model)).view(B, T, self.d_model)

        # temporal self-attention
        h = self.temporal_pe(h)
        pad_t = None
        if mask is not None:
            pad_t = (mask < 0.5).all(dim=-1)  # (B, T)
            # Same stability issue: if an entire sequence is padded, disable
            # mask and let temporal attn operate on zeros.
            all_pad_t = pad_t.all(dim=1)  # (B,)
            if all_pad_t.any():
                h = h.clone()
                h[all_pad_t] = 0.0
                pad_t = pad_t.clone()
                pad_t[all_pad_t] = False
        h = self.temporal_attn(h, src_key_padding_mask=pad_t)
        return h


# ──────────────── 2. Layer-Domain Encoder ────────────────

class LayerDomainEncoder(nn.Module):
    """
    Input :  (B, T, L, F)   L = 4 layers, F = 4 features per layer
    Output:  (B, T, d_model)

    Pipeline:
      1. Project per-layer feature vector → d_model, add layer PE
      2. Per-timestep self-attention across L layer-tokens
      3. Mean-pool over layers → (B, T, d_model)
      4. Temporal self-attention
    """

    def __init__(
        self,
        n_layers: int,
        n_layer_features: int,
        d_model: int,
        n_heads: int,
        n_temporal_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model

        self.layer_proj = nn.Linear(n_layer_features, d_model)
        self.layer_pe = nn.Embedding(n_layers, d_model)

        l_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.layer_attn = nn.TransformerEncoder(l_layer, num_layers=1)

        self.pool_proj = nn.Linear(d_model, d_model)
        self.temporal_pe = SinusoidalPE(d_model)

        t_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(t_layer, num_layers=n_temporal_layers)

    def forward(self, x: torch.Tensor):
        B, T, L, F = x.shape

        h = self.layer_proj(x)                      # (B, T, L, d)
        ids = torch.arange(L, device=x.device)
        h = h + self.layer_pe(ids)

        h = h.reshape(B * T, L, self.d_model)
        h = self.layer_attn(h)

        h = h.mean(dim=1)                           # (B*T, d)
        h = self.pool_proj(h).view(B, T, self.d_model)

        h = self.temporal_pe(h)
        h = self.temporal_attn(h)
        return h


# ──────────────── 3. Smoke / Optical / Heat Encoder ────────────────

class SmokeEncoder(nn.Module):
    """
    Input :  (B, T, D_smoke)
    Output:  (B, T, d_model)

    Simple projection + temporal self-attention.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        n_heads: int,
        n_temporal_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_pe = SinusoidalPE(d_model)
        t_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(t_layer, num_layers=n_temporal_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        h = self.proj(x)
        h = self.temporal_pe(h)
        pad = None
        if mask is not None:
            pad = (mask < 0.5).all(dim=-1)
            # If an entire sequence is padded (all time steps masked),
            # Transformer attention can produce NaNs. Stabilize by zeroing
            # features and disabling masking for those samples.
            all_pad = pad.all(dim=1)  # (B,)
            if all_pad.any():
                h = h.clone()
                h[all_pad] = 0.0
                pad = pad.clone()
                pad[all_pad] = False
        h = self.temporal_attn(h, src_key_padding_mask=pad)
        return h
