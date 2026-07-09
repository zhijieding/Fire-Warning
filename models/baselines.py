"""
Baseline models for fire early-warning comparison.

All baselines share the same forward signature and output dict as
``TriModalFireModel`` so that ``train.py`` / ``evaluate.py`` can run them
unchanged:

    forward(temp_field, layer_domain, smoke, mask_temp=None, mask_smoke=None)
        → dict(mu, logvar, warn_logits, Z_fused)

Each baseline follows the same top-level recipe:

    1.  (Per-modality) linear projection of raw features → d_model
        - temp_field  (B, T, 160)  →  (B, T, d_model)
        - layer_domain(B, T, 4, 4) →  (B, T, d_model)
        - smoke       (B, T, D_s)  →  (B, T, d_model)
        with optional masked zeroing (mask=1 valid, 0 missing).
    2.  Concatenate on feature axis → (B, T, 3·d_model).
    3.  Fuse to d_model via linear.
    4.  Apply a sequence backbone (LSTM / BiLSTM / GRU / TCN / Informer /
        plain-Transformer-encoder without any cross-attention).
    5.  Pool (last or mean) → z_global (B, d_model).
    6.  Share ``PredictionHead`` and ``WarningHead`` with the main model.

Therefore the classification/regression heads, loss, temperature scaling
and metric pipelines remain identical, isolating the comparison to the
sequence backbone itself.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from .heads import PredictionHead, WarningHead


# ═════════════════════════════════════════════════════════════════
#  Common utilities
# ═════════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    """Additive sinusoidal positional encoding (same as encoders.SinusoidalPE)."""

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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ModalityProjector(nn.Module):
    """
    Per-modality linear projection + concat + fuse to d_model.

    Inputs:
        temp_field   (B, T, 160)
        layer_domain (B, T, 4, 4)
        smoke        (B, T, D_smoke)
        mask_temp    (B, T, 160)  optional
        mask_smoke   (B, T, D_smoke)  optional

    Output:
        h (B, T, d_model)
    """

    def __init__(
        self,
        n_temp_channels: int,
        n_layers: int,
        n_layer_features: int,
        n_smoke_channels: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_temp = nn.Linear(n_temp_channels, d_model)
        self.proj_layer = nn.Linear(n_layers * n_layer_features, d_model)
        self.proj_smoke = nn.Linear(n_smoke_channels, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _apply_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x
        return x * mask.clamp(0.0, 1.0)

    def forward(
        self,
        temp_field: torch.Tensor,
        layer_domain: torch.Tensor,
        smoke: torch.Tensor,
        mask_temp: torch.Tensor | None = None,
        mask_smoke: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temp = self._apply_mask(temp_field, mask_temp)
        smk = self._apply_mask(smoke, mask_smoke)
        B, T, L, Fm = layer_domain.shape
        ld = layer_domain.reshape(B, T, L * Fm)

        z_t = self.proj_temp(temp)       # (B, T, d)
        z_l = self.proj_layer(ld)
        z_s = self.proj_smoke(smk)

        h = torch.cat([z_t, z_l, z_s], dim=-1)   # (B, T, 3d)
        return self.fuse(h)                      # (B, T, d)


def _pool(z: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "last":
        return z[:, -1, :]
    if mode == "mean":
        return z.mean(dim=1)
    raise ValueError(f"Unknown pooling mode {mode!r}")


# ═════════════════════════════════════════════════════════════════
#  1. LSTM / BiLSTM / GRU baselines (feature concatenation)
# ═════════════════════════════════════════════════════════════════

class LSTMBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bi = bidirectional
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        if bidirectional:
            self.proj = nn.Linear(2 * d_model, d_model)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.proj(out)


class GRUBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bi = bidirectional
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.proj = nn.Linear(2 * d_model, d_model) if bidirectional else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.proj(out)


# ═════════════════════════════════════════════════════════════════
#  2. TCN baseline (dilated causal Conv1D)
# ═════════════════════════════════════════════════════════════════

class _Chomp1d(nn.Module):
    """Remove the rightmost ``chomp`` time-steps after causal conv."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp == 0:
            return x
        return x[..., : -self.chomp].contiguous()


class _TemporalBlock(nn.Module):
    """Causal dilated Conv1D block with residual connection (Bai et al. 2018)."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = _Chomp1d(padding)
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = _Chomp1d(padding)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()

        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.chomp1(y)
        y = F.relu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.chomp2(y)
        y = F.relu(y)
        y = self.drop(y)
        res = self.res(x)
        return F.relu(y + res)


class TCNBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for i in range(n_levels):
            dil = 2 ** i
            layers.append(
                _TemporalBlock(d_model, d_model, kernel_size, dil, dropout)
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d) → (B, d, T) for Conv1D → back to (B, T, d)
        y = self.net(x.transpose(1, 2))
        return y.transpose(1, 2).contiguous()


# ═════════════════════════════════════════════════════════════════
#  3. Informer baseline (ProbSparse self-attention, simplified)
#
#  We follow the core idea of Zhou et al. 2021: for each query, only
#  the top-u queries (ranked by max-mean sparsity score against a
#  random subset of keys) attend over all keys; the remaining queries
#  receive the mean of V as output.  Distillation is omitted because
#  the sequence length in this task is short (T=50).
# ═════════════════════════════════════════════════════════════════

class ProbSparseSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_qk(
        self,
        Q: torch.Tensor,     # (B, H, L_Q, d_k)
        K: torch.Tensor,     # (B, H, L_K, d_k)
        sample_k: int,
        top_u: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, L_K, D = K.shape
        L_Q = Q.size(2)

        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        K_sample = K_expand[:, :, torch.arange(L_Q, device=Q.device).unsqueeze(1), index_sample, :]
        # (B, H, L_Q, sample_k)
        QK_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = QK_sample.max(dim=-1)[0] - QK_sample.mean(dim=-1)
        top_u = min(top_u, L_Q)
        M_top = M.topk(top_u, sorted=False)[1]              # (B, H, top_u)
        b_idx = torch.arange(B, device=Q.device)[:, None, None]
        h_idx = torch.arange(H, device=Q.device)[None, :, None]
        Q_reduce = Q[b_idx, h_idx, M_top, :]                # (B, H, top_u, d_k)
        QK = torch.matmul(Q_reduce, K.transpose(-2, -1))    # (B, H, top_u, L_K)
        return QK, M_top

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, L, 3, self.n_heads, self.d_k)
            .permute(2, 0, 3, 1, 4)
        )
        Q, K, V = qkv[0], qkv[1], qkv[2]   # each (B, H, L, d_k)

        sample_k = max(1, min(L, int(self.factor * math.ceil(math.log(max(L, 2))))))
        top_u = max(1, min(L, int(self.factor * math.ceil(math.log(max(L, 2))))))

        scores_top, index = self._prob_qk(Q, K, sample_k, top_u)
        scores_top = scores_top / math.sqrt(self.d_k)

        attn = torch.softmax(scores_top, dim=-1)
        attn = self.dropout(attn)
        context_top = torch.matmul(attn, V)   # (B, H, top_u, d_k)

        V_mean = V.mean(dim=-2, keepdim=True).expand(-1, -1, L, -1).contiguous()
        out = V_mean.clone()
        b_idx = torch.arange(B, device=x.device)[:, None, None]
        h_idx = torch.arange(self.n_heads, device=x.device)[None, :, None]
        out[b_idx, h_idx, index, :] = context_top

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class _InformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads, factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class InformerBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int = 2,
        factor: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pe = SinusoidalPE(d_model)
        self.layers = nn.ModuleList([
            _InformerEncoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ═════════════════════════════════════════════════════════════════
#  4. Plain Transformer encoder baseline (no cross-attention)
# ═════════════════════════════════════════════════════════════════

class TransformerNoXABackbone(nn.Module):
    """Standard TransformerEncoder stack on the concatenated modalities."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pe = SinusoidalPE(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)
        return self.encoder(x)


# ═════════════════════════════════════════════════════════════════
#  5. PatchTST baseline (patch + Transformer encoder, Nie et al. 2023)
# ═════════════════════════════════════════════════════════════════

class PatchTSTBackbone(nn.Module):
    """Patch time axis, embed patches, Transformer; linear upsample back to T."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int = 2,
        patch_len: int = 8,
        stride: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_embed = nn.Linear(patch_len * d_model, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → patches (B, n_patches, patch_len * D)
        patches = x.unfold(1, self.patch_len, self.stride)
        B, n_p, pl, D = patches.shape
        return patches.reshape(B, n_p, pl * D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        patches = self._patchify(x)
        h = self.patch_embed(patches)
        h = self.encoder(h)
        h = self.norm(h)
        # (B, n_patches, D) → (B, D, n_patches) → upsample to T
        h = h.transpose(1, 2)
        h = F.interpolate(h, size=T, mode="linear", align_corners=False)
        return h.transpose(1, 2).contiguous()


# ═════════════════════════════════════════════════════════════════
#  6. iTransformer baseline (inverted dims, Liu et al. 2024)
# ═════════════════════════════════════════════════════════════════

class iTransformerBackbone(nn.Module):
    """Treat each feature dim as a token; time axis is embedded per token."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int = 2,
        seq_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_embed = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.token_proj = nn.Linear(d_model, seq_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) → variate tokens (B, D, T)
        tokens = x.transpose(1, 2)
        if tokens.size(-1) != self.seq_len:
            tokens = F.interpolate(
                tokens, size=self.seq_len, mode="linear", align_corners=False,
            )
        h = self.token_embed(tokens)
        h = self.encoder(self.norm(h))
        out = self.token_proj(h).transpose(1, 2)
        if out.size(1) != x.size(1):
            out = F.interpolate(
                out.transpose(1, 2), size=x.size(1),
                mode="linear", align_corners=False,
            ).transpose(1, 2)
        return out.contiguous()


# ═════════════════════════════════════════════════════════════════
#  7. TimesNet baseline (multi-period 2D conv, Wu et al. 2023)
# ═════════════════════════════════════════════════════════════════

class _InceptionBlock2d(nn.Module):
    """Multi-kernel 2D inception block used inside TimesBlock."""

    def __init__(self, in_ch: int, out_ch: int, kernel_sizes: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        n_k = len(kernel_sizes)
        ch_each = max(1, out_ch // n_k)
        branches = []
        for k in kernel_sizes:
            pad = k // 2
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, ch_each, kernel_size=k, padding=pad),
                    nn.GELU(),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.proj = nn.Conv2d(ch_each * n_k, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        return self.proj(torch.cat(outs, dim=1))


class _TimesBlock(nn.Module):
    def __init__(self, d_model: int, top_k: int = 3):
        super().__init__()
        self.top_k = top_k
        self.conv = _InceptionBlock2d(d_model, d_model)

    @staticmethod
    def _fft_periods(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        B, T, _ = x.shape
        xf = torch.fft.rfft(x.float(), dim=1)
        amp = xf.abs().mean(dim=(0, 2))
        amp = amp.clone()
        amp[0] = 0.0
        k = min(k, max(1, amp.numel() - 1))
        _, top_idx = torch.topk(amp, k)
        periods = (T / top_idx.float()).clamp(min=2.0).long()
        period_weight = amp[top_idx]
        period_weight = period_weight / period_weight.sum().clamp(min=1e-8)
        return periods, period_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        periods, p_w = self._fft_periods(x, self.top_k)
        res = torch.zeros_like(x)
        for i, period in enumerate(periods):
            p = int(period.item())
            p = max(2, min(p, T))
            pad = (p - T % p) % p
            xp = F.pad(x, (0, 0, 0, pad))
            Tp = xp.size(1)
            out = xp.reshape(B, Tp // p, p, D).permute(0, 3, 1, 2)
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, Tp, D)
            out = out[:, :T, :]
            res = res + p_w[i] * out
        return res


class TimesNetBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            _TimesBlock(d_model, top_k=top_k) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = x + self.drop(blk(x))
        return self.norm(x)


# ═════════════════════════════════════════════════════════════════
#  Unified baseline wrapper — drop-in replacement for TriModalFireModel
# ═════════════════════════════════════════════════════════════════

_BACKBONE_KEYS = (
    "lstm", "bilstm", "gru", "tcn", "informer", "transformer_noxa",
    "patchtst", "itransformer", "timesnet",
)


def _build_backbone(model_type: str, cfg: Config) -> nn.Module:
    d = cfg.d_model
    dr = float(getattr(cfg, "baseline_dropout", cfg.dropout))

    if model_type == "lstm":
        return LSTMBackbone(
            d_model=d,
            n_layers=int(getattr(cfg, "baseline_rnn_layers", 2)),
            dropout=dr,
            bidirectional=bool(getattr(cfg, "baseline_bidirectional", False)),
        )
    if model_type == "bilstm":
        return LSTMBackbone(
            d_model=d,
            n_layers=int(getattr(cfg, "baseline_rnn_layers", 2)),
            dropout=dr,
            bidirectional=True,
        )
    if model_type == "gru":
        return GRUBackbone(
            d_model=d,
            n_layers=int(getattr(cfg, "baseline_rnn_layers", 2)),
            dropout=dr,
            bidirectional=bool(getattr(cfg, "baseline_bidirectional", False)),
        )
    if model_type == "tcn":
        return TCNBackbone(
            d_model=d,
            n_levels=int(getattr(cfg, "baseline_tcn_levels", 4)),
            kernel_size=int(getattr(cfg, "baseline_tcn_kernel", 3)),
            dropout=dr,
        )
    if model_type == "informer":
        return InformerBackbone(
            d_model=d,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=int(getattr(cfg, "baseline_transformer_layers", 2)),
            factor=int(getattr(cfg, "baseline_informer_factor", 5)),
            dropout=dr,
        )
    if model_type == "transformer_noxa":
        return TransformerNoXABackbone(
            d_model=d,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=int(getattr(cfg, "baseline_transformer_layers", 2)),
            dropout=dr,
        )
    n_tr_layers = int(getattr(cfg, "baseline_transformer_layers", 2))
    if model_type == "patchtst":
        return PatchTSTBackbone(
            d_model=d,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=n_tr_layers,
            patch_len=int(getattr(cfg, "baseline_patch_len", 8)),
            stride=int(getattr(cfg, "baseline_patch_stride", 4)),
            dropout=dr,
        )
    if model_type == "itransformer":
        return iTransformerBackbone(
            d_model=d,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=n_tr_layers,
            seq_len=int(getattr(cfg, "history_window", 50)),
            dropout=dr,
        )
    if model_type == "timesnet":
        return TimesNetBackbone(
            d_model=d,
            n_layers=n_tr_layers,
            top_k=int(getattr(cfg, "baseline_timesnet_top_k", 3)),
            dropout=dr,
        )
    raise ValueError(
        f"Unknown baseline model_type {model_type!r}; expected one of {_BACKBONE_KEYS}"
    )


class BaselineFireModel(nn.Module):
    """
    Multi-modal baseline fire early-warning model.

    Same I/O contract as ``TriModalFireModel`` so it can be plugged into
    the existing ``train`` / ``evaluate`` pipeline via ``build_model``.

    Supported ``cfg.model_type`` values (besides ``"trimodal"``):
        - ``"lstm"``              — 2-layer uni-directional LSTM
        - ``"bilstm"``            — 2-layer bi-directional LSTM
        - ``"gru"``               — 2-layer GRU
        - ``"tcn"``               — 4-level dilated causal Conv1D
        - ``"informer"``          — ProbSparse self-attention encoder
        - ``"transformer_noxa"``  — plain Transformer encoder (no cross-attn)
        - ``"patchtst"``          — PatchTST (patch + Transformer)
        - ``"itransformer"``      — iTransformer (inverted variate tokens)
        - ``"timesnet"``          — TimesNet (FFT multi-period 2D conv)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        model_type = str(getattr(cfg, "model_type", "trimodal")).lower()
        assert model_type in _BACKBONE_KEYS, (
            f"BaselineFireModel constructed with unsupported model_type={model_type!r}"
        )
        self.model_type = model_type
        d = cfg.d_model
        dr = float(getattr(cfg, "baseline_dropout", cfg.dropout))

        self.input_proj = ModalityProjector(
            n_temp_channels=cfg.n_temp_channels,
            n_layers=cfg.n_layers,
            n_layer_features=cfg.n_layer_features,
            n_smoke_channels=cfg.n_smoke_channels,
            d_model=d,
            dropout=dr,
        )
        self.backbone = _build_backbone(model_type, cfg)

        head_drop = getattr(cfg, "head_dropout", 0.0)
        self.pred_head = PredictionHead(
            d_model=d,
            horizon=cfg.prediction_horizon,
            n_targets=cfg.n_pred_targets,
            dropout=head_drop,
        )
        self.warn_head = WarningHead(d_model=d, dropout=head_drop)
        self.pooling = cfg.pooling

    def forward(
        self,
        temp_field: torch.Tensor,
        layer_domain: torch.Tensor,
        smoke: torch.Tensor,
        mask_temp: torch.Tensor | None = None,
        mask_smoke: torch.Tensor | None = None,
    ) -> dict:
        h = self.input_proj(temp_field, layer_domain, smoke, mask_temp, mask_smoke)
        Z = self.backbone(h)                     # (B, T, d)
        z_global = _pool(Z, self.pooling)        # (B, d)

        mu, logvar = self.pred_head(z_global)
        warn_logits = self.warn_head(z_global)

        if bool(getattr(self.cfg, "debug_nan_checks", False)):
            to_check = {
                "h": h, "Z": Z, "z_global": z_global,
                "mu": mu, "logvar": logvar, "warn_logits": warn_logits,
            }
            bad = {
                k: (~torch.isfinite(v)).sum().item()
                for k, v in to_check.items()
                if v is not None and v.numel() > 0 and not torch.isfinite(v).all()
            }
            if bad:
                raise RuntimeError(
                    f"[{self.model_type}] debug_nan_checks: NaN/Inf in tensors: {bad}"
                )

        return dict(
            mu=mu,
            logvar=logvar,
            warn_logits=warn_logits,
            Z_fused=Z,
        )
