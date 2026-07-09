"""
Layer-Bridge Cross-Attention Fusion.

Three-step coupling (repeated n_stages times):
  Step 1:  Z_layer' = CrossAttn(Q=Z_layer, KV=Z_temp)
           → layer-domain absorbs fine-grained temperature details
  Step 2:  Z_smoke' = CrossAttn(Q=Z_smoke, KV=Z_layer')
           → smoke absorbs thermal stratification structure
  Step 3:  Z_smoke''= CrossAttn(Q=Z_smoke', KV=Z_temp)
           → smoke directly supplements local hot-spot information

Gate fusion:
  g = σ(W_g [Z_smoke'' ; Z_layer'])
  Z_fused = g ⊙ Z_smoke'' + (1 − g) ⊙ Z_layer'
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import Config


class CrossAttentionBlock(nn.Module):
    """Standard cross-attention + FFN with pre-norm residuals."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, attn_weights = self.cross_attn(
            q, kv, kv, need_weights=True, average_attn_weights=True,
        )
        x = query + attn_out
        x = x + self.ff(self.norm_ff(x))
        if return_attention:
            return x, attn_weights
        return x


class LayerBridgeFusion(nn.Module):
    """
    Multi-stage 3-step cross-attention with gated fusion.

    Inputs:
        Z_temp   (B, T, d)  – temperature field representation
        Z_layer  (B, T, d)  – layer-domain representation
        Z_smoke  (B, T, d)  – smoke/optical/heat representation

    Output:
        Z_fused  (B, T, d)  – gate-fused representation
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_stages: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        for _ in range(n_stages):
            self.stages.append(nn.ModuleDict({
                "layer_from_temp":  CrossAttentionBlock(d_model, n_heads, dropout),
                "smoke_from_layer": CrossAttentionBlock(d_model, n_heads, dropout),
                "smoke_from_temp":  CrossAttentionBlock(d_model, n_heads, dropout),
            }))

        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    _ATTN_KEYS = ("layer_from_temp", "smoke_from_layer", "smoke_from_temp")
    _ATTN_LABELS = (
        "Layer-domain ← Temperature field",
        "Hazard-related ← Layer-domain",
        "Hazard-related ← Temperature field",
    )

    def forward(
        self,
        Z_temp: torch.Tensor,
        Z_layer: torch.Tensor,
        Z_smoke: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        attn_by_stage: list[dict[str, torch.Tensor]] = []
        for stage in self.stages:
            stage_attn: dict[str, torch.Tensor] = {}
            if return_attention:
                Z_layer, w_lt = stage["layer_from_temp"](
                    query=Z_layer, key_value=Z_temp, return_attention=True,
                )
                stage_attn["layer_from_temp"] = w_lt
            else:
                Z_layer = stage["layer_from_temp"](query=Z_layer, key_value=Z_temp)
            if return_attention:
                Z_smoke, w_sl = stage["smoke_from_layer"](
                    query=Z_smoke, key_value=Z_layer, return_attention=True,
                )
                stage_attn["smoke_from_layer"] = w_sl
            else:
                Z_smoke = stage["smoke_from_layer"](query=Z_smoke, key_value=Z_layer)
            if return_attention:
                Z_smoke, w_st = stage["smoke_from_temp"](
                    query=Z_smoke, key_value=Z_temp, return_attention=True,
                )
                stage_attn["smoke_from_temp"] = w_st
            else:
                Z_smoke = stage["smoke_from_temp"](query=Z_smoke, key_value=Z_temp)
            if return_attention:
                attn_by_stage.append(stage_attn)

        g = self.gate_proj(torch.cat([Z_smoke, Z_layer], dim=-1))
        Z_fused = g * Z_smoke + (1.0 - g) * Z_layer
        if return_attention:
            return Z_fused, {"stages": attn_by_stage, "labels": list(self._ATTN_LABELS)}
        return Z_fused


class DirectSmokeTempFusion(nn.Module):
    """
    Ablation (w/o bridge): drop layer↔temp and smoke↔layer cross-attn chains.
    Only Smoke cross-attends to Temp (per stage); Z_layer stays the encoder
    output and enters the gate unchanged.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_stages: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stages = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_stages)
        ])
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    _ATTN_LABELS = ("Hazard-related ← Temperature field",)

    def forward(
        self,
        Z_temp: torch.Tensor,
        Z_layer: torch.Tensor,
        Z_smoke: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        Z_layer_static = Z_layer
        attn_by_stage: list[dict[str, torch.Tensor]] = []
        for block in self.stages:
            if return_attention:
                Z_smoke, w = block(
                    query=Z_smoke, key_value=Z_temp, return_attention=True,
                )
                attn_by_stage.append({"smoke_from_temp": w})
            else:
                Z_smoke = block(query=Z_smoke, key_value=Z_temp)
        g = self.gate_proj(torch.cat([Z_smoke, Z_layer_static], dim=-1))
        Z_fused = g * Z_smoke + (1.0 - g) * Z_layer_static
        if return_attention:
            return Z_fused, {"stages": attn_by_stage, "labels": list(self._ATTN_LABELS)}
        return Z_fused


class LateConcatFusion(nn.Module):
    """
    Ablation (w/o cross-attn): pool each modality, concat → Linear → d_model,
    then broadcast to (B, T, d) so downstream pooling matches TriModalFireModel.
    """

    def __init__(self, d_model: int, pooling: str = "last"):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def _pool(Z: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "last":
            return Z[:, -1, :]
        if mode == "mean":
            return Z.mean(dim=1)
        raise ValueError(f"LateConcatFusion: unknown pooling {mode!r}")

    def forward(
        self,
        Z_temp: torch.Tensor,
        Z_layer: torch.Tensor,
        Z_smoke: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        z_t = self._pool(Z_temp, self.pooling)
        z_l = self._pool(Z_layer, self.pooling)
        z_s = self._pool(Z_smoke, self.pooling)
        z = self.proj(torch.cat([z_t, z_l, z_s], dim=-1))
        B, T, _d = Z_temp.shape
        Z_fused = z.unsqueeze(1).expand(B, T, -1).contiguous()
        if return_attention:
            return Z_fused, None
        return Z_fused


def build_fusion(cfg: Config) -> nn.Module:
    """Construct fusion from cfg.fusion_mode (supervised & contrastive)."""
    mode = getattr(cfg, "fusion_mode", "layer_bridge")
    d, h, st, dr = cfg.d_model, cfg.n_heads, cfg.n_cross_attn_stages, cfg.dropout

    if mode == "layer_bridge":
        return LayerBridgeFusion(
            d_model=d, n_heads=h, n_stages=st, dropout=dr,
        )
    if mode == "direct_smoke_temp":
        return DirectSmokeTempFusion(
            d_model=d, n_heads=h, n_stages=st, dropout=dr,
        )
    if mode == "late_concat":
        return LateConcatFusion(d_model=d, pooling=cfg.pooling)

    raise ValueError(
        f"Unknown fusion_mode {mode!r}; expected "
        "'layer_bridge', 'direct_smoke_temp', or 'late_concat'"
    )
