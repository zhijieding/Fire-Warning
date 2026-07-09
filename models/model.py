"""
TriModalFireModel – full model assembling the three encoders,
layer-bridge cross-attention fusion, and prediction / warning heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import Config
from .encoders import TempFieldEncoder, LayerDomainEncoder, SmokeEncoder
from .fusion import build_fusion
from .heads import PredictionHead, WarningHead


class TriModalFireModel(nn.Module):
    """
    Forward signature:
        temp_field   (B, w, 160)
        layer_domain (B, w, 4, 4)
        smoke        (B, w, D_smoke)
        mask_temp    (B, w, 160)     [optional]
        mask_smoke   (B, w, D_smoke) [optional]

    Returns dict with keys:
        mu, logvar, warn_logits, Z_fused
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ── encoders ──
        self.temp_encoder = TempFieldEncoder(
            n_channels=cfg.n_temp_channels,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_spatial_layers=cfg.n_spatial_layers,
            n_temporal_layers=cfg.n_temporal_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        )
        self.layer_encoder = LayerDomainEncoder(
            n_layers=cfg.n_layers,
            n_layer_features=cfg.n_layer_features,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_temporal_layers=cfg.n_temporal_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        )
        self.smoke_encoder = SmokeEncoder(
            n_channels=cfg.n_smoke_channels,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_temporal_layers=cfg.n_temporal_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
        )

        # ── fusion (mode from cfg.fusion_mode) ──
        self.fusion = build_fusion(cfg)

        # ── heads ──
        head_drop = getattr(cfg, "head_dropout", 0.0)
        self.pred_head = PredictionHead(
            d_model=cfg.d_model,
            horizon=cfg.prediction_horizon,
            n_targets=cfg.n_pred_targets,
            dropout=head_drop,
        )
        self.warn_head = WarningHead(d_model=cfg.d_model, dropout=head_drop)

        self.pooling = cfg.pooling

    # ─────────── forward ───────────
    def forward(
        self,
        temp_field: torch.Tensor,
        layer_domain: torch.Tensor,
        smoke: torch.Tensor,
        mask_temp: torch.Tensor | None = None,
        mask_smoke: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict:
        Z_temp  = self.temp_encoder(temp_field, mask=mask_temp)
        Z_layer = self.layer_encoder(layer_domain)
        Z_smoke = self.smoke_encoder(smoke, mask=mask_smoke)

        fusion_out = self.fusion(
            Z_temp, Z_layer, Z_smoke, return_attention=return_attention,
        )
        attention = None
        if return_attention:
            Z_fused, attention = fusion_out
        else:
            Z_fused = fusion_out

        if self.pooling == "last":
            z_global = Z_fused[:, -1, :]
        else:
            z_global = Z_fused.mean(dim=1)

        mu, logvar = self.pred_head(z_global)
        warn_logits = self.warn_head(z_global)

        # Optional: fail fast when NaN/Inf appears.
        # This helps pinpoint which intermediate becomes non-finite.
        if bool(getattr(self.cfg, "debug_nan_checks", False)):
            to_check = {
                "Z_temp": Z_temp,
                "Z_layer": Z_layer,
                "Z_smoke": Z_smoke,
                "Z_fused": Z_fused,
                "z_global": z_global,
                "mu": mu,
                "logvar": logvar,
                "warn_logits": warn_logits,
            }
            bad = {
                k: (~torch.isfinite(v)).sum().item()
                for k, v in to_check.items()
                if v is not None and v.numel() > 0 and not torch.isfinite(v).all()
            }
            if bad:
                # Print only counts to keep message short.
                raise RuntimeError(f"debug_nan_checks: NaN/Inf in tensors: {bad}")

        out = dict(
            mu=mu,                     # (B, H, D_target)
            logvar=logvar,             # (B, H, D_target)
            warn_logits=warn_logits,   # (B,)
            Z_fused=Z_fused,           # (B, T, d)  for interpretability
        )
        if attention is not None:
            out["attention"] = attention
        return out
