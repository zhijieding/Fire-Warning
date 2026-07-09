"""
Loss functions for joint prediction + warning training.

  L_total = λ_pred · L_pred  +  λ_warn · L_warn
          + λ_trend · L_trend  +  λ_consist · L_consist

  L_pred    – Gaussian negative log-likelihood on multi-step forecast
  L_warn    – Asymmetric focal loss on early-warning logits
  L_trend   – MSE on predicted vs true slopes (first derivative)
  L_consist – physical consistency penalty on k predictions
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class AsymmetricFocalLoss(nn.Module):
    """
    Focal loss with asymmetric gamma for fire safety:
    - gamma_pos low  → don't down-weight ANY positive (miss = catastrophic)
    - gamma_neg high → focus on hard negatives
    - alpha > 0.5    → overall higher weight for positive class
    - label_smoothing → soft targets for better calibration
    """
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                 alpha: float = 0.75, pos_weight: float | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = torch.sigmoid(logits)
        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            ce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=pw,
            )
        else:
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** gamma) * ce
        return loss.mean()


class GaussianNLLPredLoss(nn.Module):
    """
    L_pred = 0.5 Σ [ log σ² + (y − μ)² / σ² ] + λ_reg * logvar²
    logvar clamping + stronger regularisation prevents variance explosion
    that causes negative total loss and training instability.
    """
    def __init__(self, logvar_reg: float = 0.1):
        super().__init__()
        self.logvar_reg = logvar_reg

    def forward(
        self, mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        logvar = logvar.clamp(-6.0, 2.0)
        var = torch.exp(logvar).clamp(min=1e-6)
        nll = 0.5 * (logvar + (target - mu).pow(2) / var)
        reg = self.logvar_reg * (logvar ** 2).mean()
        return nll.mean() + reg


class TrendLoss(nn.Module):
    """
    L_trend = MSE( Δŷ , Δy )
    where Δy_h = y_{h+1} − y_h   (temporal first-difference).
    """
    def forward(self, mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if mu.size(1) < 2:
            return mu.new_tensor(0.0)
        slope_pred = mu[:, 1:, :] - mu[:, :-1, :]
        slope_true = target[:, 1:, :] - target[:, :-1, :]
        return F.mse_loss(slope_pred, slope_true)


class ConsistencyLoss(nn.Module):
    """Penalise physically impossible Trans predictions: should stay in (0, 1]."""
    def __init__(self, trans_index: int = 1):
        super().__init__()
        self.ti = trans_index

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        trans_pred = mu[:, :, self.ti]
        below_penalty = F.relu(-trans_pred).mean()
        over_penalty = F.relu(trans_pred - 1.0).mean()
        return below_penalty + over_penalty


class CombinedLoss(nn.Module):
    """Aggregates all four loss terms with configurable weights."""

    def __init__(self, cfg: Config, pos_weight: float | None = None):
        super().__init__()
        self.pred_loss = GaussianNLLPredLoss(
            logvar_reg=getattr(cfg, "logvar_reg", 0.1),
        )
        self.warn_loss = AsymmetricFocalLoss(
            gamma_pos=0.0,
            gamma_neg=float(getattr(cfg, "warn_focal_gamma_neg", 4.0)),
            alpha=float(getattr(cfg, "warn_focal_alpha", 0.62)),
            pos_weight=pos_weight,
            label_smoothing=getattr(cfg, "label_smoothing", 0.0),
        )
        self.trend_loss = TrendLoss()

        trans_idx = (
            cfg.smoke_col_names.index("Trans")
            if "Trans" in cfg.smoke_col_names else 1
        )
        self.consist_loss = ConsistencyLoss(trans_index=trans_idx)

        self.lp = cfg.lambda_pred
        self.lw = cfg.lambda_warn
        self.lt = cfg.lambda_trend
        self.lc = cfg.lambda_consist

    def forward(
        self,
        outputs: dict,
        targets: dict,
    ) -> dict:
        mu       = outputs["mu"]
        logvar   = outputs["logvar"]
        warn_log = outputs["warn_logits"]

        future   = targets["future"]
        warn_lbl = targets["warn"]
        warn_valid = targets.get("warn_valid")

        l_pred    = self.pred_loss(mu, logvar, future)
        l_trend   = self.trend_loss(mu, future)
        l_consist = self.consist_loss(mu)

        if warn_valid is not None and warn_valid.sum() > 0:
            valid_mask = warn_valid.bool()
            l_warn = self.warn_loss(warn_log[valid_mask], warn_lbl[valid_mask])
        elif warn_valid is not None and warn_valid.sum() == 0:
            l_warn = warn_log.new_tensor(0.0)
        else:
            l_warn = self.warn_loss(warn_log, warn_lbl)

        total = (
            self.lp * l_pred
            + self.lw * l_warn
            + self.lt * l_trend
            + self.lc * l_consist
        )

        return dict(
            total=total,
            pred=l_pred.detach(),
            warn=l_warn.detach(),
            trend=l_trend.detach(),
            consist=l_consist.detach(),
        )
