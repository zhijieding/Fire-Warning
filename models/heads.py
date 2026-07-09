"""
Prediction head  – multi-step Gaussian output  (μ, log σ²)
Warning head     – binary P(warn)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    Multi-step forecasting head with learned step embeddings.

    For each future step h = 0 … H−1 the head outputs per-target
    mean μ and log-variance log σ² (for Gaussian NLL).

    Input :  z_global  (B, d_model)
    Output:  μ (B, H, D_target),  logvar (B, H, D_target)
    """

    def __init__(
        self,
        d_model: int,
        horizon: int,
        n_targets: int,
        d_step: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets

        self.step_embed = nn.Embedding(horizon, d_step)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_step, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets * 2),
        )

    def forward(self, z_global: torch.Tensor):
        B = z_global.size(0)
        device = z_global.device

        step_ids = torch.arange(self.horizon, device=device)
        step_emb = self.step_embed(step_ids)             # (H, d_step)

        z_exp = z_global.unsqueeze(1).expand(B, self.horizon, -1)
        s_exp = step_emb.unsqueeze(0).expand(B, -1, -1)

        out = self.mlp(torch.cat([z_exp, s_exp], dim=-1))  # (B, H, 2*D)
        mu, logvar = out.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-6.0, max=2.0)
        return mu, logvar


class WarningHead(nn.Module):
    """
    Binary early-warning head.

    Input :  z_global (B, d_model)
    Output:  logits   (B,)
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, z_global: torch.Tensor) -> torch.Tensor:
        return self.mlp(z_global).squeeze(-1)
