from __future__ import annotations

import torch
from torch import nn


class ResidualBudgetController(nn.Module):
    """Produce one bounded residual budget for each input sample."""

    def __init__(
        self,
        condition_dim: int = 8,
        hidden_dim: int = 32,
        beta_max: float = 0.5,
        beta_bias: float = -3.0,
        dropout: float = 0.1,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < beta_max <= 1.0:
            raise ValueError(f"beta_max must be in (0,1], got {beta_max}")
        if condition_dim < 1:
            raise ValueError(f"condition_dim must be positive, got {condition_dim}")

        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.beta_bias = nn.Parameter(torch.tensor(float(beta_bias)))
        self.beta_max = float(beta_max)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError(
                f"Expected condition [B,D], got shape {tuple(condition.shape)}"
            )
        residual_logit = self.net(condition)
        beta = self.beta_max * torch.sigmoid(self.beta_bias + residual_logit)
        return beta.view(-1, 1, 1, 1, 1)
