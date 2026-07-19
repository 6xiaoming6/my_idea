from __future__ import annotations

import torch
from torch import nn


class ContinuousResidualCalibrator(nn.Module):
    """Predict a bounded, per-sample residual strength from observable signals."""

    def __init__(
        self,
        condition_dim: int = 12,
        hidden_dim: int = 32,
        fixed_bias: float = -2.0,
        dropout: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if condition_dim not in {9, 12}:
            raise ValueError(
                "V16 calibration condition must contain 12 values "
                f"(or 9 for the documented ablation), got {condition_dim}"
            )
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.condition_dim = int(condition_dim)
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
        self.register_buffer("fixed_bias", torch.tensor(float(fixed_bias)))

    def forward_logits(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"Expected condition [B,{self.condition_dim}], "
                f"got {tuple(condition.shape)}"
            )
        return self.fixed_bias.to(
            device=condition.device,
            dtype=condition.dtype,
        ) + self.net(condition)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(condition)).view(-1, 1, 1, 1, 1)
