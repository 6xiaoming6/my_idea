from __future__ import annotations

import torch
from torch import nn


class ResidualAcceptanceGate(nn.Module):
    """Estimate whether a bounded residual candidate should be accepted."""

    def __init__(
        self,
        condition_dim: int = 9,
        hidden_dim: int = 24,
        fixed_bias: float = -1.5,
        dropout: float = 0.1,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if condition_dim < 1 or hidden_dim < 1:
            raise ValueError(
                f"condition_dim/hidden_dim must be positive, got {condition_dim}/{hidden_dim}"
            )
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        if zero_init:
            nn.init.zeros_(self.encoder[-1].weight)
        # Deliberately not trainable: a global learned bias caused V15 budgets
        # to saturate across nearly every sample.
        self.register_buffer("fixed_bias", torch.tensor(float(fixed_bias)))

    def forward_logits(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError(
                f"Expected acceptance condition [B,D], got {tuple(condition.shape)}"
            )
        return self.fixed_bias.to(dtype=condition.dtype) + self.encoder(condition)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(condition)).view(-1, 1, 1, 1, 1)
