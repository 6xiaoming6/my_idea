from __future__ import annotations

import math

import torch
from torch import nn


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class BoundedResidualBudgetController(nn.Module):
    """Produce V18's sole residual-magnitude variables."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        rho_coarse_max: float = 0.15,
        rho_mid_max: float = 0.15,
        rho_fine_max: float = 0.20,
        rho_init: float = 0.02,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        maxima = (rho_coarse_max, rho_mid_max, rho_fine_max)
        for name, value in zip(
            ("rho_coarse_max", "rho_mid_max", "rho_fine_max"), maxima
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0,1], got {value}")
        if not 0.0 < rho_init < min(maxima):
            raise ValueError(
                "rho_init must be positive and smaller than every rho_max"
            )
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")

        hidden_half = max(16, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_half),
            nn.GELU(),
            nn.Linear(hidden_half, 3),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        self.register_buffer(
            "rho_max", torch.tensor(maxima, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.tensor(
                [_logit(rho_init / maximum) for maximum in maxima],
                dtype=torch.float32,
            )
        )

    def forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if condition.ndim != 2:
            raise ValueError(
                f"condition must have shape [B,D], got {tuple(condition.shape)}"
            )
        residual = self.net(condition)
        rho = self.rho_max.to(
            dtype=condition.dtype, device=condition.device
        ) * torch.sigmoid(
            self.bias.to(dtype=condition.dtype, device=condition.device)
            + residual
        )
        rho_c = rho[:, 0].view(-1, 1, 1, 1, 1)
        rho_m = rho[:, 1].view(-1, 1, 1, 1, 1)
        rho_f = rho[:, 2].view(-1, 1, 1, 1, 1)
        return rho_c, rho_m, rho_f
