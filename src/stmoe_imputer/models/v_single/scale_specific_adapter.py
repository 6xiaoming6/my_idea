from __future__ import annotations

import torch
from torch import nn


class ScaleSpecificAdapter(nn.Module):
    """A zero-initialized bottleneck residual adapter for one spatial scale."""

    def __init__(
        self,
        dim: int = 64,
        bottleneck_dim: int = 16,
        dropout: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.down = nn.Conv3d(dim, bottleneck_dim, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Conv3d(bottleneck_dim, dim, kernel_size=1)
        if zero_init:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(self.act(self.down(x))))
        return x + residual


class IdentityScaleAdapter(nn.Module):
    """A config-friendly identity used by the No-Scale-Adapter ablation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
