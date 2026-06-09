from __future__ import annotations

import torch
from torch import nn

from .blocks import valid_num_groups


class ScaleTokenEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        dim: int,
        max_t: int,
        h: int,
        w: int,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        groups = valid_num_groups(dim, num_groups)
        self.value_embed = nn.Sequential(
            nn.Conv3d(c_in, dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim),
            nn.GELU(),
        )
        self.mask_embed = nn.Conv3d(1, dim, kernel_size=3, padding=1)
        self.scale_embed = nn.Parameter(torch.randn(1, dim, 1, 1, 1) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(1, dim, max_t, 1, 1) * 0.02)
        self.space_embed = nn.Parameter(torch.randn(1, dim, 1, h, w) * 0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.shape
        if t > self.time_embed.shape[2]:
            raise ValueError(f"T={t} exceeds configured max_t={self.time_embed.shape[2]}")
        if h > self.space_embed.shape[-2] or w > self.space_embed.shape[-1]:
            raise ValueError(
                f"Input spatial size {(h, w)} exceeds configured "
                f"{tuple(self.space_embed.shape[-2:])}"
            )
        value = self.value_embed(x)
        mask_value = self.mask_embed(mask)
        time = self.time_embed[:, :, :t]
        space = self.space_embed[:, :, :, :h, :w]
        return value + mask_value + self.scale_embed + time + space


ScaleEmbedding = ScaleTokenEncoder
