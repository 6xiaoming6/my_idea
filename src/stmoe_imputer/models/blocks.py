from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def valid_num_groups(dim: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, dim), 0, -1):
        if dim % groups == 0:
            return groups
    return 1


class ResidualSTBlock(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        groups = valid_num_groups(dim, num_groups)
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, dim)
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, dim)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.gelu(out + identity)
        return out


ResBlock3D = ResidualSTBlock
