from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..blocks import ResidualSTBlock, valid_num_groups


class FeatureAdapter(nn.Module):
    """Lightweight feature-space adaptation used at one pyramid level."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        groups = valid_num_groups(out_channels, num_groups)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            ResidualSTBlock(out_channels, num_groups=num_groups, dropout=dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class CompactResidualPyramid(nn.Module):
    """Aggregate coarse-to-fine features and predict only a base residual."""

    def __init__(
        self,
        dim: int,
        c_out: int,
        num_groups: int = 8,
        dropout: float = 0.1,
        zero_init: bool = True,
        use_pyramid: bool = True,
    ) -> None:
        super().__init__()
        self.use_pyramid = use_pyramid
        if use_pyramid:
            self.coarse_adapter = FeatureAdapter(
                dim, dim, num_groups=num_groups, dropout=dropout
            )
            self.mid_fusion = FeatureAdapter(
                dim * 2, dim, num_groups=num_groups, dropout=dropout
            )
            fine_channels = dim * 3
        else:
            self.coarse_adapter = nn.Identity()
            self.mid_fusion = nn.Identity()
            fine_channels = dim * 2
        self.fine_fusion = FeatureAdapter(
            fine_channels, dim, num_groups=num_groups, dropout=dropout
        )

        hidden = max(16, dim // 2)
        groups = valid_num_groups(hidden, num_groups)
        self.residual_head = nn.Sequential(
            nn.Conv3d(dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(hidden, c_out, kernel_size=1),
        )
        if zero_init:
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_c: torch.Tensor,
        h_main: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.use_pyramid:
            p_c = self.coarse_adapter(z_c)
            p_c_up = F.interpolate(
                p_c, size=z_m.shape[-3:], mode="trilinear", align_corners=False
            )
            p_m = self.mid_fusion(torch.cat((z_m, p_c_up), dim=1))
            p_m_up = F.interpolate(
                p_m, size=z_f.shape[-3:], mode="trilinear", align_corners=False
            )
            fine_input = torch.cat((z_f, h_main, p_m_up), dim=1)
        else:
            p_c = torch.zeros_like(z_c)
            p_m = torch.zeros_like(z_m)
            fine_input = torch.cat((z_f, h_main), dim=1)
        p_f = self.fine_fusion(fine_input)
        delta_raw = self.residual_head(p_f)
        return {
            "pyramid_coarse": p_c,
            "pyramid_mid": p_m,
            "pyramid_fine": p_f,
            "delta_raw": delta_raw,
        }
