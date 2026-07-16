from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..blocks import ResidualSTBlock, valid_num_groups


class ProjectionBlock(nn.Module):
    """Project a main-backbone feature into the lightweight residual space."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(valid_num_groups(out_channels, num_groups), out_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ScaleGuidedResidualAdapter(nn.Module):
    """Predict a compact residual direction using only active backbone scales."""

    def __init__(
        self,
        main_dim: int,
        residual_dim: int,
        out_channels: int,
        active_scales: tuple[bool, bool, bool],
        num_groups: int = 8,
        dropout: float = 0.1,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if residual_dim < 1:
            raise ValueError(f"residual_dim must be positive, got {residual_dim}")
        if len(active_scales) != 3:
            raise ValueError(f"active_scales must contain fine/mid/coarse, got {active_scales}")
        self.active_fine, self.active_mid, self.active_coarse = active_scales
        if not self.active_fine:
            raise ValueError("ScaleGuidedResidualAdapter requires the fine scale")
        if self.active_coarse and not self.active_mid:
            raise ValueError("The coarse residual scale requires the mid scale")
        self.residual_dim = int(residual_dim)

        self.proj_f = ProjectionBlock(main_dim, residual_dim, num_groups=num_groups)
        self.proj_m = ProjectionBlock(main_dim, residual_dim, num_groups=num_groups)
        self.proj_c = ProjectionBlock(main_dim, residual_dim, num_groups=num_groups)
        self.proj_main = ProjectionBlock(main_dim, residual_dim, num_groups=num_groups)

        self.mid_fuse = nn.Sequential(
            nn.Conv3d(residual_dim * 2, residual_dim, kernel_size=1),
            nn.GroupNorm(valid_num_groups(residual_dim, num_groups), residual_dim),
            nn.GELU(),
        )
        self.fine_fuse = nn.Sequential(
            nn.Conv3d(residual_dim * 3, residual_dim, kernel_size=1),
            nn.GroupNorm(valid_num_groups(residual_dim, num_groups), residual_dim),
            nn.GELU(),
            ResidualSTBlock(
                residual_dim,
                num_groups=num_groups,
                dropout=dropout,
            ),
        )

        head_dim = max(16, residual_dim // 2)
        self.residual_head = nn.Sequential(
            nn.Conv3d(residual_dim, head_dim, kernel_size=3, padding=1),
            nn.GroupNorm(valid_num_groups(head_dim, num_groups), head_dim),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(head_dim, out_channels, kernel_size=1),
        )
        if zero_init:
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)

    def _zeros_at_scale(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(
            reference.shape[0],
            self.residual_dim,
            *reference.shape[-3:],
        )

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_c: torch.Tensor,
        h_main: torch.Tensor,
        scale_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if scale_weight.ndim != 2 or scale_weight.shape[1] != 3:
            raise ValueError(
                f"Expected scale_weight [B,3], got {tuple(scale_weight.shape)}"
            )
        w_f = scale_weight[:, 0].view(-1, 1, 1, 1, 1)
        w_m = scale_weight[:, 1].view(-1, 1, 1, 1, 1)
        w_c = scale_weight[:, 2].view(-1, 1, 1, 1, 1)

        p_f_expert = w_f * self.proj_f(z_f)
        p_f_main = self.proj_main(h_main)

        if self.active_coarse:
            p_c = w_c * self.proj_c(z_c)
            p_c_up = F.interpolate(
                p_c,
                size=z_m.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        else:
            # Do not evaluate proj_c: inactive coarse features cannot affect output
            # and receive no residual-branch gradient.
            p_c = self._zeros_at_scale(z_c)
            p_c_up = self._zeros_at_scale(z_m)

        if self.active_mid:
            p_m_local = w_m * self.proj_m(z_m)
            p_m = self.mid_fuse(torch.cat((p_m_local, p_c_up), dim=1))
        else:
            # This also prevents fusion biases from creating an inactive mid signal.
            p_m = self._zeros_at_scale(z_m)

        p_m_up = F.interpolate(
            p_m,
            size=z_f.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        p_f = self.fine_fuse(torch.cat((p_f_expert, p_f_main, p_m_up), dim=1))
        delta_raw = self.residual_head(p_f)
        return {
            "delta_raw": delta_raw,
            "residual_fine_feature": p_f,
            "residual_mid_feature": p_m,
            "residual_coarse_feature": p_c,
        }

