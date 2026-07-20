from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..blocks import ResidualSTBlock


class FinePreservedScaleWeight(nn.Module):
    """Apply the active-scale mask while reserving a minimum Fine contribution."""

    def __init__(self, fine_floor: float = 0.25) -> None:
        super().__init__()
        if not 0.0 <= fine_floor < 1.0:
            raise ValueError("fine_floor must lie in [0, 1)")
        self.fine_floor = float(fine_floor)

    def forward(
        self,
        scale_weight: torch.Tensor,
        active_scale_mask: torch.Tensor,
    ) -> torch.Tensor:
        if scale_weight.ndim != 2 or scale_weight.shape[1] != 3:
            raise ValueError("scale_weight must have shape [B,3]")
        if active_scale_mask.shape != scale_weight.shape:
            raise ValueError("active_scale_mask must match scale_weight")
        if not active_scale_mask[:, 0].all():
            raise ValueError("Fine scale must remain active in V17")
        weight = scale_weight * active_scale_mask.to(scale_weight.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        safe_weight = (1.0 - self.fine_floor) * weight
        safe_weight = safe_weight.clone()
        safe_weight[:, 0] = safe_weight[:, 0] + self.fine_floor
        safe_weight = safe_weight * active_scale_mask.to(safe_weight.dtype)
        safe_weight = safe_weight / safe_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return safe_weight


class FinePreservedParallelRouteFusion(nn.Module):
    """Fuse all active routed scales directly at Fine resolution."""

    def __init__(
        self,
        dim: int = 64,
        num_groups: int = 8,
        dropout: float = 0.0,
        mid_projection: bool = True,
        coarse_projection: bool = True,
    ) -> None:
        super().__init__()
        self.mid_projection = (
            nn.Conv3d(dim, dim, kernel_size=1) if mid_projection else nn.Identity()
        )
        self.coarse_projection = (
            nn.Conv3d(dim, dim, kernel_size=1) if coarse_projection else nn.Identity()
        )
        self.refine = ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout)

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_c: torch.Tensor,
        scale_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = z_f.shape[0]
        if scale_weight.shape != (batch_size, 3):
            raise ValueError(
                f"scale_weight must have shape {(batch_size, 3)}, "
                f"got {tuple(scale_weight.shape)}"
            )
        target_size = z_f.shape[-3:]
        z_m_up = self.mid_projection(
            F.interpolate(z_m, size=target_size, mode="trilinear", align_corners=False)
        )
        z_c_up = self.coarse_projection(
            F.interpolate(z_c, size=target_size, mode="trilinear", align_corners=False)
        )
        weights = scale_weight.view(batch_size, 3, 1, 1, 1, 1)
        stacked = torch.stack([z_f, z_m_up, z_c_up], dim=1)
        h_route_mix = (weights * stacked).sum(dim=1)
        h_route = self.refine(h_route_mix)
        return {
            "h_route": h_route,
            "h_route_mix": h_route_mix,
            "z_m_to_f": z_m_up,
            "z_c_to_f": z_c_up,
        }
