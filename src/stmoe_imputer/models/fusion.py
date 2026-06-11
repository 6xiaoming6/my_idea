from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ResidualSTBlock, valid_num_groups


class LearnableUpsample3D(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        groups = valid_num_groups(dim, num_groups)
        self.refine = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim),
            nn.GELU(),
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
        )

    def forward(self, x: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        return self.refine(x)


class GatedFusion2(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Conv3d(dim * 2, 2, kernel_size=1)
        self.refine = ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.softmax(self.gate(torch.cat([x1, x2], dim=1)), dim=1)
        out = gate[:, 0:1] * x1 + gate[:, 1:2] * x2
        return self.refine(out), gate


class GatedFusion3(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Conv3d(dim * 3, 3, kernel_size=1)
        self.refine = ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.softmax(self.gate(torch.cat([x1, x2, x3], dim=1)), dim=1)
        out = gate[:, 0:1] * x1 + gate[:, 1:2] * x2 + gate[:, 2:3] * x3
        return self.refine(out), gate


class ProgressiveScaleGatedFusion(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.up_c_to_m = LearnableUpsample3D(dim, num_groups=num_groups, dropout=dropout)
        self.fuse_m_c = GatedFusion2(dim, num_groups=num_groups, dropout=dropout)
        self.up_mc_to_f = LearnableUpsample3D(dim, num_groups=num_groups, dropout=dropout)
        self.fuse_f_mc_shared = GatedFusion3(dim, num_groups=num_groups, dropout=dropout)

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_c: torch.Tensor,
        z_shared: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _, _, t, h, w = z_f.shape
        _, _, _, h_m, w_m = z_m.shape

        z_c_to_m = self.up_c_to_m(z_c, target_size=(t, h_m, w_m))
        z_mc, gate_16 = self.fuse_m_c(z_m, z_c_to_m)
        z_mc_to_f = self.up_mc_to_f(z_mc, target_size=(t, h, w))
        h_main, gate_32 = self.fuse_f_mc_shared(z_f, z_mc_to_f, z_shared)

        return {
            "h_main": h_main,
            "z_c_to_m": z_c_to_m,
            "z_mc": z_mc,
            "z_mc_to_f": z_mc_to_f,
            "gate_16": gate_16,
            "gate_32": gate_32,
        }


class ProgressiveRouteFusion(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.up_c_to_m = LearnableUpsample3D(dim, num_groups=num_groups, dropout=dropout)
        self.fuse_m_c = GatedFusion2(dim, num_groups=num_groups, dropout=dropout)
        self.up_mc_to_f = LearnableUpsample3D(dim, num_groups=num_groups, dropout=dropout)
        self.fuse_f_mc = GatedFusion2(dim, num_groups=num_groups, dropout=dropout)

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_c: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _, _, t, h, w = z_f.shape
        _, _, _, h_m, w_m = z_m.shape

        z_c_to_m = self.up_c_to_m(z_c, target_size=(t, h_m, w_m))
        z_mc, gate_16 = self.fuse_m_c(z_m, z_c_to_m)
        z_mc_to_f = self.up_mc_to_f(z_mc, target_size=(t, h, w))
        h_route, gate_32_route = self.fuse_f_mc(z_f, z_mc_to_f)

        return {
            "h_route": h_route,
            "z_c_to_m": z_c_to_m,
            "z_mc": z_mc,
            "z_mc_to_f": z_mc_to_f,
            "gate_16": gate_16,
            "gate_32_route": gate_32_route,
        }


class SharedRoutedResidualFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_groups: int = 8,
        dropout: float = 0.0,
        route_gamma_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.shared_refine = nn.Sequential(
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
        )
        self.route_proj = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
        )
        self.route_gamma = nn.Parameter(torch.tensor(float(route_gamma_init)))

    def refine_shared(self, z_shared: torch.Tensor) -> torch.Tensor:
        return self.shared_refine(z_shared)

    def project_route(self, h_route: torch.Tensor) -> torch.Tensor:
        return self.route_proj(h_route)

    def forward(self, z_shared: torch.Tensor, h_route: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_shared = self.refine_shared(z_shared)
        h_route_proj = self.project_route(h_route)
        gamma = torch.sigmoid(self.route_gamma)
        h_main = h_shared + gamma * h_route_proj
        return h_main, h_shared, h_route_proj
