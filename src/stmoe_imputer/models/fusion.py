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


class ReliabilityAwareScaleGate(nn.Module):
    def __init__(
        self,
        dim: int,
        stat_dim: int = 5,
        hidden_dim: int = 128,
        num_scales: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        input_dim = dim * 3 + stat_dim * 3 + 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_scales),
        )

    def forward(
        self,
        h_f: torch.Tensor,
        h_m: torch.Tensor,
        h_c: torch.Tensor,
        q_f: torch.Tensor,
        q_m: torch.Tensor,
        q_c: torch.Tensor,
        r_m: torch.Tensor | None = None,
        r_c: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = h_f.shape[0]
        device = h_f.device
        p_f = h_f.mean(dim=(2, 3, 4))
        p_m = h_m.mean(dim=(2, 3, 4))
        p_c = h_c.mean(dim=(2, 3, 4))

        if r_m is None:
            r_m_mean = torch.ones(batch_size, 1, device=device, dtype=h_f.dtype)
        else:
            r_m_mean = r_m.mean(dim=(1, 2, 3, 4), keepdim=False).view(batch_size, 1)

        if r_c is None:
            r_c_mean = torch.ones(batch_size, 1, device=device, dtype=h_f.dtype)
        else:
            r_c_mean = r_c.mean(dim=(1, 2, 3, 4), keepdim=False).view(batch_size, 1)

        gate_input = torch.cat([p_f, p_m, p_c, q_f, q_m, q_c, r_m_mean, r_c_mean], dim=-1)
        logits = self.mlp(gate_input)
        if active_mask is not None:
            logits = logits.masked_fill(~active_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)


class GatedCrossScaleSharedExpert(nn.Module):
    def __init__(
        self,
        dim: int,
        stat_dim: int = 5,
        num_groups: int = 8,
        dropout: float = 0.0,
        use_scale_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_scale_gate = use_scale_gate
        self.scale_gate = ReliabilityAwareScaleGate(
            dim=dim,
            stat_dim=stat_dim,
            hidden_dim=max(dim * 2, 32),
            dropout=dropout,
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(dim * 3, dim, kernel_size=1),
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
        )

    def forward(
        self,
        h_f: torch.Tensor,
        h_m: torch.Tensor,
        h_c: torch.Tensor,
        q_f: torch.Tensor,
        q_m: torch.Tensor,
        q_c: torch.Tensor,
        r_m: torch.Tensor | None = None,
        r_c: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = h_f.shape[0]
        target_size = h_f.shape[-3:]
        h_m_up = F.interpolate(h_m, size=target_size, mode="trilinear", align_corners=False)
        h_c_up = F.interpolate(h_c, size=target_size, mode="trilinear", align_corners=False)

        if active_mask is None:
            active_mask = torch.ones(batch_size, 3, device=h_f.device, dtype=torch.bool)

        if self.use_scale_gate:
            scale_weight = self.scale_gate(
                h_f=h_f,
                h_m=h_m,
                h_c=h_c,
                q_f=q_f,
                q_m=q_m,
                q_c=q_c,
                r_m=r_m,
                r_c=r_c,
                active_mask=active_mask,
            )
        else:
            scale_weight = active_mask.to(dtype=h_f.dtype)
            scale_weight = scale_weight / scale_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        w_f = scale_weight[:, 0].view(batch_size, 1, 1, 1, 1)
        w_m = scale_weight[:, 1].view(batch_size, 1, 1, 1, 1)
        w_c = scale_weight[:, 2].view(batch_size, 1, 1, 1, 1)
        z_shared = self.fuse(torch.cat([w_f * h_f, w_m * h_m_up, w_c * h_c_up], dim=1))
        return z_shared, h_m_up, h_c_up, scale_weight


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
        self.up_m_to_f = LearnableUpsample3D(dim, num_groups=num_groups, dropout=dropout)
        self.up_mc_to_f = LearnableUpsample3D(dim, num_groups=num_groups, dropout=dropout)
        self.fuse_f_m = GatedFusion2(dim, num_groups=num_groups, dropout=dropout)
        self.fuse_f_mc = GatedFusion2(dim, num_groups=num_groups, dropout=dropout)
        self.fine_refine = ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout)

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor | None = None,
        z_c: torch.Tensor | None = None,
        scale_mode: str = "fine_mid_coarse",
    ) -> dict[str, torch.Tensor]:
        _, _, t, h, w = z_f.shape
        gate_16_zero = torch.zeros(
            z_f.shape[0],
            2,
            t,
            z_m.shape[3] if z_m is not None else h,
            z_m.shape[4] if z_m is not None else w,
            device=z_f.device,
            dtype=z_f.dtype,
        )
        gate_32_zero = torch.zeros(z_f.shape[0], 2, t, h, w, device=z_f.device, dtype=z_f.dtype)

        if scale_mode == "fine":
            h_route = self.fine_refine(z_f)
            return {
                "h_route": h_route,
                "z_c_to_m": torch.zeros_like(z_m) if z_m is not None else torch.zeros_like(z_f),
                "z_m_to_f": torch.zeros_like(z_f),
                "z_mc": torch.zeros_like(z_m) if z_m is not None else torch.zeros_like(z_f),
                "z_mc_to_f": torch.zeros_like(z_f),
                "gate_16": gate_16_zero,
                "gate_32_route": gate_32_zero,
            }

        if z_m is None:
            raise ValueError(f"z_m is required for scale_mode={scale_mode}")

        if scale_mode == "fine_mid":
            z_m_to_f = self.up_m_to_f(z_m, target_size=(t, h, w))
            h_route, gate_32_route = self.fuse_f_m(z_f, z_m_to_f)
            return {
                "h_route": h_route,
                "z_c_to_m": torch.zeros_like(z_m),
                "z_m_to_f": z_m_to_f,
                "z_mc": z_m,
                "z_mc_to_f": z_m_to_f,
                "gate_16": gate_16_zero,
                "gate_32_route": gate_32_route,
            }

        if scale_mode != "fine_mid_coarse":
            raise ValueError(f"Unknown scale_mode: {scale_mode}")
        if z_c is None:
            raise ValueError("z_c is required for scale_mode=fine_mid_coarse")
        _, _, _, h_m, w_m = z_m.shape

        z_c_to_m = self.up_c_to_m(z_c, target_size=(t, h_m, w_m))
        z_mc, gate_16 = self.fuse_m_c(z_m, z_c_to_m)
        z_mc_to_f = self.up_mc_to_f(z_mc, target_size=(t, h, w))
        h_route, gate_32_route = self.fuse_f_mc(z_f, z_mc_to_f)

        return {
            "h_route": h_route,
            "z_c_to_m": z_c_to_m,
            "z_m_to_f": torch.zeros_like(z_f),
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
