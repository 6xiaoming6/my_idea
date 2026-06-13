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


class ExpertEnhancedSharedInput(nn.Module):
    def __init__(
        self,
        dim: int,
        mode: str = "pre",
        beta_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.mode = mode
        beta = torch.tensor(float(beta_init)).clamp(1e-4, 1.0 - 1e-4)
        self.beta = nn.Parameter(torch.full((3,), torch.logit(beta).item()))
        if mode == "concat_hz":
            self.proj_f = nn.Conv3d(dim * 2, dim, kernel_size=1)
            self.proj_m = nn.Conv3d(dim * 2, dim, kernel_size=1)
            self.proj_c = nn.Conv3d(dim * 2, dim, kernel_size=1)

    def beta_values(self) -> torch.Tensor:
        return torch.sigmoid(self.beta)

    def forward(
        self,
        h_f: torch.Tensor,
        h_m: torch.Tensor,
        h_c: torch.Tensor,
        z_f: torch.Tensor | None = None,
        z_m: torch.Tensor | None = None,
        z_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "pre" or z_f is None or z_m is None or z_c is None:
            return h_f, h_m, h_c
        if self.mode == "post":
            return z_f, z_m, z_c
        if self.mode == "hybrid":
            beta = self.beta_values()
            return h_f + beta[0] * z_f, h_m + beta[1] * z_m, h_c + beta[2] * z_c
        if self.mode == "concat_hz":
            return (
                self.proj_f(torch.cat([h_f, z_f], dim=1)),
                self.proj_m(torch.cat([h_m, z_m], dim=1)),
                self.proj_c(torch.cat([h_c, z_c], dim=1)),
            )
        raise ValueError(f"Unsupported shared_input_mode: {self.mode}")


class AdaptiveBranchGate(nn.Module):
    def __init__(
        self,
        dim: int,
        q_dim: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        init_mode: str = "balanced",
    ) -> None:
        super().__init__()
        self.init_mode = init_mode
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2 + q_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        if self.init_mode == "shared_bias":
            last.bias.data = torch.tensor([1.0, -1.0], dtype=last.bias.dtype)
        elif self.init_mode == "route_bias":
            last.bias.data = torch.tensor([-1.0, 1.0], dtype=last.bias.dtype)
        elif self.init_mode == "balanced":
            nn.init.zeros_(last.bias)
        else:
            raise ValueError(f"Unsupported branch_gate_init: {self.init_mode}")

    def forward(
        self,
        h_shared: torch.Tensor,
        h_route: torch.Tensor,
        q_f: torch.Tensor,
        scale_gate: torch.Tensor,
    ) -> torch.Tensor:
        p_shared = h_shared.mean(dim=(2, 3, 4))
        p_route = h_route.mean(dim=(2, 3, 4))
        gate_input = torch.cat([p_shared, p_route, q_f, scale_gate], dim=-1)
        return torch.softmax(self.mlp(gate_input), dim=-1)


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
        branch_fusion_mode: str = "residual",
        branch_gate_init: str = "balanced",
        q_dim: int = 5,
        route_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.branch_fusion_mode = branch_fusion_mode
        self.shared_refine = nn.Sequential(
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
        )
        self.route_proj = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
        )
        self.route_dropout = nn.Dropout3d(route_dropout) if route_dropout > 0 else nn.Identity()
        self.route_gamma = nn.Parameter(torch.tensor(float(route_gamma_init)))
        self.shared_gamma = nn.Parameter(torch.tensor(float(route_gamma_init)))
        self.branch_gate = AdaptiveBranchGate(
            dim=dim,
            q_dim=q_dim,
            hidden_dim=max(dim * 2, 32),
            dropout=dropout,
            init_mode=branch_gate_init,
        )

    def refine_shared(self, z_shared: torch.Tensor) -> torch.Tensor:
        return self.shared_refine(z_shared)

    def project_route(self, h_route: torch.Tensor) -> torch.Tensor:
        return self.route_proj(h_route)

    def forward(
        self,
        z_shared: torch.Tensor,
        h_route: torch.Tensor,
        q_f: torch.Tensor | None = None,
        scale_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_shared = self.refine_shared(z_shared)
        h_route_proj = self.project_route(h_route)
        h_route_proj = self.route_dropout(h_route_proj)
        if self.branch_fusion_mode in {"residual", "shared_plus_routed_residual"}:
            gamma = torch.sigmoid(self.route_gamma)
            h_main = h_shared + gamma * h_route_proj
            branch_gate = torch.stack([torch.ones_like(gamma), gamma], dim=0).view(1, 2)
            branch_gate = branch_gate.expand(h_shared.shape[0], 2)
            return h_main, h_shared, h_route_proj, branch_gate
        if self.branch_fusion_mode == "routed_primary":
            gamma = torch.sigmoid(self.shared_gamma)
            h_main = h_route_proj + gamma * h_shared
            branch_gate = torch.stack([gamma, torch.ones_like(gamma)], dim=0).view(1, 2)
            branch_gate = branch_gate.expand(h_shared.shape[0], 2)
            return h_main, h_shared, h_route_proj, branch_gate
        if self.branch_fusion_mode == "adaptive_gate":
            if q_f is None or scale_gate is None:
                raise ValueError("q_f and scale_gate are required for adaptive_gate")
            branch_gate = self.branch_gate(h_shared, h_route_proj, q_f, scale_gate)
            w_shared = branch_gate[:, 0].view(h_shared.shape[0], 1, 1, 1, 1)
            w_route = branch_gate[:, 1].view(h_shared.shape[0], 1, 1, 1, 1)
            h_main = w_shared * h_shared + w_route * h_route_proj
            return h_main, h_shared, h_route_proj, branch_gate
        raise ValueError(f"Unsupported branch_fusion_mode: {self.branch_fusion_mode}")
