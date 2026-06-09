from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ResidualSTBlock, valid_num_groups


class STExpert(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        groups = valid_num_groups(dim, num_groups)
        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim),
            nn.GELU(),
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TopKRoutedExpertPool(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        self.num_experts = num_experts
        self.top_k = min(max(1, top_k), num_experts)
        self.experts = nn.ModuleList(
            [STExpert(dim, num_groups=num_groups, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, h: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        top_values, top_indices = torch.topk(gate, k=self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        expert_outputs = [expert(h) for expert in self.experts]
        z = torch.zeros_like(expert_outputs[0])
        batch_size = h.shape[0]
        for slot in range(self.top_k):
            indices = top_indices[:, slot]
            weights = top_weights[:, slot].view(batch_size, 1, 1, 1, 1)
            for expert_idx, expert_out in enumerate(expert_outputs):
                selected = (indices == expert_idx).to(h.dtype).view(batch_size, 1, 1, 1, 1)
                z = z + selected * weights * expert_out
        return z, top_indices, top_weights


class CrossScaleSharedExpert(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim * 3, dim, kernel_size=1),
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
            ResidualSTBlock(dim, num_groups=num_groups, dropout=dropout),
        )

    def forward(
        self,
        h_f: torch.Tensor,
        h_m: torch.Tensor,
        h_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, t, h, w = h_f.shape
        h_m_up = F.interpolate(h_m, size=(t, h, w), mode="trilinear", align_corners=False)
        h_c_up = F.interpolate(h_c, size=(t, h, w), mode="trilinear", align_corners=False)
        z_shared = self.net(torch.cat([h_f, h_m_up, h_c_up], dim=1))
        return z_shared, h_m_up, h_c_up


Conv3DExpert = STExpert
ExpertMixer = TopKRoutedExpertPool
SharedExpertPool = TopKRoutedExpertPool
ScaleInteractionBlock = CrossScaleSharedExpert
CrossScaleExpert = CrossScaleSharedExpert
