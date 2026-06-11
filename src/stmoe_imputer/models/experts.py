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

    def forward(
        self,
        h: torch.Tensor,
        gate: torch.Tensor,
        routing_mode: str = "topk",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=1)
        batch_size = h.shape[0]

        if routing_mode == "dense":
            weights = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            z = (weights[:, :, None, None, None, None] * expert_outputs).sum(dim=1)
            expert_range = torch.arange(self.num_experts, device=h.device, dtype=torch.long)
            top_indices = expert_range.view(1, -1).expand(batch_size, -1)
            selected_mask = torch.ones_like(weights)
            return z, top_indices, weights, selected_mask

        if routing_mode in {"topk", "soft_topk"}:
            top_values, top_indices = torch.topk(gate, k=self.top_k, dim=-1)
            top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            selected_mask = torch.zeros_like(gate)
            selected_mask.scatter_(1, top_indices, 1.0)

            if routing_mode == "soft_topk":
                masked_gate = gate * selected_mask
                weights = masked_gate / masked_gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                z = (weights[:, :, None, None, None, None] * expert_outputs).sum(dim=1)
                return z, top_indices, top_weights, selected_mask

            z = torch.zeros_like(expert_outputs[:, 0])
            for slot in range(self.top_k):
                indices = top_indices[:, slot]
                weights = top_weights[:, slot].view(batch_size, 1, 1, 1, 1)
                for expert_idx in range(self.num_experts):
                    selected = (indices == expert_idx).to(h.dtype).view(batch_size, 1, 1, 1, 1)
                    z = z + selected * weights * expert_outputs[:, expert_idx]
            return z, top_indices, top_weights, selected_mask

        raise ValueError(f"Unsupported routing_mode: {routing_mode}")


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
