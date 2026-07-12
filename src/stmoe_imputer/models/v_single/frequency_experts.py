from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..blocks import valid_num_groups


def _zero_init_last_conv(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Conv3d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


def _norm(dim: int, num_groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(valid_num_groups(dim, num_groups), dim)


class SmoothTrendExpert(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            _norm(dim, num_groups),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        context = F.avg_pool3d(
            F.pad(h, (2, 2, 2, 2, 1, 1), mode="replicate"),
            kernel_size=(3, 5, 5),
            stride=1,
        )
        return h + self.net(context)


class TemporalTrendExpert(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            _norm(dim, num_groups),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return h + self.net(h)


class CoarseContextExpert(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            _norm(dim, num_groups),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if h.shape[-2] >= 2 and h.shape[-1] >= 2:
            context = F.avg_pool3d(
                h,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                ceil_mode=True,
            )
            context = F.interpolate(context, size=h.shape[-3:], mode="trilinear", align_corners=False)
        else:
            context = h
        return h + self.net(context)


class LocalDetailExpert(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            _norm(dim, num_groups),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=1),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return h + self.net(h)


class DynamicDetailExpert(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            _norm(dim, num_groups),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=1),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        temporal_residual = h - h.mean(dim=2, keepdim=True)
        return h + self.net(temporal_residual)


class BoundaryExpert(nn.Module):
    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim + 1, dim, kernel_size=3, padding=1),
            _norm(dim, num_groups),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            edge = h.new_zeros((h.shape[0], 1, *h.shape[2:]))
        else:
            mask = mask.to(device=h.device, dtype=h.dtype)
            if mask.shape[0] != h.shape[0] or mask.shape[1] != 1 or mask.shape[2:] != h.shape[2:]:
                raise ValueError(f"Mask {tuple(mask.shape)} is incompatible with {tuple(h.shape)}")
            smooth = F.avg_pool3d(
                F.pad(mask, (1, 1, 1, 1, 0, 0), mode="replicate"),
                kernel_size=(1, 3, 3),
                stride=1,
            )
            edge = (mask - smooth).abs()
        return h + self.net(torch.cat([h, edge], dim=1))


class RoutedFrequencyExpertPool(nn.Module):
    def __init__(self, experts: list[nn.Module], top_k: int = 1) -> None:
        super().__init__()
        if not experts:
            raise ValueError("At least one frequency expert is required")
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.top_k = min(max(1, top_k), self.num_experts)

    def forward(
        self,
        h: torch.Tensor,
        gate: torch.Tensor,
        mask: torch.Tensor | None,
        routing_mode: str = "topk",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if gate.shape[-1] != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} gate values, got {gate.shape[-1]}")
        outputs = torch.stack([expert(h, mask=mask) for expert in self.experts], dim=1)
        batch_size = h.shape[0]
        if routing_mode == "dense":
            weights = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            z = (weights[:, :, None, None, None, None] * outputs).sum(dim=1)
            indices = torch.arange(self.num_experts, device=h.device).view(1, -1).expand(batch_size, -1)
            return z, indices, weights, torch.ones_like(weights)
        if routing_mode not in {"topk", "soft_topk"}:
            raise ValueError(f"Unsupported frequency routing mode: {routing_mode}")
        top_values, top_indices = torch.topk(gate, k=self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        selected = torch.zeros_like(gate)
        selected.scatter_(1, top_indices, 1.0)
        effective = gate * selected
        effective = effective / effective.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        z = (effective[:, :, None, None, None, None] * outputs).sum(dim=1)
        return z, top_indices, top_weights, selected
