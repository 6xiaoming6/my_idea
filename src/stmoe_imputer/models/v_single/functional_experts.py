from __future__ import annotations

import torch
from torch import nn


FUNCTIONAL_EXPERT_NAMES = ("smooth", "local", "temporal", "missing", "dynamic")


def _zero_init_last_conv(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Conv3d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


class SmoothExpert(nn.Module):
    """Low-frequency spatial smoothing expert."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.smooth = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return h + self.net(self.smooth(h))


class LocalSpatialExpert(nn.Module):
    """Local spatial-neighborhood expert."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=1),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return h + self.net(h)


class TemporalExpert(nn.Module):
    """Temporal trend expert."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return h + self.net(h)


class MissingPatternExpert(nn.Module):
    """Mask-aware missing-pattern expert."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mask_embed = nn.Conv3d(1, dim, kernel_size=1)
        self.net = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = h.new_ones((h.shape[0], 1, h.shape[2], h.shape[3], h.shape[4]))
        mask = mask.to(device=h.device, dtype=h.dtype)
        if mask.shape[2:] != h.shape[2:]:
            raise ValueError(f"mask shape {tuple(mask.shape)} is incompatible with h shape {tuple(h.shape)}")
        mask_embed = self.mask_embed(mask)
        return h + self.net(torch.cat([h, mask_embed], dim=1))


class DynamicExpert(nn.Module):
    """High-frequency dynamic residual expert."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(dim, dim, kernel_size=1),
        )
        _zero_init_last_conv(self.net)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        trend = h.mean(dim=2, keepdim=True)
        return h + self.net(h - trend)


class FunctionalExpertPool(nn.Module):
    """Top-K routed functional expert pool.

    The returned tuple intentionally matches ``TopKRoutedExpertPool``:
    ``z, top_indices, top_weights, selected_mask``.
    """

    def __init__(
        self,
        dim: int,
        top_k: int = 2,
        dropout: float = 0.0,
        expert_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        names = tuple(expert_names or FUNCTIONAL_EXPERT_NAMES)
        if not names:
            raise ValueError("FunctionalExpertPool requires at least one expert.")
        unsupported = sorted(set(names) - set(FUNCTIONAL_EXPERT_NAMES))
        if unsupported:
            raise ValueError(f"Unsupported functional experts: {', '.join(unsupported)}")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate functional experts are not allowed: {names}")

        self.expert_names = names
        self.num_experts = len(names)
        self.top_k = min(max(1, top_k), self.num_experts)
        factories = {
            "smooth": SmoothExpert,
            "local": LocalSpatialExpert,
            "temporal": TemporalExpert,
            "missing": MissingPatternExpert,
            "dynamic": DynamicExpert,
        }
        self.experts = nn.ModuleList([factories[name](dim, dropout=dropout) for name in names])

    def forward(
        self,
        h: torch.Tensor,
        gate: torch.Tensor,
        routing_mode: str = "topk",
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if gate.shape[-1] != self.num_experts:
            raise ValueError(
                f"gate has {gate.shape[-1]} experts, but FunctionalExpertPool "
                f"has {self.num_experts}: {self.expert_names}"
            )
        expert_outputs = torch.stack([expert(h, mask=mask) for expert in self.experts], dim=1)
        batch_size = h.shape[0]

        if routing_mode == "dense":
            weights = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            z = (weights[:, :, None, None, None, None] * expert_outputs).sum(dim=1)
            expert_range = torch.arange(self.num_experts, device=h.device, dtype=torch.long)
            top_indices = expert_range.view(1, -1).expand(batch_size, -1)
            selected_mask = torch.ones_like(weights)
            return z, top_indices, weights, selected_mask

        if routing_mode not in {"topk", "soft_topk"}:
            raise ValueError(f"Unsupported routing_mode: {routing_mode}")

        top_values, top_indices = torch.topk(gate, k=self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        selected_mask = torch.zeros_like(gate)
        selected_mask.scatter_(1, top_indices, 1.0)

        if routing_mode == "soft_topk":
            weights = gate * selected_mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
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
