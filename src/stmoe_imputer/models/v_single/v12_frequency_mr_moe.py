from __future__ import annotations

import torch
from torch import nn

from ..router import QualityRouter, uniform_gate
from .frequency_decomposition import FrequencyDecomposition
from .frequency_experts import (
    BoundaryExpert,
    CoarseContextExpert,
    DynamicDetailExpert,
    LocalDetailExpert,
    RoutedFrequencyExpertPool,
    SmoothTrendExpert,
    TemporalTrendExpert,
)


class FrequencyGate(nn.Module):
    def __init__(
        self,
        q_dim: int = 5,
        hidden_dim: int = 64,
        eta_init: float = -3.0,
        eta_trainable: bool = True,
        eta_fixed: float = 0.05,
        zero_init: bool = True,
        branch_mode: str = "low_plus_high",
    ) -> None:
        super().__init__()
        if branch_mode not in {"low_plus_high", "low_only", "high_only"}:
            raise ValueError(f"Unsupported frequency branch mode: {branch_mode}")
        if not 0.0 <= eta_fixed <= 1.0:
            raise ValueError(f"eta_fixed must be in [0,1], got {eta_fixed}")
        self.branch_mode = branch_mode
        self.eta_trainable = eta_trainable
        if eta_trainable:
            self.high_eta_logit = nn.Parameter(torch.tensor(float(eta_init)))
        else:
            self.register_buffer("high_eta_fixed", torch.tensor(float(eta_fixed)))
        self.mlp = nn.Sequential(
            nn.Linear(q_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def eta_value(self) -> torch.Tensor:
        if self.eta_trainable:
            return torch.sigmoid(self.high_eta_logit)
        return self.high_eta_fixed

    def forward(
        self,
        z_low: torch.Tensor,
        z_high: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        low_energy = z_low.abs().mean(dim=(1, 2, 3, 4)).view(-1, 1)
        high_energy = z_high.abs().mean(dim=(1, 2, 3, 4)).view(-1, 1)
        high_ratio = high_energy / low_energy.clamp_min(1e-6)
        gate = torch.sigmoid(self.mlp(torch.cat([q, low_energy, high_energy], dim=-1)))
        eta = self.eta_value().to(device=z_low.device, dtype=z_low.dtype)

        if self.branch_mode == "low_only":
            low_coefficient = torch.ones_like(gate)
            high_coefficient = torch.zeros_like(gate)
            z = z_low
        elif self.branch_mode == "high_only":
            low_coefficient = torch.zeros_like(gate)
            high_coefficient = torch.ones_like(gate)
            z = z_high
        else:
            low_coefficient = torch.ones_like(gate)
            high_coefficient = eta * gate
            z = z_low + high_coefficient.view(-1, 1, 1, 1, 1) * z_high
        return z, {
            "frequency_high_gate": gate,
            "eta_high": eta.reshape(1),
            "low_energy": low_energy,
            "high_energy": high_energy,
            "high_energy_ratio": high_ratio,
            "low_coefficient": low_coefficient,
            "high_coefficient": high_coefficient,
        }


class FrequencyMultiResolutionExpertPool(nn.Module):
    """Trend/detail MoE for one spatial scale."""

    LOW_EXPERTS = (SmoothTrendExpert, TemporalTrendExpert, CoarseContextExpert)
    HIGH_EXPERTS = (LocalDetailExpert, DynamicDetailExpert, BoundaryExpert)

    def __init__(
        self,
        dim: int,
        q_dim: int = 5,
        low_num_experts: int = 3,
        high_num_experts: int = 3,
        low_top_k: int = 1,
        high_top_k: int = 1,
        num_groups: int = 8,
        dropout: float = 0.0,
        frequency_mode: str = "avg_residual",
        frequency_kernel_t: int = 3,
        frequency_kernel_s: int = 3,
        use_fft: bool = False,
        high_eta_init: float = -3.0,
        high_eta_trainable: bool = True,
        high_eta_fixed: float = 0.05,
        frequency_gate_zero_init: bool = True,
        branch_mode: str = "low_plus_high",
    ) -> None:
        super().__init__()
        if not 1 <= low_num_experts <= len(self.LOW_EXPERTS):
            raise ValueError(f"low_num_experts must be in [1,3], got {low_num_experts}")
        if not 1 <= high_num_experts <= len(self.HIGH_EXPERTS):
            raise ValueError(f"high_num_experts must be in [1,3], got {high_num_experts}")
        self.low_num_experts = low_num_experts
        self.high_num_experts = high_num_experts
        self.branch_mode = branch_mode
        self.decomposition = FrequencyDecomposition(
            kernel_t=frequency_kernel_t,
            kernel_s=frequency_kernel_s,
            mode=frequency_mode,
            use_fft=use_fft,
        )
        self.low_router = QualityRouter(dim, q_dim, low_num_experts)
        self.high_router = QualityRouter(dim, q_dim, high_num_experts)
        self.low_pool = RoutedFrequencyExpertPool(
            [factory(dim, num_groups=num_groups, dropout=dropout) for factory in self.LOW_EXPERTS[:low_num_experts]],
            top_k=low_top_k,
        )
        self.high_pool = RoutedFrequencyExpertPool(
            [factory(dim, num_groups=num_groups, dropout=dropout) for factory in self.HIGH_EXPERTS[:high_num_experts]],
            top_k=high_top_k,
        )
        self.frequency_gate = FrequencyGate(
            q_dim=q_dim,
            hidden_dim=dim,
            eta_init=high_eta_init,
            eta_trainable=high_eta_trainable,
            eta_fixed=high_eta_fixed,
            zero_init=frequency_gate_zero_init,
            branch_mode=branch_mode,
        )

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        q: torch.Tensor,
        scale_embed_vec: torch.Tensor,
        routing_mode: str = "topk",
        use_router: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        h_low, h_high = self.decomposition(h)
        if use_router:
            low_gate = self.low_router(h_low, q, scale_embed_vec)
            high_gate = self.high_router(h_high, q, scale_embed_vec)
        else:
            low_gate = uniform_gate(h.shape[0], self.low_num_experts, h.device, h.dtype)
            high_gate = uniform_gate(h.shape[0], self.high_num_experts, h.device, h.dtype)
        z_low, low_idx, low_w, low_selected = self.low_pool(
            h_low, low_gate, mask, routing_mode=routing_mode
        )
        z_high, high_idx, high_w, high_selected = self.high_pool(
            h_high, high_gate, mask, routing_mode=routing_mode
        )
        z, frequency = self.frequency_gate(z_low, z_high, q)

        if self.branch_mode == "low_only":
            combined_gate = low_gate
            combined_selected = low_selected
            top_indices = low_idx
            top_weights = low_w
        elif self.branch_mode == "high_only":
            combined_gate = high_gate
            combined_selected = high_selected
            top_indices = high_idx
            top_weights = high_w
        else:
            combined_gate = torch.cat([0.5 * low_gate, 0.5 * high_gate], dim=-1)
            combined_selected = torch.cat([low_selected, high_selected], dim=-1)
            top_indices = torch.cat([low_idx, high_idx + self.low_num_experts], dim=-1)
            top_weights = torch.cat([0.5 * low_w, 0.5 * high_w], dim=-1)

        aux = {
            **frequency,
            "h_low_energy": h_low.abs().mean(dim=(1, 2, 3, 4)).view(-1, 1),
            "h_high_energy": h_high.abs().mean(dim=(1, 2, 3, 4)).view(-1, 1),
            "low_gate": low_gate,
            "high_router_gate": high_gate,
            "low_selected": low_selected,
            "high_selected": high_selected,
            "combined_gate": combined_gate,
        }
        return z, top_indices, top_weights, combined_selected, aux
