from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        return -20.0
    return math.log(math.expm1(value))


class HierarchicalScaleExpertRouter(nn.Module):
    """Jointly route scales, experts within each scale, and the routed branch."""

    def __init__(
        self,
        dim: int = 64,
        q_dim: int = 5,
        num_experts: int = 4,
        local_dim: int = 32,
        global_dim: int = 64,
        scale_embed_dim: int = 8,
        scale_temperature: float = 1.0,
        reliability_prior_enabled: bool = True,
        reliability_prior_init: float = 1.0,
        route_gate_bias: float = -3.0,
        route_gate_zero_init: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(dim, q_dim, num_experts, local_dim, global_dim, scale_embed_dim) <= 0:
            raise ValueError("all router dimensions must be positive")
        if scale_temperature <= 0:
            raise ValueError("scale_temperature must be positive")
        if reliability_prior_init < 0:
            raise ValueError("reliability_prior_init must be non-negative")

        self.num_experts = num_experts
        self.scale_temperature = float(scale_temperature)
        self.reliability_prior_enabled = bool(reliability_prior_enabled)
        self.eps = float(eps)
        self.scale_embedding = nn.Embedding(3, scale_embed_dim)
        descriptor_dim = dim + q_dim + 1 + scale_embed_dim
        self.local_projector = nn.Sequential(
            nn.Linear(descriptor_dim, local_dim),
            nn.LayerNorm(local_dim),
            nn.GELU(),
        )
        self.global_projector = nn.Sequential(
            nn.Linear(local_dim * 3, global_dim),
            nn.LayerNorm(global_dim),
            nn.GELU(),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(global_dim, local_dim),
            nn.GELU(),
            nn.Linear(local_dim, 3),
        )
        self.expert_head = nn.Sequential(
            nn.Linear(local_dim + global_dim, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, num_experts),
        )
        self.route_head = nn.Sequential(
            nn.Linear(global_dim, local_dim),
            nn.GELU(),
            nn.Linear(local_dim, 1, bias=False),
        )
        self.register_buffer(
            "route_gate_bias", torch.tensor(float(route_gate_bias)), persistent=True
        )
        if route_gate_zero_init:
            nn.init.zeros_(self.route_head[-1].weight)

        if self.reliability_prior_enabled:
            self.reliability_strength_raw = nn.Parameter(
                torch.tensor(_inverse_softplus(float(reliability_prior_init)))
            )
        else:
            self.register_buffer(
                "reliability_strength_raw", torch.tensor(-20.0), persistent=True
            )

    @property
    def reliability_strength(self) -> torch.Tensor:
        if not self.reliability_prior_enabled:
            return self.reliability_strength_raw.new_zeros(())
        return F.softplus(self.reliability_strength_raw)

    @staticmethod
    def _check_scale_inputs(
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        reliabilities: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        active_scale_mask: torch.Tensor,
    ) -> None:
        batch_size = features[0].shape[0]
        if active_scale_mask.shape != (batch_size, 3):
            raise ValueError(
                f"active_scale_mask must have shape {(batch_size, 3)}, "
                f"got {tuple(active_scale_mask.shape)}"
            )
        if not active_scale_mask.any(dim=1).all():
            raise ValueError("each sample must have at least one active scale")
        for feature, stats, reliability in zip(features, statistics, reliabilities):
            if feature.ndim != 5 or feature.shape[0] != batch_size:
                raise ValueError("scale features must have shape [B,D,T,H,W]")
            if stats.ndim != 2 or stats.shape[0] != batch_size:
                raise ValueError("scale statistics must have shape [B,Q]")
            if reliability.shape != (batch_size, 1):
                raise ValueError("scale reliability must have shape [B,1]")

    def forward(
        self,
        h_f: torch.Tensor,
        h_m: torch.Tensor,
        h_c: torch.Tensor,
        q_f: torch.Tensor,
        q_m: torch.Tensor,
        q_c: torch.Tensor,
        rel_f: torch.Tensor,
        rel_m: torch.Tensor,
        rel_c: torch.Tensor,
        active_scale_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        features = (h_f, h_m, h_c)
        statistics = (q_f, q_m, q_c)
        reliabilities = (rel_f, rel_m, rel_c)
        self._check_scale_inputs(features, statistics, reliabilities, active_scale_mask)
        active_scale_mask = active_scale_mask.to(device=h_f.device, dtype=torch.bool)
        batch_size = h_f.shape[0]

        local_tokens: list[torch.Tensor] = []
        for scale_index, (feature, stats, reliability) in enumerate(
            zip(features, statistics, reliabilities)
        ):
            pooled = feature.mean(dim=(2, 3, 4))
            scale_ids = torch.full(
                (batch_size,), scale_index, device=h_f.device, dtype=torch.long
            )
            scale_embedding = self.scale_embedding(scale_ids).to(dtype=pooled.dtype)
            descriptor = torch.cat(
                [pooled, stats.to(pooled.dtype), reliability.to(pooled.dtype), scale_embedding],
                dim=-1,
            )
            token = self.local_projector(descriptor)
            token = token * active_scale_mask[:, scale_index : scale_index + 1].to(
                token.dtype
            )
            local_tokens.append(token)

        global_context = self.global_projector(torch.cat(local_tokens, dim=-1))
        learned_scale_logits = self.scale_head(global_context)
        reliability = torch.cat(reliabilities, dim=1).to(learned_scale_logits.dtype)
        reliability = reliability.clamp_min(self.eps).clamp_max(1.0)
        scale_logits = learned_scale_logits + self.reliability_strength * reliability.log()
        scale_logits = scale_logits.masked_fill(
            ~active_scale_mask, torch.finfo(scale_logits.dtype).min
        )
        scale_weight = torch.softmax(scale_logits / self.scale_temperature, dim=-1)

        expert_gates = []
        for token in local_tokens:
            expert_logits = self.expert_head(torch.cat([token, global_context], dim=-1))
            expert_gates.append(torch.softmax(expert_logits, dim=-1))

        route_logit = self.route_head(global_context) + self.route_gate_bias.to(
            dtype=global_context.dtype
        )
        route_branch_gate = torch.sigmoid(route_logit).view(batch_size, 1, 1, 1, 1)

        tensors = [scale_weight, route_branch_gate, *expert_gates, global_context]
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise FloatingPointError("V17 router produced NaN or Inf")

        return {
            "scale_weight": scale_weight,
            "expert_gate_f": expert_gates[0],
            "expert_gate_m": expert_gates[1],
            "expert_gate_c": expert_gates[2],
            "route_branch_gate": route_branch_gate,
            "global_context": global_context,
            "scale_tokens": {
                "fine": local_tokens[0],
                "mid": local_tokens[1],
                "coarse": local_tokens[2],
            },
            "reliability_strength": self.reliability_strength,
        }
