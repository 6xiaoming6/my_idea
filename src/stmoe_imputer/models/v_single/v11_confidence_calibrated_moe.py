from __future__ import annotations

import torch
from torch import nn

from ..experts import STExpert
from .confidence_heads import CalibratedWeightComposer, ExpertConfidenceHead


class ConfidenceCalibratedExpertPool(nn.Module):
    """Homogeneous ST experts with sample-level confidence calibration.

    The first four returned values match ``TopKRoutedExpertPool``.  The fifth
    value contains v11 diagnostics used by the backbone and logger.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        num_groups: int = 8,
        dropout: float = 0.0,
        confidence_hidden_dim: int | None = None,
        confidence_dropout: float = 0.1,
        confidence_beta_init: float = 0.05,
        confidence_beta_max: float = 0.5,
        confidence_zero_init: bool = True,
        confidence_use_mask: bool = True,
        confidence_use_input_feature: bool = True,
        confidence_enabled: bool = True,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        self.num_experts = num_experts
        self.top_k = min(max(1, top_k), num_experts)
        self.confidence_enabled = confidence_enabled
        self.experts = nn.ModuleList(
            [STExpert(dim, num_groups=num_groups, dropout=dropout) for _ in range(num_experts)]
        )
        self.confidence_heads = nn.ModuleList(
            [
                ExpertConfidenceHead(
                    dim=dim,
                    hidden_dim=confidence_hidden_dim,
                    dropout=confidence_dropout,
                    use_input_feature=confidence_use_input_feature,
                    use_mask=confidence_use_mask,
                    zero_init=confidence_zero_init,
                )
                for _ in range(num_experts)
            ]
        )
        self.composer = CalibratedWeightComposer(
            beta_init=confidence_beta_init,
            beta_max=confidence_beta_max,
        )

    def forward(
        self,
        h: torch.Tensor,
        gate: torch.Tensor,
        routing_mode: str = "soft",
        mask: torch.Tensor | None = None,
        router_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if gate.shape[-1] != self.num_experts:
            raise ValueError(
                f"gate has {gate.shape[-1]} experts, expected {self.num_experts}"
            )
        router_weight = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        if router_logits is None:
            router_logits = torch.log(router_weight.clamp_min(1e-6))

        outputs = [expert(h) for expert in self.experts]
        expert_outputs = torch.stack(outputs, dim=1)
        confidence_pairs = [
            head(expert_output, h, mask)
            for head, expert_output in zip(self.confidence_heads, outputs)
        ]
        confidence = torch.cat([pair[0] for pair in confidence_pairs], dim=1)
        confidence_logits = torch.cat([pair[1] for pair in confidence_pairs], dim=1)
        calibrated_weight, calibrated_logits, beta = self.composer(
            router_logits,
            confidence,
            enabled=self.confidence_enabled,
        )
        if mask is None:
            missing_rate = h.new_zeros((h.shape[0], 1))
        else:
            missing_rate = 1.0 - mask.to(dtype=h.dtype).mean(dim=(1, 2, 3, 4)).view(-1, 1)

        batch_size = h.shape[0]
        if routing_mode in {"soft", "dense"}:
            effective_weight = calibrated_weight
            z = (effective_weight[:, :, None, None, None, None] * expert_outputs).sum(dim=1)
            expert_range = torch.arange(self.num_experts, device=h.device, dtype=torch.long)
            top_indices = expert_range.view(1, -1).expand(batch_size, -1)
            top_weights = effective_weight
            selected_mask = torch.ones_like(effective_weight)
        elif routing_mode in {"topk", "soft_topk"}:
            top_values, top_indices = torch.topk(calibrated_weight, k=self.top_k, dim=-1)
            top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            selected_mask = torch.zeros_like(calibrated_weight)
            selected_mask.scatter_(1, top_indices, 1.0)
            effective_weight = calibrated_weight * selected_mask
            effective_weight = effective_weight / effective_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            z = (effective_weight[:, :, None, None, None, None] * expert_outputs).sum(dim=1)
        else:
            raise ValueError(f"Unsupported confidence-calibrated routing_mode: {routing_mode}")

        aux = {
            "router_logits": router_logits,
            "router_weight": router_weight,
            "confidence": confidence,
            "confidence_logits": confidence_logits,
            "calibrated_logits": calibrated_logits,
            "calibrated_weight": calibrated_weight,
            "effective_weight": effective_weight,
            "beta_conf": beta,
            "missing_rate": missing_rate,
        }
        return z, top_indices, top_weights, selected_mask, aux
