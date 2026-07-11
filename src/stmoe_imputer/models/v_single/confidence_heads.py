from __future__ import annotations

import math

import torch
from torch import nn


class ExpertConfidenceHead(nn.Module):
    """Estimate one sample-level reliability value for one expert output."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_input_feature: bool = True,
        use_mask: bool = True,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.use_input_feature = use_input_feature
        self.use_mask = use_mask
        input_dim = dim
        if use_input_feature:
            input_dim += dim
        if use_mask:
            input_dim += 1
        hidden_dim = hidden_dim or dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if zero_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        expert_output: torch.Tensor,
        input_feature: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if expert_output.shape != input_feature.shape:
            raise ValueError(
                "expert_output and input_feature must have the same shape, got "
                f"{tuple(expert_output.shape)} and {tuple(input_feature.shape)}"
            )
        pooled = [expert_output.mean(dim=(2, 3, 4))]
        if self.use_input_feature:
            pooled.append(input_feature.mean(dim=(2, 3, 4)))
        if self.use_mask:
            if mask is None:
                observed_ratio = expert_output.new_ones((expert_output.shape[0], 1))
            else:
                if mask.ndim != 5 or mask.shape[0] != expert_output.shape[0] or mask.shape[1] != 1:
                    raise ValueError(
                        f"Expected mask [B,1,T,H,W], got {tuple(mask.shape)} for "
                        f"expert output {tuple(expert_output.shape)}"
                    )
                observed_ratio = mask.to(dtype=expert_output.dtype).mean(dim=(1, 2, 3, 4)).view(-1, 1)
            pooled.append(observed_ratio)
        confidence_logit = self.mlp(torch.cat(pooled, dim=-1))
        return torch.sigmoid(confidence_logit), confidence_logit


class CalibratedWeightComposer(nn.Module):
    """Compose router preference and expert confidence in logit space."""

    def __init__(
        self,
        beta_init: float = 0.05,
        beta_max: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if beta_max <= 0:
            raise ValueError(f"beta_max must be positive, got {beta_max}")
        if not 0 <= beta_init <= beta_max:
            raise ValueError(f"beta_init must be in [0, {beta_max}], got {beta_init}")
        self.beta_max = float(beta_max)
        self.eps = float(eps)

        # A bounded sigmoid keeps beta non-negative.  Exact beta=0 would block
        # all gradients to zero-initialized confidence heads, so zero is
        # represented by a numerically negligible value instead.
        fraction = min(max(beta_init / beta_max, 1e-6), 1.0 - 1e-6)
        raw_init = math.log(fraction / (1.0 - fraction))
        self.beta_conf_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def effective_beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_conf_raw) * self.beta_max

    def forward(
        self,
        router_logits: torch.Tensor,
        confidence: torch.Tensor,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if router_logits.shape != confidence.shape:
            raise ValueError(
                f"router_logits and confidence must match, got {tuple(router_logits.shape)} "
                f"and {tuple(confidence.shape)}"
            )
        beta = self.effective_beta()
        if enabled:
            confidence_adjustment = torch.log(confidence.clamp_min(self.eps))
            calibrated_logits = router_logits + beta * confidence_adjustment
        else:
            calibrated_logits = router_logits
        return torch.softmax(calibrated_logits, dim=-1), calibrated_logits, beta
