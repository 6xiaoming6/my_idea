from __future__ import annotations

import torch
from torch import nn


def _zero_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


class ObservedConsistencyEvaluator(nn.Module):
    """Target-free sample statistics measured only at observed positions."""

    output_dim = 5

    def forward(
        self,
        x_base: torch.Tensor,
        x_ctf: torch.Tensor,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        base = x_base.detach().float()
        ctf = x_ctf.detach().float()
        obs = x_obs.detach().float()
        observed = mask.detach().float().expand_as(base)
        count = observed.flatten(1).sum(dim=1).clamp_min(1.0)
        base_error = ((base - obs).abs() * observed).flatten(1).sum(dim=1) / count
        ctf_error = ((ctf - obs).abs() * observed).flatten(1).sum(dim=1) / count
        delta = ((ctf - base).abs() * observed).flatten(1)
        delta_mean = delta.sum(dim=1) / count
        observed_flat = observed.flatten(1).bool()
        delta_q95 = torch.stack([
            torch.quantile(values[valid], 0.95) if valid.any() else values.new_zeros(())
            for values, valid in zip(delta, observed_flat)
        ])
        return torch.stack(
            (base_error, ctf_error, ctf_error - base_error, delta_mean, delta_q95), dim=1
        ).to(dtype=x_base.dtype)


class SafetyController(nn.Module):
    def __init__(
        self,
        precondition_dim: int,
        consistency_dim: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        alpha_mid_bias: float = -3.0,
        alpha_fine_bias: float = -3.0,
        alpha_final_bias: float = -5.0,
        alpha_mid_max: float = 0.8,
        alpha_fine_max: float = 0.8,
        alpha_final_max: float = 0.5,
        zero_init: bool = True,
        dynamic_gate: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (
            ("alpha_mid_max", alpha_mid_max),
            ("alpha_fine_max", alpha_fine_max),
            ("alpha_final_max", alpha_final_max),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        hidden_half = max(8, hidden_dim // 2)
        self.refinement_net = nn.Sequential(
            nn.Linear(precondition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_half),
            nn.GELU(),
            nn.Linear(hidden_half, 2),
        )
        self.final_net = nn.Sequential(
            nn.Linear(precondition_dim + consistency_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_half),
            nn.GELU(),
            nn.Linear(hidden_half, 1),
        )
        if zero_init:
            _zero_last_linear(self.refinement_net)
            _zero_last_linear(self.final_net)
        self.mid_bias = nn.Parameter(torch.tensor(float(alpha_mid_bias)))
        self.fine_bias = nn.Parameter(torch.tensor(float(alpha_fine_bias)))
        self.final_bias = nn.Parameter(torch.tensor(float(alpha_final_bias)))
        self.alpha_mid_max = alpha_mid_max
        self.alpha_fine_max = alpha_fine_max
        self.alpha_final_max = alpha_final_max
        self.dynamic_gate = dynamic_gate

    def refinement_gates(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.refinement_net(condition) if self.dynamic_gate else torch.zeros(
            condition.shape[0], 2, device=condition.device, dtype=condition.dtype
        )
        alpha_mid = self.alpha_mid_max * torch.sigmoid(self.mid_bias + residual[:, 0])
        alpha_fine = self.alpha_fine_max * torch.sigmoid(self.fine_bias + residual[:, 1])
        return alpha_mid.view(-1, 1, 1, 1, 1), alpha_fine.view(-1, 1, 1, 1, 1)

    def final_gate(self, condition: torch.Tensor, consistency: torch.Tensor) -> torch.Tensor:
        residual = self.final_net(torch.cat([condition, consistency], dim=-1)) if self.dynamic_gate else torch.zeros(
            condition.shape[0], 1, device=condition.device, dtype=condition.dtype
        )
        return (
            self.alpha_final_max * torch.sigmoid(self.final_bias + residual[:, 0])
        ).view(-1, 1, 1, 1, 1)
