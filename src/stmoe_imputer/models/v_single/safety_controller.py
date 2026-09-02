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
        c_out: int = 1,
        channel_final_gate: bool = False,
        channel_stats_dim: int = 5,
        channel_gain_delta: float = 0.2,
        final_gate_mode: str = "legacy",
        monotonic_advantage_gain: float = 1.0,
        monotonic_context_bound: float = 0.5,
        monotonic_eps: float = 1e-6,
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
        self.channel_final_gate = bool(channel_final_gate and c_out > 1)
        self.c_out = int(c_out)
        self.channel_gain_delta = float(channel_gain_delta)
        if not 0.0 <= self.channel_gain_delta < 1.0:
            raise ValueError(
                f"channel_gain_delta must be in [0,1), got {channel_gain_delta}"
            )
        self.channel_stats_dim = int(channel_stats_dim)
        self.final_gate_mode = str(final_gate_mode)
        if self.final_gate_mode not in {"legacy", "monotonic_consistency"}:
            raise ValueError(
                "final_gate_mode must be 'legacy' or 'monotonic_consistency', "
                f"got {self.final_gate_mode!r}"
            )
        self.monotonic_advantage_gain = float(monotonic_advantage_gain)
        self.monotonic_context_bound = float(monotonic_context_bound)
        self.monotonic_eps = float(monotonic_eps)
        if self.monotonic_advantage_gain < 0.0:
            raise ValueError("monotonic_advantage_gain must be non-negative")
        if self.monotonic_context_bound < 0.0:
            raise ValueError("monotonic_context_bound must be non-negative")
        if self.monotonic_eps <= 0.0:
            raise ValueError("monotonic_eps must be positive")
        if self.final_gate_mode == "monotonic_consistency" and self.channel_final_gate:
            raise ValueError(
                "monotonic_consistency and channel_final_gate are separate "
                "single-variable explorations and cannot be enabled together"
            )
        if self.channel_final_gate:
            # Do not advance the outer RNG state: enabling the optional channel
            # calibrator must not change initialization of the existing refiner.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1402)
                self.channel_final_net = nn.Sequential(
                    nn.Linear(
                        precondition_dim + consistency_dim + self.channel_stats_dim,
                        hidden_dim,
                    ),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_half),
                    nn.GELU(),
                    nn.Linear(hidden_half, 1),
                )
            _zero_last_linear(self.channel_final_net)
        else:
            self.channel_final_net = None
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

    def final_gate(
        self,
        condition: torch.Tensor,
        consistency: torch.Tensor,
        channel_stats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.dynamic_gate:
            if self.final_gate_mode == "monotonic_consistency":
                # Keep the original MLP and parameter count, but prevent its
                # unconstrained consistency path from reversing the observed
                # gain ordering.  It remains a bounded context adjustment.
                context_input = torch.cat(
                    (condition, torch.zeros_like(consistency)), dim=-1
                )
                raw_context = self.final_net(context_input)[:, 0]
                context = self.monotonic_context_bound * torch.tanh(raw_context)
                observed_advantage = -consistency[:, 2]
                observed_error_scale = (
                    consistency[:, 0].abs() + consistency[:, 1].abs()
                ).clamp_min(self.monotonic_eps)
                relative_advantage = (
                    observed_advantage / observed_error_scale
                ).clamp(-1.0, 1.0)
                residual = context + self.monotonic_advantage_gain * relative_advantage
            else:
                residual = self.final_net(
                    torch.cat([condition, consistency], dim=-1)
                )[:, 0]
                context = torch.zeros_like(residual)
                relative_advantage = torch.zeros_like(residual)
        else:
            residual = torch.zeros(
                condition.shape[0], device=condition.device, dtype=condition.dtype
            )
            context = torch.zeros_like(residual)
            relative_advantage = torch.zeros_like(residual)
        alpha = (
            self.alpha_final_max * torch.sigmoid(self.final_bias + residual)
        ).view(-1, 1)
        if not self.channel_final_gate:
            return alpha.view(-1, 1, 1, 1, 1), {
                "final_gate_context": context,
                "relative_observed_advantage": relative_advantage,
            }
        if channel_stats is None:
            raise ValueError("channel_stats are required when channel_final_gate is enabled")
        if channel_stats.shape != (
            condition.shape[0],
            self.c_out,
            self.channel_stats_dim,
        ):
            raise ValueError(
                "Expected channel_stats shape "
                f"{(condition.shape[0], self.c_out, self.channel_stats_dim)}, "
                f"got {tuple(channel_stats.shape)}"
            )
        global_condition = torch.cat((condition, consistency), dim=-1)
        global_condition = global_condition.unsqueeze(1).expand(-1, self.c_out, -1)
        channel_input = torch.cat((global_condition, channel_stats), dim=-1)
        channel_logits = self.channel_final_net(
            channel_input.reshape(-1, channel_input.shape[-1])
        ).reshape(condition.shape[0], self.c_out)
        channel_gain = 1.0 + self.channel_gain_delta * torch.tanh(channel_logits)
        alpha_channel = alpha * channel_gain
        saturation = (
            channel_gain.sub(1.0).abs()
            >= self.channel_gain_delta * 0.95
        ).to(channel_gain.dtype)
        return alpha_channel.view(-1, self.c_out, 1, 1, 1), {
            "channel_gain": channel_gain,
            "channel_gain_saturation": saturation,
        }
