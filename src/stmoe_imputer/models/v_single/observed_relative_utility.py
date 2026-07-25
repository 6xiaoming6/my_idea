from __future__ import annotations

import torch
from torch import nn


class ObservedRelativeUtilityEvaluator(nn.Module):
    """Compute detached, target-free probe utility at observed positions."""

    output_dim = 5

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = float(eps)

    @staticmethod
    def _validate(
        x_base: torch.Tensor,
        x_probe: torch.Tensor,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if x_base.ndim != 5 or mask.ndim != 5:
            raise ValueError("predictions, observations, and mask must be 5-D tensors")
        if x_probe.shape != x_base.shape or x_obs.shape != x_base.shape:
            raise ValueError(
                "x_base, x_probe, and x_obs must have identical shapes, got "
                f"{tuple(x_base.shape)}, {tuple(x_probe.shape)}, and {tuple(x_obs.shape)}"
            )
        if mask.shape[0] != x_base.shape[0] or mask.shape[2:] != x_base.shape[2:]:
            raise ValueError(
                "mask must match prediction batch and spatiotemporal dimensions"
            )
        if mask.shape[1] != 1:
            raise ValueError(
                f"mask channel dimension must be 1, got {mask.shape[1]}"
            )

    @staticmethod
    def _masked_mean(
        value: torch.Tensor,
        observed: torch.Tensor,
        count: torch.Tensor,
    ) -> torch.Tensor:
        return (value * observed).flatten(1).sum(dim=1) / count

    def forward(
        self,
        x_base: torch.Tensor,
        x_probe: torch.Tensor,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(x_base, x_probe, x_obs, mask)
        base = x_base.detach().float()
        probe = x_probe.detach().float()
        obs = x_obs.detach().float()
        observed = mask.detach().float().expand_as(base)
        count = observed.flatten(1).sum(dim=1).clamp_min(1.0)

        observed_scale = self._masked_mean(
            obs.abs(), observed, count
        ).clamp_min(self.eps)
        base_error = self._masked_mean((base - obs).abs(), observed, count)
        probe_error = self._masked_mean((probe - obs).abs(), observed, count)
        base_relative_error = base_error / observed_scale
        probe_relative_error = probe_error / observed_scale
        relative_gain = (
            (base_error - probe_error) / base_error.clamp_min(self.eps)
        )

        delta = ((probe - base).abs() * observed).flatten(1)
        delta_mean_relative = delta.sum(dim=1) / count / observed_scale
        valid = observed.flatten(1).bool()
        delta_q95 = torch.stack(
            [
                (
                    torch.quantile(values[is_valid], 0.95)
                    if is_valid.any()
                    else values.new_zeros(())
                )
                for values, is_valid in zip(delta, valid)
            ]
        )
        delta_q95_relative = delta_q95 / observed_scale

        utility = torch.stack(
            (
                base_relative_error,
                probe_relative_error,
                relative_gain,
                delta_mean_relative,
                delta_q95_relative,
            ),
            dim=1,
        )
        return torch.nan_to_num(
            utility, nan=0.0, posinf=1e4, neginf=-1e4
        ).to(dtype=x_base.dtype).detach()
