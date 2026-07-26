from __future__ import annotations

import torch
from torch import nn


class ChannelResidualGain(nn.Module):
    """Predict a bounded per-sample, per-channel gain for V14's residual."""

    feature_names = (
        "observed_scale",
        "base_observed_error_relative",
        "v14_observed_error_relative",
        "v14_observed_gain_relative",
        "effective_residual_mean_relative",
        "effective_residual_q95_relative",
        "observed_zero_ratio",
    )

    def __init__(
        self,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        gain_range: float = 0.5,
        scale_eps: float = 1e-3,
        zero_eps: float = 1e-6,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if not 0.0 < gain_range < 1.0:
            raise ValueError("gain_range must be in (0,1)")
        if scale_eps <= 0.0 or zero_eps < 0.0:
            raise ValueError("scale_eps must be positive and zero_eps non-negative")

        hidden_half = max(8, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(len(self.feature_names), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_half),
            nn.GELU(),
            nn.Linear(hidden_half, 1),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.gain_range = float(gain_range)
        self.scale_eps = float(scale_eps)
        self.zero_eps = float(zero_eps)

    @staticmethod
    def _validate(
        x_base: torch.Tensor,
        x_v14: torch.Tensor,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if x_base.ndim != 5:
            raise ValueError(f"predictions must be 5-D, got {tuple(x_base.shape)}")
        if x_v14.shape != x_base.shape or x_obs.shape != x_base.shape:
            raise ValueError("x_base, x_v14, and x_obs must have identical shapes")
        if mask.ndim != 5 or mask.shape[1] != 1:
            raise ValueError("mask must have shape [B,1,T,H,W]")
        if mask.shape[0] != x_base.shape[0] or mask.shape[2:] != x_base.shape[2:]:
            raise ValueError("mask must match prediction batch and spatiotemporal dimensions")

    @staticmethod
    def _observed_q95(
        value: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        value_flat = value.detach().float().flatten(2)
        observed_flat = observed.detach().bool().flatten(2)
        masked_values = value_flat.masked_fill(
            ~observed_flat,
            float("nan"),
        )
        return torch.nan_to_num(
            torch.nanquantile(masked_values, 0.95, dim=-1),
            nan=0.0,
        )

    def forward(
        self,
        x_base: torch.Tensor,
        x_v14: torch.Tensor,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate(x_base, x_v14, x_obs, mask)
        base = x_base.detach().float()
        v14 = x_v14.detach().float()
        observed_values = x_obs.detach().float()
        observed = mask.detach().float().expand_as(base)
        count = observed.sum(dim=(2, 3, 4)).clamp_min(1.0)

        scale = (
            (observed_values.abs() * observed).sum(dim=(2, 3, 4)) / count
        ).clamp_min(self.scale_eps)
        base_error = (
            ((base - observed_values).abs() * observed).sum(dim=(2, 3, 4))
            / count
        )
        v14_error = (
            ((v14 - observed_values).abs() * observed).sum(dim=(2, 3, 4))
            / count
        )
        base_error_relative = base_error / scale
        v14_error_relative = v14_error / scale
        gain_relative = (
            (base_error - v14_error) / base_error.clamp_min(self.scale_eps)
        )

        residual_absolute = (v14 - base).abs()
        residual_mean_relative = (
            (residual_absolute * observed).sum(dim=(2, 3, 4)) / count / scale
        )
        residual_q95_relative = (
            self._observed_q95(residual_absolute, observed) / scale
        )
        zero_ratio = (
            (
                (observed_values.abs() <= self.zero_eps).to(observed.dtype)
                * observed
            ).sum(dim=(2, 3, 4))
            / count
        )

        features = torch.stack(
            (
                scale,
                base_error_relative,
                v14_error_relative,
                gain_relative,
                residual_mean_relative,
                residual_q95_relative,
                zero_ratio,
            ),
            dim=-1,
        )
        features = torch.nan_to_num(
            features, nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(min=-1e4, max=1e4).detach()
        raw_gain = self.net(features)
        gain = 1.0 + self.gain_range * torch.tanh(raw_gain)
        gain = gain.squeeze(-1).view(
            x_base.shape[0], x_base.shape[1], 1, 1, 1
        )
        diagnostics = {
            name: features[..., index]
            for index, name in enumerate(self.feature_names)
        }
        diagnostics["gain"] = gain.flatten(1)
        diagnostics["raw_gain"] = raw_gain.squeeze(-1)
        return gain.to(dtype=x_base.dtype), diagnostics
