from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..blocks import valid_num_groups


class BoundedLocalResidualGate(nn.Module):
    """Conservative target-free modulation of V14's global final gate.

    The module predicts either one modulation per time step or a smooth
    low-resolution regional map.  Its last convolution is zero-initialized, so
    enabling it starts exactly at the original V14 prediction.
    """

    MODES = {"temporal", "regional"}

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 16,
        mode: str = "regional",
        max_relative_delta: float = 0.2,
        spatial_divisor: int = 4,
        num_groups: int = 8,
        detach_inputs: bool = True,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(
                f"local residual gate mode must be one of {sorted(self.MODES)}, "
                f"got {mode!r}"
            )
        if not 0.0 < max_relative_delta < 1.0:
            raise ValueError("local gate max_relative_delta must be in (0, 1)")
        if spatial_divisor < 1:
            raise ValueError("local gate spatial_divisor must be at least 1")
        if hidden_dim < 1:
            raise ValueError("local gate hidden_dim must be at least 1")
        self.mode = mode
        self.max_relative_delta = float(max_relative_delta)
        self.spatial_divisor = int(spatial_divisor)
        self.detach_inputs = bool(detach_inputs)
        self.feature_proj = nn.Conv3d(feature_dim, hidden_dim, kernel_size=1)
        groups = valid_num_groups(hidden_dim, num_groups)
        # Four risk maps: raw correction magnitude, C2F/base disagreement,
        # observation density, and observed-position base error.
        self.net = nn.Sequential(
            nn.Conv3d(hidden_dim + 4, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _target_size(self, value: torch.Tensor) -> tuple[int, int, int]:
        time, height, width = value.shape[-3:]
        if self.mode == "temporal":
            return time, 1, 1
        return (
            time,
            max(1, height // self.spatial_divisor),
            max(1, width // self.spatial_divisor),
        )

    def forward(
        self,
        alpha_global: torch.Tensor,
        h_main: torch.Tensor,
        delta_ctf: torch.Tensor,
        x_ctf: torch.Tensor,
        x_base: torch.Tensor,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        alpha_max: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if alpha_global.shape[1:] != (1, 1, 1, 1):
            raise ValueError(
                "BRLG requires the original sample-level scalar final gate; "
                f"got {tuple(alpha_global.shape)}"
            )
        target_size = self._target_size(h_main)
        sources = (h_main, delta_ctf, x_ctf, x_base, x_obs, mask)
        if self.detach_inputs:
            h_main, delta_ctf, x_ctf, x_base, x_obs, mask = (
                value.detach() for value in sources
            )

        feature = F.adaptive_avg_pool3d(self.feature_proj(h_main), target_size)
        correction_magnitude = delta_ctf.abs().mean(dim=1, keepdim=True)
        ctf_disagreement = (x_ctf - x_base).abs().mean(dim=1, keepdim=True)
        observed_density = mask.to(dtype=x_base.dtype)
        observed_error = (
            (x_base - x_obs).abs() * observed_density
        ).mean(dim=1, keepdim=True)
        risk_maps = torch.cat(
            (
                correction_magnitude,
                ctf_disagreement,
                observed_density,
                observed_error,
            ),
            dim=1,
        )
        risk_maps = F.adaptive_avg_pool3d(risk_maps, target_size)
        logits = self.net(torch.cat((feature, risk_maps), dim=1))
        modulation_lowres = 1.0 + self.max_relative_delta * torch.tanh(logits)
        modulation = F.interpolate(
            modulation_lowres,
            size=h_main.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        alpha_local = (alpha_global * modulation).clamp(0.0, float(alpha_max))
        return alpha_local, {
            "local_gate_logits": logits,
            "local_gate_modulation": modulation,
            "local_gate_modulation_lowres": modulation_lowres,
        }

