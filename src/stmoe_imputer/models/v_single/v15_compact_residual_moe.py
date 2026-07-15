from __future__ import annotations

import torch
from torch import nn

from ..main_branch import MultiScaleMoEBackbone
from ..stats import compute_observation_stats
from .compact_residual_pyramid import CompactResidualPyramid
from .residual_budget import ResidualBudgetController


class V15CompactResidualMoE(nn.Module):
    """Main backbone with a compact, explicitly bounded residual pyramid."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.main_backbone = MultiScaleMoEBackbone.from_config(cfg)
        model_cfg = cfg["model"]
        main_cfg = model_cfg["main"]
        v15_cfg = model_cfg.get("v15", {})

        self.enabled = bool(v15_cfg.get("enabled", True))
        self.reuse_main_features = bool(v15_cfg.get("reuse_main_features", True))
        if not self.reuse_main_features:
            raise ValueError(
                "V15 requires reuse_main_features=true; duplicating the main encoder or "
                "expert pool is intentionally unsupported."
            )

        main_dim = int(main_cfg["dim"])
        pyramid_dim = int(v15_cfg.get("pyramid_dim", main_dim))
        if pyramid_dim != main_dim:
            raise ValueError(
                f"V15 pyramid_dim must match model.main.dim ({main_dim}), got {pyramid_dim}"
            )
        condition_dim = int(v15_cfg.get("condition_dim", 8))
        if condition_dim != 8:
            raise ValueError(
                f"V15 compact condition has exactly 8 statistics, got condition_dim={condition_dim}"
            )

        self.detach_scale_gate = bool(v15_cfg.get("detach_scale_gate", True))
        self.dynamic_budget = bool(v15_cfg.get("dynamic_budget", True))
        self.bounded_residual = bool(v15_cfg.get("bounded_residual", True))
        self.fixed_beta = float(v15_cfg.get("fixed_beta", 0.05))
        self.scale_floor = float(v15_cfg.get("scale_floor", 1e-3))
        if self.scale_floor <= 0:
            raise ValueError(f"scale_floor must be positive, got {self.scale_floor}")

        self.residual_pyramid = CompactResidualPyramid(
            dim=main_dim,
            c_out=int(model_cfg["c_in"]),
            num_groups=int(main_cfg.get("num_groups", 8)),
            dropout=float(v15_cfg.get("pyramid_dropout", 0.1)),
            zero_init=bool(v15_cfg.get("residual_zero_init", True)),
            use_pyramid=bool(v15_cfg.get("use_pyramid", True)),
        )
        self.budget_controller = ResidualBudgetController(
            condition_dim=condition_dim,
            hidden_dim=int(v15_cfg.get("controller_hidden", 32)),
            beta_max=float(v15_cfg.get("beta_max", 0.5)),
            beta_bias=float(v15_cfg.get("beta_bias", -3.0)),
            dropout=float(v15_cfg.get("controller_dropout", 0.1)),
            zero_init=bool(v15_cfg.get("controller_zero_init", True)),
        )
        if not 0.0 <= self.fixed_beta <= self.beta_max:
            raise ValueError(
                f"fixed_beta must be in [0,beta_max={self.beta_max}], got {self.fixed_beta}"
            )
        if not self.dynamic_budget:
            for parameter in self.budget_controller.parameters():
                parameter.requires_grad_(False)
        if bool(v15_cfg.get("freeze_main", False)):
            for parameter in self.main_backbone.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg: dict) -> "V15CompactResidualMoE":
        return cls(cfg)

    @property
    def beta_max(self) -> float:
        return self.budget_controller.beta_max

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().square().mean(dim=(1, 2, 3, 4)).sqrt()

    def _compute_scale_ref(self, x_base: torch.Tensor) -> torch.Tensor:
        return (
            x_base.detach()
            .float()
            .square()
            .mean(dim=(2, 3, 4), keepdim=True)
            .add(1e-6)
            .sqrt()
            .clamp_min(self.scale_floor)
            .to(dtype=x_base.dtype)
        )

    @staticmethod
    def _mean_reliability(
        reliability: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if reliability is None:
            return torch.zeros(reference.shape[0], device=reference.device, dtype=reference.dtype)
        return reliability.float().mean(dim=(1, 2, 3, 4)).to(dtype=reference.dtype)

    def _build_condition(
        self,
        m_f: torch.Tensor,
        r_m: torch.Tensor | None,
        r_c: torch.Tensor | None,
        scale_gate: torch.Tensor,
    ) -> torch.Tensor:
        q_f = compute_observation_stats(m_f)
        difficulty = q_f[:, (0, 2, 3)]
        reliability = torch.stack(
            (
                self._mean_reliability(r_m, q_f),
                self._mean_reliability(r_c, q_f),
            ),
            dim=1,
        )
        if scale_gate.ndim != 2 or scale_gate.shape[1] != 3:
            raise ValueError(
                f"Expected main scale_gate [B,3], got shape {tuple(scale_gate.shape)}"
            )
        return torch.cat(
            (difficulty, reliability, scale_gate.to(dtype=q_f.dtype)), dim=1
        )

    def forward(
        self,
        x_f: torch.Tensor,
        m_f: torch.Tensor,
        x_m: torch.Tensor,
        m_m: torch.Tensor,
        x_c: torch.Tensor,
        m_c: torch.Tensor,
        r_m: torch.Tensor | None = None,
        r_c: torch.Tensor | None = None,
    ) -> dict:
        base_outputs = self.main_backbone(
            x_f=x_f,
            m_f=m_f,
            x_m=x_m,
            m_m=m_m,
            x_c=x_c,
            m_c=m_c,
            r_m=r_m,
            r_c=r_c,
        )
        if not self.enabled:
            return base_outputs

        features = base_outputs["features"]
        x_base = base_outputs["x_hat_main"]
        pyramid = self.residual_pyramid(
            z_f=features["z_f"],
            z_m=features["z_m"],
            z_c=features["z_c"],
            h_main=features["h_main"],
        )

        scale_gate = base_outputs["gates"]["scale_gate"]
        if self.detach_scale_gate:
            scale_gate = scale_gate.detach()
        condition = self._build_condition(m_f, r_m, r_c, scale_gate)
        if self.dynamic_budget:
            beta = self.budget_controller(condition)
        else:
            beta = x_base.new_full((x_base.shape[0], 1, 1, 1, 1), self.fixed_beta)
        scale_ref = self._compute_scale_ref(x_base)
        if self.bounded_residual:
            direction = torch.tanh(pyramid["delta_raw"])
            effective_delta = beta * scale_ref * direction
        else:
            direction = pyramid["delta_raw"]
            effective_delta = beta * direction
        x_final = x_base + effective_delta

        output_features = dict(features)
        output_features.update({
            "h_main_base": features["h_main"],
            "pyramid_coarse": pyramid["pyramid_coarse"],
            "pyramid_mid": pyramid["pyramid_mid"],
            "pyramid_fine": pyramid["pyramid_fine"],
            "delta_raw": pyramid["delta_raw"],
            "effective_delta": effective_delta,
        })
        output_diagnostics = dict(base_outputs.get("diagnostics", {}))
        normalized_effective = effective_delta.float() / scale_ref.float().clamp_min(1e-6)
        output_diagnostics["v15"] = {
            "beta": beta.flatten(1).mean(dim=1),
            "scale_ref": scale_ref.detach().float().flatten(1).mean(dim=1),
            "raw_delta_rms": self._rms(pyramid["delta_raw"]),
            "direction_rms": self._rms(direction),
            "effective_delta_rms": self._rms(effective_delta),
            "effective_relative_rms": self._rms(normalized_effective),
        }

        outputs = dict(base_outputs)
        outputs.update({
            "x_hat_main": x_final,
            "x_hat_base": x_base,
            "delta_effective": effective_delta,
            "residual_budget": beta,
            "scale_ref": scale_ref,
            "features": output_features,
            "diagnostics": output_diagnostics,
            "branch_mode": "v15_compact_residual",
            "v15_enabled": True,
            "v15_bounded_residual": self.bounded_residual,
        })
        return outputs
