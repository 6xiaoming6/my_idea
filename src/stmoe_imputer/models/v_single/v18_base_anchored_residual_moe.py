from __future__ import annotations

import torch
from torch import nn

from ..main_branch import MultiScaleMoEBackbone
from .base_anchored_residual_pyramid import (
    AbsoluteCoarseToFinePyramid,
    BaseAnchoredResidualPyramid,
)
from .bounded_residual_controller import BoundedResidualBudgetController
from .difficulty_condition import DifficultyConditionEncoder
from .observed_relative_utility import ObservedRelativeUtilityEvaluator
from .observed_scale import masked_channel_rms


class V18BaseAnchoredResidualMoE(nn.Module):
    """BARP-MoE: Base-Anchored Bounded Residual Pyramid MoE."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.main_backbone = MultiScaleMoEBackbone.from_config(cfg)
        model_cfg = cfg["model"]
        main_cfg = model_cfg["main"]
        v18_cfg = model_cfg.get("v18", {})

        self.enabled = bool(v18_cfg.get("enabled", True))
        if not bool(v18_cfg.get("reuse_main_features", True)):
            raise ValueError(
                "V18 requires reuse_main_features=true; a second backbone is unsupported"
            )
        if not bool(v18_cfg.get("main_bypass_required", True)):
            raise ValueError(
                "V18 requires main_bypass_required=true to preserve the base safety path"
            )

        self.use_observed_relative_utility = bool(
            v18_cfg.get("use_observed_relative_utility", True)
        )
        self.pyramid_mode = str(
            v18_cfg.get("pyramid_mode", "base_anchored_direction")
        )
        if self.pyramid_mode not in {
            "base_anchored_direction",
            "absolute_c2f",
        }:
            raise ValueError(
                "model.v18.pyramid_mode must be "
                "'base_anchored_direction' or 'absolute_c2f'"
            )
        self.bounded_directions = bool(
            v18_cfg.get("bounded_directions", True)
        )
        self.fine_only_residual = bool(
            v18_cfg.get("fine_only_residual", False)
        )
        if self.pyramid_mode == "absolute_c2f" and self.fine_only_residual:
            raise ValueError(
                "absolute_c2f and fine_only_residual are separate ablations"
            )
        fixed_budget_value = v18_cfg.get("fixed_budget", False)
        fixed_budget = bool(fixed_budget_value)
        self.dynamic_budget = bool(
            v18_cfg.get("dynamic_budget", not fixed_budget)
        )
        if fixed_budget and self.dynamic_budget:
            raise ValueError(
                "fixed_budget=true and dynamic_budget=true are mutually exclusive"
            )
        if isinstance(fixed_budget_value, (int, float)) and not isinstance(
            fixed_budget_value, bool
        ):
            self.fixed_rho = float(fixed_budget_value)
        else:
            self.fixed_rho = float(v18_cfg.get("fixed_rho", 0.05))
        self.rho_probe = float(v18_cfg.get("rho_probe", 0.05))
        self.scale_eps = float(v18_cfg.get("scale_eps", 1e-3))
        if self.rho_probe <= 0.0:
            raise ValueError("rho_probe must be positive")

        self.rho_coarse_max = float(v18_cfg.get("rho_coarse_max", 0.15))
        self.rho_mid_max = float(v18_cfg.get("rho_mid_max", 0.15))
        self.rho_fine_max = float(v18_cfg.get("rho_fine_max", 0.20))
        if not self.dynamic_budget and not (
            0.0 < self.fixed_rho <= min(
                self.rho_coarse_max,
                self.rho_mid_max,
                self.rho_fine_max,
            )
        ):
            raise ValueError(
                "fixed_rho must be positive and no larger than every rho maximum"
            )

        difficulty_out = int(v18_cfg.get("difficulty_out_dim", 32))
        self.condition_encoder = DifficultyConditionEncoder(
            hidden_dim=int(v18_cfg.get("difficulty_hidden", 32)),
            out_dim=difficulty_out,
            dropout=float(v18_cfg.get("controller_dropout", 0.1)),
            enabled=bool(v18_cfg.get("difficulty_enabled", True)),
            use_spatial_block=bool(
                v18_cfg.get("difficulty_use_spatial_block", True)
            ),
            use_cross_scale_consistency=bool(
                v18_cfg.get("difficulty_use_cross_scale_consistency", True)
            ),
        )
        if not self.dynamic_budget:
            # Fixed-budget ablations do not consume the learned condition
            # embedding. Keep only parameter-free difficulty diagnostics and
            # avoid DDP unused-parameter failures.
            self.condition_encoder.enabled = False
            for parameter in self.condition_encoder.parameters():
                parameter.requires_grad_(False)
        residual_common = {
            "dim": int(main_cfg["dim"]),
            "c_out": int(model_cfg["c_in"]),
            "hidden": int(v18_cfg.get("residual_hidden", 32)),
            "num_groups": int(main_cfg.get("num_groups", 8)),
            "dropout": float(
                v18_cfg.get(
                    "residual_dropout", main_cfg.get("dropout", 0.0)
                )
            ),
            "use_reliability_filtered_propagation": bool(
                v18_cfg.get("use_reliability_filtered_propagation", True)
            ),
        }
        if self.pyramid_mode == "absolute_c2f":
            self.residual_pyramid = AbsoluteCoarseToFinePyramid(
                **residual_common,
                prediction_embed_dim=int(
                    v18_cfg.get("direction_embed_dim", 16)
                ),
            )
        else:
            self.residual_pyramid = BaseAnchoredResidualPyramid(
                **residual_common,
                anchor_embed_dim=int(v18_cfg.get("anchor_embed_dim", 16)),
                direction_embed_dim=int(
                    v18_cfg.get("direction_embed_dim", 16)
                ),
                zero_init=bool(v18_cfg.get("direction_zero_init", True)),
                bounded_directions=self.bounded_directions,
                fine_only_residual=self.fine_only_residual,
            )
        self.utility_evaluator = (
            ObservedRelativeUtilityEvaluator(
                eps=float(v18_cfg.get("utility_eps", 1e-6))
            )
            if self.use_observed_relative_utility
            else None
        )

        controller_input_dim = difficulty_out + 2 + 3 + 5
        if self.use_observed_relative_utility:
            controller_input_dim += ObservedRelativeUtilityEvaluator.output_dim
        self.controller = (
            BoundedResidualBudgetController(
                input_dim=controller_input_dim,
                hidden_dim=int(v18_cfg.get("controller_hidden", 64)),
                dropout=float(v18_cfg.get("controller_dropout", 0.1)),
                rho_coarse_max=self.rho_coarse_max,
                rho_mid_max=self.rho_mid_max,
                rho_fine_max=self.rho_fine_max,
                rho_init=float(v18_cfg.get("rho_init", 0.02)),
                zero_init=bool(v18_cfg.get("controller_zero_init", True)),
            )
            if self.dynamic_budget
            else None
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "V18BaseAnchoredResidualMoE":
        return cls(cfg)

    @staticmethod
    def _geometry(z_f: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        batch_size = z_f.shape[0]
        height, width = z_f.shape[-2:]
        coarse_height, coarse_width = z_c.shape[-2:]
        values = torch.tensor(
            (
                height / 32.0,
                width / 32.0,
                height / max(float(width), 1.0),
                min(coarse_height, coarse_width) / 8.0,
                coarse_height
                * coarse_width
                / max(float(height * width), 1.0),
            ),
            device=z_f.device,
            dtype=z_f.dtype,
        )
        return values.view(1, 5).expand(batch_size, -1)

    @staticmethod
    def _sample_rms(value: torch.Tensor) -> torch.Tensor:
        return (
            value.detach()
            .float()
            .square()
            .mean(dim=(1, 2, 3, 4))
            .sqrt()
        )

    @staticmethod
    def _sample_abs_q95(value: torch.Tensor) -> torch.Tensor:
        return torch.quantile(
            value.detach().float().abs().flatten(1), 0.95, dim=1
        )

    @staticmethod
    def _reliability_scalar(
        reliability: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        source = reliability if reliability is not None else mask
        return source.detach().float().mean(dim=(1, 2, 3, 4))

    def _budgets(
        self,
        controller_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.controller is not None:
            return self.controller(controller_input)
        shape = (controller_input.shape[0], 1, 1, 1, 1)
        fixed = controller_input.new_full(shape, self.fixed_rho)
        return fixed, fixed.clone(), fixed.clone()

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
        gates = base_outputs["gates"]
        x_base = base_outputs["x_hat_main"]
        z_f = features["z_f"]
        z_m = features["z_m"]
        z_c = features["z_c"]
        h_main = features["h_main"]

        rel_m_scalar = self._reliability_scalar(r_m, m_m)
        rel_c_scalar = self._reliability_scalar(r_c, m_c)
        reliability = torch.stack(
            (rel_m_scalar, rel_c_scalar), dim=1
        ).to(dtype=x_f.dtype)
        rel_m = rel_m_scalar.to(dtype=x_f.dtype).view(-1, 1, 1, 1, 1)
        rel_c = rel_c_scalar.to(dtype=x_f.dtype).view(-1, 1, 1, 1, 1)
        directions = self.residual_pyramid(
            z_f=z_f,
            z_m=z_m,
            z_c=z_c,
            h_main=h_main,
            x_base=x_base,
            reliability_m=rel_m,
            reliability_c=rel_c,
        )

        scale_f = masked_channel_rms(x_f, m_f, eps=self.scale_eps)
        scale_m = masked_channel_rms(x_m, m_m, eps=self.scale_eps)
        scale_c = masked_channel_rms(x_c, m_c, eps=self.scale_eps)
        probe_scale = (
            1.0
            if self.pyramid_mode == "absolute_c2f"
            else scale_f
        )
        x_probe = (
            x_base.detach()
            + self.rho_probe
            * probe_scale
            * directions["direction_f"].detach()
        )
        if self.utility_evaluator is not None:
            utility = self.utility_evaluator(
                x_base=x_base,
                x_probe=x_probe,
                x_obs=x_f,
                mask=m_f,
            )
        else:
            utility = x_f.new_zeros(
                x_f.shape[0], ObservedRelativeUtilityEvaluator.output_dim
            )

        condition_embedding, difficulty = self.condition_encoder(
            x_f=x_f,
            m_f=m_f,
            x_m=x_m,
            m_m=m_m,
            x_c=x_c,
            m_c=m_c,
            r_m=r_m,
            r_c=r_c,
        )
        geometry = self._geometry(z_f, z_c)
        scale_gate = gates["scale_gate"].to(dtype=x_f.dtype)
        controller_parts = (
            condition_embedding,
            reliability,
            scale_gate,
            geometry,
        )
        if self.use_observed_relative_utility:
            controller_parts += (utility,)
        controller_input = torch.cat(controller_parts, dim=-1)
        rho_c, rho_m, rho_f = self._budgets(controller_input)

        if self.pyramid_mode == "absolute_c2f":
            x_hat_coarse = (
                directions["anchor_c"]
                + rho_c * directions["direction_c"]
            )
            x_hat_mid = (
                directions["anchor_m"]
                + rho_m * directions["direction_m"]
            )
            effective_residual = rho_f * directions["direction_f"]
        else:
            x_hat_coarse = (
                directions["anchor_c"]
                + rho_c * scale_c * directions["direction_c"]
            )
            x_hat_mid = (
                directions["anchor_m"]
                + rho_m * scale_m * directions["direction_m"]
            )
            effective_residual = (
                rho_f * scale_f * directions["direction_f"]
            )
        x_final = x_base + effective_residual

        output_features = dict(features)
        output_features.update(
            {
                "h_main_base": h_main,
                "direction_c": directions["direction_c"],
                "direction_m": directions["direction_m"],
                "direction_f": directions["direction_f"],
                "effective_residual": effective_residual,
                "observed_scale_f": scale_f,
            }
        )
        diagnostics = dict(base_outputs.get("diagnostics", {}))
        scale_f_sample = scale_f.detach().float().flatten(1).mean(dim=1)
        effective_rms = self._sample_rms(effective_residual)
        diagnostics["v18"] = {
            "rho_c": rho_c.flatten(1).mean(dim=1),
            "rho_m": rho_m.flatten(1).mean(dim=1),
            "rho_f": rho_f.flatten(1).mean(dim=1),
            "utility_base_rel": utility[:, 0],
            "utility_probe_rel": utility[:, 1],
            "utility_gain": utility[:, 2],
            "probe_delta_mean_rel": utility[:, 3],
            "probe_delta_q95_rel": utility[:, 4],
            "scale_f_mean": scale_f_sample,
            "direction_c_rms": self._sample_rms(directions["direction_c"]),
            "direction_m_rms": self._sample_rms(directions["direction_m"]),
            "direction_f_rms": self._sample_rms(directions["direction_f"]),
            "direction_f_abs_q95": self._sample_abs_q95(
                directions["direction_f"]
            ),
            "effective_residual_rms": effective_rms,
            "effective_residual_q95": self._sample_abs_q95(
                effective_residual
            ),
            "effective_residual_ratio": effective_rms
            / self._sample_rms(x_base).clamp_min(1e-6),
            "effective_residual_scale_ratio": effective_rms
            / scale_f_sample.clamp_min(1e-6),
            "difficulty_f": difficulty["score_f"],
            "difficulty_m": difficulty["score_m"],
            "difficulty_c": difficulty["score_c"],
            "scale_reliability_m": reliability[:, 0],
            "scale_reliability_c": reliability[:, 1],
            "geometry_h": geometry[:, 0],
            "geometry_w": geometry[:, 1],
            "geometry_aspect": geometry[:, 2],
            "geometry_coarse_min": geometry[:, 3],
            "geometry_coarse_ratio": geometry[:, 4],
        }

        outputs = dict(base_outputs)
        outputs.update(
            {
                "x_hat_main": x_final,
                "x_hat_base": x_base,
                "x_hat_mid": x_hat_mid,
                "x_hat_coarse": x_hat_coarse,
                "x_hat_probe": x_probe,
                "features": output_features,
                "diagnostics": diagnostics,
                "branch_mode": (
                    "v18_absolute_c2f_ablation"
                    if self.pyramid_mode == "absolute_c2f"
                    else (
                        "v18_unbounded_residual_ablation"
                        if not self.bounded_directions
                        else "v18_barp_moe"
                    )
                ),
                "v18_enabled": True,
            }
        )
        return outputs
