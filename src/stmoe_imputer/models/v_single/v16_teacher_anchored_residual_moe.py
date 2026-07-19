from __future__ import annotations

import torch
from torch import nn

from ..main_branch import MultiScaleMoEBackbone
from ..scale_utils import get_active_scales
from ..stats import compute_observation_stats
from .continuous_residual_calibrator import ContinuousResidualCalibrator
from .scale_guided_residual_adapter import ScaleGuidedResidualAdapter


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mask.expand(target.shape[0], target.shape[1], *target.shape[2:])


def observed_mae_per_sample(
    prediction: torch.Tensor,
    observed_value: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    mask = _expand_mask(observed_mask, prediction).to(dtype=prediction.dtype)
    numerator = ((prediction - observed_value).abs() * mask).flatten(1).sum(dim=1)
    return numerator / mask.flatten(1).sum(dim=1).clamp_min(1.0)


def masked_rms_per_sample(
    value: torch.Tensor,
    selected_mask: torch.Tensor,
) -> torch.Tensor:
    mask = _expand_mask(selected_mask, value).float()
    numerator = (value.float().square() * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (numerator / denominator).sqrt()


class V16TeacherAnchoredResidualMoE(nn.Module):
    """Teacher-anchored base with a continuously calibrated residual proposal.

    The V14 teacher deliberately does not live inside this module: it is owned by
    the training engine, so inference and student checkpoints contain no teacher.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.student_backbone = MultiScaleMoEBackbone.from_config(cfg)
        model_cfg = cfg["model"]
        main_cfg = model_cfg["main"]
        version_cfg = model_cfg.get("v16", {})

        self.enabled = bool(version_cfg.get("enabled", True))
        self.detach_residual_inputs = bool(
            version_cfg.get("detach_residual_inputs", True)
        )
        self.detach_scale_gate = bool(version_cfg.get("detach_scale_gate", True))
        self.use_scale_guidance = bool(version_cfg.get("use_scale_guidance", True))
        self.rho = float(version_cfg.get("rho", 0.05))
        self.scale_floor = float(version_cfg.get("scale_floor", 1e-3))
        self.warmup_epochs = int(version_cfg.get("warmup_epochs", 12))
        self.calibration_mode = str(version_cfg.get("calibration_mode", "learned"))
        self.calibration_supervision = str(
            version_cfg.get("calibration_supervision", "oracle")
        )
        if self.calibration_mode not in {"learned", "fixed_one"}:
            raise ValueError(
                "calibration_mode must be 'learned' or 'fixed_one', "
                f"got {self.calibration_mode!r}"
            )
        if self.calibration_supervision not in {"oracle", "binary"}:
            raise ValueError(
                "calibration_supervision must be 'oracle' or 'binary', "
                f"got {self.calibration_supervision!r}"
            )
        if not 0.0 < self.rho <= 1.0:
            raise ValueError(f"rho must be in (0,1], got {self.rho}")
        if self.scale_floor <= 0.0:
            raise ValueError(f"scale_floor must be positive, got {self.scale_floor}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {self.warmup_epochs}")

        main_scale_mode = str(main_cfg.get("scale_mode", "fine_mid_coarse"))
        configured_residual_mode = str(
            version_cfg.get("residual_scale_mode", "inherit")
        )
        residual_scale_mode = (
            main_scale_mode
            if configured_residual_mode == "inherit"
            else configured_residual_mode
        )
        active_names = set(get_active_scales(residual_scale_mode))
        active_scales = tuple(
            name in active_names for name in ("fine", "mid", "coarse")
        )
        self.scale_mode = main_scale_mode
        self.residual_scale_mode = residual_scale_mode
        self.register_buffer(
            "active_scale_mask",
            torch.tensor(active_scales, dtype=torch.bool),
            persistent=True,
        )

        self.residual_proposer = ScaleGuidedResidualAdapter(
            main_dim=int(main_cfg["dim"]),
            residual_dim=int(version_cfg.get("residual_dim", 24)),
            out_channels=int(model_cfg["c_in"]),
            active_scales=active_scales,
            num_groups=int(main_cfg.get("num_groups", 8)),
            dropout=float(version_cfg.get("residual_dropout", 0.1)),
            zero_init=bool(version_cfg.get("residual_zero_init", True)),
        )
        self.condition_dim = int(version_cfg.get("calibration_condition_dim", 12))
        self.calibrator = ContinuousResidualCalibrator(
            condition_dim=self.condition_dim,
            hidden_dim=int(version_cfg.get("calibration_hidden_dim", 32)),
            fixed_bias=float(version_cfg.get("calibration_fixed_bias", -2.0)),
            dropout=float(version_cfg.get("calibration_dropout", 0.0)),
            zero_init=bool(version_cfg.get("calibration_zero_init", True)),
        )
        self.training_stage = "joint"

    @classmethod
    def from_config(cls, cfg: dict) -> "V16TeacherAnchoredResidualMoE":
        return cls(cfg)

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().square().mean(dim=(1, 2, 3, 4)).sqrt()

    def _scale_ref(self, x_base: torch.Tensor) -> torch.Tensor:
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
            return torch.zeros(
                reference.shape[0], device=reference.device, dtype=reference.dtype
            )
        return reliability.float().mean(dim=(1, 2, 3, 4)).to(dtype=reference.dtype)

    def _active_scale_weight(self, scale_gate: torch.Tensor) -> torch.Tensor:
        if scale_gate.ndim != 2 or scale_gate.shape[1] != 3:
            raise ValueError(f"Expected scale_gate [B,3], got {tuple(scale_gate.shape)}")
        active = self.active_scale_mask.to(
            device=scale_gate.device,
            dtype=scale_gate.dtype,
        ).view(1, 3)
        weights = (
            scale_gate * active
            if self.use_scale_guidance
            else active.expand_as(scale_gate)
        )
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _build_condition(
        self,
        x_f: torch.Tensor,
        m_f: torch.Tensor,
        r_m: torch.Tensor | None,
        r_c: torch.Tensor | None,
        x_base: torch.Tensor,
        x_candidate: torch.Tensor,
        delta_candidate: torch.Tensor,
        scale_weight: torch.Tensor,
        x_hat_shared: torch.Tensor | None,
        x_hat_route: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q_f = compute_observation_stats(m_f)
        difficulty = q_f[:, (0, 2, 3)]
        reliability = torch.stack(
            (
                self._mean_reliability(r_m, q_f),
                self._mean_reliability(r_c, q_f),
            ),
            dim=1,
        )
        scale_mean = self._scale_ref(x_base).float().flatten(1).mean(dim=1)
        candidate_relative_rms = self._rms(delta_candidate) / scale_mean.clamp_min(1e-6)

        if x_hat_shared is not None and x_hat_route is not None:
            disagreement = masked_rms_per_sample(
                x_hat_shared.detach() - x_hat_route.detach(),
                1.0 - m_f,
            )
        else:
            disagreement = torch.zeros_like(candidate_relative_rms)
        base_obs = observed_mae_per_sample(x_base.detach(), x_f, m_f)
        candidate_obs = observed_mae_per_sample(x_candidate.detach(), x_f, m_f)
        observed_gain = (base_obs - candidate_obs) / base_obs.clamp_min(1e-6)

        original_nine = torch.cat(
            (
                difficulty,
                reliability,
                scale_weight.detach().to(dtype=q_f.dtype),
                candidate_relative_rms.detach().to(dtype=q_f.dtype).view(-1, 1),
            ),
            dim=1,
        )
        if self.condition_dim == 9:
            condition = original_nine
        else:
            condition = torch.cat(
                (
                    original_nine,
                    disagreement.detach().to(dtype=q_f.dtype).view(-1, 1),
                    base_obs.detach().to(dtype=q_f.dtype).view(-1, 1),
                    observed_gain.detach().to(dtype=q_f.dtype).view(-1, 1),
                ),
                dim=1,
            )
        components = {
            "candidate_relative_rms": candidate_relative_rms,
            "branch_disagreement": disagreement,
            "observed_base_mae": base_obs,
            "observed_candidate_gain": observed_gain,
        }
        return condition, components

    def configure_training_stage(self, epoch: int) -> str:
        """Apply the documented two-stage freeze policy for the given epoch."""
        self.training_stage = "warmup" if epoch <= self.warmup_epochs else "joint"
        warmup = self.training_stage == "warmup"
        for parameter in self.student_backbone.parameters():
            parameter.requires_grad_(not warmup)
        for parameter in self.residual_proposer.parameters():
            parameter.requires_grad_(True)
        for parameter in self.calibrator.parameters():
            parameter.requires_grad_(not warmup and self.calibration_mode == "learned")
        return self.training_stage

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
        student_outputs = self.student_backbone(
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
            return student_outputs

        features = student_outputs["features"]
        x_base = student_outputs["x_hat_main"]
        scale_gate = student_outputs["gates"]["scale_gate"]
        scale_source = scale_gate.detach() if self.detach_scale_gate else scale_gate
        scale_weight = self._active_scale_weight(scale_source)

        residual_inputs = {
            key: features[key] for key in ("z_f", "z_m", "z_c", "h_main")
        }
        if self.detach_residual_inputs:
            residual_inputs = {
                key: value.detach() for key, value in residual_inputs.items()
            }
        residual_outputs = self.residual_proposer(
            z_f=residual_inputs["z_f"],
            z_m=residual_inputs["z_m"],
            z_c=residual_inputs["z_c"],
            h_main=residual_inputs["h_main"],
            scale_weight=scale_weight,
        )
        delta_raw = residual_outputs["delta_raw"]
        direction = torch.tanh(delta_raw)
        scale_ref = self._scale_ref(x_base)
        delta_candidate = self.rho * scale_ref * direction
        x_candidate = x_base.detach() + delta_candidate

        condition, condition_parts = self._build_condition(
            x_f=x_f,
            m_f=m_f,
            r_m=r_m,
            r_c=r_c,
            x_base=x_base,
            x_candidate=x_candidate,
            delta_candidate=delta_candidate,
            scale_weight=scale_weight,
            x_hat_shared=student_outputs.get("x_hat_shared"),
            x_hat_route=student_outputs.get("x_hat_route"),
        )
        if self.training_stage == "warmup" or self.calibration_mode == "fixed_one":
            alpha_logit = x_base.new_full((x_base.shape[0], 1), 20.0)
            alpha = x_base.new_ones((x_base.shape[0], 1, 1, 1, 1))
        else:
            alpha_logit = self.calibrator.forward_logits(condition)
            alpha = torch.sigmoid(alpha_logit).view(-1, 1, 1, 1, 1)
        effective_delta = alpha * delta_candidate
        x_final = x_base + effective_delta

        output_features = dict(features)
        output_features.update(residual_outputs)
        output_features["delta_candidate"] = delta_candidate
        output_features["effective_delta"] = effective_delta
        diagnostics = dict(student_outputs.get("diagnostics", {}))
        diagnostics["v16"] = {
            "active_scale_weight_f": scale_weight[:, 0],
            "active_scale_weight_m": scale_weight[:, 1],
            "active_scale_weight_c": scale_weight[:, 2],
            "delta_raw_rms": self._rms(delta_raw),
            "direction_rms": self._rms(direction),
            "delta_candidate_rms": self._rms(delta_candidate),
            "effective_delta_rms": self._rms(effective_delta),
            "residual_alpha": alpha.flatten(1).mean(dim=1),
            "scale_ref": scale_ref.detach().float().flatten(1).mean(dim=1),
            **condition_parts,
        }

        outputs = dict(student_outputs)
        outputs.update({
            "x_hat_main": x_final,
            "x_hat_base": x_base,
            "x_hat_candidate": x_candidate,
            "delta_raw": delta_raw,
            "delta_candidate": delta_candidate,
            "delta_effective": effective_delta,
            "residual_alpha": alpha,
            "residual_alpha_logit": alpha_logit,
            "calibration_condition": condition,
            "active_scale_weight": scale_weight,
            "scale_ref": scale_ref,
            "features": output_features,
            "diagnostics": diagnostics,
            "v16_enabled": True,
            "v16_stage": self.training_stage,
            "branch_mode": "v16_teacher_anchored_continuous_calibration",
            "calibration_mode": self.calibration_mode,
            "calibration_supervision": self.calibration_supervision,
            "residual_scale_mode": self.residual_scale_mode,
        })
        return outputs
