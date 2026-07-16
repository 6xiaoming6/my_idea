from __future__ import annotations

import torch
from torch import nn

from ..main_branch import MultiScaleMoEBackbone
from ..scale_utils import get_active_scales
from ..stats import compute_observation_stats
from .residual_acceptance import ResidualAcceptanceGate
from .scale_guided_residual_adapter import ScaleGuidedResidualAdapter


class V15_1ScaleGuidedResidualMoE(nn.Module):
    """V15.1: lightweight scale-guided residual candidate with acceptance."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.main_backbone = MultiScaleMoEBackbone.from_config(cfg)
        model_cfg = cfg["model"]
        main_cfg = model_cfg["main"]
        version_cfg = model_cfg.get("v15_1", {})

        self.enabled = bool(version_cfg.get("enabled", True))
        self.detach_residual_inputs = bool(
            version_cfg.get("detach_residual_inputs", True)
        )
        self.detach_scale_gate = bool(version_cfg.get("detach_scale_gate", True))
        self.use_scale_guidance = bool(version_cfg.get("use_scale_guidance", True))
        self.acceptance_mode = str(version_cfg.get("acceptance_mode", "learned"))
        if self.acceptance_mode not in {"learned", "fixed_one"}:
            raise ValueError(
                f"acceptance_mode must be 'learned' or 'fixed_one', got {self.acceptance_mode!r}"
            )
        self.rho = float(version_cfg.get("rho", 0.05))
        self.scale_floor = float(version_cfg.get("scale_floor", 1e-3))
        if not 0.0 < self.rho <= 1.0:
            raise ValueError(f"rho must be in (0,1], got {self.rho}")
        if self.scale_floor <= 0.0:
            raise ValueError(f"scale_floor must be positive, got {self.scale_floor}")

        main_scale_mode = str(main_cfg.get("scale_mode", "fine_mid_coarse"))
        configured_residual_mode = str(
            version_cfg.get("residual_scale_mode", "inherit")
        )
        scale_mode = (
            main_scale_mode
            if configured_residual_mode == "inherit"
            else configured_residual_mode
        )
        active_names = set(get_active_scales(scale_mode))
        active_scales = tuple(
            name in active_names for name in ("fine", "mid", "coarse")
        )
        self.scale_mode = main_scale_mode
        self.residual_scale_mode = scale_mode
        self.register_buffer(
            "active_scale_mask",
            torch.tensor(active_scales, dtype=torch.bool),
            persistent=True,
        )

        residual_dim = int(version_cfg.get("residual_dim", 24))
        self.residual_adapter = ScaleGuidedResidualAdapter(
            main_dim=int(main_cfg["dim"]),
            residual_dim=residual_dim,
            out_channels=int(model_cfg["c_in"]),
            active_scales=active_scales,
            num_groups=int(main_cfg.get("num_groups", 8)),
            dropout=float(version_cfg.get("residual_dropout", 0.1)),
            zero_init=bool(version_cfg.get("residual_zero_init", True)),
        )
        condition_dim = int(version_cfg.get("accept_condition_dim", 9))
        if condition_dim != 9:
            raise ValueError(
                f"V15.1 acceptance condition has exactly 9 values, got {condition_dim}"
            )
        self.acceptance_gate = ResidualAcceptanceGate(
            condition_dim=condition_dim,
            hidden_dim=int(version_cfg.get("accept_hidden_dim", 24)),
            fixed_bias=float(version_cfg.get("accept_fixed_bias", -1.5)),
            dropout=float(version_cfg.get("accept_dropout", 0.1)),
            zero_init=bool(version_cfg.get("accept_zero_init", True)),
        )

        if bool(version_cfg.get("freeze_main", False)):
            for parameter in self.main_backbone.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg: dict) -> "V15_1ScaleGuidedResidualMoE":
        return cls(cfg)

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
        weight = scale_gate * active if self.use_scale_guidance else active.expand_as(scale_gate)
        return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _build_condition(
        self,
        m_f: torch.Tensor,
        r_m: torch.Tensor | None,
        r_c: torch.Tensor | None,
        scale_weight: torch.Tensor,
        candidate_relative_rms: torch.Tensor,
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
        # Candidate magnitude is a descriptor for the gate. Detaching it keeps
        # acceptance supervision from reshaping the candidate through this side path.
        relative = candidate_relative_rms.detach().to(dtype=q_f.dtype).view(-1, 1)
        return torch.cat(
            (difficulty, reliability, scale_weight.to(dtype=q_f.dtype), relative),
            dim=1,
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
        scale_gate = base_outputs["gates"]["scale_gate"]
        scale_weight_source = scale_gate.detach() if self.detach_scale_gate else scale_gate
        scale_weight = self._active_scale_weight(scale_weight_source)

        z_f = features["z_f"]
        z_m = features["z_m"]
        z_c = features["z_c"]
        h_main = features["h_main"]
        if self.detach_residual_inputs:
            z_f = z_f.detach()
            z_m = z_m.detach()
            z_c = z_c.detach()
            h_main = h_main.detach()

        residual_outputs = self.residual_adapter(
            z_f=z_f,
            z_m=z_m,
            z_c=z_c,
            h_main=h_main,
            scale_weight=scale_weight,
        )
        delta_raw = residual_outputs["delta_raw"]
        direction = torch.tanh(delta_raw)
        scale_ref = self._compute_scale_ref(x_base)
        delta_candidate = self.rho * scale_ref * direction
        candidate_rms = self._rms(delta_candidate)
        scale_mean = scale_ref.detach().float().flatten(1).mean(dim=1).clamp_min(1e-6)
        candidate_relative_rms = candidate_rms / scale_mean

        condition = self._build_condition(
            m_f=m_f,
            r_m=r_m,
            r_c=r_c,
            scale_weight=scale_weight,
            candidate_relative_rms=candidate_relative_rms,
        )
        if self.acceptance_mode == "learned":
            accept_logit = self.acceptance_gate.forward_logits(condition)
            accept_gate = torch.sigmoid(accept_logit).view(-1, 1, 1, 1, 1)
        else:
            accept_logit = x_base.new_full((x_base.shape[0], 1), 20.0)
            accept_gate = x_base.new_ones((x_base.shape[0], 1, 1, 1, 1))
        effective_delta = accept_gate * delta_candidate
        x_candidate = x_base.detach() + delta_candidate
        x_final = x_base + effective_delta

        output_features = dict(features)
        output_features.update(residual_outputs)
        output_features["delta_candidate"] = delta_candidate
        output_features["effective_delta"] = effective_delta

        normalized_effective = (
            effective_delta.float() / scale_ref.float().clamp_min(1e-6)
        )
        diagnostics = dict(base_outputs.get("diagnostics", {}))
        diagnostics["v15_1"] = {
            "active_scale_weight_f": scale_weight[:, 0],
            "active_scale_weight_m": scale_weight[:, 1],
            "active_scale_weight_c": scale_weight[:, 2],
            "delta_raw_rms": self._rms(delta_raw),
            "direction_rms": self._rms(direction),
            "delta_candidate_rms": candidate_rms,
            "effective_delta_rms": self._rms(effective_delta),
            "candidate_relative_rms": candidate_relative_rms,
            "effective_relative_rms": self._rms(normalized_effective),
            "accept_gate": accept_gate.flatten(1).mean(dim=1),
            "scale_ref": scale_ref.detach().float().flatten(1).mean(dim=1),
        }

        outputs = dict(base_outputs)
        outputs.update({
            "x_hat_main": x_final,
            "x_hat_base": x_base,
            "x_hat_candidate": x_candidate,
            "delta_raw": delta_raw,
            "delta_candidate": delta_candidate,
            "delta_effective": effective_delta,
            "accept_gate": accept_gate,
            "accept_logit": accept_logit,
            "active_scale_weight": scale_weight,
            "scale_ref": scale_ref,
            "features": output_features,
            "diagnostics": diagnostics,
            "branch_mode": "v15_1_scale_guided_residual_acceptance",
            "acceptance_mode": self.acceptance_mode,
            "residual_scale_mode": self.residual_scale_mode,
            "v15_1_enabled": True,
        })
        return outputs
