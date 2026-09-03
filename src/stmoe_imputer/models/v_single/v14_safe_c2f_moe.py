from __future__ import annotations

import torch
from torch import nn

from ..main_branch import MultiScaleMoEBackbone
from .difficulty_condition import DifficultyConditionEncoder
from .local_residual_gate import BoundedLocalResidualGate
from .safe_c2f_refiner import SafeCoarseToFineRefiner
from .safety_controller import ObservedConsistencyEvaluator, SafetyController


class V14SafeC2FMoE(nn.Module):
    """Main-compatible safeguarded coarse-to-fine residual wrapper."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.main_backbone = MultiScaleMoEBackbone.from_config(cfg)
        model_cfg = cfg["model"]
        main_cfg = model_cfg["main"]
        v14_cfg = model_cfg.get("v14", {})
        self.enabled = bool(v14_cfg.get("enabled", True))
        self.reuse_main_features = bool(v14_cfg.get("reuse_main_features", True))
        if not self.reuse_main_features:
            raise ValueError(
                "V14 requires reuse_main_features=true; duplicating the main encoder/expert pool "
                "is intentionally unsupported."
            )
        self.use_main_bypass = bool(v14_cfg.get("use_main_bypass", True))
        self.use_observed_consistency = bool(v14_cfg.get("use_observed_consistency", True))
        self.use_geometry_descriptor = bool(v14_cfg.get("use_geometry_descriptor", True))
        self.use_scale_reliability = bool(v14_cfg.get("use_scale_reliability", True))
        self.effective_residual_mode = str(
            v14_cfg.get("effective_residual_mode", "legacy")
        )
        if self.effective_residual_mode not in {"legacy", "identifiable"}:
            raise ValueError(
                "model.v14.effective_residual_mode must be 'legacy' or "
                f"'identifiable', got {self.effective_residual_mode!r}"
            )
        self.identifiable_eps = float(v14_cfg.get("identifiable_eps", 1e-6))
        self.local_final_gate_mode = str(
            v14_cfg.get("local_final_gate_mode", "legacy")
        )
        if self.local_final_gate_mode not in {"legacy", "temporal", "regional"}:
            raise ValueError(
                "model.v14.local_final_gate_mode must be legacy/temporal/regional, "
                f"got {self.local_final_gate_mode!r}"
            )

        difficulty_out = int(v14_cfg.get("difficulty_out_dim", 32))
        self.condition_encoder = DifficultyConditionEncoder(
            hidden_dim=int(v14_cfg.get("difficulty_hidden", 32)),
            out_dim=difficulty_out,
            dropout=float(v14_cfg.get("controller_dropout", 0.1)),
            enabled=bool(v14_cfg.get("difficulty_enabled", True)),
            use_spatial_block=bool(v14_cfg.get("difficulty_use_spatial_block", True)),
            use_cross_scale_consistency=bool(
                v14_cfg.get("difficulty_use_cross_scale_consistency", True)
            ),
        )
        self.consistency_evaluator = ObservedConsistencyEvaluator()
        precondition_dim = difficulty_out + 2 + 3 + 5
        self.controller = SafetyController(
            precondition_dim=precondition_dim,
            consistency_dim=ObservedConsistencyEvaluator.output_dim,
            hidden_dim=int(v14_cfg.get("controller_hidden", 64)),
            dropout=float(v14_cfg.get("controller_dropout", 0.1)),
            alpha_mid_bias=float(v14_cfg.get("alpha_mid_bias", -3.0)),
            alpha_fine_bias=float(v14_cfg.get("alpha_fine_bias", -3.0)),
            alpha_final_bias=float(v14_cfg.get("alpha_final_bias", -5.0)),
            alpha_mid_max=float(v14_cfg.get("alpha_mid_max", 0.8)),
            alpha_fine_max=float(v14_cfg.get("alpha_fine_max", 0.8)),
            alpha_final_max=float(v14_cfg.get("alpha_final_max", 0.5)),
            zero_init=bool(v14_cfg.get("controller_zero_init", True)),
            dynamic_gate=bool(v14_cfg.get("dynamic_gate", True)),
            c_out=int(model_cfg["c_in"]),
            channel_final_gate=bool(v14_cfg.get("channel_final_gate", False)),
            channel_stats_dim=5,
            channel_gain_delta=float(v14_cfg.get("channel_gain_delta", 0.2)),
            final_gate_mode=str(v14_cfg.get("final_gate_mode", "legacy")),
            monotonic_advantage_gain=float(
                v14_cfg.get("monotonic_advantage_gain", 1.0)
            ),
            monotonic_context_bound=float(
                v14_cfg.get("monotonic_context_bound", 0.5)
            ),
            monotonic_eps=float(v14_cfg.get("monotonic_eps", 1e-6)),
        )
        self.refiner = SafeCoarseToFineRefiner(
            dim=int(main_cfg["dim"]),
            c_out=int(model_cfg["c_in"]),
            hidden=int(v14_cfg.get("refiner_hidden", 32)),
            prediction_embed_dim=int(v14_cfg.get("prediction_embed_dim", 16)),
            num_groups=int(main_cfg.get("num_groups", 8)),
            dropout=float(v14_cfg.get("refiner_dropout", main_cfg.get("dropout", 0.0))),
            correction_hidden=int(v14_cfg.get("correction_hidden", 16)),
            correction_zero_init=bool(v14_cfg.get("correction_zero_init", True)),
            fine_uses_main_feature=bool(v14_cfg.get("fine_uses_main_feature", True)),
        )
        self.local_residual_gate: BoundedLocalResidualGate | None = None
        if self.local_final_gate_mode != "legacy":
            if self.controller.channel_final_gate:
                raise ValueError(
                    "BRLG and channel_final_gate are separate structural candidates "
                    "and cannot be enabled together"
                )
            if self.controller.final_gate_mode != "legacy":
                raise ValueError(
                    "BRLG requires the original legacy global final gate so that "
                    "the exploration changes one structural variable"
                )
            # Do not advance the outer RNG state. Common V14 parameters and
            # subsequent data-loader/dropout randomness remain seed-matched.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1419)
                self.local_residual_gate = BoundedLocalResidualGate(
                    feature_dim=int(main_cfg["dim"]),
                    hidden_dim=int(v14_cfg.get("local_gate_hidden", 16)),
                    mode=self.local_final_gate_mode,
                    max_relative_delta=float(
                        v14_cfg.get("local_gate_max_relative_delta", 0.2)
                    ),
                    spatial_divisor=int(
                        v14_cfg.get("local_gate_spatial_divisor", 4)
                    ),
                    num_groups=int(main_cfg.get("num_groups", 8)),
                    detach_inputs=bool(
                        v14_cfg.get("local_gate_detach_inputs", True)
                    ),
                )
        if bool(v14_cfg.get("freeze_main", False)):
            for parameter in self.main_backbone.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg: dict) -> "V14SafeC2FMoE":
        return cls(cfg)

    @staticmethod
    def _geometry(
        z_f: torch.Tensor,
        z_c: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        batch_size = z_f.shape[0]
        if not enabled:
            return torch.zeros(batch_size, 5, device=z_f.device, dtype=z_f.dtype)
        height, width = z_f.shape[-2:]
        coarse_height, coarse_width = z_c.shape[-2:]
        values = torch.tensor(
            (
                height / 32.0,
                width / 32.0,
                height / max(float(width), 1.0),
                min(coarse_height, coarse_width) / 8.0,
                coarse_height * coarse_width / max(float(height * width), 1.0),
            ),
            device=z_f.device,
            dtype=z_f.dtype,
        )
        return values.view(1, 5).expand(batch_size, -1)

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().square().mean(dim=(1, 2, 3, 4)).sqrt()

    @staticmethod
    def _channel_observed_stats(
        x_base: torch.Tensor,
        x_ctf: torch.Tensor,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        base = x_base.detach().float()
        ctf = x_ctf.detach().float()
        obs = x_obs.detach().float()
        observed = mask.detach().float().expand_as(base)
        reduce_dims = (2, 3, 4)
        count = observed.sum(dim=reduce_dims).clamp_min(1.0)
        observed_abs_mean = (obs.abs() * observed).sum(dim=reduce_dims) / count
        observed_rms = (
            (obs.square() * observed).sum(dim=reduce_dims) / count + 1e-6
        ).sqrt()
        base_error = ((base - obs).abs() * observed).sum(dim=reduce_dims) / count
        ctf_error = ((ctf - obs).abs() * observed).sum(dim=reduce_dims) / count
        delta_mean = ((ctf - base).abs() * observed).sum(dim=reduce_dims) / count
        return torch.stack(
            (observed_abs_mean, observed_rms, base_error, ctf_error, delta_mean),
            dim=-1,
        ).to(dtype=x_base.dtype)

    @staticmethod
    def _identifiable_effective_residual(
        delta: torch.Tensor,
        alpha: torch.Tensor,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduce_dims = tuple(range(1, delta.ndim))
        delta_rms = (
            delta.float().square().mean(dim=reduce_dims) + eps
        ).sqrt()
        direction = delta / delta_rms.to(dtype=delta.dtype).view(-1, 1, 1, 1, 1)
        observed = mask.float().expand_as(x_obs)
        observed_count = observed.sum(dim=reduce_dims).clamp_min(1.0)
        observed_rms = (
            (x_obs.float().square() * observed).sum(dim=reduce_dims)
            / observed_count
            + eps
        ).sqrt()
        magnitude = alpha * observed_rms.to(dtype=alpha.dtype).view(-1, 1, 1, 1, 1)
        return magnitude * direction, observed_rms

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
        routing_evidence: dict[str, dict[str, object]] | None = None,
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
            routing_evidence=routing_evidence,
        )
        if not self.enabled:
            return base_outputs

        features = base_outputs["features"]
        gates = base_outputs["gates"]
        x_base = base_outputs["x_hat_main"]
        z_f, z_m, z_c = features["z_f"], features["z_m"], features["z_c"]
        h_main = features["h_main"]
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
        batch_size = x_f.shape[0]
        if self.use_scale_reliability:
            reliability = torch.stack(
                (
                    r_m.float().mean(dim=(1, 2, 3, 4)) if r_m is not None else m_m.float().mean(dim=(1, 2, 3, 4)),
                    r_c.float().mean(dim=(1, 2, 3, 4)) if r_c is not None else m_c.float().mean(dim=(1, 2, 3, 4)),
                ),
                dim=1,
            ).to(dtype=x_f.dtype)
        else:
            reliability = torch.zeros(batch_size, 2, device=x_f.device, dtype=x_f.dtype)
        geometry = self._geometry(z_f, z_c, self.use_geometry_descriptor)
        scale_gate = gates["scale_gate"].to(dtype=x_f.dtype)
        precondition = torch.cat(
            (condition_embedding, reliability, scale_gate, geometry), dim=-1
        )
        alpha_mid, alpha_fine = self.controller.refinement_gates(precondition)
        refine = self.refiner(
            z_f=z_f,
            z_m=z_m,
            z_c=z_c,
            h_main=h_main,
            x_base=x_base,
            alpha_mid=alpha_mid,
            alpha_fine=alpha_fine,
        )
        if self.use_observed_consistency:
            consistency = self.consistency_evaluator(
                x_base=x_base,
                x_ctf=refine["x_hat_ctf"],
                x_obs=x_f,
                mask=m_f,
            )
        else:
            consistency = torch.zeros(
                batch_size,
                ObservedConsistencyEvaluator.output_dim,
                device=x_f.device,
                dtype=x_f.dtype,
            )
        channel_stats = (
            self._channel_observed_stats(x_base, refine["x_hat_ctf"], x_f, m_f)
            if self.controller.channel_final_gate
            else None
        )
        alpha_final_global, final_gate_diagnostics = self.controller.final_gate(
            precondition, consistency, channel_stats=channel_stats
        )
        alpha_final = alpha_final_global
        if self.local_residual_gate is not None:
            alpha_final, local_gate_diagnostics = self.local_residual_gate(
                alpha_global=alpha_final_global,
                h_main=h_main,
                delta_ctf=refine["delta_ctf"],
                x_ctf=refine["x_hat_ctf"],
                x_base=x_base,
                x_obs=x_f,
                mask=m_f,
                alpha_max=self.controller.alpha_final_max,
            )
            final_gate_diagnostics.update(local_gate_diagnostics)
        if self.use_main_bypass:
            if self.effective_residual_mode == "identifiable":
                effective_residual, observed_scale = (
                    self._identifiable_effective_residual(
                        refine["delta_ctf"],
                        alpha_final,
                        x_f,
                        m_f,
                        eps=self.identifiable_eps,
                    )
                )
            else:
                effective_residual = alpha_final * refine["delta_ctf"]
                observed_scale = torch.zeros(
                    batch_size, device=x_f.device, dtype=torch.float32
                )
            x_final = x_base + effective_residual
        else:
            x_final = refine["x_hat_ctf"]
            alpha_final = torch.ones_like(alpha_final)
            effective_residual = x_final - x_base
            observed_scale = torch.zeros(
                batch_size, device=x_f.device, dtype=torch.float32
            )

        outputs = dict(base_outputs)
        output_features = dict(features)
        output_features.update({
            "h_main_base": h_main,
            "delta_mid": refine["delta_mid"],
            "delta_fine": refine["delta_fine"],
            "delta_ctf": refine["delta_ctf"],
        })
        output_diagnostics = dict(base_outputs.get("diagnostics", {}))
        output_diagnostics["v14"] = {
            "alpha_mid": alpha_mid.flatten(1).mean(dim=1),
            "alpha_fine": alpha_fine.flatten(1).mean(dim=1),
            "alpha_final": alpha_final.flatten(1).mean(dim=1),
            "alpha_final_global": alpha_final_global.flatten(1).mean(dim=1),
            "difficulty_f": difficulty["score_f"],
            "difficulty_m": difficulty["score_m"],
            "difficulty_c": difficulty["score_c"],
            "scale_reliability_m": reliability[:, 0],
            "scale_reliability_c": reliability[:, 1],
            "coarse_geometry_score": geometry[:, 3],
            "observed_error_base": consistency[:, 0],
            "observed_error_ctf": consistency[:, 1],
            "observed_advantage": -consistency[:, 2],
            "observed_delta_mean": consistency[:, 3],
            "observed_delta_q95": consistency[:, 4],
            "delta_mid_norm": self._rms(refine["delta_mid"]),
            "delta_fine_norm": self._rms(refine["delta_fine"]),
            "delta_ctf_norm": self._rms(refine["delta_ctf"]),
            "effective_residual_norm": self._rms(effective_residual),
            "identifiable_observed_scale": observed_scale,
        }
        channel_gain = final_gate_diagnostics.get("channel_gain")
        if torch.is_tensor(channel_gain):
            for channel_index in range(channel_gain.shape[1]):
                output_diagnostics["v14"][
                    f"channel_gain_{channel_index}"
                ] = channel_gain[:, channel_index]
            output_diagnostics["v14"]["channel_gain_saturation"] = (
                final_gate_diagnostics["channel_gain_saturation"].mean(dim=1)
            )
        for key in ("final_gate_context", "relative_observed_advantage"):
            value = final_gate_diagnostics.get(key)
            if torch.is_tensor(value):
                output_diagnostics["v14"][key] = value
        for key in (
            "local_gate_logits",
            "local_gate_modulation",
            "local_gate_modulation_lowres",
        ):
            value = final_gate_diagnostics.get(key)
            if torch.is_tensor(value):
                output_diagnostics["v14"][key] = value
        outputs.update({
            "x_hat_main": x_final,
            "x_hat_base": x_base,
            "x_hat_ctf": refine["x_hat_ctf"],
            "x_hat_mid": refine["x_hat_mid"],
            "x_hat_coarse": refine["x_hat_coarse"],
            "features": output_features,
            "diagnostics": output_diagnostics,
            "branch_mode": "v14_safe_c2f",
            "v14_enabled": True,
        })
        return outputs
