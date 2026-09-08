from __future__ import annotations

import math

import torch
from torch import nn

from .v14_safe_c2f_moe import V14SafeC2FMoE


class DistortionAcceptanceGate(nn.Module):
    """Bounded local gate between the V14 and measure-preserving pyramids.

    The gate sees observation-only descriptors.  Its output is initialized
    close to zero, so the complete model starts from the V14 representation
    and must learn evidence for accepting a measure-path correction.
    """

    def __init__(
        self,
        evidence_dim: int = 7,
        hidden_dim: int = 16,
        alpha_max: float = 1.0,
        alpha_init: float = 0.01,
        use_explicit_distortion: bool = True,
    ) -> None:
        super().__init__()
        if evidence_dim < 0:
            raise ValueError("evidence_dim must be non-negative")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 < alpha_max <= 1.0:
            raise ValueError("alpha_max must be in (0, 1]")
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError("alpha_init must be in (0, alpha_max)")
        self.evidence_dim = int(evidence_dim)
        self.alpha_max = float(alpha_max)
        self.use_explicit_distortion = bool(use_explicit_distortion)
        self.descriptor_dim = 3 + self.evidence_dim
        self.net = nn.Sequential(
            nn.Conv3d(self.descriptor_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        initial_probability = alpha_init / alpha_max
        initial_logit = math.log(initial_probability / (1.0 - initial_probability))
        nn.init.constant_(self.net[-1].bias, initial_logit)

    def forward(
        self,
        legacy: torch.Tensor,
        measure: torch.Tensor,
        legacy_reliability: torch.Tensor,
        measure_reliability: torch.Tensor,
        evidence: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if legacy.shape != measure.shape:
            raise ValueError(
                f"Legacy/measure shape mismatch: {tuple(legacy.shape)} vs {tuple(measure.shape)}"
            )
        if legacy_reliability.shape != measure_reliability.shape:
            raise ValueError(
                "Legacy/measure reliability shape mismatch: "
                f"{tuple(legacy_reliability.shape)} vs {tuple(measure_reliability.shape)}"
            )
        delta = measure - legacy
        magnitude = legacy.abs().mean(dim=1, keepdim=True) + measure.abs().mean(
            dim=1, keepdim=True
        )
        denominator = magnitude.clamp_min(1e-4)
        relative_abs = (delta.abs().mean(dim=1, keepdim=True) / denominator).clamp(0.0, 1.0)
        relative_signed = (delta.mean(dim=1, keepdim=True) / denominator).clamp(-1.0, 1.0)
        support_gap = (measure_reliability - legacy_reliability).abs().clamp(0.0, 1.0)
        distortion = torch.cat((relative_abs, relative_signed, support_gap), dim=1)
        if not self.use_explicit_distortion:
            distortion = torch.zeros_like(distortion)

        if self.evidence_dim == 0:
            evidence_value = legacy.new_zeros(
                legacy.shape[0], 0, *legacy.shape[2:]
            )
        elif evidence is None:
            evidence_value = legacy.new_zeros(
                legacy.shape[0], self.evidence_dim, *legacy.shape[2:]
            )
        else:
            if evidence.shape[1] != self.evidence_dim or evidence.shape[2:] != legacy.shape[2:]:
                raise ValueError(
                    f"Expected evidence [B,{self.evidence_dim},T,H,W], got {tuple(evidence.shape)}"
                )
            evidence_value = evidence.to(dtype=legacy.dtype)

        descriptor = torch.cat((distortion, evidence_value), dim=1)
        alpha = self.alpha_max * torch.sigmoid(self.net(descriptor))
        calibrated = legacy + alpha * delta
        reliability = legacy_reliability + alpha * (
            measure_reliability - legacy_reliability
        )
        return {
            "value": calibrated,
            "reliability": reliability.clamp(0.0, 1.0),
            "alpha": alpha,
            "delta": delta,
            "relative_abs": relative_abs,
            "relative_signed": relative_signed,
            "support_gap": support_gap,
        }


class V21DistortionCalibratedMoE(V14SafeC2FMoE):
    """V14 anchored by a bounded, distortion-aware dual-pyramid correction."""

    def __init__(self, cfg: dict) -> None:
        # Construct V14 first so every common parameter keeps the same seeded
        # initialization as the strict anchor.
        super().__init__(cfg)
        settings = cfg["model"].get("v21_1", {})
        self.use_explicit_distortion = bool(
            settings.get("use_explicit_distortion", True)
        )
        evidence_dim = int(settings.get("evidence_dim", 7))
        gate_kwargs = {
            "evidence_dim": evidence_dim,
            "hidden_dim": int(settings.get("gate_hidden", 16)),
            "alpha_max": float(settings.get("alpha_max", 1.0)),
            "alpha_init": float(settings.get("alpha_init", 0.01)),
            "use_explicit_distortion": self.use_explicit_distortion,
        }
        # Optional V21.1 parameters must not shift any later stochastic state
        # used by data loading or dropout in the controlled experiment.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(settings.get("module_seed", 2111)))
            self.coarse_acceptance = DistortionAcceptanceGate(**gate_kwargs)

    @classmethod
    def from_config(cls, cfg: dict) -> "V21DistortionCalibratedMoE":
        return cls(cfg)

    @staticmethod
    def _sample_mean(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().mean(dim=tuple(range(1, value.ndim)))

    @staticmethod
    def _sample_rms(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().square().mean(
            dim=tuple(range(1, value.ndim))
        ).sqrt()

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
        e_f: torch.Tensor | None = None,
        e_m: torch.Tensor | None = None,
        e_c: torch.Tensor | None = None,
        x_m_measure: torch.Tensor | None = None,
        m_m_measure: torch.Tensor | None = None,
        r_m_measure: torch.Tensor | None = None,
        x_c_measure: torch.Tensor | None = None,
        m_c_measure: torch.Tensor | None = None,
        r_c_measure: torch.Tensor | None = None,
    ) -> dict:
        required = {
            "x_m_measure": x_m_measure,
            "r_m_measure": r_m_measure,
            "x_c_measure": x_c_measure,
            "r_c_measure": r_c_measure,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "V21.1 requires data.scales.pyramid_mode='dual_observation_moment'; "
                f"missing batch keys: {', '.join(missing)}"
            )
        if r_m is None:
            r_m = m_m.float()
        if r_c is None:
            r_c = m_c.float()

        coarse = self.coarse_acceptance(x_c, x_c_measure, r_c, r_c_measure, e_c)
        outputs = super().forward(
            x_f=x_f,
            m_f=m_f,
            # At the first pooling stage, V14 and the direct measure mean are
            # already identical.  Calibrating Mid would add a dead module; the
            # actual mask-scale non-commutativity begins at hierarchical Coarse.
            x_m=x_m,
            m_m=m_m,
            x_c=coarse["value"],
            m_c=m_c,
            r_m=r_m,
            r_c=coarse["reliability"],
            e_f=None,
            e_m=None,
            e_c=None,
        )
        diagnostics = dict(outputs.get("diagnostics", {}))
        mid_delta = x_m_measure - x_m
        mid_denominator = (
            x_m.abs().mean(dim=1, keepdim=True)
            + x_m_measure.abs().mean(dim=1, keepdim=True)
        ).clamp_min(1e-4)
        mid_distortion = (
            mid_delta.abs().mean(dim=1, keepdim=True) / mid_denominator
        ).clamp(0.0, 1.0)
        alpha_mid = torch.zeros_like(r_m)
        diagnostics["v21_1"] = {
            "alpha_mid": self._sample_mean(alpha_mid),
            "alpha_coarse": self._sample_mean(coarse["alpha"]),
            "alpha_mid_std": alpha_mid.detach().float().flatten(1).std(
                dim=1, unbiased=False
            ),
            "alpha_coarse_std": coarse["alpha"].detach().float().flatten(1).std(
                dim=1, unbiased=False
            ),
            "distortion_mid": self._sample_mean(mid_distortion),
            "distortion_coarse": self._sample_mean(coarse["relative_abs"]),
            "support_gap_mid": self._sample_mean(
                (r_m_measure - r_m).abs().clamp(0.0, 1.0)
            ),
            "support_gap_coarse": self._sample_mean(coarse["support_gap"]),
            "correction_mid_norm": self._sample_rms(torch.zeros_like(mid_delta)),
            "correction_coarse_norm": self._sample_rms(coarse["alpha"] * coarse["delta"]),
        }
        features = dict(outputs.get("features", {}))
        features["v21_1"] = {
            "alpha_mid": alpha_mid,
            "alpha_coarse": coarse["alpha"],
            "distortion_mid": mid_distortion,
            "distortion_coarse": coarse["relative_abs"],
            "legacy_mid": x_m,
            "legacy_coarse": x_c,
            "measure_mid": x_m_measure,
            "measure_coarse": x_c_measure,
            "calibrated_mid": x_m,
            "calibrated_coarse": coarse["value"],
        }
        outputs.update({
            "diagnostics": diagnostics,
            "features": features,
            "branch_mode": "v21_1_distortion_calibrated_v14",
            "v21_1_enabled": True,
            "v21_1_use_explicit_distortion": self.use_explicit_distortion,
        })
        return outputs


__all__ = ["DistortionAcceptanceGate", "V21DistortionCalibratedMoE"]
