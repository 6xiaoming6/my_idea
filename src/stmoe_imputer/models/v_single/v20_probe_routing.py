from __future__ import annotations

from collections.abc import Mapping
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..experts import TopKRoutedExpertPool
from ..main_branch import MultiScaleMoEBackbone
from ..scale_utils import get_active_scales
from .v20_probe_mask import GeometryMatchedProbeBuilder


class SharedProbeDecoder(nn.Module):
    """One decoder shared by every expert and every spatial scale."""

    def __init__(
        self,
        dim: int,
        c_out: int,
        dropout: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        hidden = max(16, dim // 2)
        self.in_proj = nn.Conv3d(dim, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout3d(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.out_proj = nn.Conv3d(hidden, c_out, kernel_size=1)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, expert_features: torch.Tensor) -> torch.Tensor:
        if expert_features.ndim != 6:
            raise ValueError(
                "expert_features must have shape [B,E,D,T,H,W], got "
                f"{tuple(expert_features.shape)}"
            )
        b, e, d, t, h, w = expert_features.shape
        value = expert_features.reshape(b * e, d, t, h, w)
        prediction = self.out_proj(self.dropout(self.act(self.in_proj(value))))
        return prediction.reshape(b, e, prediction.shape[1], t, h, w)


class ProbeCompetenceEvaluator(nn.Module):
    """Run geometry-matched on-site expert exams without leaking hidden targets."""

    _SCALE_MODULES = {
        "fine": ("embed_f", "routed_expert_pool"),
        "mid": ("embed_m", "routed_expert_pool_m"),
        "coarse": ("embed_c", "routed_expert_pool_c"),
    }

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        main_cfg = model_cfg["main"]
        v20_cfg = model_cfg.get("v20", {})
        self.enabled = bool(v20_cfg.get("enabled", True))
        self.probe_enabled = bool(v20_cfg.get("probe_enabled", True))
        self.temperature = float(v20_cfg.get("probe_temperature", 0.5))
        self.eta_max = float(v20_cfg.get("probe_eta_max", 0.85))
        self.eps = float(v20_cfg.get("probe_eps", 1e-6))
        self.routing_fusion = str(v20_cfg.get("routing_fusion", "hybrid"))
        self.confidence_mode = str(v20_cfg.get("confidence_mode", "legacy_best"))
        self.entropy_threshold = float(
            v20_cfg.get("competence_entropy_threshold", 0.02)
        )
        self.entropy_saturation = float(
            v20_cfg.get("competence_entropy_saturation", 0.08)
        )
        self.apply_evidence_during_training = bool(
            v20_cfg.get("apply_evidence_during_training", True)
        )
        evidence_scales = v20_cfg.get(
            "routing_evidence_scales", ("fine", "mid", "coarse")
        )
        if isinstance(evidence_scales, str):
            evidence_scales = (evidence_scales,)
        self.routing_evidence_scales = frozenset(str(scale) for scale in evidence_scales)
        if self.temperature <= 0.0:
            raise ValueError("model.v20.probe_temperature must be positive")
        if not 0.0 <= self.eta_max <= 1.0:
            raise ValueError("model.v20.probe_eta_max must be in [0,1]")
        if self.routing_fusion not in {
            "hybrid", "neutral_hybrid", "exam_only", "prior_only"
        }:
            raise ValueError(
                "model.v20.routing_fusion must be "
                "hybrid/neutral_hybrid/exam_only/prior_only"
            )
        if self.confidence_mode not in {"legacy_best", "entropy_threshold"}:
            raise ValueError(
                "model.v20.confidence_mode must be legacy_best/entropy_threshold"
            )
        if not 0.0 <= self.entropy_threshold < self.entropy_saturation <= 1.0:
            raise ValueError(
                "V20 entropy confidence requires 0 <= threshold < saturation <= 1"
            )
        invalid_scales = self.routing_evidence_scales - {"fine", "mid", "coarse"}
        if invalid_scales:
            raise ValueError(f"Unknown V20 routing evidence scales: {sorted(invalid_scales)}")
        if not bool(v20_cfg.get("detach_probe_features", True)):
            raise ValueError("Full V20 requires detach_probe_features=true")
        if not bool(v20_cfg.get("detach_routing_evidence", True)):
            raise ValueError("Full V20 requires detach_routing_evidence=true")
        descriptor_cfg = v20_cfg.get("descriptor_weights", {})
        descriptor_weights = (
            float(descriptor_cfg.get("spatial_small", 1.0)),
            float(descriptor_cfg.get("spatial_large", 1.0)),
            float(descriptor_cfg.get("temporal", 0.5)),
            float(descriptor_cfg.get("reliability", 1.0)),
        )
        self.probe_builder = GeometryMatchedProbeBuilder(
            probe_ratio=float(v20_cfg.get("probe_ratio", 0.08)),
            min_count=int(v20_cfg.get("probe_min_count", 8)),
            max_count=int(v20_cfg.get("probe_max_count", 128)),
            min_remaining=int(v20_cfg.get("probe_min_remaining", 16)),
            spatial_kernel_small=int(v20_cfg.get("spatial_kernel_small", 3)),
            spatial_kernel_large=int(v20_cfg.get("spatial_kernel_large", 7)),
            temporal_kernel=int(v20_cfg.get("temporal_kernel", 3)),
            reliability_kernel=int(v20_cfg.get("reliability_kernel", 5)),
            descriptor_weights=descriptor_weights,
            selection_mode=str(v20_cfg.get("probe_mode", "geometry_matched")),
            eps=self.eps,
        )
        self.probe_decoder = SharedProbeDecoder(
            dim=int(main_cfg["dim"]),
            c_out=int(model_cfg["c_in"]),
            dropout=float(v20_cfg.get("probe_decoder_dropout", 0.0)),
            zero_init=bool(v20_cfg.get("probe_decoder_zero_init", True)),
        )

    @staticmethod
    def _exam_features(
        embed: nn.Module,
        expert_pool: TopKRoutedExpertPool,
        x_exam: torch.Tensor,
        m_exam: torch.Tensor,
    ) -> torch.Tensor:
        # Probe measurement is deterministic and must not consume the main path's
        # dropout RNG or update/train the examined modules through probe loss.
        embed_training = embed.training
        pool_training = expert_pool.training
        embed.eval()
        expert_pool.eval()
        try:
            with torch.no_grad():
                encoded = embed(x_exam, m_exam)
                features = expert_pool.forward_all(encoded)
        finally:
            embed.train(embed_training)
            expert_pool.train(pool_training)
        return features.detach()

    def _evaluate_scale(
        self,
        *,
        x: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        embed: nn.Module,
        expert_pool: TopKRoutedExpertPool,
    ) -> dict[str, torch.Tensor]:
        probe = self.probe_builder.build(mask, reliability)
        probe_mask = probe["probe_mask"]
        exam_mask = mask * (1.0 - probe_mask)
        exam_value = x * exam_mask
        features = self._exam_features(embed, expert_pool, exam_value, exam_mask)
        prediction = self.probe_decoder(features)

        selected = probe_mask[:, None].expand(
            -1, expert_pool.num_experts, x.shape[1], -1, -1, -1
        ).float()
        target = x[:, None].expand_as(prediction).float()
        prediction_f = prediction.float()
        count = selected.sum(dim=(2, 3, 4, 5)).clamp_min(1.0)
        raw_error = ((prediction_f - target).abs() * selected).sum(
            dim=(2, 3, 4, 5)
        ) / count
        smooth = F.smooth_l1_loss(prediction_f, target, reduction="none")
        sample_expert_loss = (smooth * selected).sum(dim=(2, 3, 4, 5)) / count
        valid_f = probe["valid"].float()
        valid_expert_count = (valid_f.sum() * expert_pool.num_experts).clamp_min(1.0)
        probe_loss = (sample_expert_loss * valid_f[:, None]).sum() / valid_expert_count

        evidence_error = raw_error.detach()
        mean_error = evidence_error.mean(dim=1, keepdim=True).clamp_min(self.eps)
        normalized_error = evidence_error / mean_error
        competence = torch.softmax(-normalized_error / self.temperature, dim=-1)
        uniform = torch.full_like(competence, 1.0 / expert_pool.num_experts)
        competence = torch.where(probe["valid"][:, None], competence, uniform)
        legacy_confidence = (
            1.0 - normalized_error.min(dim=1).values
        ).clamp(0.0, 1.0)
        competence_entropy = -(
            competence * competence.clamp_min(self.eps).log()
        ).sum(dim=-1)
        competence_certainty = (
            1.0 - competence_entropy / math.log(float(expert_pool.num_experts))
        ).clamp(0.0, 1.0)
        sorted_error = normalized_error.sort(dim=-1).values
        rank_margin = sorted_error[:, 1] - sorted_error[:, 0]
        if self.confidence_mode == "entropy_threshold":
            confidence = (
                (competence_certainty - self.entropy_threshold)
                / (self.entropy_saturation - self.entropy_threshold)
            ).clamp(0.0, 1.0)
        else:
            confidence = legacy_confidence
        confidence = confidence * valid_f
        if self.routing_fusion in {"hybrid", "neutral_hybrid"}:
            eta = self.eta_max * confidence
        elif self.routing_fusion == "exam_only":
            eta = valid_f
        else:
            eta = torch.zeros_like(confidence)

        return {
            **probe,
            "exam_mask": exam_mask,
            "exam_value": exam_value,
            "probe_loss": probe_loss,
            "raw_error": raw_error.detach(),
            "normalized_error": normalized_error.detach(),
            "competence": competence.detach(),
            "competence_certainty": competence_certainty.detach(),
            "rank_margin": rank_margin.detach(),
            "legacy_confidence": legacy_confidence.detach(),
            "confidence": confidence.detach(),
            "eta": eta[:, None].detach(),
        }

    def forward(
        self,
        *,
        backbone: MultiScaleMoEBackbone,
        x_f: torch.Tensor,
        m_f: torch.Tensor,
        r_f: torch.Tensor,
        x_m: torch.Tensor,
        m_m: torch.Tensor,
        r_m: torch.Tensor,
        x_c: torch.Tensor,
        m_c: torch.Tensor,
        r_c: torch.Tensor,
        scale_mode: str,
    ) -> dict[str, object]:
        zero = self.probe_decoder.out_proj.weight.sum() * 0.0
        result: dict[str, object] = {"probe_loss": zero, "routing_evidence": {}}
        if not self.enabled or not self.probe_enabled:
            return result
        values: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
            "fine": (x_f, m_f, r_f),
            "mid": (x_m, m_m, r_m),
            "coarse": (x_c, m_c, r_c),
        }
        losses = []
        routing_evidence: dict[str, dict[str, object]] = {}
        for scale in get_active_scales(scale_mode):
            embed_name, pool_name = self._SCALE_MODULES[scale]
            x, mask, reliability = values[scale]
            scale_result = self._evaluate_scale(
                x=x,
                mask=mask,
                reliability=reliability,
                embed=getattr(backbone, embed_name),
                expert_pool=getattr(backbone, pool_name),
            )
            proposed_eta = scale_result["eta"]
            evidence_allowed = scale in self.routing_evidence_scales
            training_allowed = self.apply_evidence_during_training or not self.training
            if not evidence_allowed or not training_allowed:
                scale_result["eta"] = torch.zeros_like(proposed_eta)
            scale_result["proposed_eta"] = proposed_eta
            scale_result["evidence_allowed"] = evidence_allowed
            result[scale] = scale_result
            losses.append(scale_result["probe_loss"])
            if evidence_allowed:
                routing_evidence[scale] = {
                    "competence": scale_result["competence"],
                    "eta": scale_result["eta"],
                    "fusion_mode": (
                        "neutral_multiplicative"
                        if self.routing_fusion == "neutral_hybrid"
                        else "convex_geometric"
                    ),
                }
        result["probe_loss"] = torch.stack(losses).mean() if losses else zero
        result["routing_evidence"] = routing_evidence
        return result
