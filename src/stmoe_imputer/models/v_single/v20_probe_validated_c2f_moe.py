from __future__ import annotations

import torch

from .v14_safe_c2f_moe import V14SafeC2FMoE
from .v20_probe_routing import ProbeCompetenceEvaluator


class V20ProbeValidatedC2FMoE(V14SafeC2FMoE):
    """GMSV-MoE: measure current-sample expert competence before routing."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        v20_cfg = cfg.get("model", {}).get("v20", {})
        self.v20_enabled = bool(v20_cfg.get("enabled", True))
        # Keep common V14 initialization and the global training RNG independent
        # from the newly introduced decoder parameters.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(v20_cfg.get("probe_decoder_seed", 2020)))
            self.probe_evaluator = ProbeCompetenceEvaluator(cfg)

    @classmethod
    def from_config(cls, cfg: dict) -> "V20ProbeValidatedC2FMoE":
        return cls(cfg)

    @staticmethod
    def _entropy(probability: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        value = probability.detach().float()
        return -(value * value.clamp_min(eps).log()).sum(dim=-1)

    @staticmethod
    def _topk_overlap(
        left: torch.Tensor,
        right: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        left_indices = left.topk(top_k, dim=-1).indices
        right_indices = right.topk(top_k, dim=-1).indices
        overlap = (
            left_indices[:, :, None] == right_indices[:, None, :]
        ).any(dim=-1).float().sum(dim=-1)
        return overlap / float(max(top_k, 1))

    def _attach_routing_diagnostics(
        self,
        outputs: dict,
        probe_result: dict[str, object],
    ) -> None:
        prior_gates = outputs.get("prior_gates", {})
        final_gates = outputs.get("gates", {})
        for scale in ("fine", "mid", "coarse"):
            scale_result = probe_result.get(scale)
            if not isinstance(scale_result, dict):
                continue
            prior = prior_gates.get(scale)
            final = final_gates.get(scale)
            competence = scale_result.get("competence")
            if not all(torch.is_tensor(value) for value in (prior, final, competence)):
                continue
            prior_top1 = prior.detach().argmax(dim=-1)
            probe_top1 = competence.detach().argmax(dim=-1)
            final_top1 = final.detach().argmax(dim=-1)
            scale_result.update({
                "prior_gate": prior.detach(),
                "final_gate": final.detach(),
                "prior_gate_entropy": self._entropy(prior),
                "competence_entropy": self._entropy(competence),
                "final_gate_entropy": self._entropy(final),
                "prior_top1": prior_top1,
                "probe_top1": probe_top1,
                "final_top1": final_top1,
                "prior_probe_top1_agreement": (prior_top1 == probe_top1).float(),
                "prior_final_topk_overlap": self._topk_overlap(
                    prior, final, self.main_backbone.top_k
                ),
                "probe_final_topk_overlap": self._topk_overlap(
                    competence, final, self.main_backbone.top_k
                ),
            })

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
        if r_m is None:
            r_m = m_m.float()
        if r_c is None:
            r_c = m_c.float()
        probe_result = self.probe_evaluator(
            backbone=self.main_backbone,
            x_f=x_f,
            m_f=m_f,
            r_f=m_f.float(),
            x_m=x_m,
            m_m=m_m,
            r_m=r_m,
            x_c=x_c,
            m_c=m_c,
            r_c=r_c,
            scale_mode=self.main_backbone.scale_mode,
        )
        outputs = super().forward(
            x_f=x_f,
            m_f=m_f,
            x_m=x_m,
            m_m=m_m,
            x_c=x_c,
            m_c=m_c,
            r_m=r_m,
            r_c=r_c,
            routing_evidence=probe_result["routing_evidence"],
        )
        self._attach_routing_diagnostics(outputs, probe_result)
        outputs["v20_enabled"] = self.v20_enabled
        outputs["v20_probe"] = probe_result
        outputs["branch_mode"] = "v20_probe_validated_c2f"
        return outputs
