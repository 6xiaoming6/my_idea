"""Exact target-only diagnostics for selective repair, merged by counts."""
from __future__ import annotations

from collections import defaultdict

import torch

from stmoe_imputer.engine import _CoEQualityMetrics
from stmoe_imputer.losses import supervision_mask
from .losses import _point_errors


def _zero_pair() -> list[float]:
    return [0.0, 0.0]


class RASQualityMetrics(_CoEQualityMetrics):
    def __init__(self) -> None:
        super().__init__()
        self.ratios: dict[str, list[float]] = defaultdict(_zero_pair)
        self.confusion = [0.0, 0.0, 0.0, 0.0]  # TP, FP, TN, FN
        self.monotonic_all = [0.0, 0.0]

    def _add(self, key: str, numerator: torch.Tensor | float,
             denominator: torch.Tensor | float) -> None:
        self.ratios[key][0] += float(numerator)
        self.ratios[key][1] += float(denominator)

    @torch.no_grad()
    def update(self, outputs: dict, batch: dict) -> None:
        super().update(outputs, batch)
        coe = outputs.get("coe", {})
        old = coe.get("initial_completion")
        completions = coe.get("completions", ())
        candidates = coe.get("candidate_completions", ())
        if old is None or not completions or len(completions) != len(candidates):
            return
        target = batch["x_f_gt"]
        selected = supervision_mask(target, batch["m_f"], batch.get("target_mask"))
        q_count = selected.sum()
        if not bool(q_count):
            return
        safe_target = torch.where(selected, target.float(), 0.0)
        initial = torch.where(selected, old.float(), 0.0)
        self._add("coe_initial_mae", (initial - safe_target).abs().sum(), q_count)
        all_nonworse = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
        sample_valid = selected.flatten(1).any(dim=1)
        accepted_weights = coe.get("repair_acceptance_weights")
        for step, (candidate, accepted) in enumerate(zip(candidates, completions), start=1):
            prefix = f"coe_step{step}"
            accepted_safe = torch.where(selected, accepted.float(), 0.0)
            self._add(f"{prefix}_mae", (accepted_safe - safe_target).abs().sum(), q_count)
            old_error, candidate_error, valid = _point_errors(old, candidate, target, selected)
            _, accepted_error, _ = _point_errors(old, accepted, target, selected)
            valid_count = valid.sum()
            candidate_harm = valid & (candidate_error > old_error)
            accepted_harm = valid & (accepted_error > old_error)
            oracle_accept = valid & (candidate_error < old_error)
            self._add(f"{prefix}_candidate_harm_rate", candidate_harm.sum(), valid_count)
            self._add(f"{prefix}_accepted_harm_rate", accepted_harm.sum(), valid_count)
            self._add(f"{prefix}_oracle_accept_rate", oracle_accept.sum(), valid_count)
            self._add(f"{prefix}_overrepair_prevention_rate",
                      (candidate_harm.sum() - accepted_harm.sum()), candidate_harm.sum())
            self._add(f"{prefix}_oracle_selective_mae",
                      torch.minimum(old_error, candidate_error)[valid].sum(), valid_count)
            self._add(f"{prefix}_acceptance_oracle_gap",
                      (accepted_error - torch.minimum(old_error, candidate_error))[valid].sum(), valid_count)
            if accepted_weights is not None:
                weight = accepted_weights[:, step - 1].detach().float()
                predicted = weight > 0.5
                self._add(f"{prefix}_pred_accept_rate", (predicted & valid).sum(), valid_count)
                self._add(f"{prefix}_repair_accept_mean", weight[valid].sum(), valid_count)
                self._add(f"{prefix}_accept_on_harm_mean",
                          weight[candidate_harm].sum(), candidate_harm.sum())
                self._add(f"{prefix}_accept_on_benefit_mean",
                          weight[oracle_accept].sum(), oracle_accept.sum())
                self._add(f"{prefix}_missed_benefit_rate",
                          (oracle_accept & ~predicted).sum(), oracle_accept.sum())
                self._add("coe_repair_accept_mean", weight[valid].sum(), valid_count)
                tp = (predicted & oracle_accept).sum()
                fp = (predicted & valid & ~oracle_accept).sum()
                tn = (~predicted & valid & ~oracle_accept).sum()
                fn = (~predicted & oracle_accept).sum()
                for i, count in enumerate((tp, fp, tn, fn)):
                    self.confusion[i] += float(count)
                self._add(f"{prefix}_accept_accuracy", tp + tn, valid_count)
            old_safe = torch.where(selected, old.float(), 0.0)
            old_sample = (old_safe - safe_target).abs().flatten(1).sum(1) / selected.flatten(1).sum(1).clamp_min(1)
            accepted_sample = (accepted_safe - safe_target).abs().flatten(1).sum(1) / selected.flatten(1).sum(1).clamp_min(1)
            nonworse = accepted_sample <= old_sample
            self._add(f"{prefix}_nonworse_sample_rate", (nonworse & sample_valid).sum(), sample_valid.sum())
            all_nonworse &= nonworse
            old = accepted
        self.monotonic_all[0] += float((all_nonworse & sample_valid).sum())
        self.monotonic_all[1] += float(sample_valid.sum())

    def merge(self, other: "RASQualityMetrics") -> None:
        super().merge(other)
        for key, pair in other.ratios.items():
            self.ratios[key][0] += pair[0]
            self.ratios[key][1] += pair[1]
        for index, count in enumerate(other.confusion):
            self.confusion[index] += count
        self.monotonic_all[0] += other.monotonic_all[0]
        self.monotonic_all[1] += other.monotonic_all[1]

    def compute(self) -> dict[str, float]:
        result = super().compute()
        for key, (numerator, denominator) in self.ratios.items():
            if denominator:
                result[key] = numerator / denominator
        if self.monotonic_all[1]:
            result["coe_all_steps_monotonic_sample_rate"] = self.monotonic_all[0] / self.monotonic_all[1]
        tp, fp, tn, fn = self.confusion
        if tp + fp + tn + fn:
            result["coe_accept_accuracy"] = (tp + tn) / (tp + fp + tn + fn)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            result["coe_accept_precision"] = precision
            result["coe_accept_recall"] = recall
            result["coe_accept_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return result
