"""Exact, target-free diagnostics for sample-level Top-K expert routing."""

from __future__ import annotations

import math
from collections import defaultdict

import torch


def active_routing_scales(scale_mode: str) -> tuple[str, ...]:
    mapping = {
        "fine": ("fine",),
        "fine_mid": ("fine", "mid"),
        "fine_mid_coarse": ("fine", "mid", "coarse"),
    }
    try:
        return mapping[scale_mode]
    except KeyError as error:
        raise ValueError(f"Unknown scale_mode: {scale_mode}") from error


class RoutingMetricAccumulator:
    """Accumulate routing statistics over every sample instead of batch means."""

    def __init__(
        self,
        scale_names: tuple[str, ...],
        dead_threshold: float = 0.01,
        always_threshold: float = 0.99,
        eps: float = 1e-12,
    ) -> None:
        if not scale_names:
            raise ValueError("scale_names must contain at least one active scale")
        self.scale_names = tuple(scale_names)
        self.dead_threshold = float(dead_threshold)
        self.always_threshold = float(always_threshold)
        self.eps = float(eps)
        self._gate_sum: dict[str, torch.Tensor] = {}
        self._load_sum: dict[str, torch.Tensor] = {}
        self._sample_count: dict[str, int] = defaultdict(int)
        self._margin_sum: dict[str, float] = defaultdict(float)
        self._margin_count: dict[str, int] = defaultdict(int)

    def _add(
        self,
        name: str,
        gate: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> None:
        if gate.ndim != 2 or selected_mask.shape != gate.shape:
            raise ValueError(
                "Expected gate and selected_mask shaped [batch, experts], got "
                f"{tuple(gate.shape)} and {tuple(selected_mask.shape)}"
            )
        gate_cpu = gate.detach().to(device="cpu", dtype=torch.float64)
        selected_cpu = selected_mask.detach().to(device="cpu", dtype=torch.float64)
        if name not in self._gate_sum:
            self._gate_sum[name] = torch.zeros(gate_cpu.shape[1], dtype=torch.float64)
            self._load_sum[name] = torch.zeros(gate_cpu.shape[1], dtype=torch.float64)
        if self._gate_sum[name].numel() != gate_cpu.shape[1]:
            raise ValueError(f"Expert count changed while accumulating {name}")

        self._gate_sum[name] += gate_cpu.sum(dim=0)
        self._load_sum[name] += selected_cpu.sum(dim=0)
        self._sample_count[name] += int(gate_cpu.shape[0])

        selected_per_sample = selected_cpu.sum(dim=1)
        if selected_per_sample.numel() == 0:
            return
        top_k = int(round(float(selected_per_sample[0])))
        if (
            0 < top_k < gate_cpu.shape[1]
            and torch.allclose(
                selected_per_sample,
                torch.full_like(selected_per_sample, float(top_k)),
            )
        ):
            sorted_gate = gate_cpu.sort(dim=1, descending=True).values
            margins = sorted_gate[:, top_k - 1] - sorted_gate[:, top_k]
            self._margin_sum[name] += float(margins.sum())
            self._margin_count[name] += int(margins.numel())

    def update(
        self,
        gates: dict[str, torch.Tensor],
        selected_masks: dict[str, torch.Tensor] | None,
    ) -> None:
        if selected_masks is None:
            return
        active: list[tuple[torch.Tensor, torch.Tensor]] = []
        for scale in self.scale_names:
            gate = gates.get(scale)
            selected = selected_masks.get(scale)
            if not torch.is_tensor(gate) or not torch.is_tensor(selected):
                continue
            self._add(scale, gate, selected)
            active.append((gate, selected))
        if active:
            self._add(
                "all",
                torch.cat([item[0] for item in active], dim=0),
                torch.cat([item[1] for item in active], dim=0),
            )

    def _compute_group(self, name: str) -> dict[str, float]:
        count = self._sample_count[name]
        if count <= 0:
            return {}
        importance = self._gate_sum[name] / count
        hard_load = self._load_sum[name] / count
        load_distribution = hard_load / hard_load.sum().clamp_min(self.eps)
        mean_load = hard_load.mean()
        load_cv = hard_load.std(unbiased=False) / mean_load.clamp_min(self.eps)
        entropy = -(
            load_distribution
            * load_distribution.clamp_min(self.eps).log()
        ).sum()
        if load_distribution.numel() > 1:
            entropy = entropy / math.log(load_distribution.numel())
        soft_hard_gap = (importance - load_distribution).abs().sum()
        prefix = f"routing_{name}"
        result = {
            f"{prefix}_hard_load_cv": float(load_cv),
            f"{prefix}_selection_entropy": float(entropy),
            f"{prefix}_dead_expert_rate": float(
                (hard_load < self.dead_threshold).to(torch.float64).mean()
            ),
            f"{prefix}_always_selected_rate": float(
                (hard_load > self.always_threshold).to(torch.float64).mean()
            ),
            f"{prefix}_soft_hard_l1_gap": float(soft_hard_gap),
        }
        if self._margin_count[name] > 0:
            result[f"{prefix}_topk_boundary_margin"] = (
                self._margin_sum[name] / self._margin_count[name]
            )
        for index, value in enumerate(importance):
            result[f"{prefix}_soft_importance_{index}"] = float(value)
        for index, value in enumerate(hard_load):
            result[f"{prefix}_hard_load_{index}"] = float(value)
        return result

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in (*self.scale_names, "all"):
            if self._sample_count[name] > 0:
                result.update(self._compute_group(name))
        active_names = [
            name for name in self.scale_names if self._sample_count[name] > 0
        ]
        for metric in (
            "hard_load_cv",
            "selection_entropy",
            "dead_expert_rate",
            "always_selected_rate",
            "soft_hard_l1_gap",
            "topk_boundary_margin",
        ):
            values = [
                result[f"routing_{name}_{metric}"]
                for name in active_names
                if f"routing_{name}_{metric}" in result
            ]
            if values:
                result[f"routing_scales_mean_{metric}"] = sum(values) / len(values)
                result[f"routing_scales_max_{metric}"] = max(values)
        return result
