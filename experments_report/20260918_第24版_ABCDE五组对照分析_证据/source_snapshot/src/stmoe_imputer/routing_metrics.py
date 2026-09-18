"""Exact, target-free diagnostics for sample-level Top-K expert routing."""

from __future__ import annotations

import math
from collections import Counter
from collections import defaultdict
from itertools import product

import torch


class _CoERoutingTotals:
    """Routing totals shared by the global and observation-condition groups."""

    def __init__(self) -> None:
        self.count = 0
        self.probability_sum = None
        self.weight_sum = None
        self.entropy_sum = None
        self.path_counts: Counter = Counter()
        self.mode = None
        self.expert_names: tuple[str, ...] | None = None

    @torch.no_grad()
    def update(self, coe: dict) -> None:
        weights = coe["route_weights"].detach().double().cpu()
        if not weights.numel() or not bool(weights.sum()):
            return  # Shared-only has no routed chain.
        probs = coe["route_probs"].detach().double().cpu()
        names = tuple(coe.get("expert_names", ("T", "S")))
        mode = coe["routing_mode"]
        if weights.ndim != 3 or probs.shape != weights.shape or len(names) != weights.shape[-1]:
            raise ValueError("CoE routing tensors must match [batch, steps, len(expert_names)]")
        if len(set(names)) != len(names):
            raise ValueError("CoE expert_names must be unique")
        if self.weight_sum is not None and (
            names != self.expert_names or mode != self.mode
            or weights.shape[1:] != self.weight_sum.shape
        ):
            raise ValueError("CoE routing configuration changed during accumulation")
        self.mode, self.expert_names = mode, names
        if self.weight_sum is None:
            self.weight_sum = torch.zeros_like(weights[0])
            self.probability_sum = torch.zeros_like(probs[0])
            self.entropy_sum = torch.zeros(probs.shape[1], dtype=torch.float64)
        self.count += weights.shape[0]
        self.weight_sum += weights.sum(dim=0)
        self.probability_sum += probs.sum(dim=0)
        self.entropy_sum += -(probs * probs.clamp_min(1e-12).log()).sum(dim=(0, 2))
        if self.mode not in {"soft", "parallel"}:
            paths = coe["paths"].detach().cpu()
            if paths.shape != weights.shape[:2] or bool(((paths < 0) | (paths >= len(names))).any()):
                raise ValueError("CoE paths must contain valid expert indices for every step")
            for path in paths.tolist():
                self.path_counts[tuple(path)] += 1

    def compute(self) -> dict[str, float]:
        if not self.count:
            return {}
        result = {
            "coe_routing_sample_count": float(self.count),
            "coe_num_experts": float(len(self.expert_names)),
            "coe_num_steps": float(self.weight_sum.shape[0]),
        }
        for step in range(self.weight_sum.shape[0]):
            prefix = f"coe_step{step + 1}"
            for expert, label in enumerate(self.expert_names):
                suffix = "weight" if self.mode in {"soft", "parallel"} else "usage"
                result[f"{prefix}_{label}_{suffix}"] = float(
                    self.weight_sum[step, expert] / self.count
                )
                if self.mode != "fixed":
                    result[f"{prefix}_{label}_prob"] = float(
                        self.probability_sum[step, expert] / self.count
                    )
            if self.mode != "fixed":
                result[f"{prefix}_router_entropy"] = float(self.entropy_sum[step] / self.count)
        if self.mode not in {"soft", "parallel"}:
            # Keep legacy binary-path keys. For richer pools, separate labels
            # to distinguish a joint ST operation from consecutive S and T.
            paths = set(self.path_counts)
            binary_pool = self.expert_names == ("T", "S")
            if binary_pool and self.weight_sum.shape[0] <= 4:
                paths.update(product(range(2), repeat=self.weight_sum.shape[0]))
            separator = "" if binary_pool else "__"
            for path in sorted(paths):
                name = separator.join(self.expert_names[expert] for expert in path)
                result[f"coe_path_{name}_fraction"] = self.path_counts[path] / self.count
            fractions = [count / self.count for count in self.path_counts.values()]
            result["coe_path_unique_count"] = float(len(fractions))
            result["coe_path_entropy"] = -sum(p * math.log(p) for p in fractions)
            result["coe_path_max_fraction"] = max(fractions)
        return result


class CoERoutingMetricAccumulator(_CoERoutingTotals):
    """Sample-weighted routing, globally and by target-free missing conditions.

    Missing fractions are measured across all variables and grid/time positions.
    Nonempty missing windows belong to exactly one fraction bin: (0, .25],
    (.25, .5], or (.5, 1]. Fully observed windows are a separate group. Temporal
    and spatial support groups use each window's mean coverage over its missing
    variable positions, split at <= .5 versus > .5. These three groupings
    overlap; they are diagnostics, not labels or constraints on expert roles.

    Legacy routing records without observation metadata keep global metrics.
    Mixtures (soft or parallel) report weights, never invented argmax paths.
    """

    def __init__(self) -> None:
        super().__init__()
        self._condition_totals: dict[str, _CoERoutingTotals] = {}

    @staticmethod
    def _condition_groups(coe: dict) -> dict[str, torch.Tensor]:
        mask = coe.get("observation_mask")
        if mask is None:
            return {}
        if not torch.is_tensor(mask) or mask.ndim != 5 or any(size < 1 for size in mask.shape):
            raise ValueError("CoE observation_mask must have nonempty shape [batch, channels, T, H, W]")
        if mask.shape[0] != coe["route_weights"].shape[0]:
            raise ValueError("CoE observation_mask and route_weights must have the same batch size")
        mask = mask.detach()
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("CoE observation_mask must contain finite binary values")
        missing = ~mask.bool()
        missing_fraction = missing.float().mean(dim=(1, 2, 3, 4))
        has_missing = missing_fraction > 0
        groups = {
            "fully_observed": ~has_missing,
            "missing_low": has_missing & (missing_fraction <= .25),
            "missing_medium": (missing_fraction > .25) & (missing_fraction <= .5),
            "missing_high": missing_fraction > .5,
        }

        support = coe.get("support")
        feature_names = coe.get("support_feature_names")
        if support is None or feature_names is None:
            return {name: selected.cpu() for name, selected in groups.items()}
        feature_names = tuple(feature_names)
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("CoE support_feature_names must be nonempty and unique")
        if (
            not torch.is_tensor(support) or support.ndim != 5
            or support.shape[0] != mask.shape[0] or support.shape[2:] != mask.shape[2:]
            or support.shape[1] < 1 or support.shape[1] % len(feature_names)
        ):
            raise ValueError("CoE support must have one channel group per support feature")
        support_channels = support.shape[1] // len(feature_names)
        if mask.shape[1] not in (1, support_channels):
            raise ValueError("CoE observation_mask channels must be 1 or match support variables")
        # Feature-major layout: all C temporal channels, all C spatial channels,
        # etc. Expand a shared mask before pooling, preserving variable weights.
        missing = missing.to(device=support.device).expand(-1, support_channels, -1, -1, -1)
        missing_count = missing.sum(dim=(1, 2, 3, 4)).clamp_min(1)
        for direction in ("temporal", "spatial"):
            feature_name = f"{direction}_coverage"
            if feature_name not in feature_names:
                continue
            index = feature_names.index(feature_name) * support_channels
            coverage = support.detach()[:, index:index + support_channels].float()
            support_mean = torch.where(missing, coverage, 0).sum(dim=(1, 2, 3, 4)) / missing_count
            present = has_missing.to(device=support.device)
            groups[f"{direction}_support_low"] = present & (support_mean <= .5)
            groups[f"{direction}_support_high"] = present & (support_mean > .5)
        return {name: selected.cpu() for name, selected in groups.items()}

    @torch.no_grad()
    def update(self, coe: dict) -> None:
        weights = coe["route_weights"]
        if not weights.numel() or not bool(weights.detach().sum()):
            return  # Shared-only has neither global nor conditional routes.
        groups = self._condition_groups(coe)
        if 'mask_family' in coe:
            from .data.diverse_masks import FAMILIES
            labels = coe['mask_family'].detach().cpu()
            if labels.shape != (weights.shape[0],) or not bool(((labels >= 0) & (labels < len(FAMILIES))).all()):
                raise ValueError('Invalid diagnostic mask family labels')
            groups.update({f'family_{name}': labels == i for i, name in enumerate(FAMILIES)})
        super().update(coe)
        # Route tensors are tiny [B, K, E]; condition accumulators never see
        # observation/support tensors, so grouping is performed exactly once.
        route_record = {
            name: coe[name].detach().cpu()
            for name in ("route_weights", "route_probs")
        }
        if coe["routing_mode"] not in {"soft", "parallel"}:
            route_record["paths"] = coe["paths"].detach().cpu()
        for group, selected in groups.items():
            totals = self._condition_totals.setdefault(group, _CoERoutingTotals())
            if not bool(selected.any()):
                continue
            record = {name: value[selected] for name, value in route_record.items()}
            record.update(routing_mode=coe["routing_mode"], expert_names=self.expert_names)
            totals.update(record)

    def compute(self) -> dict[str, float]:
        result = super().compute()
        for group, totals in self._condition_totals.items():
            prefix = f"coe_condition_{group}"
            result[f"{prefix}_sample_count"] = float(totals.count)
            for name, value in totals.compute().items():
                if name in {"coe_routing_sample_count", "coe_num_experts", "coe_num_steps"}:
                    continue
                result[f"{prefix}_{name.removeprefix('coe_')}"] = value
        return result


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
