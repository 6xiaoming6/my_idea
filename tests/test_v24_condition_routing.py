"""Target-free routing diagnostics for missingness and visible support."""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from stmoe_imputer.models.temporal_spatial_coe import (
    SUPPORT_FEATURE_NAMES,
    compute_observation_support,
)
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator


def routing_record(mask: torch.Tensor, paths: torch.Tensor, mode: str = "hard") -> dict:
    weights = F.one_hot(paths, num_classes=2).float()
    probabilities = .2 + .6 * weights
    return {
        "expert_names": ("T", "S"),
        "routing_mode": mode,
        "route_weights": probabilities if mode in {"soft", "parallel"} else weights,
        "route_probs": probabilities,
        "paths": paths,
        "observation_mask": mask,
        "support": compute_observation_support(mask),
        "support_feature_names": SUPPORT_FEATURE_NAMES,
    }


class ConditionalCoERoutingTests(unittest.TestCase):
    def test_conditions_use_missing_positions_and_exact_fraction_boundaries(self) -> None:
        mask = torch.ones(5, 1, 4, 1, 1)
        for sample in range(1, 5):
            mask[sample, :, :sample] = 0
        # An interior isolated missing point has temporal coverage 2/3.
        mask[1].fill_(1)
        mask[1, :, 1] = 0
        paths = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0], [0, 1]])
        accumulator = CoERoutingMetricAccumulator()
        accumulator.update(routing_record(mask, paths))
        result = accumulator.compute()
        for name, count in {
            "fully_observed": 1, "missing_low": 1, "missing_medium": 1,
            "missing_high": 2, "temporal_support_low": 3,
            "temporal_support_high": 1, "spatial_support_low": 4,
            "spatial_support_high": 0,
        }.items():
            self.assertEqual(result[f"coe_condition_{name}_sample_count"], count)
        self.assertEqual(result["coe_condition_missing_low_path_TS_fraction"], 1)
        self.assertEqual(result["coe_condition_missing_medium_path_SS_fraction"], 1)
        self.assertEqual(result["coe_condition_missing_high_path_ST_fraction"], .5)
        self.assertEqual(result["coe_condition_missing_high_path_TS_fraction"], .5)
        self.assertEqual(result["coe_condition_fully_observed_step1_T_usage"], 1)
        self.assertFalse(any(
            name.startswith("coe_condition_spatial_support_high_") and not name.endswith("sample_count")
            for name in result
        ))

    def test_sample_weighted_condition_aggregation_across_uneven_batches(self) -> None:
        mask = torch.ones(7, 1, 4, 1, 1)
        for sample, count in enumerate((0, 1, 1, 1, 2, 3, 4)):
            mask[sample, :, :count] = 0
        paths = torch.tensor([[0, 0], [0, 1], [0, 1], [1, 0], [1, 1], [0, 1], [1, 0]])
        record = routing_record(mask, paths)
        entire = CoERoutingMetricAccumulator()
        entire.update(record)
        batched = CoERoutingMetricAccumulator()
        for selection in (slice(0, 3), slice(3, 5), slice(5, 7)):
            batched.update({
                name: value[selection] if torch.is_tensor(value) else value
                for name, value in record.items()
            })
        self.assertEqual(entire.compute(), batched.compute())
        result = batched.compute()
        self.assertEqual(result["coe_condition_missing_low_sample_count"], 3)
        self.assertAlmostEqual(result["coe_condition_missing_low_step1_T_usage"], 2 / 3)
        self.assertAlmostEqual(result["coe_condition_missing_low_step1_T_prob"], .6, places=6)
        self.assertAlmostEqual(result["coe_condition_missing_low_path_TS_fraction"], 2 / 3)
        self.assertEqual(result["coe_condition_missing_low_path_unique_count"], 2)

    def test_support_pools_missing_variables_with_feature_major_layout(self) -> None:
        # Missing positions are channel 0 at both times and channel 1 at t=0.
        # Temporal mean = (0 + 0 + 1)/3, not a channel-mean average of .5.
        mask = torch.tensor([0, 0, 0, 1], dtype=torch.float32).reshape(1, 2, 2, 1, 1)
        paths = torch.tensor([[0, 1]])
        record = routing_record(mask, paths)
        support = torch.full((1, len(SUPPORT_FEATURE_NAMES) * 2, 2, 1, 1), float("nan"))
        temporal = SUPPORT_FEATURE_NAMES.index("temporal_coverage") * 2
        spatial = SUPPORT_FEATURE_NAMES.index("spatial_coverage") * 2
        support[:, temporal] = 0
        support[:, temporal + 1] = 1
        support[:, spatial] = .75
        support[:, spatial + 1] = .25
        # Observed-position values do not contribute even when non-finite.
        support[:, temporal + 1, 1] = float("nan")
        support[:, spatial + 1, 1] = float("nan")
        record["support"] = support.requires_grad_()
        accumulator = CoERoutingMetricAccumulator()
        accumulator.update(record)
        result = accumulator.compute()
        self.assertEqual(result["coe_condition_temporal_support_low_sample_count"], 1)
        self.assertEqual(result["coe_condition_spatial_support_high_sample_count"], 1)
        self.assertEqual(result["coe_condition_missing_high_sample_count"], 1)
        self.assertIsNone(record["support"].grad)

    def test_single_channel_mask_expands_to_support_variables(self) -> None:
        mask = torch.tensor([0, 1], dtype=torch.float32).reshape(1, 1, 2, 1, 1)
        paths = torch.tensor([[1, 0]])
        record = routing_record(mask, paths)
        support = torch.zeros(1, len(SUPPORT_FEATURE_NAMES) * 2, 2, 1, 1)
        temporal = SUPPORT_FEATURE_NAMES.index("temporal_coverage") * 2
        spatial = SUPPORT_FEATURE_NAMES.index("spatial_coverage") * 2
        support[:, temporal:temporal + 2] = .5
        support[:, spatial:spatial + 2] = .75
        record["support"] = support
        accumulator = CoERoutingMetricAccumulator()
        accumulator.update(record)
        result = accumulator.compute()
        self.assertEqual(result["coe_condition_missing_medium_sample_count"], 1)
        self.assertEqual(result["coe_condition_temporal_support_low_sample_count"], 1)
        self.assertEqual(result["coe_condition_temporal_support_high_sample_count"], 0)
        self.assertEqual(result["coe_condition_spatial_support_high_sample_count"], 1)

    def test_spatial_coverage_distinguishes_visible_and_empty_neighbors(self) -> None:
        mask = torch.ones(2, 1, 1, 2, 2)
        mask[0, :, :, 0, 0] = 0
        mask[1].zero_()
        accumulator = CoERoutingMetricAccumulator()
        accumulator.update(routing_record(mask, torch.tensor([[1], [0]])))
        result = accumulator.compute()
        self.assertEqual(result["coe_condition_spatial_support_high_step1_S_usage"], 1)
        self.assertEqual(result["coe_condition_spatial_support_low_step1_T_usage"], 1)
        self.assertEqual(result["coe_condition_temporal_support_low_sample_count"], 2)

    def test_soft_and_parallel_report_mixture_weights_without_paths(self) -> None:
        mask = torch.zeros(2, 1, 2, 1, 1)
        for mode in ("soft", "parallel"):
            with self.subTest(mode=mode):
                record = routing_record(mask, torch.tensor([[0], [1]]), mode=mode)
                del record["paths"]  # Mixtures need no discrete path metadata.
                accumulator = CoERoutingMetricAccumulator()
                accumulator.update(record)
                result = accumulator.compute()
                self.assertFalse(any("_path_" in name or name.endswith("_usage") for name in result))
                self.assertAlmostEqual(result["coe_condition_missing_high_step1_T_weight"], .5)
                self.assertAlmostEqual(result["coe_condition_missing_high_step1_S_prob"], .5)

    def test_legacy_metadata_and_shared_only_are_supported(self) -> None:
        record = routing_record(torch.zeros(1, 1, 2, 1, 1), torch.tensor([[0]]))
        legacy = {name: value for name, value in record.items() if name not in {
            "observation_mask", "support", "support_feature_names",
        }}
        accumulator = CoERoutingMetricAccumulator()
        accumulator.update(legacy)
        self.assertFalse(any(name.startswith("coe_condition_") for name in accumulator.compute()))
        mask_only = {name: value for name, value in record.items() if name not in {
            "support", "support_feature_names",
        }}
        accumulator = CoERoutingMetricAccumulator()
        accumulator.update(mask_only)
        result = accumulator.compute()
        self.assertEqual(result["coe_condition_missing_high_sample_count"], 1)
        self.assertFalse(any("support_" in name for name in result))
        accumulator = CoERoutingMetricAccumulator()
        record["route_weights"] = torch.zeros_like(record["route_weights"])
        accumulator.update(record)
        self.assertEqual(accumulator.compute(), {})

    def test_malformed_condition_metadata_does_not_corrupt_accumulated_metrics(self) -> None:
        record = routing_record(torch.zeros(1, 1, 2, 1, 1), torch.tensor([[0]]))
        accumulator = CoERoutingMetricAccumulator()
        accumulator.update(record)
        expected = accumulator.compute()
        for change in (
            {"observation_mask": torch.ones(2, 1, 2, 1, 1)},
            {"observation_mask": torch.full_like(record["observation_mask"], .5)},
            {"support": torch.zeros(1, 3, 2, 1, 1)},
            {"support_feature_names": ("temporal_coverage", "temporal_coverage")},
        ):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                accumulator.update({**record, **change})
            self.assertEqual(accumulator.compute(), expected)


if __name__ == "__main__":
    unittest.main()
