from __future__ import annotations

import math
import unittest

import torch

from stmoe_imputer.engine import _RunningDistribution, _mean_logs
from stmoe_imputer.metrics import MaskedMetricAccumulator


class GlobalMaskedMetricsTest(unittest.TestCase):
    def test_unequal_last_batch_is_weighted_by_missing_elements(self) -> None:
        accumulator = MaskedMetricAccumulator()

        target_full = torch.ones(2, 1, 1, 1, 4)
        prediction_full = target_full + 1.0
        mask_full = torch.zeros(2, 1, 1, 1, 4)
        accumulator.update(prediction_full, target_full, mask_full)

        target_tail = torch.ones(1, 1, 1, 1, 4)
        prediction_tail = target_tail.clone()
        prediction_tail[..., 0] += 10.0
        mask_tail = torch.ones(1, 1, 1, 1, 4)
        mask_tail[..., 0] = 0.0
        accumulator.update(prediction_tail, target_tail, mask_tail)

        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(12.0))
        self.assertAlmostEqual(metrics["mape"], 2.0)

    def test_empty_missing_region_stays_finite(self) -> None:
        accumulator = MaskedMetricAccumulator()
        target = torch.ones(1, 1, 1, 2, 2)
        accumulator.update(target, target, torch.ones(1, 1, 1, 2, 2))
        self.assertEqual(
            accumulator.compute(),
            {"mae": 0.0, "rmse": 0.0, "mape": 0.0},
        )

    def test_epoch_distribution_preserves_small_rho_variance(self) -> None:
        distribution = _RunningDistribution()
        values = torch.tensor(
            [0.0201127, 0.0201133],
            dtype=torch.float64,
        )
        distribution.update(
            values,
            high_threshold=0.19,
            low_threshold=0.01,
        )
        statistics = distribution.compute()
        self.assertGreater(statistics["std"], 0.0)
        self.assertAlmostEqual(statistics["min"], 0.0201127)
        self.assertAlmostEqual(statistics["max"], 0.0201133)

    def test_secondary_logs_are_weighted_by_batch_size(self) -> None:
        logs = {
            "__batch_size__": [4.0, 1.0],
            "diagnostic": [1.0, 11.0],
        }
        self.assertAlmostEqual(_mean_logs(logs)["diagnostic"], 3.0)
        self.assertNotIn("__batch_size__", _mean_logs(logs))


if __name__ == "__main__":
    unittest.main()
