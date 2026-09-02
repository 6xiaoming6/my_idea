from __future__ import annotations

import unittest

import torch

from stmoe_imputer.metrics import MaskedMetricAccumulator


class TestMaskedMetricAccumulator(unittest.TestCase):
    def test_weights_batches_by_missing_element_count(self) -> None:
        accumulator = MaskedMetricAccumulator()
        accumulator.update(
            torch.tensor([[[[[2.0]]]]]),
            torch.tensor([[[[[0.0]]]]]),
            torch.zeros(1, 1, 1, 1, 1),
        )
        accumulator.update(
            torch.tensor([[[[[1.0, 1.0, 1.0]]]]]),
            torch.zeros(1, 1, 1, 1, 3),
            torch.zeros(1, 1, 1, 1, 3),
        )
        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["mae"], 1.25)
        self.assertAlmostEqual(metrics["rmse"], (7.0 / 4.0) ** 0.5)
        self.assertEqual(metrics["metric_missing_count"], 4.0)


if __name__ == "__main__":
    unittest.main()
