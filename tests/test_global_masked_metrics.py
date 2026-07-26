from __future__ import annotations

import math
import unittest

import torch

from stmoe_imputer.metrics import MaskedMetricAccumulator


class GlobalMaskedMetricsTest(unittest.TestCase):
    def test_accumulator_uses_all_missing_elements_not_batch_averages(
        self,
    ) -> None:
        accumulator = MaskedMetricAccumulator()
        first_prediction = torch.tensor([[[[[1.0, 3.0]]]]])
        first_target = torch.zeros_like(first_prediction)
        first_mask = torch.tensor([[[[[0.0, 1.0]]]]])
        second_prediction = torch.tensor([[[[[2.0, 4.0, 6.0]]]]])
        second_target = torch.zeros_like(second_prediction)
        second_mask = torch.zeros(1, 1, 1, 1, 3)
        accumulator.update(first_prediction, first_target, first_mask)
        accumulator.update(second_prediction, second_target, second_mask)
        metrics = accumulator.compute()
        expected_errors = (1.0, 2.0, 4.0, 6.0)
        self.assertAlmostEqual(
            metrics["mae"],
            sum(expected_errors) / len(expected_errors),
        )
        self.assertAlmostEqual(
            metrics["rmse"],
            math.sqrt(
                sum(value * value for value in expected_errors)
                / len(expected_errors)
            ),
        )


if __name__ == "__main__":
    unittest.main()
