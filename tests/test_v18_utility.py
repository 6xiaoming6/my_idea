from __future__ import annotations

import unittest

import torch

from stmoe_imputer.models.v_single import ObservedRelativeUtilityEvaluator


class V18ObservedUtilityTest(unittest.TestCase):
    def test_relative_utility_has_expected_scale_normalized_values(self) -> None:
        evaluator = ObservedRelativeUtilityEvaluator()
        observed = torch.ones((1, 1, 1, 1, 4))
        mask = torch.ones((1, 1, 1, 1, 4))
        base = torch.zeros_like(observed, requires_grad=True)
        probe = torch.full_like(observed, 0.5, requires_grad=True)
        utility = evaluator(base, probe, observed, mask)
        torch.testing.assert_close(
            utility,
            torch.tensor([[1.0, 0.5, 0.5, 0.5, 0.5]]),
        )
        self.assertFalse(utility.requires_grad)

    def test_hidden_positions_do_not_affect_utility(self) -> None:
        evaluator = ObservedRelativeUtilityEvaluator()
        mask = torch.tensor([[[[[1.0, 0.0]]]]])
        obs = torch.tensor([[[[[2.0, 0.0]]]]])
        base = torch.tensor([[[[[1.0, 0.0]]]]])
        probe = torch.tensor([[[[[1.5, 0.0]]]]])
        expected = evaluator(base, probe, obs, mask)
        changed = evaluator(
            base + (1.0 - mask) * 1000.0,
            probe - (1.0 - mask) * 1000.0,
            obs,
            mask,
        )
        torch.testing.assert_close(changed, expected, atol=0.0, rtol=0.0)
