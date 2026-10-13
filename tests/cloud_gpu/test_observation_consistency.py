"""Standalone CPU checks for cloud I05; no datasets or server artifacts required.

The hierarchy fixtures and exhaustive KKT oracle intentionally stay independent
of model/queue fixtures so a shallow cloud checkout can run these tests directly.
"""
import itertools
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def aggregate(value, factor, pooling='sum'):
    b, c, t, h, w = value.shape
    result = value.reshape(b, c, t, h // factor, factor,
                           w // factor, factor).sum((4, 6))
    return result if pooling == 'sum' else result / (factor * factor)


def coarse_expand(value, factor):
    return value.repeat_interleave(factor, -2).repeat_interleave(factor, -1)


def independent_simplex(values, weights, eligible, budget):
    """Enumerate active sets, an intentionally independent tiny KKT oracle."""
    answer = torch.zeros_like(values)
    if float(budget) == 0:
        return answer
    indices = eligible.nonzero().flatten().tolist()
    for count in range(1, len(indices) + 1):
        for active in itertools.combinations(indices, count):
            active = list(active)
            multiplier = (budget - values[active].sum()) / weights[active].sum()
            candidate = values + multiplier * weights
            inactive = [i for i in indices if i not in active]
            if bool((candidate[active] > 0).all()) and (
                    not inactive or bool((candidate[inactive] <= 0).all())):
                answer[active] = candidate[active]
                return answer
    raise AssertionError('No feasible KKT active set')


class ResidualRegionProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, pooling='sum'):
        truth = (torch.arange(64).reshape(1, 1, 1, 8, 8) % 7).float() + .25
        masks = [torch.zeros_like(aggregate(truth, f)) for f in (1, 2, 4)]
        masks[0][..., ::2, ::3] = 1
        masks[1][..., 0, :2] = 1
        masks[1][..., 2, 3] = 1
        masks[2][..., 0, 0] = 1
        masks[2][..., 1, 1] = 1
        observed = [torch.where(m.bool(), aggregate(truth, factor, pooling), 0.)
                    for m, factor in zip(masks, (1, 2, 4))]
        raw = torch.linspace(-3., 4., 64).reshape_as(truth)
        weights = torch.linspace(.4, 2., 64).reshape_as(truth)
        return truth, raw, weights, masks, observed

    def project(self, raw, weights, masks, observed, pooling='sum'):
        from stmoe_imputer.models.observed_consistency_coe import project_observed_residual_regions
        return project_observed_residual_regions(
            raw, weights, observed[0], masks[0], observed[1], masks[1],
            observed[2], masks[2], pooling=pooling)[0]

    def test_nested_constraints_mean_sum_hidden_poison_and_unconstrained_values(self):
        for pooling in ['sum', 'mean']:
            _, raw, weights, masks, observed = self.fixture(pooling)
            answer = self.project(raw, weights, masks, observed, pooling)
            torch.testing.assert_close(answer[masks[0].bool()], observed[0][masks[0].bool()], rtol=0, atol=0)
            covered = coarse_expand(masks[1], 2).bool() | coarse_expand(masks[2], 4).bool()
            unconstrained = ~covered & ~masks[0].bool()
            self.assertTrue(bool(unconstrained.any()))
            torch.testing.assert_close(answer[unconstrained], raw[unconstrained], rtol=0, atol=0)
            self.assertTrue(bool((answer[covered & ~masks[0].bool()] >= 0).all()))
            for factor, mask, obs in zip((2, 4), masks[1:], observed[1:]):
                torch.testing.assert_close(aggregate(answer, factor, pooling)[mask.bool()],
                                           obs[mask.bool()], rtol=2e-6, atol=2e-5)
            poisoned = [value.masked_fill(~mask.bool(), float('nan')) for value, mask in zip(observed, masks)]
            torch.testing.assert_close(self.project(raw, weights, masks, poisoned, pooling), answer, rtol=0, atol=0)

    def test_edge_cases_redundant_constraints_zero_budget_and_single_unknown(self):
        truth, raw, weights, masks, _ = self.fixture()
        # No observations leaves the complete raw result, including negatives.
        empty = [torch.zeros_like(mask) for mask in masks]
        unknown = [torch.full_like(mask, float('nan')) for mask in masks]
        torch.testing.assert_close(self.project(raw, weights, empty, unknown), raw, rtol=0, atol=0)
        all_masks = [torch.ones_like(mask) for mask in masks]
        observed = [aggregate(truth, factor) for factor in (1, 2, 4)]
        torch.testing.assert_close(self.project(raw, weights, all_masks, observed), truth, rtol=0, atol=0)
        one_unknown = [mask.clone() for mask in all_masks]
        one_unknown[0][..., 1, 1] = 0
        torch.testing.assert_close(self.project(raw, weights, one_unknown, observed), truth, rtol=0, atol=1e-6)
        # Every M is visible: an exact C adds no independent equation.
        no_f = [torch.zeros_like(masks[0]), all_masks[1], all_masks[2]]
        without_c = [no_f[0], no_f[1], torch.zeros_like(masks[2])]
        torch.testing.assert_close(self.project(raw, weights, no_f, observed),
                                   self.project(raw, weights, without_c, observed), rtol=0, atol=0)
        zero_obs = [torch.zeros_like(mask) for mask in masks]
        only_c = [torch.zeros_like(masks[0]), torch.zeros_like(masks[1]), torch.ones_like(masks[2])]
        zero = self.project(raw, weights, only_c, zero_obs)
        self.assertFalse(bool(zero.any()))
        self.assertTrue(bool(torch.isfinite(zero).all()))

    def test_simplex_matches_independent_kkt_and_has_correct_smooth_gradients(self):
        from stmoe_imputer.models.observed_consistency_coe import weighted_simplex_projection
        values = torch.tensor([[.8, -2., 1.7, .4], [-3., -1., -2., -4.]], dtype=torch.float64)
        weights = torch.tensor([[.7, 1.3, 2., .9], [1., 2., 3., 4.]], dtype=torch.float64)
        eligible = torch.tensor([[True, True, True, True], [True, False, True, True]])
        budgets = torch.tensor([1.2, .8], dtype=torch.float64)
        for candidate_weights in (weights, torch.ones_like(weights)):
            output = weighted_simplex_projection(values, candidate_weights, eligible, budgets)
            expected = torch.stack([independent_simplex(v, w, e, b) for v, w, e, b
                                    in zip(values, candidate_weights, eligible, budgets)])
            torch.testing.assert_close(output, expected, rtol=1e-10, atol=1e-10)
        values.requires_grad_()
        weights.requires_grad_()
        budgets.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda v, w, b: weighted_simplex_projection(v, w, eligible, b),
            (values, weights, budgets), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_large_raw_predictions_and_tiny_positive_budget(self):
        from stmoe_imputer.models.observed_consistency_coe import weighted_simplex_projection
        raw = torch.tensor([[1e6, 1e6]], dtype=torch.float32, requires_grad=True)
        weights = torch.ones_like(raw, requires_grad=True)
        eligible = torch.ones_like(raw, dtype=torch.bool)
        budget = torch.tensor([.001], dtype=torch.float32)
        result = weighted_simplex_projection(raw, weights, eligible, budget)
        torch.testing.assert_close(result, torch.full_like(raw, .0005), rtol=1e-6, atol=1e-10)
        torch.testing.assert_close(result.sum(-1), budget, rtol=1e-6, atol=1e-10)
        (result * torch.tensor([[1., 2.]])).sum().backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertTrue(bool(torch.isfinite(weights.grad).all()))

    def test_rejects_observed_mid_total_exceeding_coarse_total(self):
        truth, raw, weights, masks, _ = self.fixture()
        masks = [torch.zeros_like(mask) for mask in masks]
        masks[1][..., 0, 0] = 1
        masks[2][..., 0, 0] = 1
        observed = [torch.zeros_like(mask) for mask in masks]
        observed[1][..., 0, 0] = 10.
        observed[2][..., 0, 0] = 9.
        with self.assertRaisesRegex(ValueError, 'Infeasible observed C constraint'):
            self.project(raw, weights, masks, observed)

    def test_rejects_mid_total_inconsistent_with_all_visible_fine(self):
        truth, raw, weights, masks, _ = self.fixture()
        masks = [torch.ones_like(masks[0]), torch.zeros_like(masks[1]),
                 torch.zeros_like(masks[2])]
        masks[1][..., 0, 0] = 1
        observed = [truth.clone(), torch.zeros_like(masks[1]),
                    torch.zeros_like(masks[2])]
        observed[1][..., 0, 0] = aggregate(truth, 2)[..., 0, 0] + 1.
        with self.assertRaisesRegex(ValueError, 'Infeasible observed M constraint'):
            self.project(raw, weights, masks, observed)


if __name__ == '__main__':
    unittest.main()
