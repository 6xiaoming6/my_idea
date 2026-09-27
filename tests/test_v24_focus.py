"""Focused CoE comparisons and sparse expert execution."""
from __future__ import annotations

import unittest

import torch

from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE


class FocusedCoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, **patch):
        settings = dict(c_in=2, dim=16, num_steps=3, top_k=2,
                        expert_pool=['T', 'S', 'TD', 'SD', 'TA', 'ST'],
                        routing_mode='hard', attention_heads=4,
                        expert_sharing='per_step')
        settings.update(patch)
        return TemporalSpatialCoE(**settings)

    def test_sparse_top2_executes_only_selected_windows(self):
        model = self.model()
        inputs = torch.randn(3, 16, 4, 6, 6)
        weights = torch.tensor([[.7, 0, .3, 0, 0, 0],
                                [0, .4, 0, 0, .6, 0],
                                [.2, 0, 0, 0, .8, 0]], requires_grad=True)
        calls = {}
        hooks = [expert.register_forward_hook(
            lambda _module, args, _output, index=index: calls.setdefault(index, []).append(args[0].shape[0]))
            for index, expert in enumerate(model.routed_experts(0))]
        actual = model._dispatch_weighted(inputs, weights)
        for hook in hooks:
            hook.remove()
        self.assertEqual(calls, {0: [2], 1: [1], 2: [1], 4: [2]})
        expected = sum(weights[:, i, None, None, None, None] * expert(inputs)
                       for i, expert in enumerate(model.routed_experts(0)))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        actual.square().mean().backward()
        self.assertIsNone(next(model.routed_experts(0)[3].parameters()).grad)
        self.assertIsNone(next(model.routed_experts(0)[5].parameters()).grad)
        self.assertIsNotNone(weights.grad)

    def test_layered_moe_matches_first_round_then_omits_completion_feedback(self):
        torch.manual_seed(24)
        chain = self.model(completion_feedback=True)
        torch.manual_seed(24)
        layered = self.model(completion_feedback=False)
        self.assertTrue(all(torch.equal(chain.state_dict()[key], value)
                            for key, value in layered.state_dict().items()))
        target = torch.randn(4, 2, 4, 6, 6)
        mask = (torch.rand_like(target) > .4).float()
        chain.eval(); layered.eval()
        with torch.no_grad():
            a = chain(target * mask, mask)['coe']['predictions']
            b = layered(target * mask, mask)['coe']['predictions']
        self.assertTrue(torch.allclose(a[0], b[0], atol=1e-6, rtol=1e-6))
        self.assertGreater((a[-1] - b[-1]).abs().max().item(), 1e-7)

    def test_learned_pair_can_choose_beyond_individual_top2(self):
        torch.manual_seed(31)
        model = self.model(expert_sharing='shared', pair_mode='interaction').eval()
        values = torch.randn(4, 2, 4, 6, 6)
        mask = (torch.rand_like(values) > .4).float()
        with torch.no_grad():
            baseline = model(values * mask, mask)
            native_ids = set(baseline['coe']['pair_ids'][:, 0].tolist())
            desired = next(index for index in range(len(model.pair_indices)) if index not in native_ids)
            for router in model.routers:
                router.pair_head.bias[desired] = 100.
            result = model(values * mask, mask)
        self.assertTrue((result['coe']['pair_ids'] == desired).all())
        self.assertGreater(result['diagnostics']['coe']['step1_pair_vs_top2_disagreement_rate'].item(), 0.)

    def test_invalid_no_feedback_acceptance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'does not use an acceptance gate'):
            self.model(completion_feedback=False, acceptance='point')


if __name__ == '__main__':
    unittest.main()
