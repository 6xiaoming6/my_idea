from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from stmoe_imputer.engine import _CoEQualityMetrics, build_optimizer
from stmoe_imputer.models.imputer import DualBranchSTImputer
from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator


class TeamAcceptanceV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def small(self, **patch):
        args = dict(c_in=2, dim=16, num_steps=3, top_k=2,
                    expert_pool=['T', 'S', 'TD', 'SD', 'TA', 'ST'],
                    routing_mode='hard', attention_heads=4,
                    routing_warmup_epochs=3, routing_transition_epochs=3)
        args.update(patch)
        return TemporalSpatialCoE(**args)

    def inputs(self):
        torch.manual_seed(23)
        target = torch.randn(3, 2, 4, 6, 6)
        mask = (torch.rand_like(target) > .4).float()
        return target * mask, mask, target

    def test_additive_pair_matches_native_forward_but_changes_gradient(self):
        observed, mask, _ = self.inputs()
        torch.manual_seed(11); native = self.small()
        torch.manual_seed(11); additive = self.small(pair_mode='additive')
        native.train(); additive.train()
        native.set_routing_epoch(1); additive.set_routing_epoch(1)
        a, b = native(observed, mask), additive(observed, mask)
        self.assertTrue(torch.allclose(a['x_hat_main'], b['x_hat_main'], atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(a['coe']['route_weights'], b['coe']['route_weights'], atol=2e-6))
        a['x_hat_main'].square().mean().backward()
        b['x_hat_main'].square().mean().backward()
        self.assertGreater((native.routers[0][-1].weight.grad -
                            additive.routers[0].layers[-1].weight.grad).abs().sum().item(), 1e-8)

    def test_interaction_is_zero_initialized_and_gets_task_gradient(self):
        observed, mask, _ = self.inputs()
        torch.manual_seed(11); additive = self.small(pair_mode='additive')
        torch.manual_seed(11); interaction = self.small(pair_mode='interaction')
        a, b = additive(observed, mask), interaction(observed, mask)
        self.assertTrue(torch.allclose(a['x_hat_main'], b['x_hat_main'], atol=2e-6, rtol=2e-6))
        b['x_hat_main'].square().mean().backward()
        gradient = sum(p.grad.abs().sum().item() for r in interaction.routers
                       for p in r.pair_head.parameters())
        self.assertGreater(gradient, 0.)
        extreme = torch.tensor([[1000., -1000., 999., -500., 0., 1.]])
        weights, ids, logits, probs = interaction._pair_route(extreme, None, 2.)
        self.assertTrue(torch.isfinite(weights).all() and torch.isfinite(probs).all())
        self.assertAlmostEqual(weights.sum().item(), 1., places=6)
        self.assertEqual(torch.count_nonzero(weights).item(), 2)

    def test_acceptance_endpoints_and_final_output(self):
        cfg = json.loads((ROOT / 'configs/v24/smoke.json').read_text())
        cfg['model']['coe'].update({'num_steps': 1, 'top_k': 2,
                                    'expert_pool': ['T', 'S'], 'fixed_path': None,
                                    'pair_mode': 'additive', 'acceptance': 'point'})
        torch.manual_seed(9); model = DualBranchSTImputer.from_config(cfg)
        model.eval()
        observed, mask, target = self.inputs()
        with torch.no_grad():
            model.main_branch.acceptance_head[-1].bias.fill_(40.)
            accepted = model({'x_f_obs': observed, 'm_f': mask})
            candidate = accepted['coe']['candidate_predictions'][0]
            self.assertTrue(torch.allclose(accepted['x_hat_final'], candidate, atol=2e-6))
            model.main_branch.acceptance_head[-1].bias.fill_(-40.)
            rejected = model({'x_f_obs': observed, 'm_f': mask})
            initial = rejected['coe']['initial_completion']
            self.assertTrue(torch.allclose(rejected['x_comp'], initial, atol=2e-6))
            self.assertTrue(torch.equal(rejected['x_comp'][mask.bool()], observed[mask.bool()]))
            self.assertGreater((accepted['x_hat_final'] - rejected['x_hat_final']).abs().sum().item(), 1e-5)

    def test_independent_pools_preserve_common_initialization_and_optimizer_coverage(self):
        torch.manual_seed(11); shared = self.small(pair_mode='additive')
        torch.manual_seed(11); separate = self.small(pair_mode='additive', expert_sharing='per_step')
        old = shared.state_dict(); new = separate.state_dict()
        for name, value in old.items():
            self.assertTrue(torch.equal(value, new[name]), name)
        self.assertEqual(len(separate.step_pattern_experts), 2)
        self.assertIsNot(separate.routed_experts(0)[0], separate.routed_experts(1)[0])
        cfg = {'train': {'lr_main': .001, 'weight_decay': .0001}}
        wrapped = DualBranchSTImputer(separate, torch.nn.Identity())
        opt = build_optimizer(wrapped, cfg)
        model_ids = {id(p) for p in wrapped.parameters() if p.requires_grad}
        optimizer_ids = [id(p) for group in opt.param_groups for p in group['params']]
        self.assertEqual(model_ids, set(optimizer_ids))
        self.assertEqual(len(optimizer_ids), len(set(optimizer_ids)))

    def test_top2_metrics_distinguish_frequency_weight_and_pair_path(self):
        observed, mask, _ = self.inputs()
        torch.manual_seed(11); model = self.small(pair_mode='interaction')
        model.eval()
        result = model(observed, mask)['coe']
        metrics = CoERoutingMetricAccumulator(); metrics.update(result)
        stats = metrics.compute()
        self.assertEqual(stats['coe_top_k'], 2.)
        self.assertGreaterEqual(stats['coe_pair_path_unique_count'], 1.)
        for step in range(1, 4):
            self.assertAlmostEqual(sum(stats[f'coe_step{step}_{name}_selection_rate']
                                       for name in model.expert_names), 2.)
            self.assertAlmostEqual(sum(stats[f'coe_step{step}_{name}_usage']
                                       for name in model.expert_names), 1.)
        counts = [v for k, v in stats.items() if k.startswith('coe_condition_') and
                  k.endswith('_sample_count') and v]
        self.assertTrue(counts)
        self.assertTrue(any(v == 2. for k, v in stats.items() if k.startswith('coe_condition_')
                            and k.endswith('_top_k')))

    def test_pair_ties_follow_native_top2(self):
        model = self.small(pair_mode='additive')
        for logits in (torch.zeros(2, 6),
                       torch.tensor([[1000., 1000., 1000., -1000., -1000., -1000.]])):
            _, ids, _, _ = model._pair_route(logits, None, 1.)
            selected = model.pair_indices[ids]
            expected = logits.topk(2, dim=-1).indices.sort(dim=-1).values
            self.assertTrue(torch.equal(selected, expected))

    def test_family_and_acceptance_metrics_use_exact_hidden_counts(self):
        quality = _CoEQualityMetrics()
        target = torch.tensor([[[[[2., 4.]]]], [[[[10., 20.]]]]])
        mask = torch.tensor([[[[[0., 1.]]]], [[[[0., 0.]]]]])
        old = torch.zeros_like(target)
        candidate = torch.tensor([[[[[5., 0.]]]], [[[[7., 25.]]]]])
        accepted = torch.tensor([[[[[2., 0.]]]], [[[[7., 10.]]]]])
        batch = {'x_f_gt': target, 'm_f': mask, 'mask_family': torch.tensor([0, 1])}
        outputs = {'x_hat_final': accepted,
                   'coe': {'initial_completion': old,
                           'candidate_completions': [candidate], 'completions': [accepted]}}
        quality.update(outputs, batch)
        result = quality.compute()
        self.assertEqual(result['coe_family_random_point_count'], 1.)
        self.assertEqual(result['coe_family_random_point_mae'], 0.)
        self.assertEqual(result['coe_family_node_outage_count'], 2.)
        self.assertEqual(result['coe_family_node_outage_mae'], 6.5)
        self.assertAlmostEqual(result['coe_step1_candidate_harm'], 1/3)
        self.assertAlmostEqual(result['coe_step1_accepted_harm'], 0.)
        self.assertGreater(result['coe_step1_accepted_benefit'], 0.)

    def test_router_parameter_match(self):
        a = self.small(dim=64, router_hidden_dim=64, pair_mode='additive')
        b = self.small(dim=64, router_hidden_dim=64, pair_mode='interaction')
        c = self.small(dim=64, router_hidden_dim=68, pair_mode='additive')
        counts = [sum(p.numel() for p in model.routers[0].parameters()) for model in (a, b, c)]
        self.assertEqual(counts, [14842, 15817, 15742])


if __name__ == '__main__':
    unittest.main()
