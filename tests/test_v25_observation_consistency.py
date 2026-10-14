"""CPU-only checks for I04 supervised consistency and I05 observed constraints.

These tests use tiny synthetic tensors. They never launch a training queue or
read test labels through the model's forward path.
"""
import copy
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'src'),
                str(ROOT / 'scripts/v25'), str(ROOT / 'scripts/v24')]
import test_v25_observed_multiscale as fixtures
from stmoe_imputer.data import build_datasets
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash


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


class ObservationConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.helper = fixtures.V25Tests()
        self.base = self.helper.setup_jobs(Path(self.temp.name))['taxibj_B2_seed7']
        self.batch = self.helper.batch(self.base)

    def job(self, mode):
        result = copy.deepcopy(self.base)
        result['config']['model']['coe']['observation_consistency'] = {
            'enabled': True, 'mode': mode}
        if mode == 'full_target_loss':
            result['config']['loss']['lambda_observation_consistency'] = .1
            result['config']['data']['return_multiscale_targets'] = True
        else:
            result['config']['model']['coe']['observation_consistency']['weight_bound'] = 2.
        return result

    def test_disabled_and_full_target_modes_preserve_b2_forward(self):
        baseline = self.helper.model(self.base).eval()
        enabled = self.helper.model(self.job('full_target_loss')).eval()
        disabled_job = self.job('residual_projection')
        disabled_job['config']['model']['coe']['observation_consistency']['enabled'] = False
        disabled = self.helper.model(disabled_job).eval()
        self.assertEqual(state_hash(baseline.state_dict()), state_hash(enabled.state_dict()))
        self.assertEqual(state_hash(baseline.state_dict()), state_hash(disabled.state_dict()))
        poison = copy.deepcopy(self.batch)
        for scale, factor in [('f', 1), ('m', 2), ('c', 4)]:
            poison['x_' + scale + '_gt'] = torch.full_like(
                aggregate(self.batch['x_f_gt'], factor), float('nan'))
            poison['x_' + scale + '_obs'][~poison['m_' + scale].bool()] = float('nan')
        with torch.no_grad():
            expected = baseline(self.batch)['x_hat_main']
            for model in (enabled, disabled):
                torch.testing.assert_close(model(self.batch)['x_hat_main'], expected, rtol=0, atol=0)
                torch.testing.assert_close(model(poison)['x_hat_main'], expected, rtol=0, atol=0)

    def test_full_target_labels_are_opt_in_and_loss_uses_complete_clamped_f(self):
        job = self.job('full_target_loss')
        plain, _ = build_datasets(self.base['config'], self.base['sources']['train'], self.base['sources']['val'])
        labelled, _ = build_datasets(job['config'], job['sources']['train'], job['sources']['val'])
        self.assertNotIn('x_m_gt', plain[0])
        self.assertNotIn('x_c_gt', plain[0])
        for scale, factor in [('m', 2), ('c', 4)]:
            torch.testing.assert_close(labelled[0]['x_' + scale + '_gt'],
                                       aggregate(labelled[0]['x_f_gt'][None], factor)[0], rtol=0, atol=0)
        batch = self.helper.batch(job)
        # Hide all coarse observations: supervision must still use full labels.
        batch['m_m'].zero_()
        batch['m_c'].zero_()
        batch['x_m_obs'].zero_()
        batch['x_c_obs'].zero_()
        pred = (batch['x_f_gt'] + 2.).detach().requires_grad_(True)
        outputs = {'x_hat_main': pred, 'coe': {'predictions': [pred] * 4, 'use_routed': False}}
        baseline, _ = compute_coe_loss(outputs, batch, self.base['config'])
        total, _ = compute_coe_loss(outputs, batch, job['config'])
        completed = torch.where(batch['m_f'].bool(), batch['x_f_obs'], pred)
        expected = torch.stack([
            ((aggregate(completed, factor) - batch['x_' + name + '_gt']) / factor ** 2).abs().mean()
            for name, factor in [('m', 2), ('c', 4)]
        ]).mean()
        torch.testing.assert_close(total - baseline, .1 * expected)
        (total - baseline).backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()))
        self.assertGreater(float(pred.grad[~batch['m_f'].bool()].abs().sum()), 0.)
        self.assertFalse(bool(pred.grad[batch['m_f'].bool()].any()))
        # Mean pooling uses no additional division by block area.
        mean_cfg = copy.deepcopy(job['config'])
        mean_cfg['data']['scales']['pooling_mode'] = 'mean'
        mean_batch = copy.deepcopy(batch)
        for name, factor in [('m', 2), ('c', 4)]:
            mean_batch['x_' + name + '_gt'] /= factor ** 2
        mean_total, _ = compute_coe_loss(outputs, mean_batch, mean_cfg)
        torch.testing.assert_close(mean_total, total)
        # The requested baseline supervises the precomputed coarse labels;
        # silently reaggregating the fine label would bypass this contract.
        changed = copy.deepcopy(batch)
        changed['x_m_gt'] += 40.
        changed_total, _ = compute_coe_loss(outputs, changed, job['config'])
        changed_expected = torch.stack([
            ((aggregate(completed, factor) - changed['x_' + name + '_gt']) / factor ** 2).abs().mean()
            for name, factor in [('m', 2), ('c', 4)]
        ]).mean()
        torch.testing.assert_close(changed_total - baseline, .1 * changed_expected)
        self.assertGreater(float((changed_total - total).abs()), 0.)
        missing_labels = copy.deepcopy(batch)
        del missing_labels['x_c_gt']
        with self.assertRaises((ValueError, KeyError)):
            compute_coe_loss(outputs, missing_labels, job['config'])

    def test_projection_keeps_common_initialization_and_has_trainable_weights(self):
        job = self.job('residual_projection')
        baseline = self.helper.model(self.base)
        model = self.helper.model(job)
        common = {key: value for key, value in model.state_dict().items()
                  if not key.startswith('main_branch.x_consistency.')}
        self.assertEqual(state_hash(common), state_hash(baseline.state_dict()))
        model.eval()
        with torch.no_grad():
            first = model(self.batch)
            poison = copy.deepcopy(self.batch)
            for name, factor in [('f', 1), ('m', 2), ('c', 4)]:
                poison['x_' + name + '_gt'] = torch.full_like(aggregate(self.batch['x_f_gt'], factor), float('nan'))
                poison['x_' + name + '_obs'][~poison['m_' + name].bool()] = float('nan')
            torch.testing.assert_close(model(poison)['x_hat_main'], first['x_hat_main'], rtol=0, atol=0)
        model.train()
        output = model(self.batch)
        self.assertIn('x_hat_raw_consistency', output)
        # Nonuniform coefficients ensure this probe does not cancel under a
        # sum constraint, unlike summing all constrained output locations.
        coefficients = torch.arange(output['x_hat_main'].numel(), dtype=torch.float32)
        coefficients = coefficients.sin().reshape_as(output['x_hat_main'])
        probe = (output['x_hat_main'] * coefficients).sum()
        probe.backward()
        gradients = [p.grad for p in model.main_branch.x_consistency.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertTrue(any(bool(g.abs().sum() > 0) for g in gradients))

    def test_jobs_append_two_isolated_interventions_with_original_budget(self):
        from run_taxibj_input_communication import jobs
        actual = jobs(methods=['I01','I02','I03','I04','I05'])
        self.assertEqual(list(actual), [f'taxibj_I{i:02d}_seed7' for i in range(1, 6)])
        original = actual['taxibj_I01_seed7']['config']
        for number, mode in [(4, 'full_target_loss'), (5, 'residual_projection')]:
            config = actual[f'taxibj_I{number:02d}_seed7']['config']
            self.assertEqual(config['train'], original['train'])
            self.assertEqual(config['experiment_plan']['protocol'], original['experiment_plan']['protocol'])
            self.assertEqual(config['model']['coe']['initial_observation_scales'], 'FMC')
            self.assertEqual(config['model']['coe']['core_validation']['communication'], 'conditional')
            self.assertFalse(config['model']['coe'].get('observed_history', {}).get('enabled', False))
            self.assertFalse(config['model']['coe'].get('scale_delta_exploration', {}).get('enabled', False))
            self.assertEqual(config['model']['coe']['observation_consistency']['mode'], mode)
            data = copy.deepcopy(config['data'])
            self.assertEqual(bool(data.pop('return_multiscale_targets', False)), number == 4)
            self.assertEqual(data, original['data'])


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


if __name__ == '__main__':
    unittest.main()
