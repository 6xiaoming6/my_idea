"""I11: all three scales once in the first three rounds, then final F."""
import copy
import itertools
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'src'),
                str(ROOT / 'scripts/v25'), str(ROOT / 'scripts/v24')]
import test_v25_observed_multiscale as fixtures
from run_taxibj_input_communication import jobs
from stmoe_imputer.losses import compute_coe_loss
from stmoe_imputer.utils.deterministic import state_hash
from preflight_taxibj_input_communication import check_consistency


class PermutationScaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.helper = fixtures.V25Tests()
        self.base = self.helper.setup_jobs(self.root)['taxibj_B2_seed7']
        self.batch = self.helper.batch(self.base)

    def job(self, unique=True):
        job = copy.deepcopy(self.base)
        method = 'I11' if unique else 'I10'
        full = jobs(methods=[method])[f'taxibj_{method}_seed7']
        job.update(method=method, variant=full['variant'])
        for key in ('scale_delta_exploration', 'observed_history', 'observation_consistency'):
            job['config']['model']['coe'][key] = copy.deepcopy(full['config']['model']['coe'][key])
        return job

    @staticmethod
    def favor_path(branch, path):
        with torch.no_grad():
            for head, sid in zip(branch.scale_heads, path):
                head[-1].weight.zero_()
                head[-1].bias.zero_()
                head[-1].bias[sid] = 20

    def test_config_data_initialization_and_rejected_combinations(self):
        a, b = self.job(False), self.job()
        for key in ('data', 'train', 'loss', 'seed'):
            self.assertEqual(a['config'][key], b['config'][key])
        self.assertEqual(a['config']['experiment_plan']['protocol'], b['config']['experiment_plan']['protocol'])
        self.assertEqual(state_hash(self.helper.batch(a)), state_hash(self.helper.batch(b)))
        original = self.helper.model(a).eval()
        after_original = state_hash(torch.get_rng_state())
        model = self.helper.model(b).eval()
        self.assertEqual(after_original, state_hash(torch.get_rng_state()))
        self.assertEqual(state_hash(original.state_dict()), state_hash(model.state_dict()))
        self.assertEqual(sum(p.numel() for p in original.parameters()), sum(p.numel() for p in model.parameters()))
        with torch.no_grad():
            torch.testing.assert_close(original(self.batch)['x_hat_main'], model(self.batch)['x_hat_main'], rtol=0, atol=0)
        for key, value in (('first_three_unique', 1), ('last_round_fine', False), ('path', 'CMFF')):
            bad = self.job()
            bad['config']['model']['coe']['scale_delta_exploration'][key] = value
            with self.assertRaises(ValueError):
                self.helper.model(bad)
        for constraint in ('monotone', 'last_two_fine'):
            bad = self.job()
            bad['config']['model']['coe']['id_priority']['constraint'] = constraint
            with self.assertRaises(ValueError):
                self.helper.model(bad)

    def test_all_six_permutations_train_eval_and_reset(self):
        job = self.job()
        model = self.helper.model(job)
        for training in (True, False):
            model.train(training)
            for prefix in itertools.permutations(range(3)):
                # Even a coarse preference in round four must become F.
                self.favor_path(model.main_branch, (*prefix, 2))
                with torch.no_grad():
                    out = model(self.batch)
                    repeat = model(self.batch)
                expected = torch.tensor([*prefix, 0]).expand(len(self.batch['m_f']), -1)
                torch.testing.assert_close(out['coe']['triscale_choices'], expected)
                torch.testing.assert_close(repeat['x_hat_main'], out['x_hat_main'], rtol=0, atol=0)
                check_consistency(out, self.batch, job['config'])
                self.assertFalse(hasattr(model.main_branch, '_exploration_rows'))
            # Repeated strong logit preferences cannot override the legal mask.
            for sid in range(3):
                self.favor_path(model.main_branch, [sid] * 4)
                with torch.no_grad():
                    out = model(self.batch)
                choices = out['coe']['triscale_choices']
                self.assertTrue((choices[:, 0] == sid).all())
                torch.testing.assert_close(choices[:, :3].sort(1).values, torch.tensor([0, 1, 2]).expand(len(choices), -1))
                self.assertTrue((choices[:, 3] == 0).all())

    def test_per_sample_prefix_masks_and_illegal_forced_choices(self):
        model = self.helper.model(self.job()).eval()
        branch = model.main_branch
        paths = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 0]])
        expected = torch.ones((2, 3), dtype=torch.bool)
        original = branch._scales
        seen = []

        def per_sample(c, features, step, force=None, probe=False):
            nonlocal expected
            legal = branch.legal_scales(c, step)
            if step == 3:
                expected = torch.tensor([[True, False, False]] * 2)
            torch.testing.assert_close(legal, expected)
            seen.append(legal.clone())
            if step in (1, 2):
                with self.assertRaisesRegex(ValueError, 'Forced scale violates'):
                    original(c, features, step, force=paths[:, 0])
            if step == 3:
                with self.assertRaisesRegex(ValueError, 'Forced scale violates'):
                    original(c, features, step, force=torch.tensor([1, 2]))
            result = original(c, features, step, force=paths[:, step], probe=probe)
            expected = expected.scatter(1, paths[:, step:step+1], False)
            return result

        with patch.object(branch, '_scales', side_effect=per_sample), torch.no_grad():
            out = model(self.batch)
        torch.testing.assert_close(out['coe']['triscale_choices'], paths)
        self.assertFalse(torch.equal(seen[1][0], seen[1][1]))
        # A new forward must start with all scales available for both samples.
        with torch.no_grad():
            again = model(self.batch)
        self.assertTrue((again['coe']['triscale_choices'][:, :3].sort(1).values == torch.tensor([0, 1, 2])).all())

    def test_sparse_execution_gradients_history_projection_and_no_leak(self):
        job = self.job()
        model = self.helper.model(job).train()
        branch = model.main_branch
        self.assertEqual(branch.learnable_steps(), (0, 1))
        self.assertTrue(all(p.requires_grad for head in branch.scale_heads[:2] for p in head.parameters()))
        self.assertTrue(all(not p.requires_grad for head in branch.scale_heads[2:] for p in head.parameters()))
        calls, pools = [], []
        hooks = [e.register_forward_pre_hook(lambda mod, args: calls.append(len(args[0]))) for e in branch.routed_experts()]
        hooks.append(branch.x_history.register_forward_pre_hook(lambda mod, args: pools.append([(x.requires_grad, x.grad_fn) for x in args[1]])))
        try:
            out = model(self.batch)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(sum(calls), 8 * len(self.batch['m_f']))
        self.assertEqual([len(p) for p in pools], [1, 2, 3])
        self.assertTrue(all(not req and fn is None for p in pools for req, fn in p))
        self.assertEqual([float(out['coe']['diagnostics'][f'x_step{i}_pool_size']) for i in range(1, 5)], [0, 1, 2, 3])
        check_consistency(out, self.batch, job['config'])
        loss, _ = compute_coe_loss(out, self.batch, job['config'])
        loss.backward()
        for head in [*branch.scale_heads[:2], branch.history_gate, branch.x_history, branch.x_consistency]:
            grads = [p.grad for p in head.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(bool(g.abs().sum() > 0) for g in grads))
        self.assertTrue(all(p.grad is None for head in branch.scale_heads[2:] for p in head.parameters()))
        model.eval()
        poison = copy.deepcopy(self.batch)
        for scale in ('f', 'm', 'c'):
            poison['x_' + scale + '_gt'] = torch.full_like(poison['x_' + scale + '_obs'], float('nan'))
            poison['x_' + scale + '_obs'][~poison['m_' + scale].bool()] = float('nan')
        with torch.no_grad():
            torch.testing.assert_close(model(poison)['x_hat_main'], model(self.batch)['x_hat_main'], rtol=0, atol=0)

    def test_two_epoch_strict_resume_and_all_protocols(self):
        from train_four_direction import train
        from evaluate_four_direction import evaluate_sets
        from run_b3_c3 import digest
        job = self.job()
        job['config']['train'].update(epochs=2, val_epoch=1)
        job['config']['train']['scheduler']['total_epochs'] = 2
        resumed = self.root / 'resumed'
        full = self.root / 'full'
        train(job, resumed, self.root / 'r.json', 'cpu', stop_after=1)
        train(job, resumed, self.root / 'r.json', 'cpu')
        train(job, full, self.root / 'f.json', 'cpu')
        a = torch.load(resumed / 'checkpoints/last.pth', weights_only=False)
        b = torch.load(full / 'checkpoints/last.pth', weights_only=False)
        for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rng_states'):
            self.assertEqual(state_hash(a[key]), state_hash(b[key]), key)
        self.assertEqual(state_hash(a['training_state']['w_training']), state_hash(b['training_state']['w_training']))
        evaluation = self.root / 'evaluation.json'
        evaluate_sets(resumed / 'checkpoints/best.pth', job['config']['experiment_plan']['protocol'],
                      job['sources']['test'], evaluation, digest(job['config']), 'cpu')
        record = json.loads(evaluation.read_text())
        self.assertEqual(len(record['sets']), 6)
        for row in record['sets'].values():
            metrics = row['metrics']
            self.assertEqual(metrics['coe_expert_execution_count'], 8.)
            self.assertLess(metrics['coe_v25_observed_sum_residual_C'], 2e-3)
            paths = {key.removeprefix('coe_scale_path_').removesuffix('_fraction'): value
                     for key, value in metrics.items()
                     if re.fullmatch(r'coe_scale_path_[fmc]{4}_fraction', key) and value > 0}
            self.assertTrue(paths)
            for path in paths:
                self.assertEqual(sorted(path[:3]), ['c', 'f', 'm'])
                self.assertEqual(path[-1], 'f')


if __name__ == '__main__':
    unittest.main()
