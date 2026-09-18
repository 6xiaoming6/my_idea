from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from stmoe_imputer.utils.checkpoint import snapshot_model_state

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('v24_runner', ROOT / 'scripts/v24/run_experiments.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LifecycleTests(unittest.TestCase):
    def test_cpu_snapshot_is_independent_and_keeps_buffers_and_metadata(self):
        model = torch.nn.BatchNorm1d(3)
        state = snapshot_model_state(model)
        with torch.no_grad():
            model.weight.fill_(4)
            model.running_mean.fill_(9)
            model.num_batches_tracked.add_(5)
        self.assertTrue(torch.equal(state['weight'], torch.ones(3)))
        self.assertTrue(torch.equal(state['running_mean'], torch.zeros(3)))
        self.assertEqual(state['num_batches_tracked'].item(), 0)
        self.assertTrue(hasattr(state, '_metadata'))
        model.load_state_dict(state)
        self.assertTrue(torch.equal(model.weight, torch.ones(3)))

    def test_input_change_is_detected_before_next_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.csv'; path.write_text('1,0')
            stamps = {str(path): (runner.signature(path), runner.planner._sha256(path))}
            runner.assert_unchanged(stamps, set(runner.source_files()))
            path.write_text('0,1')
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                runner.assert_unchanged(stamps, set(runner.source_files()))

    def test_fingerprint_covers_config_and_dataset_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.csv'; path.write_text('1,0')
            args = runner.planner.parse_args(['--base-config', str(ROOT/'configs/v24/smoke.json'),
                '--output-dir', tmp, '--synthetic', '--variants', 'full'])
            plan = runner.planner.build_plan(args)
            before, _ = runner.identity(plan, [path])
            changed = copy.deepcopy(plan); changed['runs'][0]['config']['train']['val_epoch'] = 2
            after, _ = runner.identity(changed, [path])
            self.assertNotEqual(runner.digest(before), runner.digest(after))
            path.write_text('0,1')
            after, _ = runner.identity(plan, [path])
            self.assertNotEqual(runner.digest(before), runner.digest(after))

    def test_comparison_selects_fixed_path_using_validation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            variants = ['full', 'fixed_tt', 'fixed_ts', 'fixed_st', 'fixed_ss']
            manifest = {'suite_fingerprint': 'test', 'runs': [{'name': v, 'variant': v,
                'protocol': 'same_mask', 'seed': 7, 'config': {'data': {'dataset_name': 'Tiny'}}} for v in variants]}
            def result(run, *_):
                i = variants.index(run['variant'])
                return {'best_epoch': 2, 'best_val_mae': [5, 1, 2, 3, 4][i],
                        'test': {'mae': [5, 9, 1, 3, 4][i], 'rmse': 10.}, 'total_time_sec': 1., 'run_dir': 'test'}
            with patch.object(runner, 'result_for', side_effect=result):
                runner.summarize(manifest, suite)
            analysis = runner.load(suite/'comparison.json')
            self.assertEqual(analysis['validation_selected_fixed_path'][0]['variant'], 'fixed_tt')
            self.assertEqual(analysis['validation_selected_fixed_path'][0]['test_mae'], 9)
            self.assertEqual(len(analysis['paired']), 4)


class RunnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='v24_runner_test_')
        cls.directory = Path(cls.tmp.name)
        cls.policy = cls.directory/'policy.json'
        cls.env = dict(os.environ, MPLCONFIGDIR=str(cls.directory/'mpl'),
                       XDG_CACHE_HOME=str(cls.directory/'cache'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        value = {'schema_version': 1, 'output_dir': str(cls.directory/'results'), 'cpu_threads': 2,
                 'seeds': [7], 'training': {'epochs': 3, 'val_epoch': 2, 'save_best_checkpoint': False},
                 'studies': {'smoke': {'base_config': str(ROOT/'configs/v24/smoke.json'),
                                      'stage': 'pilot', 'synthetic': True, 'variants': ['full', 'fixed_ts']}}}
        cls.policy.write_text(json.dumps(value))
        cls.command = [sys.executable, str(ROOT/'scripts/v24/run_experiments.py'),
                       '--config', str(cls.policy), '--study', 'smoke']
        dry = cls.invoke('--dry-run')
        cls.suite = Path(json.loads(dry.stdout)['suite'])
        assert not cls.suite.exists(), 'dry run wrote experiment artifacts'
        cls.first = cls.invoke()
        cls.manifest = runner.load(cls.suite/'plan.json')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def invoke(cls, *args):
        result = subprocess.run([*cls.command, *args], cwd=ROOT, env=cls.env, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    def test_memory_mode_validates_schedule_final_epoch_and_tests_once(self):
        for run in self.manifest['runs']:
            receipt = runner.result_for(run, self.suite, self.manifest)
            self.assertIsNotNone(receipt)
            self.assertEqual(receipt['best_state_source'], 'cpu_memory')
            self.assertEqual(receipt['samples'], {'train': 8, 'val': 4, 'test': 4})
            directory = Path(receipt['run_dir'])
            self.assertFalse((directory/'checkpoints').exists())
            rows = [json.loads(x) for x in (directory/'logs/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in rows if r.get('val') is not None], [2, 3])
            self.assertEqual(sum(r.get('stage') == 'test' for r in rows), 1)
            self.assertTrue(any('coe_expert_' in key for key in rows[0]['train']))
            if run['variant'] == 'full':
                self.assertTrue(any('router_grad_norm' in key for key in rows[0]['train']))
        lines = [x.strip() for x in self.first.stdout.splitlines() if x.strip()]
        self.assertTrue(lines)
        self.assertTrue(all(x.startswith('train epoch ') for x in lines), self.first.stdout)
        self.assertFalse(self.first.stderr.strip())

    def test_restart_skips_completed_jobs_and_summary_only_does_not_train(self):
        before = {str(p): p.stat().st_mtime_ns for p in self.suite.rglob('metrics.jsonl')}
        rerun = self.invoke()
        self.assertFalse(rerun.stdout.strip())
        self.invoke('--summary-only')
        after = {str(p): p.stat().st_mtime_ns for p in self.suite.rglob('metrics.jsonl')}
        self.assertEqual(before, after)
        self.assertEqual(runner.load(self.suite/'comparison.json')['complete'], 2)

    def test_incomplete_records_are_not_accepted_even_with_a_receipt(self):
        run = self.manifest['runs'][0]
        receipt_file = next((self.suite/'results').glob(run['name']+'.attempt*.json'))
        receipt = runner.load(receipt_file)
        metrics = Path(receipt['run_dir'])/'logs/metrics.jsonl'
        original = metrics.read_text()
        try:
            metrics.write_text('\n'.join(original.splitlines()[:-1])+'\n')
            self.assertFalse(runner.check_complete(receipt_file, run, self.manifest))
        finally:
            metrics.write_text(original)
        self.assertTrue(runner.check_complete(receipt_file, run, self.manifest))

    def test_interrupted_job_restarts_without_rerunning_other_jobs(self):
        run, other = self.manifest['runs']
        before = runner.result_for(run, self.suite, self.manifest)
        other_before = runner.result_for(other, self.suite, self.manifest)
        metrics = Path(before['run_dir'])/'logs/metrics.jsonl'
        original = metrics.read_text()
        metrics.write_text('\n'.join(original.splitlines()[:-1])+'\n')
        self.invoke()
        after = runner.result_for(run, self.suite, self.manifest)
        self.assertNotEqual(before['run_dir'], after['run_dir'])
        self.assertEqual(after['completed_epochs'], 3)
        self.assertTrue(metrics.exists())
        self.assertEqual(runner.result_for(other, self.suite, self.manifest), other_before)
        self.assertEqual(len(list((self.suite/'launcher_logs').glob(run['name']+'.attempt*.log'))), 2)

    def test_generated_commands_use_managed_runner(self):
        directory = self.directory/'generated_plan'
        args = runner.planner.parse_args(['--base-config', str(ROOT/'configs/v24/smoke.json'),
            '--output-dir', str(directory), '--synthetic', '--variants', 'full'])
        manifest = runner.planner.build_plan(args)
        runner.planner.write_plan(manifest, directory)
        command = directory/'commands.sh'
        self.assertIn('run_experiments.py', command.read_text())
        result = subprocess.run(['bash', str(command)], cwd=ROOT, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        summaries = list((directory/'managed').glob('*/comparison.json'))
        self.assertEqual(len(summaries), 1)
        self.assertEqual(runner.load(summaries[0])['complete'], 1)

    def test_disk_and_memory_modes_produce_identical_selected_test_metrics(self):
        original = self.manifest['runs'][0]
        before = runner.result_for(original, self.suite, self.manifest)
        cfg = copy.deepcopy(original['config'])
        cfg['output_dir'] = str(self.directory/'disk_run')
        cfg['train']['save_best_checkpoint'] = True
        path = self.directory/'disk.json'; path.write_text(json.dumps(cfg))
        receipt_file = self.directory/'disk_receipt.json'
        command = [sys.executable, str(ROOT/'scripts/train.py'), '-c', str(path), '--synthetic',
                   '--no_plot', '--result-file', str(receipt_file)]
        result = subprocess.run(command, cwd=ROOT, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        after = runner.load(receipt_file)
        self.assertEqual(before['best_epoch'], after['best_epoch'])
        self.assertEqual(before['test'], after['test'])
        checkpoints = list(Path(after['run_dir']).rglob('*.pt'))
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].name, 'best.pt')


if __name__ == '__main__':
    unittest.main()
