"""Disk-only dependency gate; no GPU imports or real training subprocesses."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/v25'))
import run_after_suite as queue


class QueueDependencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dep = self.root / 'dependency'
        self.dep.mkdir()
        self.protocol = {'rate': .4, 'evaluations': {
            name: {'families': [name], 'seed': i + 1}
            for i, name in enumerate(sorted(queue.PROTOCOLS))}}
        self.config = {'train': {'epochs': 100}, 'experiment_plan': {'protocol': self.protocol}}
        self.job = {'config': self.config, 'sources': {'test': str(self.root / 'test.npz')}}
        self.plan = {'jobs': {'job': self.job}, 'order': ['job']}
        queue.write(self.dep / 'plan.json', self.plan)
        self.dependency = {'path': str(self.dep), 'plan_sha256': queue.file_sha(self.dep / 'plan.json')}

    def finish(self, suite=None, plan=None):
        suite, plan = suite or self.dep, plan or self.plan
        run = suite / 'run'
        (run / 'checkpoints').mkdir(parents=True, exist_ok=True)
        for name in ('best.pth', 'last.pth'):
            (run / 'checkpoints' / name).write_bytes(b'test receipt only')
        report = suite / 'formal_report.md'
        report.write_text('complete report')
        queue.write(suite / 'report_path.json', {'path': str(report), 'complete': True, 'issues': []})
        queue.write(suite / 'completed.json', {'count': 1, 'report': str(report)})
        queue.write(suite / 'state.json', {'phase': 'completed', 'report': str(report)})
        queue.write(suite / 'jobs/job.json', plan['jobs']['job'])
        queue.write(suite / 'results/job.json', {'status': 'finished', 'config_sha256': queue.digest(self.config),
            'completed_epochs': 100, 'best_epoch': 95, 'run_dir': str(run)})
        self.evaluation = {'status': 'finished', 'config_sha256': queue.digest(self.config),
            'protocol_sha256': queue.digest(self.protocol), 'best_epoch': 95, 'split': 'test',
            'checkpoint': str(run / 'checkpoints/best.pth'), 'npz': self.job['sources']['test'], 'sets': {}}
        for name, spec in self.protocol['evaluations'].items():
            self.evaluation['sets'][name] = {'metrics': {'mae': 1., 'rmse': 2.}, 'samples': 8,
                'mask_spec': {'families': spec['families'], 'rates': [.4], 'seed': spec['seed'],
                              'resample_each_epoch': False}}
        queue.write(suite / 'evaluations/job.json', self.evaluation)

    def waiter(self):
        suite = self.root / 'waiter'
        suite.mkdir()
        runner = suite / 'source_snapshot' / queue.RUNNER
        runner.parent.mkdir(parents=True)
        runner.write_text('# never executed by these tests\n')
        plan = copy.deepcopy(self.plan)
        plan.update(run_after_suite=self.dependency, project_root=str(self.root),
                    source={queue.RUNNER: queue.file_sha(runner)})
        queue.write(suite / 'plan.json', plan)
        return suite, plan

    def test_running_waits_and_completed_requires_full_audit(self):
        self.assertEqual(queue.dependency_status(self.dependency)[0], 'waiting')
        queue.write(self.dep / 'state.json', {'phase': 'evaluating'})
        self.assertEqual(queue.dependency_status(self.dependency)[0], 'waiting')
        self.finish()
        self.assertEqual(queue.dependency_status(self.dependency)[0], 'ready')

    def test_failed_paused_and_plan_replacement_block(self):
        for phase in ('failed', 'paused', 'paused_by_user'):
            with self.subTest(phase=phase):
                queue.write(self.dep / 'state.json', {'phase': phase})
                self.assertEqual(queue.dependency_status(self.dependency)[0], 'blocked')
        queue.write(self.dep / 'state.json', {'phase': 'training'})
        queue.write(self.dep / 'paused_evaluation.json', {'status': 'paused_by_user'})
        self.assertEqual(queue.dependency_status(self.dependency)[0], 'blocked')
        (self.dep / 'paused_evaluation.json').unlink()
        queue.write(self.dep / 'plan.json', {'unexpected': 'replacement'})
        self.assertIn('hash', queue.dependency_status(self.dependency)[1])

    def test_missing_report_or_protocol_blocks_completed_suite(self):
        self.finish()
        report = self.dep / 'formal_report.md'
        report.unlink()
        self.assertEqual(queue.dependency_status(self.dependency)[0], 'blocked')
        report.write_text('report restored')
        self.evaluation['sets'].pop('unseen_triple')
        queue.write(self.dep / 'evaluations/job.json', self.evaluation)
        self.assertIn('six', queue.dependency_status(self.dependency)[1])

    def test_config_protocol_epoch_and_mask_mismatches_block(self):
        self.finish()
        for key, wrong in [('config_sha256', 'wrong'), ('protocol_sha256', 'wrong'), ('best_epoch', 90)]:
            with self.subTest(key=key):
                altered = copy.deepcopy(self.evaluation)
                altered[key] = wrong
                queue.write(self.dep / 'evaluations/job.json', altered)
                self.assertEqual(queue.dependency_status(self.dependency)[0], 'blocked')
        altered = copy.deepcopy(self.evaluation)
        altered['sets']['in_distribution']['mask_spec']['seed'] = -1
        queue.write(self.dep / 'evaluations/job.json', altered)
        self.assertIn('mask', queue.dependency_status(self.dependency)[1])

    def test_waits_for_full_completion_and_scheduler_release_then_launches_once(self):
        suite, plan = self.waiter()
        sleeps = []
        def sleep(_):
            sleeps.append(True)
            if len(sleeps) == 1:
                self.finish()
        def child(command, **kwargs):
            self.assertEqual(command[-2:], ['--suite', str(suite)])
            self.finish(suite, plan)
            return 0
        with patch.object(queue.time, 'sleep', side_effect=sleep), \
             patch.object(queue, 'queue_locks_available', side_effect=[False, True]), \
             patch.object(queue.subprocess, 'call', side_effect=child) as launched:
            self.assertEqual(queue.run(suite), 0)
            launched.assert_called_once()
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(queue.load(suite / 'waiting.json')['phase'], 'completed')

    def test_duplicate_waiter_and_failed_dependency_never_launch(self):
        suite, _ = self.waiter()
        with queue.exclusive_waiter(suite), patch.object(queue.subprocess, 'call') as launched:
            self.assertEqual(queue.run(suite), 75)
            launched.assert_not_called()
        queue.write(self.dep / 'state.json', {'phase': 'failed'})
        with patch.object(queue.subprocess, 'call') as launched:
            self.assertEqual(queue.run(suite), 2)
            launched.assert_not_called()
        self.assertEqual(queue.load(suite / 'waiting.json')['phase'], 'blocked')

    def test_completed_waiter_is_noop_and_failed_runner_is_not_retried(self):
        self.finish()
        suite, plan = self.waiter()
        with patch.object(queue.subprocess, 'call', return_value=7) as launched:
            self.assertEqual(queue.run(suite), 7)
            launched.assert_called_once()
        self.finish(suite, plan)
        with patch.object(queue.subprocess, 'call') as launched:
            self.assertEqual(queue.run(suite), 0)
            launched.assert_not_called()


if __name__ == '__main__':
    unittest.main()
