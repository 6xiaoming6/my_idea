import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'src/stmoe_imputer/utils/gpu_monitor.py'
spec = importlib.util.spec_from_file_location('gpu_monitor', MODULE)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


class GPUMonitorTests(unittest.TestCase):
    def test_query_errors_are_data(self):
        cases = [FileNotFoundError('missing'), subprocess.TimeoutExpired('nvidia-smi', 2)]
        for error in cases:
            with patch.object(gpu.subprocess, 'run', side_effect=error):
                self.assertIn('error', gpu.query('gpu', gpu.GPU_FIELDS))
        with patch.object(gpu.subprocess, 'run', return_value=subprocess.CompletedProcess([], 9, '', 'GPU lost')):
            result = gpu.query('gpu', gpu.GPU_FIELDS)
            self.assertEqual(result['returncode'], 9)
            self.assertEqual(result['error'], 'GPU lost')

    def test_append_lock_and_context_cleanup_on_training_error(self):
        with tempfile.TemporaryDirectory() as td:
            executable = Path(td) / 'nvidia-smi'
            executable.write_text('#!' + sys.executable + '\nimport sys\n'
                                  'print("GPU-test, 123, 456" if "compute-apps" in sys.argv[1] '
                                  'else "0, GPU-test, 0000:01:00.0, 550, 60, 120, 350, 95, 456, 24576")\n')
            executable.chmod(0o755)
            path = Path(td) / 'logs/gpu_status.jsonl'
            path.parent.mkdir()
            path.write_text('{"event":"old"}\n')
            with patch.dict(os.environ, PATH=td + os.pathsep + os.environ['PATH']):
                with self.assertRaisesRegex(RuntimeError, 'training failed'):
                    with gpu.GPUMonitor(td) as monitor:
                        deadline = time.monotonic() + 8
                        while '"sample"' not in path.read_text() and time.monotonic() < deadline:
                            time.sleep(.05)
                        self.assertIn('"sample"', path.read_text())
                        with self.assertRaises(BlockingIOError):
                            gpu.Log(td, os.getpid(), gpu.process_identity(os.getpid()), 'duplicate')
                        raise RuntimeError('training failed')
                self.assertIsNotNone(monitor.process.poll())
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]['event'], 'old')
            self.assertEqual(rows[-1]['event'], 'stop')
            sample = next(r for r in rows if r['event'] == 'sample')
            self.assertEqual(sample['gpus']['rows'][0]['uuid'], 'GPU-test')
            self.assertEqual(sample['compute_processes']['rows'][0]['pid'], '123')

    def test_attached_monitor_exits_when_target_dies(self):
        with tempfile.TemporaryDirectory() as td:
            target = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            worker = subprocess.Popen([sys.executable, str(MODULE), '--pid', str(target.pid),
                                       '--run-dir', td, '--interval', '.1'],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                path = Path(td) / 'logs/gpu_status.jsonl'
                deadline = time.monotonic() + 8
                while not path.exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(path.exists())
                target.terminate()
                target.wait(timeout=3)
                worker.wait(timeout=8)
                self.assertEqual(worker.returncode, 0)
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows[-1]['event'], 'stop')
            finally:
                for process in (target, worker):
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                worker.stderr.close()

    def test_queue_rollover_routes_samples_to_separate_logs(self):
        import threading
        stop = threading.Event()
        with tempfile.TemporaryDirectory() as td:
            first, second = Path(td) / 'first', Path(td) / 'second'
            step = [0]
            def children(_):
                step[0] += 1
                return [(10, first)] if step[0] == 1 else [(20, second)]
            def identity(pid):
                return None if pid == 10 and step[0] > 1 else str(pid)
            def wait(_):
                if step[0] == 2:
                    stop.set()
            with patch.object(gpu, 'training_children', side_effect=children), \
                 patch.object(gpu, 'process_identity', side_effect=identity), \
                 patch.object(gpu, 'query', return_value={'rows': []}), \
                 patch.object(stop, 'wait', side_effect=wait):
                gpu.monitor(1, queue=True, stop=stop)
            for directory, pid in [(first, 10), (second, 20)]:
                rows = [json.loads(line) for line in (directory / 'logs/gpu_status.jsonl').read_text().splitlines()]
                self.assertEqual([r['training_pid'] for r in rows if r['event'] == 'sample'], [pid])
                self.assertEqual(rows[-1]['event'], 'stop')


if __name__ == '__main__':
    unittest.main()
