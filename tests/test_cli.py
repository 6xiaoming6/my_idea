"""Exercise public CLI end to end, including exact epoch-boundary continuation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from test_release import ROOT, make_data, small_config
from release_utils import load, write
from stmoe_imputer.utils.deterministic import state_hash


class CommandLine(unittest.TestCase):
    def test_train_resume_evaluate_infer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / 'data'; make_data(data)
            config = root / 'config.json'; write(config, small_config())
            def run(script, *args):
                result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, args)],
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 0, result.stdout[-6000:])
            common = ['--data-dir', data, '--config', config, '--device', 'cpu', '--threads', 2]
            run('train.py', *common, '--run-dir', root / 'resumed', '--stop-after', 1)
            self.assertFalse((root / 'resumed/result.json').exists())
            run('train.py', *common, '--run-dir', root / 'resumed', '--evaluate')
            run('train.py', *common, '--run-dir', root / 'continuous')
            a = torch.load(root / 'resumed/checkpoints/last.pth', weights_only=False, map_location='cpu')
            b = torch.load(root / 'continuous/checkpoints/last.pth', weights_only=False, map_location='cpu')
            for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rng_states'):
                self.assertEqual(state_hash(a[key]), state_hash(b[key]), key)
            self.assertEqual(a['epoch'], 2)
            evaluation = load(root / 'resumed/evaluation.json')
            self.assertEqual(evaluation['status'], 'finished')
            self.assertEqual(len(evaluation['sets']), 6)
            self.assertTrue(all(np.isfinite(v['metrics']['mae']) for v in evaluation['sets'].values()))
            checkpoint = root / 'resumed/checkpoints/best.pth'
            run('evaluate.py', '--checkpoint', checkpoint, '--data-dir', data,
                '--output', root / 'standalone.json', '--device', 'cpu', '--threads', 2)
            standalone = load(root / 'standalone.json')
            for name, row in evaluation['sets'].items():
                self.assertEqual(row['metrics']['mae'], standalone['sets'][name]['metrics']['mae'])
            from stmoe_imputer.data import build_datasets
            sample = build_datasets(small_config(), data / 'taxibj_train.npz', data / 'taxibj_val.npz')[1][0]
            arrays = {k: v.unsqueeze(0).numpy() for k, v in sample.items() if k.endswith('_obs') or k in ('m_f','m_m','m_c')}
            np.savez_compressed(root / 'observations.npz', **arrays)
            run('infer.py', '--checkpoint', checkpoint, '--input', root / 'observations.npz',
                '--output', root / 'completed.npz', '--device', 'cpu')
            with np.load(root / 'completed.npz') as z:
                self.assertEqual(z['F_complete'].shape, (1, 2, 4, 8, 8))
                mask = arrays['m_f'].astype(bool)
                np.testing.assert_array_equal(z['F_complete'][mask], arrays['x_f_obs'][mask])
            # An already finished run must be idempotent, not start training again.
            before = (root / 'resumed/checkpoints/last.pth').stat().st_mtime_ns
            run('train.py', *common, '--run-dir', root / 'resumed')
            self.assertEqual(before, (root / 'resumed/checkpoints/last.pth').stat().st_mtime_ns)


if __name__ == '__main__': unittest.main()
