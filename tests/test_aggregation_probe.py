import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('aggregation_probe', ROOT/'scripts/v14-exploration/run_aggregation_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def options(self):
        return {'dim': 8, 'strides': [2, 4], 'wide_dilations': [2, 4]}

    def test_mask_geometry_count_and_reproducibility(self):
        shape = (3, 2, 3, 8, 8)
        a = probe.masks(shape, 'scattered', .4, 42, [0, 3, 9])
        b = probe.masks(shape, 'block', .4, 42, [0, 3, 9])
        torch.testing.assert_close(a.sum((2, 3, 4)), b.sum((2, 3, 4)))
        torch.testing.assert_close(a, probe.masks(shape, 'scattered', .4, 42, [0, 3, 9]))
        self.assertFalse(torch.equal(a, b))
        self.assertFalse(torch.equal(a, probe.masks(shape, 'scattered', .4, 100042, [0, 3, 9])))
        self.assertTrue(torch.equal(a[:, :, 0], a[:, :, 1]))

    def test_streamed_subset_matches_original_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'tiny.npz'
            values = np.arange(10*2*3*5*7, dtype=np.float32).reshape(10, 2, 3, 5, 7)
            np.savez_compressed(path, x_f_gt=values)
            x, indices = probe.read_windows(path, 4)
            self.assertEqual(indices, [0, 3, 6, 9])
            torch.testing.assert_close(x, torch.from_numpy(values[indices]))

    def test_common_weights_capacity_and_dilation(self):
        counts = []
        common = None
        for variant in ('fine_local', 'fine_wide', 'fixed', 'geometry', 'adaptive', 'uniform', 'moe'):
            torch.manual_seed(42)
            model = probe.Probe(2, variant, self.options())
            weights = {k: v for k, v in model.state_dict().items() if not k.startswith(('query.', 'key.', 'router.'))}
            if common is None:
                common = weights
            for k, value in weights.items():
                torch.testing.assert_close(value, common[k], atol=0, rtol=0)
            counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        self.assertEqual(len(set(counts[:4])), 1)
        self.assertEqual(counts[4], counts[5])

    def test_no_hidden_leak_and_finite_gradients_all_modes(self):
        x = torch.randn(2, 2, 3, 5, 7)
        mask = (torch.rand(2, 1, 3, 5, 7) > .4).float()
        hidden = x.masked_fill(~mask.expand_as(x).bool(), float('nan'))
        for variant in probe.VARIANTS:
            with self.subTest(variant=variant):
                model = probe.Probe(2, variant, self.options())
                a, _, _ = model(x, mask)
                b, _, _ = model(hidden, mask)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                a.square().mean().backward()
                for p in model.parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad)
                        self.assertTrue(torch.isfinite(p.grad).all())
                for m in (torch.zeros_like(mask), torch.ones_like(mask)):
                    self.assertTrue(torch.isfinite(model(x, m)[0]).all())

    def test_sparse_assignments_and_fixed_observed_mean(self):
        model = probe.Probe(2, 'fixed', self.options())
        features = torch.randn(1, 8, 2, 8, 8)
        mask = (torch.rand(1, 1, 2, 8, 8) > .4).float()
        a, idx, hc, wc = model.assignments(features, mask, 2, 0)
        torch.testing.assert_close(a.sum(-1), torch.ones(2, 64))
        mass = probe.scatter(probe.flat(mask), a, idx, hc*wc)
        mean = probe.scatter(probe.flat(features)*probe.flat(mask), a, idx, hc*wc)/mass.clamp_min(1e-6)
        numerator = torch.nn.functional.avg_pool3d(features*mask, (1, 2, 2))
        denominator = torch.nn.functional.avg_pool3d(mask, (1, 2, 2))
        torch.testing.assert_close(probe.grid(mean, 1, 2, hc, wc), numerator/denominator.clamp_min(1e-6))
        for variant in ('geometry', 'adaptive', 'uniform', 'moe'):
            model = probe.Probe(2, variant, self.options())
            a, idx, hc, wc = model.assignments(features, mask, 2, 0)
            torch.testing.assert_close(a.sum(-1), torch.ones(2, 64))

    def test_training_best_reload_and_strict_completion(self):
        policy = json.loads((ROOT/'configs/v14-exploration/aggregation_probe.json').read_text())
        policy['model'] = self.options()
        policy['train'].update(epochs=2, val_epoch=2, batch_size=2)
        record, _ = probe.identity(policy, 'BikeNYC', 'adaptive', 'block', 42, 'cpu')
        data = {s: torch.randn(3, 2, 2, 8, 8) for s in ('train', 'val', 'test')}
        indices = {s: [0, 3, 6] for s in data}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = probe.train_job(record, root, data, indices, torch.device('cpu'))
            found = probe.complete(root, record)
            self.assertIsNotNone(found)
            self.assertEqual(found['best_epoch'], 2)
            self.assertEqual(len(list(Path(result['run']).glob('*.pt'))), 1)
            changed = dict(record, fingerprint='changed')
            self.assertIsNone(probe.complete(root, changed))
            (Path(result['run'])/'best.pt').unlink()
            self.assertIsNone(probe.complete(root, record))


if __name__ == '__main__':
    unittest.main()
