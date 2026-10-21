"""CPU contract tests. No private dataset, checkpoint, or GPU required."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.data import build_datasets
from stmoe_imputer.data.observed_multiscale import aggregate_sum
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.utils.deterministic import configure, state_hash
from release_utils import load


def small_config():
    cfg = load(ROOT / 'configs/i37_taxibj.json')
    cfg['model']['main'].update(max_t=4, h=8, w=8)
    cfg['data'].update(batch_size=2, pin_memory=False)
    cfg['train'].update(epochs=2, val_epoch=1, amp=False, measure_forward_latency=False)
    cfg['train']['scheduler']['total_epochs'] = 2
    return cfg


def make_data(path, n=4):
    rng = np.random.default_rng(128)
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        f = rng.integers(1, 20, (n, 2, 4, 8, 8)).astype(np.float32)
        np.savez_compressed(path / f'taxibj_{split}.npz', x_f_gt=f, x_m_gt=aggregate_sum(f, 2), x_c_gt=aggregate_sum(f, 4))


class ReleaseContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.temp.name)
        make_data(cls.data)
        cls.cfg = small_config()
        configure(cls.cfg)
        cls.ds = build_datasets(cls.cfg, cls.data / 'taxibj_train.npz', cls.data / 'taxibj_val.npz')[1]

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def batch(self): return {k: v.unsqueeze(0) for k, v in self.ds[0].items()}

    def test_frozen_initialization(self):
        cfg = load(ROOT / 'configs/i37_taxibj.json')
        torch.manual_seed(7)
        model = DualBranchSTImputer.from_config(cfg)
        self.assertEqual(state_hash(model.state_dict()), '2baf7361f1aa2308e5f2a55c07fd657e7d5feea87dd296ca071406aec3179f99')
        self.assertEqual(sum(p.numel() for p in model.parameters()), 347930)

    def test_frozen_source_integrity(self):
        for name, digest in load(ROOT / 'docs/provenance.json')['unchanged_source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)

    def test_forward_projection_no_target_leak_and_native_top2(self):
        torch.manual_seed(7)
        model = DualBranchSTImputer.from_config(self.cfg).eval()
        batch = self.batch(); calls = []
        hooks = [expert.register_forward_hook(lambda mod, args, out: calls.append(args[0].shape[0]))
                 for expert in model.main_branch.pattern_experts.values()]
        with torch.no_grad(): out = model(batch)
        for hook in hooks: hook.remove()
        self.assertEqual(sum(calls), 8)  # one sample, exactly 2 executions per round
        self.assertTrue((out['coe']['triscale_choices'][:, -1] == 0).all())
        pred = out['x_comp']; mask = batch['m_f'].bool()
        self.assertEqual(pred.shape, batch['x_f_gt'].shape)
        self.assertTrue(torch.equal(pred[mask], batch['x_f_obs'][mask]))
        for scale, factor in [('m', 2), ('c', 4)]:
            pooled = torch.from_numpy(aggregate_sum(pred.numpy(), factor))
            visible = batch['m_'+scale].bool()
            torch.testing.assert_close(pooled[visible], batch['x_'+scale+'_obs'][visible], rtol=1e-5, atol=1e-4)
        changed = {k: v.clone() for k, v in batch.items()}
        changed['x_f_gt'].fill_(123456.)
        for s in ('f', 'm', 'c'):
            changed['x_'+s+'_obs'][~changed['m_'+s].bool()] = 654321.
        with torch.no_grad(): another = model(changed)
        torch.testing.assert_close(pred, another['x_comp'], rtol=0, atol=0)

    def test_prepare_data_keeps_splits_and_rejects_overwrite(self):
        from prepare_data import prepare
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'fine'; source.mkdir()
            target = Path(directory) / 'multi'
            f = np.ones((1, 2, 12, 32, 32), dtype=np.float32)
            for i, split in enumerate(('train', 'val', 'test'), 1):
                np.savez_compressed(source / f'taxibj_{split}.npz', x_f_gt=f*i)
            manifest = prepare(source, target)
            for i, split in enumerate(('train', 'val', 'test'), 1):
                with np.load(target / f'taxibj_{split}.npz') as z:
                    np.testing.assert_array_equal(z['x_f_gt'], f*i)
                    self.assertTrue((z['x_m_gt'] == 4*i).all())
                    self.assertTrue((z['x_c_gt'] == 16*i).all())
            self.assertEqual(len(manifest['splits']), 3)
            with self.assertRaises(FileExistsError): prepare(source, target)

    def test_fixed_loss_weights_and_gradient(self):
        torch.manual_seed(7)
        model = DualBranchSTImputer.from_config(self.cfg)
        out = model(self.batch()); loss, logs = compute_main_stage_loss(out, self.batch(), self.cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(model.main_branch.observation_fusion.weight.grad.abs().sum().item(), 0)
        self.assertEqual(self.cfg['loss']['lambda_raw_f'], .5)
        self.assertEqual(self.cfg['loss']['lambda_final_f'], .5)
        self.assertFalse(any('beta' in n for n, _ in model.named_parameters()))


if __name__ == '__main__': unittest.main()
