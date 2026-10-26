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
from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.utils.deterministic import configure, state_hash
from release_utils import load


def small_config():
    cfg = load(ROOT / 'configs/i37_taxibj.json')
    cfg['model']['main'].update(max_t=4, h=8, w=8)
    cfg['data'].update(batch_size=2, pin_memory=False)
    cfg['train'].update(epochs=2, val_epoch=1, amp=False, measure_forward_latency=False)
    cfg['train']['scheduler']['total_epochs'] = 2
    return cfg


def fixed_loss_config(cfg):
    """Reconstruct the original I37 loss without altering published defaults."""
    cfg = copy.deepcopy(cfg)
    cfg['loss'].pop('learnable_f_mix', None)
    cfg['loss'].update(lambda_raw_f=0.5, lambda_final_f=0.5)
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
        cfg = fixed_loss_config(load(ROOT / 'configs/i37_taxibj.json'))
        torch.manual_seed(7)
        model = DualBranchSTImputer.from_config(cfg)
        self.assertEqual(state_hash(model.state_dict()), '2baf7361f1aa2308e5f2a55c07fd657e7d5feea87dd296ca071406aec3179f99')
        self.assertEqual(sum(p.numel() for p in model.parameters()), 347930)

    def test_frozen_source_integrity(self):
        for name, digest in load(ROOT / 'docs/provenance.json')['unchanged_source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        provenance=load(ROOT/'docs/provenance.json')
        for name, record in provenance.get('performance_modified_source_sha256',{}).items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),record['candidate_sha256'],name)
            self.assertEqual(load(ROOT/'docs/provenance_original_i37.json')['unchanged_source_sha256'][name],record['reference_sha256'])
        for name,digest in provenance.get('performance_added_source_sha256',{}).items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)

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
        cfg = fixed_loss_config(self.cfg)
        torch.manual_seed(7)
        model = DualBranchSTImputer.from_config(cfg)
        out = model(self.batch()); loss, logs = compute_main_stage_loss(out, self.batch(), cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(model.main_branch.observation_fusion.weight.grad.abs().sum().item(), 0)
        self.assertEqual(cfg['loss']['lambda_raw_f'], .5)
        self.assertEqual(cfg['loss']['lambda_final_f'], .5)
        self.assertNotIn('main_branch.x_loss_mix_logit', dict(model.named_parameters()))

    def test_learnable_loss_weights_and_gradient(self):
        # Both public configurations use the same trainable, complementary mix.
        for name in ('i37_taxibj.json', 'i37_performance.json'):
            loss_cfg = load(ROOT / 'configs' / name)['loss']
            self.assertEqual(loss_cfg['learnable_f_mix'],
                             {'enabled': True, 'initial_raw_weight': .5})
            self.assertNotIn('lambda_raw_f', loss_cfg)
            self.assertNotIn('lambda_final_f', loss_cfg)
        torch.manual_seed(7)
        model = DualBranchSTImputer.from_config(self.cfg)
        optimizer = build_optimizer(model, self.cfg)
        scalar = model.main_branch.x_loss_mix_logit
        group = next(g for g in optimizer.param_groups if any(p is scalar for p in g['params']))
        self.assertEqual(group['lr'], self.cfg['train']['lr_main'])
        self.assertEqual(group['weight_decay'], 0.)
        out = model(self.batch())
        loss, logs = compute_main_stage_loss(out, self.batch(), self.cfg)
        beta = out['f_loss_raw_weight']
        self.assertEqual(beta.item(), .5)
        self.assertEqual((logs['f_loss_raw_weight'] + logs['f_loss_final_weight']).item(), 1.)
        expected_grad = beta.detach() * (1-beta.detach()) * (logs['l_raw_f'] - logs['l_final'])
        loss.backward()
        torch.testing.assert_close(scalar.grad, expected_grad, rtol=1e-6, atol=1e-7)
        self.assertNotEqual(scalar.grad.item(), 0.)
        self.assertGreater(model.main_branch.observation_fusion.weight.grad.abs().sum().item(), 0)
        before = scalar.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, scalar.detach()))
        with torch.no_grad():
            _, updated = compute_main_stage_loss(model(self.batch()), self.batch(), self.cfg)
        self.assertGreater(updated['f_loss_raw_weight'].item(), 0.)
        self.assertLess(updated['f_loss_raw_weight'].item(), 1.)
        self.assertEqual((updated['f_loss_raw_weight'] + updated['f_loss_final_weight']).item(), 1.)

    def test_learnable_mix_preserves_initial_predictions_and_fixed_checkpoints(self):
        fixed = fixed_loss_config(self.cfg)
        torch.manual_seed(7)
        old = DualBranchSTImputer.from_config(fixed)
        torch.manual_seed(7)
        new = DualBranchSTImputer.from_config(self.cfg)
        shared = {k: v for k, v in new.state_dict().items()
                  if k != 'main_branch.x_loss_mix_logit'}
        self.assertEqual(state_hash(old.state_dict()), state_hash(shared))
        self.assertEqual(sum(p.numel() for p in new.parameters()),
                         sum(p.numel() for p in old.parameters()) + 1)
        batch = self.batch()
        a, b = old(batch), new(batch)
        for key in ('x_hat_main', 'x_hat_raw_consistency', 'x_comp'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        fixed_loss, _ = compute_main_stage_loss(a, batch, fixed)
        learned_loss, _ = compute_main_stage_loss(b, batch, self.cfg)
        torch.testing.assert_close(fixed_loss, learned_loss, rtol=0, atol=0)
        # Old checkpoints continue to construct from their saved fixed config.
        restored = DualBranchSTImputer.from_config(fixed)
        restored.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(state_hash(restored.state_dict()), state_hash(old.state_dict()))


if __name__ == '__main__': unittest.main()
