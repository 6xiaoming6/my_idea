import copy
import io
import unittest

import torch
import torch.nn.functional as F

from _v14_utils import compact_v14_config, make_batch
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.v_single.v22_coarsening_moe import LocalCoarsener
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.engine import build_optimizer, train_one_epoch, evaluate


def config(mode="moe", channels=2, h=8, w=8):
    cfg = compact_v14_config(channels=channels, height=h, width=w)
    cfg["model"]["architecture"] = "v22_coarsening_moe"
    cfg["model"]["v22"] = {"mode": mode, "dim": 8, "num_groups": 2, "dropout": 0., "strides": [2, 4]}
    cfg["model"]["main"].update(use_router=False, use_routed_branch=False)
    return cfg


class V22Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_pool_matches_masked_mean(self):
        torch.manual_seed(1)
        f = torch.randn(1, 4, 2, 6, 8)
        m = (torch.rand(1, 1, 2, 6, 8) > .4).float()
        coarse = LocalCoarsener(4, 2, 0)
        mean, stats, assignment, shape, balance, _ = coarse(f, f, m)
        expected = F.avg_pool2d((f * m).permute(0, 2, 1, 3, 4).reshape(2, 4, 6, 8), 2)
        count = F.avg_pool2d(m.permute(0, 2, 1, 3, 4).reshape(2, 1, 6, 8), 2)
        expected = (expected / count.clamp_min(1e-6)).flatten(2).transpose(1, 2)
        torch.testing.assert_close(mean, expected)
        self.assertEqual(float(balance), 0.)
        torch.testing.assert_close(assignment[0].sum(-1), torch.ones(2, 48))

    def test_sparse_scatter_and_prolong_equal_dense_reference(self):
        torch.manual_seed(4)
        f = torch.randn(2, 4, 2, 5, 7, requires_grad=True)
        m = (torch.rand(2, 1, 2, 5, 7) > .5).float()
        module = LocalCoarsener(4, 2, 1)
        mean, _, (weights, indices), (hc, wc), _, _ = module(f, f, m)
        dense = weights.new_zeros(4, 35, hc * wc)
        dense.scatter_add_(2, indices[None].expand(4, -1, -1), weights)
        values = f.permute(0, 2, 3, 4, 1).reshape(4, 35, 4)
        mask = m.permute(0, 2, 3, 4, 1).reshape(4, 35, 1)
        expected = dense.transpose(1, 2) @ (values * mask)
        expected /= (dense.transpose(1, 2) @ mask).clamp_min(1e-6)
        torch.testing.assert_close(mean, expected)
        torch.testing.assert_close(module.prolong(mean, (weights, indices)), dense @ mean)
        mean.square().mean().backward()
        self.assertTrue(torch.isfinite(f.grad).all())
        self.assertGreater(float(module.query.weight.grad.abs().sum()), 0.)

    def test_all_modes_train_val_reload_and_test(self):
        for mode in ("fixed", "fixed_stats", "single_local", "single_wide", "uniform", "moe", "moe_no_stats"):
            with self.subTest(mode=mode):
                cfg = config(mode)
                batch = make_batch(cfg)
                model = DualBranchSTImputer.from_config(cfg)
                optimizer = build_optimizer(model, cfg)
                train = train_one_epoch(model, [batch], optimizer, torch.device("cpu"), cfg, 1)
                val = evaluate(model, [batch], torch.device("cpu"), cfg, epoch=1)
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in [train["loss"], val["mae"], val["rmse"]]))
                self.assertIn("v22_s2_route_entropy", val)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = DualBranchSTImputer.from_config(cfg)
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                test = evaluate(restored, [batch], torch.device("cpu"), cfg, epoch=1)
                self.assertEqual(val["mae"], test["mae"])

    def test_no_hidden_or_external_coarse_values_enter_prediction(self):
        cfg = config()
        batch = make_batch(cfg)
        model = DualBranchSTImputer.from_config(cfg).eval()
        changed = copy.deepcopy(batch)
        changed["x_f_gt"] += (1 - batch["m_f"]) * 1e7
        changed["x_f_obs"] = torch.where(batch["m_f"].bool(), batch["x_f_obs"], torch.full_like(batch["x_f_obs"], float("nan")))
        changed["x_m_obs"].fill_(1e8)
        changed["x_c_obs"].fill_(1e8)
        with torch.no_grad():
            a, b = model(batch), model(changed)
        torch.testing.assert_close(a["x_hat_final"], b["x_hat_final"], atol=0, rtol=0)

    def test_empty_full_and_constant_inputs_are_finite(self):
        cfg = config()
        for mask_value in (0., 1.):
            batch = make_batch(cfg)
            batch["m_f"].fill_(mask_value)
            batch["x_f_obs"].fill_(3.)
            batch["x_f_gt"].fill_(3.)
            model = DualBranchSTImputer.from_config(cfg)
            outputs = model(batch)
            loss, _ = compute_main_stage_loss(outputs, batch, cfg)
            self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if mask_value == 0:
                self.assertEqual(float(outputs["diagnostics"]["v22"]["s2_e1_empty_fraction"]), 1.)

    def test_router_and_assignments_receive_task_gradients(self):
        cfg = config()
        cfg["loss"].update(lambda_v22_mass=0., lambda_v22_balance=0.)
        model = DualBranchSTImputer.from_config(cfg)
        batch = make_batch(cfg)
        loss, _ = compute_main_stage_loss(model(batch), batch, cfg)
        loss.backward()
        layer = model.main_branch.scales[0]
        self.assertGreater(float(layer.router[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(layer.coarseners[1].query.weight.grad.abs().sum()), 0)

    def test_shared_weights_seed_matched_across_controls(self):
        torch.manual_seed(10)
        a = DualBranchSTImputer.from_config(config("fixed"))
        torch.manual_seed(10)
        b = DualBranchSTImputer.from_config(config("moe"))
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)

    def test_dataset_shapes(self):
        for c, h, w in ((2, 32, 32), (2, 24, 12), (1, 32, 32)):
            cfg = config(channels=c, h=h, w=w)
            batch = make_batch(cfg)
            with torch.no_grad():
                out = DualBranchSTImputer.from_config(cfg).eval()(batch)
            self.assertEqual(out["x_hat_final"].shape, batch["x_f_gt"].shape)

    def test_bfloat16_autocast(self):
        cfg = config()
        model = DualBranchSTImputer.from_config(cfg)
        batch = make_batch(cfg)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = compute_main_stage_loss(model(batch), batch, cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
