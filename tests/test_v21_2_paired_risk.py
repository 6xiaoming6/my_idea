import copy
import io
import unittest

import torch

from _v14_utils import compact_v14_config
from test_v21_1_distortion_calibration import dual_batch
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.losses import compute_main_stage_loss


def config(mode="risk"):
    cfg = compact_v14_config()
    cfg["model"]["architecture"] = "v21_paired_risk_moe"
    cfg["model"]["main"]["evidence_dim"] = 0
    cfg["model"]["v21_2"] = {"mode": mode, "min_holdout_count": 1}
    cfg["data"]["scales"]["pyramid_mode"] = "dual_observation_moment"
    return cfg


class PairedRiskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initial_candidate_exactly_matches_v14(self):
        cfg = config()
        batch = dual_batch(cfg)
        torch.manual_seed(123)
        base = DualBranchSTImputer.from_config(compact_v14_config()).eval()
        torch.manual_seed(123)
        model = DualBranchSTImputer.from_config(cfg).eval()
        for name, value in base.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            legacy_batch = {k: v for k, v in batch.items() if "measure" not in k and not k.startswith("e_")}
            a, b = base(legacy_batch), model(batch)
        torch.testing.assert_close(a["x_hat_final"], b["x_hat_final"], rtol=0, atol=0)
        self.assertEqual(float(b["features"]["v21_2"]["alpha"].sum()), 0)

    def test_proxy_trains_risk_head_without_changing_global_rng(self):
        cfg = config()
        model = DualBranchSTImputer.from_config(cfg).train()
        batch = dual_batch(cfg)
        state = torch.random.get_rng_state()
        loss, logs = model.main_branch.proxy_loss(batch["x_f_obs"], batch["m_f"])
        torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)
        self.assertGreater(float(logs["proxy_valid_fraction"]), 0)
        loss.backward()
        self.assertGreater(float(model.main_branch.scale_risk.net[-1].bias.grad.abs().sum()), 0)

    def test_hidden_targets_never_enter_prediction_or_proxy(self):
        cfg = config()
        first = dual_batch(cfg)
        changed = copy.deepcopy(first)
        changed["x_f_gt"] += (1 - changed["m_f"]) * 1e5
        a = DualBranchSTImputer.from_config(cfg).train()
        b = copy.deepcopy(a)
        # Disable any residual dropout inherited from compact configs.
        torch.manual_seed(20)
        out1 = a(first)
        torch.manual_seed(20)
        out2 = b(changed)
        torch.testing.assert_close(out1["x_hat_final"], out2["x_hat_final"], rtol=0, atol=0)
        torch.testing.assert_close(out1["v21_2_proxy_loss"], out2["v21_2_proxy_loss"], rtol=0, atol=0)

    def test_holdout_values_do_not_enter_proxy_features(self):
        cfg = config()
        b = dual_batch(cfg)
        model = DualBranchSTImputer.from_config(cfg).main_branch
        p, mean, _, _ = model.proxy_batch(b["x_f_obs"], b["m_f"])
        held = b["m_f"] - p["m_f"]
        q, changed_mean, _, _ = model.proxy_batch(b["x_f_obs"] + held * 100, b["m_f"])
        for key in ("x_c_obs", "x_c_measure", "e_c"):
            torch.testing.assert_close(p[key], q[key], rtol=0, atol=0)
        self.assertGreater(float((mean - changed_mean).abs().sum()), 0)

    def test_reject_and_accept_bounds(self):
        cfg = config()
        model = DualBranchSTImputer.from_config(cfg).eval()
        b = dual_batch(cfg)
        with torch.no_grad():
            model.main_branch.scale_risk.net[-1].bias.fill_(-1)
            out = model(b)
        alpha = out["features"]["v21_2"]["alpha"]
        self.assertTrue(bool((alpha >= 0).all() and (alpha <= .25).all()))
        self.assertGreater(float(alpha.sum()), 0)

    def test_empty_support_has_finite_zero_auxiliary_loss(self):
        model = DualBranchSTImputer.from_config(config()).main_branch
        loss, logs = model.proxy_loss(torch.zeros(1, 2, 3, 16, 16), torch.zeros(1, 1, 3, 16, 16))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0)
        self.assertEqual(float(logs["proxy_valid_fraction"]), 0)
        loss.backward()

    def test_train_val_checkpoint_test_cycle(self):
        cfg = config()
        b = dual_batch(cfg)
        model = DualBranchSTImputer.from_config(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        out = model(b)
        loss, logs = compute_main_stage_loss(out, b, cfg, epoch=1)
        self.assertIn("l_v21_2_proxy", logs)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = model(b)
        self.assertNotIn("v21_2_proxy_loss", val)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = DualBranchSTImputer.from_config(cfg).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            test = restored(b)
        torch.testing.assert_close(val["x_hat_final"], test["x_hat_final"], rtol=0, atol=0)
        self.assertEqual(int(restored.main_branch.proxy_step), 1)

    def test_fixed_holdout_is_time_shared(self):
        cfg = config()
        cfg["data"]["mask"] = {"pattern": "fixed"}
        model = DualBranchSTImputer.from_config(cfg).main_branch
        mask = torch.ones(1, 1, 3, 16, 16)
        p, _, _, _ = model.proxy_batch(torch.ones(1, 2, 3, 16, 16), mask)
        torch.testing.assert_close(p["m_f"][:, :, 0], p["m_f"][:, :, 2], rtol=0, atol=0)

    def test_controls_have_no_proxy_loss(self):
        for mode in ("constant", "anchor"):
            cfg = config(mode)
            out = DualBranchSTImputer.from_config(cfg)(dual_batch(cfg))
            self.assertNotIn("v21_2_proxy_loss", out)


if __name__ == "__main__":
    unittest.main()
