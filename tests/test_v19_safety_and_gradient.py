from __future__ import annotations

import unittest

import torch

from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.v_single import ChannelResidualGain

from _v19_utils import compact_v19_config, make_batch


class V19SafetyAndGradientTest(unittest.TestCase):
    def test_gain_is_finite_and_bounded_for_extreme_inputs(self) -> None:
        controller = ChannelResidualGain(
            hidden_dim=8,
            dropout=0.0,
            gain_range=0.5,
        ).eval()
        base = torch.full((2, 2, 3, 4, 4), 1e20)
        v14 = torch.full_like(base, -1e20)
        observed_values = torch.full_like(base, 1e20)
        mask = torch.stack(
            (
                torch.ones(1, 3, 4, 4),
                torch.zeros(1, 3, 4, 4),
            )
        )
        with torch.no_grad():
            gain, diagnostics = controller(
                x_base=base,
                x_v14=v14,
                x_obs=observed_values,
                mask=mask,
            )
        self.assertTrue(torch.isfinite(gain).all())
        self.assertGreaterEqual(float(gain.min()), 0.5)
        self.assertLessEqual(float(gain.max()), 1.5)
        for value in diagnostics.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_hidden_ground_truth_does_not_change_prediction_or_gain(self) -> None:
        cfg = compact_v19_config()
        batch = make_batch(cfg)
        changed = {key: value.clone() for key, value in batch.items()}
        hidden = (1.0 - changed["m_f"]).expand_as(changed["x_f_gt"])
        changed["x_f_gt"] = changed["x_f_gt"] + hidden * 10000.0
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            first = model(batch)
            second = model(changed)
        torch.testing.assert_close(
            first["x_hat_final"],
            second["x_hat_final"],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            first["diagnostics"]["v19"]["gain"],
            second["diagnostics"]["v19"]["gain"],
            atol=0.0,
            rtol=0.0,
        )

    def test_single_stage_updates_v14_and_gain_controller(self) -> None:
        cfg = compact_v19_config()
        model = DualBranchSTImputer.from_config(cfg).train()
        optimizer = build_optimizer(model, cfg)
        batch = make_batch(cfg)
        received = {"v14": False, "v19": False}
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, logs = compute_main_stage_loss(
                outputs,
                batch,
                cfg,
                epoch=step + 1,
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertIn("l_v19_anchor_regret", logs)
            self.assertIn("l_v19_gain", logs)
            loss.backward()
            for name, parameter in model.named_parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                self.assertTrue(
                    torch.isfinite(gradient).all(),
                    f"non-finite gradient: {name}",
                )
                if torch.count_nonzero(gradient).item() == 0:
                    continue
                if "main_branch.v14_model" in name:
                    received["v14"] = True
                if "main_branch.gain_controller" in name:
                    received["v19"] = True
            optimizer.step()
        self.assertEqual(received, {"v14": True, "v19": True})

    def test_optimizer_has_separate_gain_groups_and_small_parameter_budget(
        self,
    ) -> None:
        cfg = compact_v19_config()
        model = DualBranchSTImputer.from_config(cfg)
        optimizer = build_optimizer(model, cfg)
        groups = {
            group["name"]: group
            for group in optimizer.param_groups
        }
        self.assertIn("v19_gain", groups)
        self.assertIn("v19_gain_no_decay", groups)
        self.assertEqual(groups["v19_gain"]["lr"], 5e-3)
        self.assertEqual(groups["v19_gain_no_decay"]["lr"], 5e-3)
        gain_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "main_branch.gain_controller" in name
        )
        self.assertLess(gain_parameters, 20_000)


if __name__ == "__main__":
    unittest.main()
