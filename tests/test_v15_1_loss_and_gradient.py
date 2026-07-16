from __future__ import annotations

import torch
import unittest

from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer

from _v15_1_utils import compact_v15_1_config, make_batch


class V15_1LossAndGradientTest(unittest.TestCase):
    def test_new_losses_and_diagnostics_are_finite(self) -> None:
        cfg = compact_v15_1_config()
        model = DualBranchSTImputer.from_config(cfg).train()
        batch = make_batch(cfg)
        outputs = model(batch)
        loss, logs = compute_main_stage_loss(outputs, batch, cfg, epoch=1)
        self.assertTrue(torch.isfinite(loss))
        for key in (
            "l_v15_1_base",
            "l_v15_1_candidate",
            "l_v15_1_accept",
            "l_v15_1_safe",
            "v15_1_accept_target_mean",
            "v15_1_accept_positive_rate",
            "v15_1_accept_negative_rate",
            "v15_1_accept_uncertain_rate",
            "v15_1_accept_accuracy",
            "v15_1_candidate_violation_rate",
            "v15_1_final_violation_rate",
        ):
            self.assertIn(key, logs)
            self.assertTrue(torch.isfinite(logs[key]))

    def test_all_new_modules_receive_gradient_after_safe_startup(self) -> None:
        cfg = compact_v15_1_config()
        model = DualBranchSTImputer.from_config(cfg).train()
        optimizer = build_optimizer(model, cfg)
        self.assertIn("v15_1", {group["name"] for group in optimizer.param_groups})
        batch = make_batch(cfg)
        received: set[str] = set()
        groups = {
            "residual_adapter": "main_branch.residual_adapter",
            "acceptance_gate": "main_branch.acceptance_gate",
        }
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, _ = compute_main_stage_loss(outputs, batch, cfg, epoch=step + 1)
            loss.backward()
            for group, token in groups.items():
                gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if token in name and parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
                if any(torch.count_nonzero(value).item() > 0 for value in gradients):
                    received.add(group)
            optimizer.step()
        self.assertEqual(received, set(groups))

