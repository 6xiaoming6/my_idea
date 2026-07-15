from __future__ import annotations

import torch
import unittest

from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer

from _v15_utils import compact_v15_config, make_batch


class V15GradientFlowTest(unittest.TestCase):
    def test_all_v15_modules_receive_finite_gradient_after_safe_startup(self) -> None:
        cfg = compact_v15_config()
        model = DualBranchSTImputer.from_config(cfg).train()
        optimizer = build_optimizer(model, cfg)
        self.assertIn("v15", {group["name"] for group in optimizer.param_groups})
        batch = make_batch(cfg)
        received: set[str] = set()
        groups = {
            "coarse_adapter": "residual_pyramid.coarse_adapter",
            "mid_fusion": "residual_pyramid.mid_fusion",
            "fine_fusion": "residual_pyramid.fine_fusion",
            "residual_head": "residual_pyramid.residual_head",
            "budget_controller": "budget_controller",
        }
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, logs = compute_main_stage_loss(outputs, batch, cfg, epoch=step + 1)
            self.assertIn("l_v15_delta", logs)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for group, token in groups.items():
                gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if token in name and parameter.grad is not None
                ]
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                if any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients):
                    received.add(group)
            optimizer.step()
        self.assertEqual(received, set(groups), f"Missing non-zero gradients: {set(groups) - received}")
