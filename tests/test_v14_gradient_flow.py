from __future__ import annotations

import torch
import unittest

from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer

from _v14_utils import compact_v14_config, make_batch


class V14GradientFlowTest(unittest.TestCase):
    def test_all_v14_modules_receive_finite_gradient_after_safe_startup(self) -> None:
        cfg = compact_v14_config()
        model = DualBranchSTImputer.from_config(cfg).train()
        optimizer = build_optimizer(model, cfg)
        batch = make_batch(cfg)
        received: set[str] = set()
        groups = {
            "coarse_head": "refiner.coarse_head",
            "mid_residual_head": "refiner.mid_residual_head",
            "fine_residual_head": "refiner.fine_residual_head",
            "correction_adapter": "refiner.correction_adapter",
            "condition_encoder": "condition_encoder",
            "controller": "controller",
        }
        for step in range(3):
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
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                if any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients):
                    received.add(group)
            optimizer.step()
        self.assertEqual(received, set(groups), f"Missing non-zero gradients: {set(groups) - received}")
