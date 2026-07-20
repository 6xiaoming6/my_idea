from __future__ import annotations

import torch
import unittest

from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer

from _v17_utils import compact_v17_config, make_batch


class V17GradientFlowTest(unittest.TestCase):
    def test_new_modules_receive_finite_nonzero_gradients(self) -> None:
        cfg = compact_v17_config()
        model = DualBranchSTImputer.from_config(cfg).train()
        optimizer = build_optimizer(model, cfg)
        batch = make_batch(cfg)
        groups = {
            "adapters": "main_branch.adapter_",
            "hierarchical_router": "main_branch.hierarchical_router",
            "parallel_fusion": "main_branch.route_fusion",
            "shared_fusion": "main_branch.cross_scale_shared_expert",
            "branch_fusion": "main_branch.branch_fusion",
        }
        received: set[str] = set()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, logs = compute_main_stage_loss(outputs, batch, cfg, epoch=step + 1)
            self.assertIn("l_scale_entropy_floor", logs)
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
        self.assertEqual(
            received, set(groups), f"Missing non-zero gradients: {set(groups) - received}"
        )
