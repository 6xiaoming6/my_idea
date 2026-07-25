from __future__ import annotations

import unittest

import torch

from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer

from _v18_utils import compact_v18_config, make_batch


class V18GradientFlowTest(unittest.TestCase):
    def test_main_direction_controller_and_condition_receive_finite_gradients(self) -> None:
        cfg = compact_v18_config()
        model = DualBranchSTImputer.from_config(cfg).train()
        optimizer = build_optimizer(model, cfg)
        batch = make_batch(cfg)
        groups = {
            "main": "main_branch.main_backbone",
            "coarse_direction": "residual_pyramid.coarse_head",
            "mid_direction": "residual_pyramid.mid_head",
            "fine_direction": "residual_pyramid.fine_head",
            "condition": "condition_encoder",
            "controller": "controller",
        }
        received: set[str] = set()
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, _ = compute_main_stage_loss(
                outputs, batch, cfg, epoch=step + 1
            )
            loss.backward()
            for group, token in groups.items():
                gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if token in name and parameter.grad is not None
                ]
                self.assertTrue(gradients, f"No gradients found for {group}")
                self.assertTrue(
                    all(torch.isfinite(gradient).all() for gradient in gradients)
                )
                if any(
                    torch.count_nonzero(gradient).item() > 0
                    for gradient in gradients
                ):
                    received.add(group)
            optimizer.step()
        self.assertEqual(
            received,
            set(groups),
            f"Missing non-zero gradients: {set(groups) - received}",
        )
