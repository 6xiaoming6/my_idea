from __future__ import annotations

import torch
import unittest

from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer

from _v14_utils import make_batch
from _v20_utils import compact_v20_config


def has_nonzero_gradient(parameters) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in parameters
    )


class V20ProbeGradientTest(unittest.TestCase):
    def test_probe_loss_only_trains_shared_probe_decoder(self) -> None:
        cfg = compact_v20_config()
        model = DualBranchSTImputer.from_config(cfg).train()
        outputs = model(make_batch(cfg))
        outputs["v20_probe"]["probe_loss"].backward()
        wrapper = model.main_branch
        self.assertTrue(has_nonzero_gradient(wrapper.probe_evaluator.probe_decoder.parameters()))
        self.assertFalse(has_nonzero_gradient(wrapper.main_backbone.routed_expert_pool.parameters()))
        self.assertFalse(has_nonzero_gradient(wrapper.main_backbone.embed_f.parameters()))
        self.assertFalse(has_nonzero_gradient(wrapper.main_backbone.router_f.parameters()))

    def test_main_loss_still_trains_router_and_experts(self) -> None:
        cfg = compact_v20_config()
        cfg["loss"]["lambda_v20_probe"] = 0.0
        model = DualBranchSTImputer.from_config(cfg).train()
        batch = make_batch(cfg)
        outputs = model(batch)
        loss, _ = compute_main_stage_loss(outputs, batch, cfg, epoch=1)
        loss.backward()
        wrapper = model.main_branch
        self.assertTrue(has_nonzero_gradient(wrapper.main_backbone.routed_expert_pool.parameters()))
        self.assertTrue(has_nonzero_gradient(wrapper.main_backbone.router_f.parameters()))
