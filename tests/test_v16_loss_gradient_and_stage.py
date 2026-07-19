from __future__ import annotations

import torch
import unittest

from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer

from _v16_utils import compact_v16_config, make_batch


class V16LossGradientAndStageTest(unittest.TestCase):
    def test_joint_losses_diagnostics_and_gradients_are_finite(self) -> None:
        cfg = compact_v16_config()
        model = DualBranchSTImputer.from_config(cfg).train()
        model.main_branch.configure_training_stage(2)
        optimizer = build_optimizer(model, cfg)
        groups = {group["name"] for group in optimizer.param_groups}
        self.assertIn("v16_residual", groups)
        self.assertIn("v16_calibrator", groups)
        batch = make_batch(cfg)
        outputs = model(batch)
        teacher_outputs = {"x_hat_main": outputs["x_hat_base"].detach() + 0.1}
        loss, logs = compute_main_stage_loss(
            outputs, batch, cfg, epoch=2, teacher_outputs=teacher_outputs
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for key in (
            "l_v16_anchor", "l_v16_candidate", "l_v16_calibration", "l_v16_safe",
            "v16_alpha_absolute_error", "v16_alpha_rmse", "v16_alpha_pearson",
            "v16_alpha_spearman", "v16_oracle_hidden_mae", "v16_calibration_regret",
        ):
            self.assertIn(key, logs)
            self.assertTrue(torch.isfinite(logs[key]))
        self.assertGreater(float(logs["l_v16_base_teacher"]), 0.0)
        for token in ("residual_proposer", "calibrator"):
            gradients = [p.grad for name, p in model.named_parameters() if token in name and p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_warmup_freezes_backbone_and_calibrator_then_unfreezes(self) -> None:
        cfg = compact_v16_config()
        branch = DualBranchSTImputer.from_config(cfg).main_branch
        self.assertEqual(branch.configure_training_stage(1), "warmup")
        self.assertFalse(any(p.requires_grad for p in branch.student_backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in branch.residual_proposer.parameters()))
        self.assertFalse(any(p.requires_grad for p in branch.calibrator.parameters()))
        self.assertEqual(branch.configure_training_stage(2), "joint")
        self.assertTrue(all(p.requires_grad for p in branch.student_backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in branch.calibrator.parameters()))


if __name__ == "__main__":
    unittest.main()
