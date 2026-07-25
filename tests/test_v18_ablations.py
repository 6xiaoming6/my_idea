from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.v_single import AbsoluteCoarseToFinePyramid

from _v18_utils import backbone_kwargs, compact_v18_config, make_batch


ROOT = Path(__file__).resolve().parents[1]


class V18AblationTest(unittest.TestCase):
    def _config(self, name: str) -> dict:
        patch = json.loads(
            (
                ROOT
                / "configs"
                / "v18-single"
                / "ablations"
                / f"{name}.json"
            ).read_text(encoding="utf-8")
        )
        return deep_update(compact_v18_config(), patch)

    def test_all_declared_ablations_complete_forward_and_backward(self) -> None:
        for name in (
            "absolute_c2f",
            "unbounded_residual",
            "no_observed_utility",
            "no_reliability_filtering",
            "fine_only_residual",
            "fixed_budget",
            "no_sample_regret",
        ):
            with self.subTest(ablation=name):
                cfg = self._config(name)
                model = DualBranchSTImputer.from_config(cfg).train()
                batch = make_batch(cfg)
                outputs = model(batch)
                loss, logs = compute_main_stage_loss(
                    outputs, batch, cfg, epoch=1
                )
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(
                    all(torch.isfinite(value) for value in logs.values())
                )
                build_optimizer(model, cfg).zero_grad(set_to_none=True)
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(
                    all(torch.isfinite(gradient).all() for gradient in gradients)
                )

    def test_ablation_switches_change_only_the_requested_v18_path(self) -> None:
        no_utility = DualBranchSTImputer.from_config(
            self._config("no_observed_utility")
        ).main_branch
        self.assertIsNone(no_utility.utility_evaluator)

        no_reliability = DualBranchSTImputer.from_config(
            self._config("no_reliability_filtering")
        ).main_branch
        self.assertFalse(
            no_reliability.residual_pyramid.use_reliability_filtered_propagation
        )

        fine_only = DualBranchSTImputer.from_config(
            self._config("fine_only_residual")
        ).main_branch
        self.assertIsNone(fine_only.residual_pyramid.coarse_head)
        self.assertIsNone(fine_only.residual_pyramid.mid_head)
        fine_only_cfg = self._config("fine_only_residual")
        self.assertEqual(fine_only_cfg["loss"]["lambda_v18_mid"], 0.0)
        self.assertEqual(fine_only_cfg["loss"]["lambda_v18_coarse"], 0.0)

        fixed = DualBranchSTImputer.from_config(
            self._config("fixed_budget")
        ).main_branch
        self.assertIsNone(fixed.controller)
        self.assertAlmostEqual(fixed.fixed_rho, 0.05)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in fixed.condition_encoder.parameters()
            )
        )

        no_regret_cfg = self._config("no_sample_regret")
        self.assertEqual(
            no_regret_cfg["loss"]["lambda_v18_sample_regret"], 0.0
        )

        absolute = DualBranchSTImputer.from_config(
            self._config("absolute_c2f")
        ).main_branch
        self.assertIsInstance(
            absolute.residual_pyramid, AbsoluteCoarseToFinePyramid
        )
        absolute_cfg = self._config("absolute_c2f")
        self.assertFalse(absolute_cfg["model"]["v18"]["bounded_directions"])
        absolute_outputs = absolute(
            **backbone_kwargs(make_batch(absolute_cfg))
        )
        self.assertEqual(
            absolute_outputs["branch_mode"],
            "v18_absolute_c2f_ablation",
        )

        unbounded = DualBranchSTImputer.from_config(
            self._config("unbounded_residual")
        ).main_branch
        self.assertFalse(
            unbounded.residual_pyramid.fine_head.bounded_output
        )

    def test_unbounded_ablation_can_exceed_the_full_model_hard_bound(self) -> None:
        cfg = self._config("unbounded_residual")
        model = DualBranchSTImputer.from_config(cfg).main_branch.eval()
        with torch.no_grad():
            model.residual_pyramid.fine_head.out_proj.bias.fill_(20.0)
        batch = make_batch(cfg)
        with torch.no_grad():
            outputs = model(**backbone_kwargs(batch))
        residual = (
            outputs["x_hat_main"] - outputs["x_hat_base"]
        ).abs()
        bound = (
            model.rho_fine_max
            * outputs["features"]["observed_scale_f"]
        )
        self.assertTrue(torch.any(residual > bound + 1e-6))


if __name__ == "__main__":
    unittest.main()
