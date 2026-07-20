from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.models import DualBranchSTImputer

from _v17_utils import ROOT, compact_v17_config, make_batch


class V17ConfigTest(unittest.TestCase):
    def test_formal_configs_are_isolated_and_complete(self) -> None:
        expected = {
            "taxibj.json": (160, 5, "fine_mid"),
            "bikenyc.json": (140, 2, "fine_mid_coarse"),
            "chap.json": (150, 5, "fine_mid_coarse"),
        }
        for filename, (epochs, val_epoch, scale_mode) in expected.items():
            with self.subTest(config=filename):
                cfg = json.loads(
                    (ROOT / "configs" / "v17-single" / filename).read_text(encoding="utf-8")
                )
                self.assertEqual(cfg["output_dir"], "outputs/v17-single")
                self.assertEqual(cfg["model"]["version"], "v17-single")
                self.assertEqual(
                    cfg["model"]["architecture"], "v17_hierarchical_scale_moe"
                )
                self.assertEqual(cfg["model"]["main"]["scale_mode"], scale_mode)
                self.assertEqual(cfg["train"]["epochs"], epochs)
                self.assertEqual(cfg["train"]["val_epoch"], val_epoch)
                self.assertEqual(cfg["train"]["lr_main"], 1e-3)

    def test_all_four_core_ablation_configs_change_real_execution(self) -> None:
        expected = {
            "no_scale_adapter.json": (False, "hierarchical", "fine_preserved_parallel", "sample_residual"),
            "decoupled_router.json": (True, "decoupled", "fine_preserved_parallel", "sample_residual"),
            "progressive_fusion.json": (True, "hierarchical", "progressive", "sample_residual"),
            "global_route_gamma.json": (True, "hierarchical", "fine_preserved_parallel", "global_residual"),
        }
        base = compact_v17_config()
        batch = make_batch(base)
        for filename, contract in expected.items():
            with self.subTest(ablation=filename):
                patch = json.loads(
                    (ROOT / "configs" / "v17-single" / "ablations" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                cfg = deep_update(base, patch)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    outputs = model(batch)
                adapter_enabled, router_mode, fusion_mode, branch_mode = contract
                self.assertEqual(model.main_branch.adapter_enabled, adapter_enabled)
                self.assertEqual(outputs["v17_router_mode"], router_mode)
                self.assertEqual(outputs["v17_route_fusion"], fusion_mode)
                self.assertEqual(outputs["branch_mode"], branch_mode)
                self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())

    def test_v17_has_no_teacher_or_output_candidate_modules(self) -> None:
        model = DualBranchSTImputer.from_config(compact_v17_config())
        names = "\n".join(name.lower() for name, _ in model.named_modules())
        for forbidden in ("teacher", "calibrator", "acceptance", "residual_pyramid"):
            self.assertNotIn(forbidden, names)
