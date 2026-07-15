from __future__ import annotations

import json
from pathlib import Path
import torch
import unittest

from stmoe_imputer.config import deep_update
from stmoe_imputer.models.registry import MODEL_REGISTRY, build_model_backbone
from stmoe_imputer.models.v_single import V15CompactResidualMoE

from _v15_utils import backbone_kwargs, compact_v15_config, make_batch


ROOT = Path(__file__).resolve().parents[1]


class V15ConfigTest(unittest.TestCase):
    def test_dataset_configs_parse_and_select_v15(self) -> None:
        for dataset_config, v15_config, epochs, val_epoch in (
            ("taxibj.json", "taxibj.json", 160, 5),
            ("bikenyc.json", "bikenyc.json", 140, 2),
            ("chap_beijing.json", "chap.json", 150, 5),
        ):
            with self.subTest(config=v15_config):
                base = json.loads(
                    (ROOT / "configs/datasets" / dataset_config).read_text(encoding="utf-8")
                )
                patch = json.loads(
                    (ROOT / "configs/v15-single" / v15_config).read_text(encoding="utf-8")
                )
                cfg = deep_update(base, patch)
                self.assertEqual(cfg["output_dir"], "outputs/v15-single")
                self.assertEqual(cfg["model"]["architecture"], "v15_compact_residual_moe")
                self.assertEqual(cfg["train"]["epochs"], epochs)
                self.assertEqual(cfg["train"]["val_epoch"], val_epoch)
                self.assertFalse(cfg["train"]["early_stopping"]["enabled"])
                self.assertIsInstance(build_model_backbone(cfg), V15CompactResidualMoE)

    def test_registry_keeps_previous_architectures(self) -> None:
        self.assertIn("main", MODEL_REGISTRY)
        self.assertIn("v14_safe_c2f_moe", MODEL_REGISTRY)
        self.assertIn("v15_compact_residual_moe", MODEL_REGISTRY)

    def test_core_ablation_switches_are_executable(self) -> None:
        no_pyramid_cfg = compact_v15_config()
        no_pyramid_cfg["model"]["v15"]["use_pyramid"] = False
        no_pyramid = V15CompactResidualMoE.from_config(no_pyramid_cfg).eval()
        with torch.no_grad():
            outputs = no_pyramid(**backbone_kwargs(make_batch(no_pyramid_cfg)))
        self.assertEqual(outputs["x_hat_main"].shape, outputs["x_hat_base"].shape)

        fixed_cfg = compact_v15_config()
        fixed_cfg["model"]["v15"].update({
            "dynamic_budget": False,
            "fixed_beta": 0.05,
        })
        fixed = V15CompactResidualMoE.from_config(fixed_cfg).eval()
        with torch.no_grad():
            outputs = fixed(**backbone_kwargs(make_batch(fixed_cfg)))
        torch.testing.assert_close(
            outputs["residual_budget"],
            torch.full_like(outputs["residual_budget"], 0.05),
        )
