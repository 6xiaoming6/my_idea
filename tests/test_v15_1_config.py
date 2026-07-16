from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.models.registry import MODEL_REGISTRY, build_model_backbone
from stmoe_imputer.models.v_single import V15_1ScaleGuidedResidualMoE

from _v15_1_utils import backbone_kwargs, compact_v15_1_config, make_batch


ROOT = Path(__file__).resolve().parents[1]


class V15_1ConfigTest(unittest.TestCase):
    def test_dataset_configs_parse_and_select_v15_1(self) -> None:
        for dataset_config, version_config, epochs, val_epoch in (
            ("taxibj.json", "taxibj.json", 160, 5),
            ("bikenyc.json", "bikenyc.json", 140, 2),
            ("chap_beijing.json", "chap.json", 150, 5),
        ):
            with self.subTest(config=version_config):
                base = json.loads(
                    (ROOT / "configs/datasets" / dataset_config).read_text(encoding="utf-8")
                )
                patch = json.loads(
                    (ROOT / "configs/v15.1-single" / version_config).read_text(encoding="utf-8")
                )
                cfg = deep_update(base, patch)
                self.assertEqual(cfg["output_dir"], "outputs/v15.1-single")
                self.assertEqual(cfg["model"]["version"], "v15.1-single")
                self.assertEqual(
                    cfg["model"]["architecture"],
                    "v15_1_scale_guided_residual_moe",
                )
                self.assertEqual(cfg["train"]["epochs"], epochs)
                self.assertEqual(cfg["train"]["val_epoch"], val_epoch)
                self.assertFalse(cfg["train"]["early_stopping"]["enabled"])
                self.assertIsInstance(
                    build_model_backbone(cfg), V15_1ScaleGuidedResidualMoE
                )

    def test_registry_preserves_all_previous_architectures(self) -> None:
        for architecture in (
            "main",
            "v14_safe_c2f_moe",
            "v15_compact_residual_moe",
            "v15_1_scale_guided_residual_moe",
        ):
            self.assertIn(architecture, MODEL_REGISTRY)

    def test_all_ablation_configs_build_and_forward(self) -> None:
        expected_weights = {
            "no_scale_guidance.json": torch.tensor([[1.0 / 3.0] * 3]),
            "fine_only_residual.json": torch.tensor([[1.0, 0.0, 0.0]]),
        }
        for path in sorted((ROOT / "configs/v15.1-single/ablations").glob("*.json")):
            with self.subTest(config=path.name):
                patch = json.loads(path.read_text(encoding="utf-8"))
                cfg = deep_update(compact_v15_1_config(), patch)
                model = build_model_backbone(cfg).eval()
                with torch.no_grad():
                    outputs = model(**backbone_kwargs(make_batch(cfg)))
                self.assertTrue(torch.isfinite(outputs["x_hat_main"]).all())
                if path.name == "fixed_acceptance.json":
                    self.assertTrue(
                        torch.equal(
                            outputs["accept_gate"],
                            torch.ones_like(outputs["accept_gate"]),
                        )
                    )
                    self.assertEqual(cfg["loss"]["lambda_v15_1_accept"], 0.0)
                if path.name == "no_acceptance_loss.json":
                    self.assertEqual(cfg["loss"]["lambda_v15_1_accept"], 0.0)
                if path.name in expected_weights:
                    torch.testing.assert_close(
                        outputs["active_scale_weight"].cpu(),
                        expected_weights[path.name],
                    )
