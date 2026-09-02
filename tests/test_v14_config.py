from __future__ import annotations

import json
from pathlib import Path

import unittest

from stmoe_imputer.config import deep_update
from stmoe_imputer.models.registry import MODEL_REGISTRY, build_model_backbone
from stmoe_imputer.models.v_single import V14SafeC2FMoE


ROOT = Path(__file__).resolve().parents[1]


class V14ConfigTest(unittest.TestCase):
    def test_dataset_v14_configs_parse_and_select_registry(self) -> None:
        for dataset_config, v14_config, epochs, val_epoch in (
            ("taxibj.json", "taxibj.json", 160, 5),
            ("bikenyc.json", "bikenyc.json", 140, 2),
            ("chap_beijing.json", "chap.json", 150, 5),
        ):
            with self.subTest(config=v14_config):
                base = json.loads((ROOT / "configs/datasets" / dataset_config).read_text(encoding="utf-8"))
                patch = json.loads((ROOT / "configs/v14-single" / v14_config).read_text(encoding="utf-8"))
                cfg = deep_update(base, patch)
                self.assertEqual(cfg["output_dir"], "outputs/v14-single")
                self.assertEqual(cfg["model"]["architecture"], "v14_safe_c2f_moe")
                self.assertEqual(
                    cfg["loss"]["load_balance_mode"], "legacy_hard"
                )
                self.assertEqual(cfg["train"]["epochs"], epochs)
                self.assertEqual(cfg["train"]["val_epoch"], val_epoch)
                self.assertFalse(cfg["train"]["early_stopping"]["enabled"])
                self.assertIsInstance(build_model_backbone(cfg), V14SafeC2FMoE)

    def test_registry_keeps_main_as_default_and_rejects_unknown_architecture(self) -> None:
        self.assertIn("main", MODEL_REGISTRY)
        self.assertIn("v14_safe_c2f_moe", MODEL_REGISTRY)
        smoke = json.loads((ROOT / "configs/presets/smoke.json").read_text(encoding="utf-8"))
        self.assertEqual(build_model_backbone(smoke).__class__.__name__, "MultiScaleMoEBackbone")
        smoke["model"]["architecture"] = "does_not_exist"
        with self.assertRaisesRegex(ValueError, "Unknown model architecture"):
            build_model_backbone(smoke)
