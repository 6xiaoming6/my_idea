from __future__ import annotations

import json
from pathlib import Path
import unittest

from stmoe_imputer.config import deep_update
from stmoe_imputer.models.registry import MODEL_REGISTRY, build_model_backbone
from stmoe_imputer.models.v_single import V20ProbeValidatedC2FMoE


ROOT = Path(__file__).resolve().parents[1]


class V20ConfigTest(unittest.TestCase):
    def test_three_dataset_configs_keep_v14_and_select_v20(self) -> None:
        for base_name, patch_name, epochs, val_epoch in (
            ("taxibj.json", "taxibj.json", 160, 5),
            ("bikenyc.json", "bikenyc.json", 140, 2),
            ("chap_beijing.json", "chap.json", 150, 5),
        ):
            with self.subTest(dataset=patch_name):
                base = json.loads(
                    (ROOT / "configs/datasets" / base_name).read_text(encoding="utf-8")
                )
                patch = json.loads(
                    (ROOT / "configs/v20-single" / patch_name).read_text(encoding="utf-8")
                )
                cfg = deep_update(base, patch)
                self.assertEqual(cfg["output_dir"], "outputs/v20-single")
                self.assertEqual(cfg["model"]["architecture"], "v20_probe_validated_c2f_moe")
                self.assertTrue(cfg["model"]["v14"]["enabled"])
                self.assertEqual(cfg["model"]["v20"]["probe_mode"], "geometry_matched")
                self.assertEqual(cfg["model"]["v20"]["routing_fusion"], "neutral_hybrid")
                self.assertEqual(cfg["model"]["v20"]["confidence_mode"], "entropy_threshold")
                self.assertEqual(cfg["model"]["v20"]["routing_evidence_scales"], ["fine"])
                self.assertFalse(cfg["model"]["v20"]["apply_evidence_during_training"])
                self.assertEqual(cfg["model"]["v20"]["probe_eta_max"], 0.25)
                self.assertEqual(
                    cfg["model"]["v20"]["validation_calibration"]["eta_candidates"],
                    [0.0, 0.05, 0.1, 0.15, 0.25],
                )
                self.assertEqual(cfg["train"]["epochs"], epochs)
                self.assertEqual(cfg["train"]["val_epoch"], val_epoch)
                self.assertEqual(
                    cfg["train"]["grad_clip_isolate_groups"],
                    ["v20", "v20_no_decay"],
                )
                self.assertEqual(cfg["loss"]["lambda_v20_probe"], 0.05)

    def test_registry_exports_v20_without_replacing_v14(self) -> None:
        self.assertIn("v14_safe_c2f_moe", MODEL_REGISTRY)
        self.assertIn("v20_probe_validated_c2f_moe", MODEL_REGISTRY)
        from _v20_utils import compact_v20_config

        self.assertIsInstance(
            build_model_backbone(compact_v20_config()), V20ProbeValidatedC2FMoE
        )
