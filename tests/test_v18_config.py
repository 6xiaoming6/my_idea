from __future__ import annotations

import json
import unittest
from pathlib import Path

from stmoe_imputer.config import deep_update
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.registry import MODEL_REGISTRY, build_model_backbone
from stmoe_imputer.models.v_single import V18BaseAnchoredResidualMoE

from _v18_utils import compact_v18_config


ROOT = Path(__file__).resolve().parents[1]


class V18ConfigTest(unittest.TestCase):
    def test_dataset_configs_select_v18_and_keep_v14_registered(self) -> None:
        for dataset_config, v18_config, epochs, val_epoch in (
            ("taxibj.json", "taxibj.json", 160, 5),
            ("bikenyc.json", "bikenyc.json", 140, 2),
            ("chap_beijing.json", "chap.json", 150, 5),
        ):
            with self.subTest(config=v18_config):
                base = json.loads(
                    (ROOT / "configs/datasets" / dataset_config).read_text(
                        encoding="utf-8"
                    )
                )
                patch = json.loads(
                    (ROOT / "configs/v18-single" / v18_config).read_text(
                        encoding="utf-8"
                    )
                )
                cfg = deep_update(base, patch)
                self.assertEqual(cfg["output_dir"], "outputs/v18-single")
                self.assertEqual(
                    cfg["model"]["architecture"],
                    "v18_base_anchored_residual_moe",
                )
                self.assertEqual(cfg["train"]["epochs"], epochs)
                self.assertEqual(cfg["train"]["val_epoch"], val_epoch)
                self.assertIsInstance(
                    build_model_backbone(cfg), V18BaseAnchoredResidualMoE
                )

        self.assertIn("main", MODEL_REGISTRY)
        self.assertIn("v14_safe_c2f_moe", MODEL_REGISTRY)
        self.assertIn("v18_base_anchored_residual_moe", MODEL_REGISTRY)

    def test_disabled_auxiliary_scalar_is_frozen_at_construction(self) -> None:
        model = DualBranchSTImputer.from_config(compact_v18_config())
        self.assertFalse(model.aux_enabled)
        self.assertFalse(model.alpha.requires_grad)
