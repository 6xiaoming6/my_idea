from __future__ import annotations

import json
from pathlib import Path
import torch
import unittest

from stmoe_imputer.config import deep_update
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.registry import MODEL_REGISTRY, resolve_architecture

from _v16_utils import compact_v16_config, make_batch


ROOT = Path(__file__).resolve().parents[1]


class V16ConfigTest(unittest.TestCase):
    def test_formal_configs_select_v16_and_keep_documented_epochs(self) -> None:
        expected = {"taxibj": 160, "bikenyc": 120, "chap": 150}
        for dataset, epochs in expected.items():
            with self.subTest(dataset=dataset):
                patch = json.loads((ROOT / f"configs/v16-single/{dataset}.json").read_text(encoding="utf-8"))
                self.assertEqual(resolve_architecture(patch), "v16_teacher_anchored_residual_moe")
                self.assertEqual(patch["train"]["epochs"], epochs)
                self.assertEqual(patch["model"]["v16"]["calibration_condition_dim"], 12)
                self.assertEqual(patch["model"]["v16"]["warmup_epochs"], 12)
                self.assertTrue(patch["teacher"]["enabled"])

    def test_all_documented_ablations_build_and_forward(self) -> None:
        for name in ("no_teacher_anchor", "fixed_alpha", "original_9d_condition", "binary_acceptance"):
            with self.subTest(ablation=name):
                cfg = compact_v16_config()
                patch = json.loads((ROOT / f"configs/v16-single/ablations/{name}.json").read_text(encoding="utf-8"))
                cfg = deep_update(cfg, patch)
                # Teacher ownership belongs to the engine; model Forward remains standalone.
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    outputs = model(make_batch(cfg))
                expected_dim = 9 if name == "original_9d_condition" else 12
                self.assertEqual(outputs["calibration_condition"].shape[1], expected_dim)
                self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())

    def test_registry_preserves_historical_architectures(self) -> None:
        self.assertTrue({
            "main",
            "v14_safe_c2f_moe",
            "v15_compact_residual_moe",
            "v15_1_scale_guided_residual_moe",
            "v16_teacher_anchored_residual_moe",
        }.issubset(MODEL_REGISTRY))


if __name__ == "__main__":
    unittest.main()
