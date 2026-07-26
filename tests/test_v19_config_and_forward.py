from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.models import (
    DualBranchSTImputer,
    V19ChannelCalibratedV14MoE,
)
from stmoe_imputer.models.registry import MODEL_REGISTRY, build_model_backbone

from _v19_utils import compact_v19_config, make_batch


ROOT = Path(__file__).resolve().parents[1]


class V19ConfigAndForwardTest(unittest.TestCase):
    def test_all_dataset_configs_select_v19_and_keep_formal_epochs(self) -> None:
        cases = (
            ("taxibj.json", "taxibj.json", 160, 5),
            ("bikenyc.json", "bikenyc.json", 140, 2),
            ("chap_beijing.json", "chap.json", 150, 5),
        )
        for dataset_config, v19_config, epochs, val_epoch in cases:
            with self.subTest(config=v19_config):
                base = json.loads(
                    (ROOT / "configs/datasets" / dataset_config).read_text(
                        encoding="utf-8"
                    )
                )
                patch = json.loads(
                    (ROOT / "configs/v19-single" / v19_config).read_text(
                        encoding="utf-8"
                    )
                )
                cfg = deep_update(base, patch)
                self.assertEqual(cfg["output_dir"], "outputs/v19-single")
                self.assertEqual(
                    cfg["model"]["architecture"],
                    "v19_channel_calibrated_v14_moe",
                )
                self.assertEqual(cfg["train"]["epochs"], epochs)
                self.assertEqual(cfg["train"]["val_epoch"], val_epoch)
                self.assertFalse(cfg["train"]["early_stopping"]["enabled"])
                self.assertIsInstance(
                    build_model_backbone(cfg),
                    V19ChannelCalibratedV14MoE,
                )
        self.assertIn("v19_channel_calibrated_v14_moe", MODEL_REGISTRY)

    def test_zero_initialized_v19_is_exactly_v14(self) -> None:
        cfg = compact_v19_config()
        model = DualBranchSTImputer.from_config(cfg).eval()
        batch = make_batch(cfg)
        with torch.no_grad():
            outputs = model(batch)
        torch.testing.assert_close(
            outputs["x_hat_final"],
            outputs["x_hat_v14"],
            atol=0.0,
            rtol=0.0,
        )
        gain = outputs["diagnostics"]["v19"]["gain"]
        torch.testing.assert_close(
            gain,
            torch.ones_like(gain),
            atol=0.0,
            rtol=0.0,
        )
        self.assertGreaterEqual(float(gain.min()), 0.5)
        self.assertLessEqual(float(gain.max()), 1.5)

    def test_three_dataset_channel_and_geometry_shapes(self) -> None:
        cases = (
            (2, 12, 32, 32),
            (2, 12, 24, 12),
            (1, 7, 32, 32),
        )
        for channels, time_steps, height, width in cases:
            with self.subTest(shape=(channels, time_steps, height, width)):
                cfg = compact_v19_config(
                    channels=channels,
                    time_steps=time_steps,
                    height=height,
                    width=width,
                )
                model = DualBranchSTImputer.from_config(cfg).eval()
                batch = make_batch(cfg)
                with torch.no_grad():
                    outputs = model(batch)
                self.assertEqual(
                    outputs["x_hat_final"].shape,
                    batch["x_f_gt"].shape,
                )
                self.assertEqual(
                    outputs["diagnostics"]["v19"]["gain"].shape,
                    (1, channels),
                )


if __name__ == "__main__":
    unittest.main()
