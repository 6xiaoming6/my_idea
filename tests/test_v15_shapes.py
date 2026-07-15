from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer

from _v15_utils import compact_v15_config, make_batch


class V15ShapeTest(unittest.TestCase):
    def test_dataset_output_shapes(self) -> None:
        for channels, time_steps, height, width in (
            (2, 12, 32, 32),
            (2, 12, 24, 12),
            (1, 7, 32, 32),
        ):
            with self.subTest(shape=(channels, time_steps, height, width)):
                cfg = compact_v15_config(channels, time_steps, height, width)
                batch = make_batch(cfg)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    outputs = model(batch)
                expected = batch["x_f_gt"].shape
                self.assertEqual(outputs["x_hat_final"].shape, expected)
                self.assertEqual(outputs["x_hat_base"].shape, expected)
                self.assertEqual(outputs["features"]["delta_raw"].shape, expected)
                self.assertEqual(outputs["delta_effective"].shape, expected)
                self.assertEqual(outputs["residual_budget"].shape, (1, 1, 1, 1, 1))
                self.assertEqual(outputs["scale_ref"].shape, (1, channels, 1, 1, 1))
                self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())

    def test_large_half_values_use_float_diagnostics(self) -> None:
        cfg = compact_v15_config(channels=1)
        model = DualBranchSTImputer.from_config(cfg).main_branch
        values = torch.full((1, 1, 1, 2, 2), 60000.0, dtype=torch.float16)
        rms = model._rms(values)
        scale_ref = model._compute_scale_ref(values)
        self.assertEqual(rms.dtype, torch.float32)
        self.assertTrue(torch.isfinite(rms).all())
        self.assertTrue(torch.isfinite(scale_ref).all())
