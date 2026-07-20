from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer

from _v17_utils import compact_v17_config, make_batch


class V17ShapeTest(unittest.TestCase):
    def test_three_dataset_output_contract(self) -> None:
        for channels, time_steps, height, width, scale_mode in (
            (2, 12, 32, 32, "fine_mid"),
            (2, 12, 24, 12, "fine_mid_coarse"),
            (1, 7, 32, 32, "fine_mid_coarse"),
        ):
            with self.subTest(shape=(channels, time_steps, height, width)):
                cfg = compact_v17_config(channels, time_steps, height, width, scale_mode)
                batch = make_batch(cfg)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    outputs = model(batch)
                expected = batch["x_f_gt"].shape
                self.assertEqual(outputs["x_hat_main"].shape, expected)
                self.assertEqual(outputs["x_hat_final"].shape, expected)
                self.assertEqual(outputs["x_comp"].shape, expected)
                self.assertEqual(outputs["x_hat_shared"].shape, expected)
                self.assertEqual(outputs["x_hat_route"].shape, expected)
                self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())
                self.assertIs(outputs["v17_enabled"], True)
