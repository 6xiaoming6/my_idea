from __future__ import annotations

import unittest

import torch

from stmoe_imputer.models import DualBranchSTImputer

from _v18_utils import compact_v18_config, make_batch


class V18ThreeDatasetShapeTest(unittest.TestCase):
    def test_taxi_bike_and_chap_shapes(self) -> None:
        for channels, time_steps, height, width in (
            (2, 12, 32, 32),
            (2, 12, 24, 12),
            (1, 7, 32, 32),
        ):
            with self.subTest(
                shape=(channels, time_steps, height, width)
            ):
                cfg = compact_v18_config(
                    channels, time_steps, height, width
                )
                batch = make_batch(cfg)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    outputs = model(batch)
                self.assertEqual(
                    outputs["x_hat_final"].shape, batch["x_f_gt"].shape
                )
                self.assertEqual(
                    outputs["x_hat_mid"].shape, batch["x_m_obs"].shape
                )
                self.assertEqual(
                    outputs["x_hat_coarse"].shape, batch["x_c_obs"].shape
                )
                self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())
