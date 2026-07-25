from __future__ import annotations

import unittest

import torch

from stmoe_imputer.models import DualBranchSTImputer

from _v18_utils import compact_v18_config, make_batch


class V18DirectionShapeTest(unittest.TestCase):
    def test_all_direction_and_auxiliary_prediction_shapes_are_correct(self) -> None:
        cfg = compact_v18_config()
        batch = make_batch(cfg)
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            outputs = model(batch)

        self.assertEqual(outputs["x_hat_final"].shape, batch["x_f_gt"].shape)
        self.assertEqual(outputs["x_hat_base"].shape, batch["x_f_gt"].shape)
        self.assertEqual(outputs["x_hat_mid"].shape, batch["x_m_obs"].shape)
        self.assertEqual(outputs["x_hat_coarse"].shape, batch["x_c_obs"].shape)
        self.assertEqual(
            outputs["features"]["direction_f"].shape, batch["x_f_obs"].shape
        )
        self.assertEqual(
            outputs["features"]["direction_m"].shape, batch["x_m_obs"].shape
        )
        self.assertEqual(
            outputs["features"]["direction_c"].shape, batch["x_c_obs"].shape
        )
        self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())
