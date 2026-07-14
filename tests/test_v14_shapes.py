from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.v_single import V14SafeC2FMoE

from _v14_utils import compact_v14_config, make_batch


class V14ShapeTest(unittest.TestCase):
    def test_v14_output_shapes(self) -> None:
        for channels, time_steps, height, width in (
            (2, 3, 32, 32), (2, 3, 24, 12), (1, 3, 40, 40)
        ):
            with self.subTest(shape=(channels, time_steps, height, width)):
                cfg = compact_v14_config(channels, time_steps, height, width)
                batch = make_batch(cfg)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    outputs = model(batch)
                self.assertEqual(outputs["x_hat_final"].shape, batch["x_f_gt"].shape)
                self.assertEqual(outputs["x_hat_base"].shape, batch["x_f_gt"].shape)
                self.assertEqual(outputs["x_hat_ctf"].shape, batch["x_f_gt"].shape)
                self.assertEqual(outputs["x_hat_mid"].shape, batch["x_m_obs"].shape)
                self.assertEqual(outputs["x_hat_coarse"].shape, batch["x_c_obs"].shape)
                self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())

    def test_amp_safe_norm_is_finite_for_large_half_values(self) -> None:
        values = torch.full((1, 1, 1, 2, 2), 60000.0, dtype=torch.float16)
        norm = V14SafeC2FMoE._rms(values)
        self.assertEqual(norm.dtype, torch.float32)
        self.assertTrue(torch.isfinite(norm).all())
