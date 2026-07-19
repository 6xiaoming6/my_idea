from __future__ import annotations

import torch
import unittest

from stmoe_imputer.losses import oracle_alpha_grid
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.main_branch import MultiScaleMoEBackbone
from stmoe_imputer.models.v_single import V16TeacherAnchoredResidualMoE

from _v16_utils import backbone_kwargs, compact_v16_config, make_batch


class V16ForwardAndOracleTest(unittest.TestCase):
    def test_zero_init_shapes_alpha_and_candidate_bound(self) -> None:
        for shape in ((2, 12, 32, 32, "fine_mid"), (2, 12, 24, 12, "fine_mid_coarse"), (1, 7, 32, 32, "fine_mid_coarse")):
            with self.subTest(shape=shape):
                cfg = compact_v16_config(*shape)
                model = DualBranchSTImputer.from_config(cfg).eval()
                batch = make_batch(cfg)
                with torch.no_grad():
                    outputs = model(batch)
                self.assertEqual(outputs["calibration_condition"].shape, (1, 12))
                self.assertTrue(bool(((outputs["residual_alpha"] >= 0) & (outputs["residual_alpha"] <= 1)).all()))
                self.assertTrue(bool((outputs["delta_candidate"].abs() <= cfg["model"]["v16"]["rho"] * outputs["scale_ref"] + 1e-7).all()))
                torch.testing.assert_close(outputs["x_hat_main"], outputs["x_hat_base"], atol=0.0, rtol=0.0)

    def test_zero_init_is_exactly_unmodified_main_backbone(self) -> None:
        cfg = compact_v16_config(scale_mode="fine_mid")
        main = MultiScaleMoEBackbone.from_config(cfg).eval()
        version = V16TeacherAnchoredResidualMoE.from_config(cfg).eval()
        version.student_backbone.load_state_dict(main.state_dict())
        batch = make_batch(cfg)
        with torch.no_grad():
            expected = main(**backbone_kwargs(batch))["x_hat_main"]
            actual = version(**backbone_kwargs(batch))["x_hat_main"]
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_oracle_selects_zero_half_and_one(self) -> None:
        base = torch.zeros(3, 1, 1, 1, 4)
        delta = torch.ones_like(base)
        target = torch.stack((base[0], base[1] + 0.5, base[2] + 1.0), dim=0)
        mask = torch.zeros(3, 1, 1, 1, 4)
        actual = oracle_alpha_grid(base, delta, target, mask, (0.0, 0.5, 1.0))
        torch.testing.assert_close(actual, torch.tensor([0.0, 0.5, 1.0]))


if __name__ == "__main__":
    unittest.main()
