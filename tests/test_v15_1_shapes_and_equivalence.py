from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.main_branch import MultiScaleMoEBackbone
from stmoe_imputer.models.v_single import V15_1ScaleGuidedResidualMoE

from _v15_1_utils import backbone_kwargs, compact_v15_1_config, make_batch


class V15_1ShapeAndEquivalenceTest(unittest.TestCase):
    def test_three_dataset_shapes(self) -> None:
        for channels, time_steps, height, width, scale_mode in (
            (2, 12, 32, 32, "fine_mid"),
            (2, 12, 24, 12, "fine_mid_coarse"),
            (1, 7, 32, 32, "fine_mid_coarse"),
        ):
            with self.subTest(shape=(channels, time_steps, height, width)):
                cfg = compact_v15_1_config(
                    channels, time_steps, height, width, scale_mode
                )
                batch = make_batch(cfg)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    outputs = model(batch)
                expected = batch["x_f_gt"].shape
                for key in (
                    "x_hat_final",
                    "x_hat_base",
                    "x_hat_candidate",
                    "delta_candidate",
                    "delta_effective",
                ):
                    self.assertEqual(outputs[key].shape, expected)
                    self.assertTrue(torch.isfinite(outputs[key]).all())
                self.assertEqual(outputs["accept_gate"].shape, (1, 1, 1, 1, 1))
                self.assertEqual(outputs["active_scale_weight"].shape, (1, 3))
                self.assertEqual(outputs["scale_ref"].shape, (1, channels, 1, 1, 1))

    def test_zero_initialization_is_exactly_main(self) -> None:
        cfg = compact_v15_1_config(scale_mode="fine_mid")
        main = MultiScaleMoEBackbone.from_config(cfg).eval()
        version = V15_1ScaleGuidedResidualMoE.from_config(cfg).eval()
        version.main_backbone.load_state_dict(main.state_dict())
        batch = make_batch(cfg)
        with torch.no_grad():
            expected = main(**backbone_kwargs(batch))["x_hat_main"]
            outputs = version(**backbone_kwargs(batch))
        torch.testing.assert_close(outputs["x_hat_main"], expected, atol=0.0, rtol=0.0)
        self.assertEqual(torch.count_nonzero(outputs["delta_candidate"]), 0)
        self.assertEqual(torch.count_nonzero(outputs["delta_effective"]), 0)

    def test_disabled_v15_1_is_exactly_main(self) -> None:
        cfg = compact_v15_1_config()
        cfg["model"]["v15_1"]["enabled"] = False
        main = MultiScaleMoEBackbone.from_config(cfg).eval()
        version = V15_1ScaleGuidedResidualMoE.from_config(cfg).eval()
        version.main_backbone.load_state_dict(main.state_dict())
        batch = make_batch(cfg)
        with torch.no_grad():
            expected = main(**backbone_kwargs(batch))["x_hat_main"]
            actual = version(**backbone_kwargs(batch))["x_hat_main"]
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

