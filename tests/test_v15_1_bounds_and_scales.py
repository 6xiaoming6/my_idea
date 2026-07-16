from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models.v_single import V15_1ScaleGuidedResidualMoE

from _v15_1_utils import backbone_kwargs, compact_v15_1_config, make_batch


class V15_1BoundsAndScalesTest(unittest.TestCase):
    def test_candidate_and_effective_residual_bounds(self) -> None:
        cfg = compact_v15_1_config()
        model = V15_1ScaleGuidedResidualMoE.from_config(cfg).eval()
        with torch.no_grad():
            head = model.residual_adapter.residual_head[-1]
            head.weight.normal_(mean=0.0, std=10.0)
            head.bias.fill_(10.0)
            outputs = model(**backbone_kwargs(make_batch(cfg)))
        candidate_bound = model.rho * outputs["scale_ref"] + 1e-6
        self.assertTrue(torch.all(outputs["delta_candidate"].abs() <= candidate_bound))
        self.assertTrue(
            torch.all(
                outputs["delta_effective"].abs()
                <= outputs["delta_candidate"].abs() + 1e-6
            )
        )
        self.assertGreater(torch.count_nonzero(outputs["delta_candidate"]), 0)

    def test_taxi_residual_coarse_is_strictly_inactive(self) -> None:
        cfg = compact_v15_1_config(scale_mode="fine_mid")
        model = V15_1ScaleGuidedResidualMoE.from_config(cfg).eval()
        batch = make_batch(cfg)
        with torch.no_grad():
            model.residual_adapter.residual_head[-1].weight.normal_(0.0, 0.1)
            first = model(**backbone_kwargs(batch))
            weights = first["active_scale_weight"]
            self.assertEqual(torch.count_nonzero(weights[:, 2]), 0)

            features = first["features"]
            adapter = model.residual_adapter
            kwargs = {
                "z_f": features["z_f"].detach(),
                "z_m": features["z_m"].detach(),
                "z_c": features["z_c"].detach(),
                "h_main": features["h_main_base"].detach()
                if "h_main_base" in features
                else features["h_main"].detach(),
                "scale_weight": weights,
            }
            before = adapter(**kwargs)["delta_raw"]
            kwargs["z_c"] = kwargs["z_c"] + 10000.0
            after = adapter(**kwargs)["delta_raw"]
        torch.testing.assert_close(before, after, atol=0.0, rtol=0.0)

    def test_fixed_bias_is_not_trainable(self) -> None:
        cfg = compact_v15_1_config()
        model = V15_1ScaleGuidedResidualMoE.from_config(cfg)
        names = dict(model.acceptance_gate.named_parameters())
        buffers = dict(model.acceptance_gate.named_buffers())
        self.assertNotIn("fixed_bias", names)
        self.assertIn("fixed_bias", buffers)

