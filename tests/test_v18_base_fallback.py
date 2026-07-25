from __future__ import annotations

import unittest

import torch

from stmoe_imputer.models.main_branch import MultiScaleMoEBackbone
from stmoe_imputer.models.v_single import V18BaseAnchoredResidualMoE

from _v18_utils import backbone_kwargs, compact_v18_config, make_batch


class V18BaseFallbackTest(unittest.TestCase):
    def test_zero_initialized_directions_start_exactly_at_v14_main_base(self) -> None:
        cfg = compact_v18_config()
        main = MultiScaleMoEBackbone.from_config(cfg).eval()
        v18 = V18BaseAnchoredResidualMoE.from_config(cfg).eval()
        v18.main_backbone.load_state_dict(main.state_dict())
        batch = make_batch(cfg)

        with torch.no_grad():
            expected = main(**backbone_kwargs(batch))["x_hat_main"]
            outputs = v18(**backbone_kwargs(batch))

        torch.testing.assert_close(
            outputs["x_hat_base"], expected, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            outputs["x_hat_main"], expected, atol=0.0, rtol=0.0
        )
        self.assertEqual(
            torch.count_nonzero(outputs["features"]["effective_residual"]), 0
        )
        for scale in ("c", "m", "f"):
            self.assertEqual(
                torch.count_nonzero(outputs["features"][f"direction_{scale}"]), 0
            )
