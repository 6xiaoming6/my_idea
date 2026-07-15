from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models.main_branch import MultiScaleMoEBackbone
from stmoe_imputer.models.v_single import V15CompactResidualMoE

from _v15_utils import backbone_kwargs, compact_v15_config, make_batch


def _paired_models(enabled: bool) -> tuple[MultiScaleMoEBackbone, V15CompactResidualMoE, dict]:
    cfg = compact_v15_config()
    cfg["model"]["v15"]["enabled"] = enabled
    main = MultiScaleMoEBackbone.from_config(cfg).eval()
    v15 = V15CompactResidualMoE.from_config(cfg).eval()
    v15.main_backbone.load_state_dict(main.state_dict())
    return main, v15, make_batch(cfg)


class V15MainEquivalenceTest(unittest.TestCase):
    def test_disabled_v15_is_exactly_main(self) -> None:
        main, v15, batch = _paired_models(enabled=False)
        with torch.no_grad():
            expected = main(**backbone_kwargs(batch))["x_hat_main"]
            actual = v15(**backbone_kwargs(batch))["x_hat_main"]
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_zero_initialized_residual_starts_exactly_at_main(self) -> None:
        main, v15, batch = _paired_models(enabled=True)
        with torch.no_grad():
            expected = main(**backbone_kwargs(batch))["x_hat_main"]
            outputs = v15(**backbone_kwargs(batch))
        torch.testing.assert_close(outputs["x_hat_main"], expected, atol=0.0, rtol=0.0)
        self.assertEqual(torch.count_nonzero(outputs["delta_effective"]), 0)
