from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models.main_branch import MultiScaleMoEBackbone
from stmoe_imputer.models.v_single import V14SafeC2FMoE

from _v14_utils import backbone_kwargs, compact_v14_config, make_batch


def _paired_models(enabled: bool) -> tuple[MultiScaleMoEBackbone, V14SafeC2FMoE, dict]:
    cfg = compact_v14_config()
    cfg["model"]["v14"]["enabled"] = enabled
    main = MultiScaleMoEBackbone.from_config(cfg).eval()
    v14 = V14SafeC2FMoE.from_config(cfg).eval()
    v14.main_backbone.load_state_dict(main.state_dict())
    return main, v14, make_batch(cfg)


class V14MainEquivalenceTest(unittest.TestCase):
    def test_disabled_v14_is_exactly_main(self) -> None:
        main, v14, batch = _paired_models(enabled=False)
        with torch.no_grad():
            expected = main(**backbone_kwargs(batch))["x_hat_main"]
            actual = v14(**backbone_kwargs(batch))["x_hat_main"]
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_zero_initialized_correction_starts_exactly_at_main(self) -> None:
        main, v14, batch = _paired_models(enabled=True)
        with torch.no_grad():
            expected = main(**backbone_kwargs(batch))["x_hat_main"]
            outputs = v14(**backbone_kwargs(batch))
        torch.testing.assert_close(outputs["x_hat_main"], expected, atol=0.0, rtol=0.0)
        self.assertEqual(torch.count_nonzero(outputs["features"]["delta_ctf"]), 0)
