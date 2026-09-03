from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer

from _v14_utils import compact_v14_config, make_batch
from _v20_utils import compact_v20_config


class V20V14CompatibilityTest(unittest.TestCase):
    def test_zero_initialized_v20_starts_at_exact_v14_prediction(self) -> None:
        v14_cfg = compact_v14_config()
        v20_cfg = compact_v20_config()
        torch.manual_seed(31)
        v14 = DualBranchSTImputer.from_config(v14_cfg).eval()
        torch.manual_seed(31)
        v20 = DualBranchSTImputer.from_config(v20_cfg).eval()
        missing, unexpected = v20.load_state_dict(v14.state_dict(), strict=False)
        self.assertTrue(all("probe_evaluator" in name for name in missing))
        self.assertEqual(unexpected, [])
        batch = make_batch(v14_cfg)
        with torch.no_grad():
            expected = v14(batch)
            actual = v20(batch)
        for key in ("x_hat_main", "x_hat_base", "x_hat_ctf", "x_hat_final"):
            torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=0.0)
        for scale in expected["gates"]:
            torch.testing.assert_close(
                actual["gates"][scale], expected["gates"][scale], atol=1e-6, rtol=0.0
            )
        for key in expected["topk"]:
            torch.testing.assert_close(
                actual["topk"][key], expected["topk"][key], atol=1e-6, rtol=0.0
            )
        for scale in expected["selected_masks"]:
            torch.testing.assert_close(
                actual["selected_masks"][scale],
                expected["selected_masks"][scale],
                atol=1e-6,
                rtol=0.0,
            )
        for key in (
            "alpha_mid",
            "alpha_fine",
            "alpha_final",
        ):
            torch.testing.assert_close(
                actual["diagnostics"]["v14"][key],
                expected["diagnostics"]["v14"][key],
                atol=1e-6,
                rtol=0.0,
            )
