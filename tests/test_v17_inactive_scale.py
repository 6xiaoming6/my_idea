from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer

from _v17_utils import compact_v17_config, make_batch


class V17InactiveScaleTest(unittest.TestCase):
    def test_inactive_coarse_cannot_change_taxi_output_or_router(self) -> None:
        cfg = compact_v17_config(scale_mode="fine_mid")
        batch = make_batch(cfg)
        changed = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        changed["x_c_obs"] = changed["x_c_obs"] + 1000.0 * torch.randn_like(
            changed["x_c_obs"]
        )
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            original = model(batch)
            modified = model(changed)
        torch.testing.assert_close(
            original["x_hat_main"], modified["x_hat_main"], rtol=0.0, atol=1e-6
        )
        torch.testing.assert_close(
            original["gates"]["scale_gate"],
            modified["gates"]["scale_gate"],
            rtol=0.0,
            atol=1e-7,
        )
        for scale in ("fine", "mid"):
            torch.testing.assert_close(
                original["gates"][scale], modified["gates"][scale], rtol=0.0, atol=1e-7
            )
