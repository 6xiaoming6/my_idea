from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer

from _v17_utils import compact_v17_config, make_batch


class V17NoTargetLeakageTest(unittest.TestCase):
    def test_hidden_target_is_not_used_by_prediction_or_routing(self) -> None:
        cfg = compact_v17_config()
        batch = make_batch(cfg)
        changed = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        changed["x_f_gt"] = changed["x_f_gt"] + (1.0 - changed["m_f"]) * 100.0
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            original = model(batch)
            modified = model(changed)
        torch.testing.assert_close(
            original["x_hat_main"], modified["x_hat_main"], rtol=0.0, atol=0.0
        )
        for key in ("scale_gate", "fine", "mid", "coarse", "route_branch_gate"):
            torch.testing.assert_close(
                original["gates"][key], modified["gates"][key], rtol=0.0, atol=0.0
            )
