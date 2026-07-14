from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models import DualBranchSTImputer

from _v14_utils import compact_v14_config, make_batch


class V14NoTargetLeakageTest(unittest.TestCase):
    def test_hidden_ground_truth_never_changes_forward_prediction(self) -> None:
        cfg = compact_v14_config()
        batch = make_batch(cfg)
        changed = {key: value.clone() for key, value in batch.items()}
        hidden = (1.0 - changed["m_f"]).expand_as(changed["x_f_gt"])
        changed["x_f_gt"] = changed["x_f_gt"] + hidden * 10000.0
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            first = model(batch)
            second = model(changed)
        torch.testing.assert_close(first["x_hat_final"], second["x_hat_final"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            first["diagnostics"]["v14"]["alpha_final"],
            second["diagnostics"]["v14"]["alpha_final"],
            atol=0.0,
            rtol=0.0,
        )
