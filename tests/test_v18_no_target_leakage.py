from __future__ import annotations

import unittest

import torch

from stmoe_imputer.models import DualBranchSTImputer

from _v18_utils import compact_v18_config, make_batch


class V18NoTargetLeakageTest(unittest.TestCase):
    def test_hidden_ground_truth_cannot_change_v18_forward_outputs(self) -> None:
        cfg = compact_v18_config()
        batch = make_batch(cfg)
        changed = {key: value.clone() for key, value in batch.items()}
        hidden = (1.0 - changed["m_f"]).expand_as(changed["x_f_gt"])
        changed["x_f_gt"] = changed["x_f_gt"] + hidden * 10000.0
        model = DualBranchSTImputer.from_config(cfg).eval()

        with torch.no_grad():
            first = model(batch)
            second = model(changed)

        for key in ("x_hat_final", "x_hat_main", "x_hat_base", "x_hat_probe"):
            torch.testing.assert_close(
                first[key], second[key], atol=0.0, rtol=0.0
            )
        for key in (
            "rho_c",
            "rho_m",
            "rho_f",
            "utility_base_rel",
            "utility_probe_rel",
            "utility_gain",
        ):
            torch.testing.assert_close(
                first["diagnostics"]["v18"][key],
                second["diagnostics"]["v18"][key],
                atol=0.0,
                rtol=0.0,
            )
