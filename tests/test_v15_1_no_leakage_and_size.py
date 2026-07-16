from __future__ import annotations

import json
from pathlib import Path
import torch
import unittest

from stmoe_imputer.config import deep_update
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.main_branch import MultiScaleMoEBackbone
from stmoe_imputer.models.v_single import V15CompactResidualMoE, V15_1ScaleGuidedResidualMoE

from _v15_1_utils import compact_v15_1_config, make_batch


ROOT = Path(__file__).resolve().parents[1]


class V15_1NoLeakageAndSizeTest(unittest.TestCase):
    def test_hidden_ground_truth_never_changes_forward(self) -> None:
        cfg = compact_v15_1_config()
        batch = make_batch(cfg)
        changed = {key: value.clone() for key, value in batch.items()}
        hidden = (1.0 - changed["m_f"]).expand_as(changed["x_f_gt"])
        changed["x_f_gt"] = changed["x_f_gt"] + 10000.0 * hidden
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            first = model(batch)
            second = model(changed)
        for key in ("x_hat_final", "x_hat_candidate", "accept_gate"):
            torch.testing.assert_close(first[key], second[key], atol=0.0, rtol=0.0)

    def test_v15_1_adds_less_than_40_percent_of_v15_parameters(self) -> None:
        base_cfg = json.loads(
            (ROOT / "configs/datasets/taxibj.json").read_text(encoding="utf-8")
        )
        v15_cfg = deep_update(
            base_cfg,
            json.loads((ROOT / "configs/v15-single/taxibj.json").read_text(encoding="utf-8")),
        )
        v15_1_cfg = deep_update(
            base_cfg,
            json.loads(
                (ROOT / "configs/v15.1-single/taxibj.json").read_text(encoding="utf-8")
            ),
        )
        main = MultiScaleMoEBackbone.from_config(base_cfg)
        v15 = V15CompactResidualMoE.from_config(v15_cfg)
        v15_1 = V15_1ScaleGuidedResidualMoE.from_config(v15_1_cfg)
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        main_count = count(main)
        v15_extra = count(v15) - main_count
        v15_1_extra = count(v15_1) - main_count
        self.assertGreater(v15_1_extra, 0)
        self.assertLess(v15_1_extra, 0.40 * v15_extra)

