from __future__ import annotations

import json
import unittest

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.models import DualBranchSTImputer

from _v17_utils import ROOT, compact_v17_config, make_batch


COMBINATIONS = {
    "c1_independent_shared_scale": (False, "linear", "sample_residual"),
    "c2_independent_shared_hard_floor": (False, "hard", "sample_residual"),
    "c3_independent_shared_hard_floor_global_gamma": (
        False,
        "hard",
        "global_residual",
    ),
}


class V171StagePipelineTest(unittest.TestCase):
    def test_c1_c2_c3_are_incremental_and_executable(self) -> None:
        for variant, expected in COMBINATIONS.items():
            with self.subTest(variant=variant):
                patch = json.loads(
                    (
                        ROOT
                        / "configs"
                        / "v17.1-single"
                        / "combinations"
                        / f"{variant}.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    patch["experiment"],
                    {
                        "group": "v17_exploratory_combination",
                        "variant": variant,
                    },
                )
                cfg = deep_update(compact_v17_config(), patch)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    outputs = model(make_batch(cfg))
                unified, floor_mode, branch_mode = expected
                self.assertEqual(outputs["v17_unified_scale_weight"], unified)
                self.assertEqual(outputs["v17_fine_floor_mode"], floor_mode)
                self.assertEqual(outputs["branch_mode"], branch_mode)
                self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())

    def test_combination_output_name_is_separate_from_ablation(self) -> None:
        from scripts.train import _experiment_parts

        self.assertEqual(
            _experiment_parts("combination_c2_independent_shared_hard_floor"),
            ("combination", "c2_independent_shared_hard_floor"),
        )


if __name__ == "__main__":
    unittest.main()
