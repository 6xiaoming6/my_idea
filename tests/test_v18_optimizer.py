from __future__ import annotations

import copy
import unittest

from stmoe_imputer.engine import build_optimizer
from stmoe_imputer.models import DualBranchSTImputer

from _v18_utils import compact_v18_config


def _parameter_protocol(
    model,
    optimizer,
    token: str = "main_branch.main_backbone.",
) -> dict[str, tuple[float, float]]:
    by_id = {
        id(parameter): (
            float(group["lr"]),
            float(group.get("weight_decay", 0.0)),
        )
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    return {
        name: by_id[id(parameter)]
        for name, parameter in model.named_parameters()
        if token in name
    }


class V18OptimizerTest(unittest.TestCase):
    def test_v18_preserves_v14_main_parameter_optimization_protocol(self) -> None:
        v18_cfg = compact_v18_config()
        v14_cfg = copy.deepcopy(v18_cfg)
        v14_cfg["model"]["version"] = "v14-single"
        v14_cfg["model"]["architecture"] = "v14_safe_c2f_moe"
        v14_cfg["train"]["lr_v14"] = v14_cfg["train"]["lr_main"]

        v18 = DualBranchSTImputer.from_config(v18_cfg)
        v14 = DualBranchSTImputer.from_config(v14_cfg)
        v18_protocol = _parameter_protocol(
            v18, build_optimizer(v18, v18_cfg)
        )
        v14_protocol = _parameter_protocol(
            v14, build_optimizer(v14, v14_cfg)
        )
        self.assertEqual(v18_protocol, v14_protocol)
        self.assertTrue(v18_protocol)

        v18_condition_protocol = _parameter_protocol(
            v18,
            build_optimizer(v18, v18_cfg),
            "main_branch.condition_encoder.",
        )
        v14_condition_protocol = _parameter_protocol(
            v14,
            build_optimizer(v14, v14_cfg),
            "main_branch.condition_encoder.",
        )
        self.assertEqual(v18_condition_protocol, v14_condition_protocol)
        self.assertTrue(v18_condition_protocol)

    def test_v18_new_modules_use_declared_learning_rates(self) -> None:
        cfg = compact_v18_config()
        model = DualBranchSTImputer.from_config(cfg)
        optimizer = build_optimizer(model, cfg)
        groups = {
            group["name"]: float(group["lr"])
            for group in optimizer.param_groups
        }
        self.assertEqual(
            groups["v18_refiner"], cfg["train"]["lr_v18_refiner"]
        )
        self.assertEqual(
            groups["v18_controller"], cfg["train"]["lr_v18_controller"]
        )


if __name__ == "__main__":
    unittest.main()
