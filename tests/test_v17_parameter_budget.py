from __future__ import annotations

from copy import deepcopy
import unittest

from stmoe_imputer.models import DualBranchSTImputer

from _v17_utils import compact_v17_config


class V17ParameterBudgetTest(unittest.TestCase):
    def test_parameter_budget_is_within_two_point_five_percent_of_main(self) -> None:
        v17_cfg = compact_v17_config()
        main_cfg = deepcopy(v17_cfg)
        main_cfg["model"]["version"] = "main"
        main_cfg["model"]["architecture"] = "main"
        main = DualBranchSTImputer.from_config(main_cfg).main_branch
        v17 = DualBranchSTImputer.from_config(v17_cfg).main_branch
        main_parameters = sum(parameter.numel() for parameter in main.parameters())
        v17_parameters = sum(parameter.numel() for parameter in v17.parameters())
        relative_increase = (v17_parameters - main_parameters) / main_parameters
        self.assertLess(
            relative_increase,
            0.025,
            f"V17 parameter increase is {relative_increase:.2%}: "
            f"main={main_parameters}, v17={v17_parameters}",
        )
