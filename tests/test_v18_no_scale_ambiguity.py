from __future__ import annotations

import unittest

from stmoe_imputer.models.v_single import V18BaseAnchoredResidualMoE

from _v18_utils import compact_v18_config


class V18NoScaleAmbiguityTest(unittest.TestCase):
    def test_v18_has_no_v14_correction_adapter_or_alpha_delta_pair(self) -> None:
        model = V18BaseAnchoredResidualMoE.from_config(compact_v18_config())
        module_names = tuple(name.lower() for name, _ in model.named_modules())
        parameter_names = tuple(name.lower() for name, _ in model.named_parameters())
        forbidden = (
            "correction_adapter",
            "alpha_final",
            "alpha_final_max",
            "alpha_final_bias",
            "delta_ctf",
        )
        for token in forbidden:
            self.assertFalse(
                any(token in name for name in (*module_names, *parameter_names)),
                token,
            )
        self.assertIsNotNone(model.controller)
        self.assertTrue(hasattr(model.controller, "rho_max"))
