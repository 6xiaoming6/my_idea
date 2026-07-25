from __future__ import annotations

import unittest

import torch

from stmoe_imputer.models.v_single import V18BaseAnchoredResidualMoE

from _v18_utils import backbone_kwargs, compact_v18_config, make_batch


class V18ResidualBoundTest(unittest.TestCase):
    def test_effective_residual_has_a_hard_observed_scale_bound(self) -> None:
        cfg = compact_v18_config()
        model = V18BaseAnchoredResidualMoE.from_config(cfg).eval()
        # Make the test non-trivial while retaining the bounded tanh direction.
        with torch.no_grad():
            for module in model.residual_pyramid.modules():
                out_proj = getattr(module, "out_proj", None)
                if out_proj is not None:
                    out_proj.bias.fill_(1.0)
        batch = make_batch(cfg)
        with torch.no_grad():
            outputs = model(**backbone_kwargs(batch))

        residual = (outputs["x_hat_main"] - outputs["x_hat_base"]).abs()
        scale = outputs["features"]["observed_scale_f"]
        bound = model.rho_fine_max * scale
        self.assertGreater(torch.count_nonzero(residual), 0)
        self.assertTrue(torch.all(residual <= bound + 1e-6))
        torch.testing.assert_close(
            residual,
            outputs["features"]["effective_residual"].abs(),
        )
        rho_f = outputs["diagnostics"]["v18"]["rho_f"]
        self.assertTrue(torch.all(rho_f >= 0.0))
        self.assertTrue(torch.all(rho_f <= model.rho_fine_max))
