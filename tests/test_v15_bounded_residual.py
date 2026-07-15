from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models.v_single import V15CompactResidualMoE

from _v15_utils import backbone_kwargs, compact_v15_config, make_batch


class V15BoundedResidualTest(unittest.TestCase):
    def test_effective_residual_is_strictly_bounded(self) -> None:
        cfg = compact_v15_config()
        model = V15CompactResidualMoE.from_config(cfg).eval()
        with torch.no_grad():
            head = model.residual_pyramid.residual_head[-1]
            head.weight.normal_(mean=0.0, std=10.0)
            head.bias.fill_(10.0)
            outputs = model(**backbone_kwargs(make_batch(cfg)))

        upper_bound = model.beta_max * outputs["scale_ref"] + 1e-6
        self.assertTrue(torch.all(outputs["delta_effective"].abs() <= upper_bound))
        relative = outputs["diagnostics"]["v15"]["effective_relative_rms"]
        self.assertTrue(torch.all(relative <= model.beta_max + 1e-6))
        self.assertGreater(torch.count_nonzero(outputs["delta_effective"]), 0)
