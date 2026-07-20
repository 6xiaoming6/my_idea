from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models.v_single import ScaleSpecificAdapter


class V17AdapterIdentityTest(unittest.TestCase):
    def test_zero_initialized_adapter_is_identity(self) -> None:
        adapter = ScaleSpecificAdapter(dim=8, bottleneck_dim=4, zero_init=True).eval()
        value = torch.randn(2, 8, 3, 4, 4)
        with torch.no_grad():
            output = adapter(value)
        torch.testing.assert_close(output, value, rtol=0.0, atol=0.0)
