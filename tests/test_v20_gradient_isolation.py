from __future__ import annotations

import unittest

import torch

from stmoe_imputer.engine import _clip_optimizer_gradients


class V20GradientIsolationTest(unittest.TestCase):
    def test_probe_gradient_norm_cannot_rescale_main_gradient(self) -> None:
        main = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        probe = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        optimizer = torch.optim.SGD([
            {"name": "main", "params": [main], "lr": 1.0},
            {"name": "v20", "params": [probe], "lr": 1.0},
        ])
        main.grad = torch.tensor([3.0, 4.0])
        probe.grad = torch.tensor([300.0, 400.0])

        _clip_optimizer_gradients(
            optimizer,
            1.0,
            isolated_group_names={"v20", "v20_no_decay"},
            isolated_max_norm=1.0,
        )

        self.assertAlmostEqual(float(main.grad.norm()), 1.0, places=5)
        self.assertAlmostEqual(float(probe.grad.norm()), 1.0, places=5)
        torch.testing.assert_close(main.grad, torch.tensor([0.6, 0.8]))

    def test_default_path_preserves_historical_whole_model_clipping(self) -> None:
        left = torch.nn.Parameter(torch.tensor([0.0]))
        right = torch.nn.Parameter(torch.tensor([0.0]))
        optimizer = torch.optim.SGD([
            {"name": "main", "params": [left], "lr": 1.0},
            {"name": "other", "params": [right], "lr": 1.0},
        ])
        left.grad = torch.tensor([3.0])
        right.grad = torch.tensor([4.0])

        _clip_optimizer_gradients(optimizer, 1.0)

        torch.testing.assert_close(left.grad, torch.tensor([0.6]))
        torch.testing.assert_close(right.grad, torch.tensor([0.8]))


if __name__ == "__main__":
    unittest.main()
