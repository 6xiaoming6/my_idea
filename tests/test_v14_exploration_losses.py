from __future__ import annotations

import unittest

import torch

from stmoe_imputer.losses import (
    v14_delta_scale_loss,
    v14_sample_rmse_regret_loss,
    v14_stage_aux_schedule_scale,
)


class TestV14ExplorationLosses(unittest.TestCase):
    def test_stage_aux_schedule_preserves_legacy_default(self) -> None:
        cfg = {"loss": {}, "train": {"epochs": 100}}
        self.assertEqual(v14_stage_aux_schedule_scale(cfg, 1), 1.0)
        self.assertEqual(v14_stage_aux_schedule_scale(cfg, 100), 1.0)

    def test_stage_aux_cosine_schedule_has_plateau_and_zero_endpoint(self) -> None:
        cfg = {
            "loss": {
                "v14_stage_aux_schedule": {
                    "mode": "cosine",
                    "start_fraction": 0.25,
                    "final_scale": 0.0,
                }
            },
            "train": {"epochs": 101},
        }
        self.assertEqual(v14_stage_aux_schedule_scale(cfg, 1), 1.0)
        self.assertEqual(v14_stage_aux_schedule_scale(cfg, 26), 1.0)
        self.assertAlmostEqual(v14_stage_aux_schedule_scale(cfg, 101), 0.0)
        self.assertGreater(
            v14_stage_aux_schedule_scale(cfg, 50),
            v14_stage_aux_schedule_scale(cfg, 75),
        )

    def test_stage_aux_schedule_rejects_invalid_range(self) -> None:
        cfg = {
            "loss": {
                "v14_stage_aux_schedule": {
                    "mode": "cosine",
                    "start_fraction": 1.0,
                }
            },
            "train": {"epochs": 100},
        }
        with self.assertRaisesRegex(ValueError, "start_fraction"):
            v14_stage_aux_schedule_scale(cfg, 1)

    def test_rmse_regret_uses_detached_base_and_penalizes_regression(self) -> None:
        target = torch.zeros(2, 1, 1, 1, 2)
        mask = torch.zeros(2, 1, 1, 1, 2)
        base = torch.ones_like(target, requires_grad=True)
        final = torch.stack(
            (torch.full((1, 1, 1, 2), 2.0), torch.full((1, 1, 1, 2), 0.5))
        ).requires_grad_()
        loss, violation = v14_sample_rmse_regret_loss(final, base, target, mask)
        self.assertAlmostEqual(float(loss), 0.5, places=6)
        self.assertAlmostEqual(float(violation), 0.5)
        loss.backward()
        self.assertIsNone(base.grad)
        self.assertIsNotNone(final.grad)

    def test_delta_scale_is_normalized_by_observed_rms(self) -> None:
        delta = torch.full((1, 1, 1, 1, 2), 4.0)
        observed = torch.full_like(delta, 2.0)
        mask = torch.ones(1, 1, 1, 1, 2)
        loss, ratio = v14_delta_scale_loss(delta, observed, mask)
        self.assertAlmostEqual(float(ratio), 2.0, places=5)
        self.assertAlmostEqual(
            float(loss), float(torch.log1p(torch.tensor(2.0)).square()), places=5
        )

    def test_zero_initialized_losses_have_finite_gradients(self) -> None:
        target = torch.zeros(2, 1, 1, 2, 2)
        mask = torch.ones(2, 1, 1, 2, 2)
        mask[..., 0, 0] = 0.0
        base = torch.zeros_like(target)
        final = torch.zeros_like(target, requires_grad=True)
        delta = torch.zeros_like(target, requires_grad=True)

        rmse_loss, _ = v14_sample_rmse_regret_loss(
            final, base, target, mask
        )
        scale_loss, _ = v14_delta_scale_loss(delta, target, mask)
        (rmse_loss + scale_loss).backward()

        self.assertTrue(torch.isfinite(final.grad).all())
        self.assertTrue(torch.isfinite(delta.grad).all())


if __name__ == "__main__":
    unittest.main()
