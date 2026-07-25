from __future__ import annotations

import unittest

import torch

from stmoe_imputer.models.v_single import masked_channel_rms


class V18ObservedScaleTest(unittest.TestCase):
    def test_masked_channel_rms_uses_observed_values_per_channel(self) -> None:
        value = torch.tensor(
            [[[[[3.0, 4.0]]], [[[6.0, 8.0]]]]], requires_grad=True
        )
        mask = torch.tensor([[[[[1.0, 0.0]]]]])
        scale = masked_channel_rms(value, mask, eps=1e-3)
        self.assertEqual(scale.shape, (1, 2, 1, 1, 1))
        torch.testing.assert_close(
            scale.flatten(), torch.tensor([3.0, 6.0])
        )
        self.assertFalse(scale.requires_grad)

    def test_empty_observation_fallback_is_target_free_and_finite(self) -> None:
        value = torch.zeros((1, 2, 1, 2, 2))
        mask = torch.zeros((1, 1, 1, 2, 2))
        scale = masked_channel_rms(value, mask, eps=0.125)
        torch.testing.assert_close(scale, torch.full_like(scale, 0.125))
        self.assertTrue(torch.isfinite(scale).all())

    def test_invalid_mask_channels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "channel dimension"):
            masked_channel_rms(
                torch.ones((1, 2, 1, 2, 2)),
                torch.ones((1, 2, 1, 2, 2)),
            )
