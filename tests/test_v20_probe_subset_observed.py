from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models.v_single import GeometryMatchedProbeBuilder


class V20ProbeMaskTest(unittest.TestCase):
    def test_probe_is_deterministic_observed_subset_and_exam_removes_answers(self) -> None:
        mask = torch.ones(2, 1, 3, 12, 12)
        mask[:, :, :, 4:8, 4:8] = 0.0
        reliability = mask.clone()
        builder = GeometryMatchedProbeBuilder(
            probe_ratio=0.1, min_count=4, max_count=20, min_remaining=8
        )
        first = builder.build(mask, reliability)
        second = builder.build(mask, reliability)
        probe = first["probe_mask"]
        self.assertTrue(first["valid"].all())
        self.assertTrue(torch.all(probe <= mask))
        torch.testing.assert_close(probe, second["probe_mask"], atol=0.0, rtol=0.0)
        value = torch.randn(2, 2, 3, 12, 12) * mask
        exam_mask = mask * (1.0 - probe)
        exam_value = value * exam_mask
        selected = probe.bool().expand_as(exam_value)
        self.assertEqual(torch.count_nonzero(exam_mask[probe.bool()]), 0)
        self.assertEqual(torch.count_nonzero(exam_value[selected]), 0)

    def test_geometry_selection_has_no_worse_match_than_deterministic_random(self) -> None:
        mask = torch.ones(1, 1, 3, 16, 16)
        mask[:, :, :, 6:10, 6:10] = 0.0
        geometry = GeometryMatchedProbeBuilder(
            probe_ratio=0.1, min_count=8, max_count=24, min_remaining=16
        ).build(mask, mask)
        random = GeometryMatchedProbeBuilder(
            probe_ratio=0.1,
            min_count=8,
            max_count=24,
            min_remaining=16,
            selection_mode="random",
        ).build(mask, mask)
        self.assertLessEqual(
            float(geometry["match_distance"].item()),
            float(random["match_distance"].item()) + 1e-7,
        )

    def test_too_few_candidates_falls_back(self) -> None:
        mask = torch.zeros(1, 1, 2, 4, 4)
        mask.reshape(-1)[:6] = 1.0
        result = GeometryMatchedProbeBuilder(
            min_count=4, max_count=8, min_remaining=4
        ).build(mask, mask)
        self.assertFalse(result["valid"].item())
        self.assertEqual(torch.count_nonzero(result["probe_mask"]), 0)

    def test_no_target_geometry_falls_back_without_nan(self) -> None:
        mask = torch.ones(1, 1, 3, 8, 8)
        result = GeometryMatchedProbeBuilder(
            min_count=4, max_count=16, min_remaining=8
        ).build(mask, mask)
        self.assertFalse(result["valid"].item())
        self.assertEqual(torch.count_nonzero(result["probe_mask"]), 0)
        for value in result.values():
            if torch.is_tensor(value) and value.is_floating_point():
                self.assertTrue(torch.isfinite(value).all())
