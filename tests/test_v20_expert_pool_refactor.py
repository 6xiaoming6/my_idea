from __future__ import annotations

import torch
import unittest

from stmoe_imputer.models.experts import TopKRoutedExpertPool


def legacy_mix(pool, expert_outputs, gate, routing_mode):
    batch_size = expert_outputs.shape[0]
    if routing_mode == "dense":
        weights = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        z = (weights[:, :, None, None, None, None] * expert_outputs).sum(dim=1)
        indices = torch.arange(pool.num_experts).view(1, -1).expand(batch_size, -1)
        return z, indices, weights, torch.ones_like(weights)
    top_values, top_indices = torch.topk(gate, k=pool.top_k, dim=-1)
    top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    selected_mask = torch.zeros_like(gate)
    selected_mask.scatter_(1, top_indices, 1.0)
    if routing_mode == "soft_topk":
        weights = gate * selected_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        z = (weights[:, :, None, None, None, None] * expert_outputs).sum(dim=1)
        return z, top_indices, top_weights, selected_mask
    z = torch.zeros_like(expert_outputs[:, 0])
    for slot in range(pool.top_k):
        indices = top_indices[:, slot]
        weights = top_weights[:, slot].view(batch_size, 1, 1, 1, 1)
        for expert_idx in range(pool.num_experts):
            selected = (indices == expert_idx).to(expert_outputs.dtype).view(
                batch_size, 1, 1, 1, 1
            )
            z = z + selected * weights * expert_outputs[:, expert_idx]
    return z, top_indices, top_weights, selected_mask


class V20ExpertPoolRefactorTest(unittest.TestCase):
    def test_all_routing_modes_are_numerically_identical(self) -> None:
        torch.manual_seed(4)
        pool = TopKRoutedExpertPool(4, 3, top_k=2, num_groups=2).eval()
        value = torch.randn(2, 4, 2, 3, 3)
        gate = torch.softmax(torch.randn(2, 3), dim=-1)
        with torch.no_grad():
            expert_outputs = pool.forward_all(value)
            for mode in ("dense", "topk", "soft_topk"):
                with self.subTest(mode=mode):
                    expected = legacy_mix(pool, expert_outputs, gate, mode)
                    actual = pool.mix_from_outputs(expert_outputs, gate, mode)
                    for expected_value, actual_value in zip(expected, actual):
                        torch.testing.assert_close(
                            actual_value, expected_value, atol=1e-7, rtol=0.0
                        )
