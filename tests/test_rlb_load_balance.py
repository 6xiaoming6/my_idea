from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from stmoe_imputer.losses import moe_balance_loss
from stmoe_imputer.routing_metrics import (
    RoutingMetricAccumulator,
    active_routing_scales,
)

ROOT = Path(__file__).resolve().parents[1]


def _topk_inputs() -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    logits = torch.tensor(
        [
            [4.0, 3.0, 0.0, -1.0],
            [3.5, 2.5, 0.1, -0.5],
            [4.2, 3.2, -0.2, -1.2],
            [3.8, 2.8, 0.2, -0.8],
        ],
        requires_grad=True,
    )
    gate = logits.softmax(dim=1)
    indices = gate.topk(k=2, dim=1).indices
    selected = torch.zeros_like(gate)
    selected.scatter_(1, indices, 1.0)
    return logits, {"fine": gate}, {"fine": selected}


class TestRLBLoadBalance(unittest.TestCase):
    def test_experiment_configs_change_only_routing_loss_settings(self) -> None:
        config_root = ROOT / "configs/v14-exploration/routing"
        expected = {
            "R01_switch_topk_1e-4.json": 1e-4,
            "R02_switch_topk_1e-3.json": 1e-3,
            "R03_switch_topk_1e-2.json": 1e-2,
        }
        for filename, weight in expected.items():
            with self.subTest(config=filename):
                patch = json.loads(
                    (config_root / filename).read_text(encoding="utf-8")
                )
                self.assertEqual(
                    set(patch), {"output_dir", "experiment", "loss"}
                )
                self.assertEqual(
                    patch["loss"],
                    {
                        "load_balance_mode": "switch_topk",
                        "lambda_load_balance": weight,
                    },
                )

    def test_legacy_mode_exactly_matches_original_formula(self) -> None:
        _, gates, masks = _topk_inputs()
        total, importance, load = moe_balance_loss(
            gates,
            masks,
            scale_names=("fine",),
            load_balance_mode="legacy_hard",
        )
        gate_all = gates["fine"]
        expected_importance = (
            (gate_all.mean(dim=0) - torch.full((4,), 0.25)) ** 2
        ).sum()
        hard_load = masks["fine"].mean(dim=0)
        expected_load = ((hard_load - hard_load.mean()) ** 2).sum()
        torch.testing.assert_close(importance, expected_importance)
        torch.testing.assert_close(load, expected_load)
        torch.testing.assert_close(total, expected_importance + expected_load)

    def test_legacy_hard_load_has_no_router_gradient(self) -> None:
        logits, gates, masks = _topk_inputs()
        total, importance, load = moe_balance_loss(
            gates,
            masks,
            scale_names=("fine",),
            load_balance_mode="legacy_hard",
        )
        self.assertFalse(masks["fine"].requires_grad)
        self.assertFalse(load.requires_grad)
        total_gradient = torch.autograd.grad(total, logits, retain_graph=True)[0]
        importance_gradient = torch.autograd.grad(importance, logits)[0]
        torch.testing.assert_close(total_gradient, importance_gradient)

    def test_switch_topk_has_nonzero_router_gradient_when_overloaded(self) -> None:
        logits, gates, masks = _topk_inputs()
        _, _, load = moe_balance_loss(
            gates,
            masks,
            scale_names=("fine",),
            load_balance_mode="switch_topk",
        )
        gradient = torch.autograd.grad(load, logits)[0]
        self.assertTrue(load.requires_grad)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().max()), 1e-8)

    def test_switch_topk_gradient_is_zero_for_uniform_hard_load(self) -> None:
        logits, gates, _ = _topk_inputs()
        uniform_mask = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        )
        _, _, load = moe_balance_loss(
            gates,
            {"fine": uniform_mask},
            scale_names=("fine",),
            load_balance_mode="switch_topk",
        )
        gradient = torch.autograd.grad(load, logits)[0]
        self.assertAlmostEqual(float(load), 1.0, places=6)
        self.assertLess(float(gradient.abs().max()), 1e-7)

    def test_only_active_scales_affect_switch_loss(self) -> None:
        logits, fine_gates, fine_masks = _topk_inputs()
        mid_logits = (-logits.detach()).requires_grad_()
        mid_gate = mid_logits.softmax(dim=1)
        mid_indices = mid_gate.topk(k=2, dim=1).indices
        mid_mask = torch.zeros_like(mid_gate)
        mid_mask.scatter_(1, mid_indices, 1.0)
        coarse_gate = torch.tensor(
            [[0.97, 0.01, 0.01, 0.01]] * 4,
            requires_grad=True,
        )
        coarse_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 4)
        gates = {
            "fine": fine_gates["fine"],
            "mid": mid_gate,
            "coarse": coarse_gate,
        }
        masks = {
            "fine": fine_masks["fine"],
            "mid": mid_mask,
            "coarse": coarse_mask,
        }
        selected, _, _ = moe_balance_loss(
            gates,
            masks,
            scale_names=("fine", "mid"),
            load_balance_mode="switch_topk",
        )
        manual, _, _ = moe_balance_loss(
            {"fine": gates["fine"], "mid": gates["mid"]},
            {"fine": masks["fine"], "mid": masks["mid"]},
            scale_names=("fine", "mid"),
            load_balance_mode="switch_topk",
        )
        torch.testing.assert_close(selected, manual)
        gradient = torch.autograd.grad(selected, coarse_gate, allow_unused=True)[0]
        self.assertIsNone(gradient)

    def test_disabled_load_balance_is_safe_and_invalid_mode_is_rejected(self) -> None:
        logits, gates, masks = _topk_inputs()
        _, _, load = moe_balance_loss(
            gates,
            masks,
            use_load_balance=False,
            scale_names=("fine",),
            load_balance_mode="switch_topk",
        )
        gradient = torch.autograd.grad(load, logits)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
        with self.assertRaisesRegex(ValueError, "load_balance_mode"):
            moe_balance_loss(
                gates,
                masks,
                scale_names=("fine",),
                load_balance_mode="unknown",
            )


class TestRoutingMetricAccumulator(unittest.TestCase):
    def test_active_scales_are_explicit(self) -> None:
        self.assertEqual(active_routing_scales("fine"), ("fine",))
        self.assertEqual(active_routing_scales("fine_mid"), ("fine", "mid"))
        self.assertEqual(
            active_routing_scales("fine_mid_coarse"),
            ("fine", "mid", "coarse"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown scale_mode"):
            active_routing_scales("bad")

    def test_accumulator_reports_exact_hard_load_statistics(self) -> None:
        accumulator = RoutingMetricAccumulator(("fine",))
        gate = torch.tensor(
            [
                [0.4, 0.3, 0.2, 0.1],
                [0.4, 0.3, 0.2, 0.1],
                [0.4, 0.3, 0.2, 0.1],
                [0.4, 0.3, 0.2, 0.1],
            ]
        )
        selected = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        )
        accumulator.update({"fine": gate}, {"fine": selected})
        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["routing_fine_hard_load_0"], 1.0)
        self.assertAlmostEqual(metrics["routing_fine_hard_load_1"], 1.0)
        self.assertAlmostEqual(metrics["routing_fine_hard_load_2"], 0.0)
        self.assertAlmostEqual(metrics["routing_fine_hard_load_3"], 0.0)
        self.assertAlmostEqual(metrics["routing_fine_dead_expert_rate"], 0.5)
        self.assertAlmostEqual(
            metrics["routing_fine_always_selected_rate"], 0.5
        )
        self.assertAlmostEqual(metrics["routing_all_hard_load_cv"], 1.0)
        self.assertAlmostEqual(metrics["routing_scales_mean_hard_load_cv"], 1.0)
        self.assertAlmostEqual(metrics["routing_scales_max_hard_load_cv"], 1.0)
        self.assertGreater(metrics["routing_all_soft_hard_l1_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
