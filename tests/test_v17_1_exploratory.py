from __future__ import annotations

import json
import math
import unittest
from collections import defaultdict

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.engine import _append_model_diagnostics
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.v_single.fine_preserved_scale_fusion import (
    FinePreservedScaleWeight,
)

from _v17_utils import ROOT, compact_v17_config, make_batch


VARIANT_EXPECTATIONS = {
    "full": {
        "expert_router": "hierarchical_shared_head",
        "fusion": "fine_preserved_parallel",
        "floor": "linear",
        "branch": "sample_residual",
        "unified_scale": True,
    },
    "no_adapter": {"adapter": False},
    "decoupled_expert_router": {"expert_router": "decoupled"},
    "progressive_fusion": {"fusion": "progressive"},
    "no_fine_floor": {"floor": "none"},
    "hard_fine_floor": {"floor": "hard"},
    "global_route_gamma": {"branch": "global_residual"},
    "independent_shared_scale": {"unified_scale": False},
}


class V171ExploratoryTest(unittest.TestCase):
    def _variant_config(self, variant: str) -> dict:
        patch = json.loads(
            (
                ROOT
                / "configs"
                / "v17.1-single"
                / "exploratory"
                / f"{variant}.json"
            ).read_text(encoding="utf-8")
        )
        return deep_update(
            compact_v17_config(),
            deep_update(patch, {"model": {"version": "v17.1-single"}}),
        )

    def test_all_e0_e7_configs_change_only_supported_execution_modes(self) -> None:
        for variant, expected in VARIANT_EXPECTATIONS.items():
            with self.subTest(variant=variant):
                cfg = self._variant_config(variant)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    outputs = model(make_batch(cfg))
                self.assertTrue(torch.isfinite(outputs["x_hat_final"]).all())
                self.assertEqual(
                    outputs["v17_expert_router_mode"],
                    expected.get("expert_router", "hierarchical_shared_head"),
                )
                self.assertEqual(
                    outputs["v17_route_fusion"],
                    expected.get("fusion", "fine_preserved_parallel"),
                )
                self.assertEqual(
                    outputs["v17_fine_floor_mode"],
                    expected.get("floor", "linear"),
                )
                self.assertEqual(
                    outputs["branch_mode"],
                    expected.get("branch", "sample_residual"),
                )
                self.assertEqual(
                    outputs["v17_unified_scale_weight"],
                    expected.get("unified_scale", True),
                )
                self.assertEqual(
                    model.main_branch.adapter_enabled,
                    expected.get("adapter", True),
                )

    def test_decoupled_expert_ablation_keeps_hierarchical_scale_router(self) -> None:
        cfg = self._variant_config("decoupled_expert_router")
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            outputs = model(make_batch(cfg))
        self.assertEqual(outputs["v17_scale_router_mode"], "hierarchical")
        self.assertEqual(outputs["v17_expert_router_mode"], "decoupled")

    def test_fine_floor_modes_follow_the_ablation_contract(self) -> None:
        weights = torch.tensor([[0.40, 0.35, 0.25], [0.10, 0.60, 0.30]])
        active = torch.ones_like(weights, dtype=torch.bool)
        none = FinePreservedScaleWeight(0.25, mode="none")(weights, active)
        hard = FinePreservedScaleWeight(0.25, mode="hard")(weights, active)
        linear = FinePreservedScaleWeight(0.25, mode="linear")(weights, active)
        torch.testing.assert_close(none, weights)
        torch.testing.assert_close(hard[0], weights[0])
        torch.testing.assert_close(hard[1], torch.tensor([0.25, 0.50, 0.25]))
        torch.testing.assert_close(linear[0], torch.tensor([0.55, 0.2625, 0.1875]))
        torch.testing.assert_close(hard.sum(dim=1), torch.ones(2))

    def test_independent_shared_gate_reports_both_scale_decisions(self) -> None:
        cfg = self._variant_config("independent_shared_scale")
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            outputs = model(make_batch(cfg))
        diagnostics = outputs["diagnostics"]["v17"]
        for key in (
            "routed_scale_weight",
            "shared_scale_weight",
            "shared_routed_scale_l1",
            "shared_routed_scale_cosine",
        ):
            self.assertIn(key, diagnostics)
            self.assertTrue(torch.isfinite(diagnostics[key]).all())
        torch.testing.assert_close(
            diagnostics["shared_scale_weight"].sum(dim=1), torch.ones(1)
        )
        torch.testing.assert_close(
            diagnostics["routed_scale_weight"].sum(dim=1), torch.ones(1)
        )

    def test_required_exploratory_diagnostics_are_aggregated(self) -> None:
        cfg = self._variant_config("full")
        cfg = deep_update(cfg, {"model": {"main": {"top_k": 2}}})
        model = DualBranchSTImputer.from_config(cfg).eval()
        with torch.no_grad():
            outputs = model(make_batch(cfg))
        logs: dict[str, list[float]] = defaultdict(list)
        _append_model_diagnostics(logs, outputs)
        required = {
            "v17_effective_scale_count",
            "v17_effective_expert_count_fine",
            "v17_top2_second_weight_fine",
            "v17_same_top1_fine_mid_rate",
            "v17_route_gate_high_saturation",
            "v17_route_gate_low_saturation",
            "v17_shared_routed_scale_l1_mean",
            "v17_effective_routed_ratio",
        }
        self.assertTrue(required.issubset(logs), required.difference(logs))
        for key in required:
            self.assertTrue(math.isfinite(logs[key][0]), key)

    def test_new_ablation_paths_receive_gradients(self) -> None:
        targets = {
            "decoupled_expert_router": "main_branch.router_f",
            "progressive_fusion": "main_branch.route_fusion",
            "global_route_gamma": "main_branch.branch_fusion.route_gamma",
            "independent_shared_scale": "main_branch.cross_scale_shared_expert.scale_gate",
        }
        for variant, prefix in targets.items():
            with self.subTest(variant=variant):
                cfg = self._variant_config(variant)
                cfg = deep_update(cfg, {"model": {"main": {"top_k": 2}}})
                model = DualBranchSTImputer.from_config(cfg).train()
                outputs = model(make_batch(cfg))
                outputs["x_hat_final"].square().mean().backward()
                gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if name == prefix or name.startswith(prefix + ".")
                ]
                self.assertTrue(gradients, prefix)
                self.assertTrue(
                    any(
                        gradient is not None
                        and torch.isfinite(gradient).all()
                        and float(gradient.abs().sum()) > 0.0
                        for gradient in gradients
                    ),
                    prefix,
                )


if __name__ == "__main__":
    unittest.main()
