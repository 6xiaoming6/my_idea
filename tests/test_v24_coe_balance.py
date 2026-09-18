from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from stmoe_imputer.losses import coe_load_balance_per_step, compute_coe_loss
from stmoe_imputer.models import DualBranchSTImputer
from test_v24_deeper_coe import CANDIDATES, compact_config, make_batch


def loss_fixture(*, steps: int = 3, experts: int = 4, mode: str = "hard"):
    logits = torch.linspace(1.0, -1.0, experts).repeat(3, steps, 1).requires_grad_()
    probabilities = logits.softmax(-1)
    paths = torch.zeros(3, steps, dtype=torch.long)
    weights = probabilities if mode == "soft" else F.one_hot(paths, experts).float()
    prediction = torch.ones(3, 1, 1, 1, 2, requires_grad=True)
    outputs = {
        "x_hat_main": prediction,
        "coe": {
            "predictions": [prediction] * steps,
            "route_probs": probabilities,
            "route_weights": weights,
            "routing_mode": mode,
            "use_routed": True,
            "num_steps": steps,
            "num_experts": experts,
        },
    }
    batch = {"x_f_gt": torch.zeros_like(prediction), "m_f": torch.zeros_like(prediction)}
    cfg = {"loss": {"type": "l1", "lambda_coe_mid": 0.0, "lambda_coe_balance": 0.001}}
    return outputs, batch, cfg, logits


class CoELoadBalanceTest(unittest.TestCase):
    def test_balanced_confident_specialists_have_lower_penalty_than_collapse(self) -> None:
        # Each window can route confidently: only aggregate usage is balanced.
        balanced = torch.eye(4).unsqueeze(1).repeat(1, 3, 1)
        collapsed = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(4, 3, 1)
        torch.testing.assert_close(coe_load_balance_per_step(balanced, balanced), torch.ones(3))
        torch.testing.assert_close(coe_load_balance_per_step(collapsed, collapsed), torch.full((3,), 4.0))

    def test_penalty_uses_actual_sampled_load_instead_of_probability_argmax(self) -> None:
        probabilities = torch.tensor([0.9, 0.1]).repeat(2, 1, 1)
        # Gumbel sampling can pick the less likely expert in every window.
        sampled_weights = torch.tensor([0.0, 1.0]).repeat(2, 1, 1)
        torch.testing.assert_close(
            coe_load_balance_per_step(probabilities, sampled_weights), torch.tensor([0.2]),
        )

    def test_gradient_discourages_overloaded_expert_and_detaches_load(self) -> None:
        logits = torch.tensor([1.0, 0.0, -1.0]).repeat(4, 2, 1).requires_grad_()
        weights = torch.tensor([1.0, 0.0, 0.0]).repeat(4, 2, 1).requires_grad_()
        coe_load_balance_per_step(logits.softmax(-1), weights).mean().backward()
        self.assertIsNone(weights.grad)
        self.assertTrue((logits.grad[..., 0] > 0).all())
        self.assertTrue((logits.grad[..., 1:] < 0).all())

    def test_soft_usage_is_detached_even_when_it_aliases_probabilities(self) -> None:
        probabilities = torch.tensor([0.8, 0.2]).repeat(2, 1, 1).requires_grad_()
        coe_load_balance_per_step(probabilities, probabilities).sum().backward()
        torch.testing.assert_close(probabilities.grad, torch.tensor([0.8, 0.2]).repeat(2, 1, 1))

    def test_repeating_batch_or_chain_does_not_change_mean_penalty(self) -> None:
        probabilities = torch.tensor([[[0.7, 0.2, 0.1]], [[0.4, 0.35, 0.25]]])
        weights = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
        expected = coe_load_balance_per_step(probabilities, weights).mean()
        for batch_repeats, steps in ((1, 2), (3, 3), (4, 4)):
            with self.subTest(batch_repeats=batch_repeats, steps=steps):
                actual = coe_load_balance_per_step(
                    probabilities.repeat(batch_repeats, steps, 1),
                    weights.repeat(batch_repeats, steps, 1),
                ).mean()
                torch.testing.assert_close(actual, expected)

    def test_rejects_mismatched_or_empty_dimensions(self) -> None:
        for probabilities, weights in (
            (torch.ones(2, 3), torch.ones(2, 3)),
            (torch.ones(2, 3, 4), torch.ones(2, 3, 3)),
            (torch.ones(0, 3, 4), torch.ones(0, 3, 4)),
            (torch.ones(2, 0, 4), torch.ones(2, 0, 4)),
            (torch.ones(2, 3, 0), torch.ones(2, 3, 0)),
        ):
            with self.subTest(shape=probabilities.shape, other=weights.shape), self.assertRaises(ValueError):
                coe_load_balance_per_step(probabilities, weights)

    def test_reduction_stays_float32_for_low_precision_inputs(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                probabilities = torch.tensor([0.75, 0.25], dtype=dtype).repeat(4, 4, 1).requires_grad_()
                weights = torch.tensor([1.0, 0.0], dtype=dtype).repeat(4, 4, 1)
                result = coe_load_balance_per_step(probabilities, weights)
                self.assertEqual(result.dtype, torch.float32)
                torch.testing.assert_close(result, torch.full((4,), 1.5))
                result.mean().backward()
                self.assertTrue(torch.isfinite(probabilities.grad).all())


class CoELoadBalanceIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def test_weighted_term_is_added_once_and_each_round_is_logged(self) -> None:
        for mode in ("hard", "soft"):
            with self.subTest(mode=mode):
                outputs, batch, cfg, _ = loss_fixture(mode=mode)
                loss, logs = compute_coe_loss(outputs, batch, cfg)
                per_step = coe_load_balance_per_step(
                    outputs["coe"]["route_probs"], outputs["coe"]["route_weights"],
                )
                torch.testing.assert_close(loss, torch.tensor(1.0) + 0.001 * per_step.mean())
                torch.testing.assert_close(logs["l_coe_balance"], per_step.mean())
                torch.testing.assert_close(logs["l_coe_balance_weighted"], 0.001 * per_step.mean())
                for step, expected in enumerate(per_step, start=1):
                    torch.testing.assert_close(logs[f"l_coe_balance_step{step}"], expected)
                self.assertTrue(all(not value.requires_grad for value in logs.values()))

    def test_zero_or_absent_weight_logs_raw_term_without_router_gradient(self) -> None:
        for missing in (False, True):
            with self.subTest(missing=missing):
                outputs, batch, cfg, logits = loss_fixture()
                if missing:
                    cfg["loss"].pop("lambda_coe_balance")
                else:
                    cfg["loss"]["lambda_coe_balance"] = 0.0
                loss, logs = compute_coe_loss(outputs, batch, cfg)
                self.assertEqual(float(loss), 1.0)
                self.assertGreater(float(logs["l_coe_balance"]), 0.0)
                self.assertEqual(float(logs["l_coe_balance_weighted"]), 0.0)
                loss.backward()
                self.assertIsNone(logits.grad)

    def test_invalid_balance_weight_is_rejected(self) -> None:
        for weight in (-0.001, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                outputs, batch, cfg, _ = loss_fixture()
                cfg["loss"]["lambda_coe_balance"] = weight
                with self.assertRaisesRegex(ValueError, "lambda_coe_balance"):
                    compute_coe_loss(outputs, batch, cfg)

    def test_fixed_shared_only_and_single_expert_skip_balance(self) -> None:
        for variant in ("fixed", "shared_only", "single_expert"):
            with self.subTest(variant=variant):
                outputs, batch, cfg, logits = loss_fixture(experts=1 if variant == "single_expert" else 4)
                if variant == "fixed":
                    outputs["coe"]["routing_mode"] = "fixed"
                elif variant == "shared_only":
                    outputs["coe"]["use_routed"] = False
                loss, logs = compute_coe_loss(outputs, batch, cfg)
                self.assertEqual(float(loss), 1.0)
                self.assertEqual(float(logs["l_coe_balance"]), 0.0)
                self.assertEqual(float(logs["l_coe_balance_weighted"]), 0.0)
                loss.backward()
                self.assertIsNone(logits.grad)

    def test_no_selected_targets_does_not_apply_auxiliary_gradient(self) -> None:
        for cause in ("observed", "no_holdouts", "nonfinite"):
            with self.subTest(cause=cause):
                outputs, batch, cfg, logits = loss_fixture()
                if cause == "observed":
                    batch["m_f"].fill_(1.0)
                elif cause == "no_holdouts":
                    batch["target_mask"] = torch.zeros_like(batch["m_f"])
                else:
                    batch["x_f_gt"].fill_(float("nan"))
                loss, logs = compute_coe_loss(outputs, batch, cfg)
                self.assertEqual(float(loss), 0.0)
                self.assertEqual(float(logs["l_coe_balance"]), 0.0)
                self.assertEqual(float(logs["l_coe_balance_weighted"]), 0.0)
                loss.backward()
                self.assertIsNone(logits.grad)

    def test_only_windows_with_finite_selected_targets_contribute_to_load(self) -> None:
        outputs, batch, cfg, _ = loss_fixture(steps=2, experts=2)
        probabilities = torch.tensor([[[0.9, 0.1]] * 2, [[0.1, 0.9]] * 2, [[0.1, 0.9]] * 2], requires_grad=True)
        outputs["coe"]["route_probs"] = probabilities
        outputs["coe"]["route_weights"] = F.one_hot(probabilities.argmax(-1), 2).float()
        batch["target_mask"] = torch.ones_like(batch["m_f"])
        batch["target_mask"][1] = 0
        batch["x_f_gt"][2] = float("nan")
        loss, logs = compute_coe_loss(outputs, batch, cfg)
        torch.testing.assert_close(logs["l_coe_balance"], torch.tensor(1.8))
        loss.backward()
        self.assertGreater(float(probabilities.grad[0].abs().sum()), 0.0)
        torch.testing.assert_close(probabilities.grad[1:], torch.zeros_like(probabilities.grad[1:]))

    def test_enabled_balance_rejects_missing_routing_outputs(self) -> None:
        outputs, batch, cfg, _ = loss_fixture()
        outputs["coe"].pop("route_probs")
        with self.assertRaises(ValueError):
            compute_coe_loss(outputs, batch, cfg)

    def test_auxiliary_loss_alone_trains_every_router_in_both_deeper_candidates(self) -> None:
        for steps, experts in CANDIDATES:
            for mode in ("hard", "soft"):
                with self.subTest(steps=steps, experts=experts, mode=mode):
                    torch.manual_seed(91)
                    cfg = compact_config(steps, experts, routing_mode=mode)
                    cfg["loss"]["lambda_coe_balance"] = 0.001
                    model = DualBranchSTImputer.from_config(cfg).train()
                    batch = make_batch()
                    outputs = model(batch)
                    # Exact detached targets make the L1 value and gradient zero,
                    # so any router gradient must come from the auxiliary loss.
                    exact_batch = dict(batch, x_f_gt=outputs["x_hat_main"].detach().clone())
                    loss, logs = compute_coe_loss(outputs, exact_batch, cfg)
                    self.assertEqual(float(logs["l_main"]), 0.0)
                    self.assertGreater(float(loss), 0.0)
                    loss.backward()
                    for step, router in enumerate(model.main_branch.routers, start=1):
                        gradients = [p.grad for p in router.parameters() if p.grad is not None]
                        self.assertTrue(gradients, f"router {step}")
                        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0, f"router {step}")


if __name__ == "__main__":
    unittest.main()
