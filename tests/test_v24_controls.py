from __future__ import annotations

import unittest

import torch

from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE


def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(29)
    target = torch.randn(2, 2, 4, 3, 5, generator=generator)
    mask = (torch.rand(target.shape, generator=generator) > 0.45).float()
    return torch.where(mask.bool(), target, 0.0), mask, target


class V24MechanismControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        torch.manual_seed(31)

    @staticmethod
    def capture_inputs(model: TemporalSpatialCoE, x: torch.Tensor, mask: torch.Tensor):
        captured = {"routers": [], "shared": [], "projection": []}
        captured.update({name: [] for name in model.expert_names})
        handles = []
        modules = [
            *[("routers", router) for router in model.routers],
            ("shared", model.shared_expert),
            ("projection", model.state_projection),
            *zip(model.expert_names, model.routed_experts()),
        ]
        for name, module in modules:
            def save_input(_module, args, name=name):
                captured[name].append(args[0].detach().clone())
            handles.append(module.register_forward_pre_hook(save_input))
        try:
            result = model(x, mask)
        finally:
            for handle in handles:
                handle.remove()
        return result, captured

    def assert_task_gradient(self, module: torch.nn.Module) -> None:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_frozen_expert_inputs_preserve_candidate_calls_and_evolving_router_states(self) -> None:
        x, mask, target = make_inputs()
        for steps in (2, 3, 4):
            with self.subTest(steps=steps):
                dynamic = TemporalSpatialCoE(2, dim=8, num_steps=steps).train()
                frozen = TemporalSpatialCoE(
                    2, dim=8, num_steps=steps, expert_state="initial"
                ).train()
                frozen.load_state_dict(dynamic.state_dict(), strict=True)
                torch.manual_seed(37)
                dynamic_output, dynamic_inputs = self.capture_inputs(dynamic, x, mask)
                torch.manual_seed(37)
                frozen_output, frozen_inputs = self.capture_inputs(frozen, x, mask)
                for name in ("T", "S", "shared", "projection"):
                    self.assertEqual(len(dynamic_inputs[name]), steps)
                    self.assertEqual(len(frozen_inputs[name]), steps)
                    torch.testing.assert_close(
                        frozen_inputs[name][0], dynamic_inputs[name][0], rtol=0, atol=0
                    )
                    for step in range(1, steps):
                        torch.testing.assert_close(
                            frozen_inputs[name][step], frozen_inputs[name][0], rtol=0, atol=0
                        )
                        self.assertFalse(torch.equal(dynamic_inputs[name][step], dynamic_inputs[name][0]))
                for step in range(1, steps):
                    # Freezing expert input does not also freeze the router or
                    # discard accumulated updates to the hidden representation.
                    self.assertFalse(torch.equal(frozen_inputs["routers"][step], frozen_inputs["routers"][0]))
                    self.assertFalse(torch.equal(
                        frozen_output["coe"]["predictions"][step],
                        frozen_output["coe"]["predictions"][step - 1],
                    ))
                self.assertFalse(torch.equal(dynamic_output["x_hat_main"], frozen_output["x_hat_main"]))
                self.assertEqual(len({id(router) for router in frozen.routers}), steps)
                parameter_ids = [{id(p) for p in router.parameters()} for router in frozen.routers]
                for left in range(steps):
                    for right in range(left + 1, steps):
                        self.assertTrue(parameter_ids[left].isdisjoint(parameter_ids[right]))
                missing = ~mask.bool()
                (frozen_output["x_hat_main"][missing] - target[missing]).square().mean().backward()
                for router in frozen.routers:
                    self.assert_task_gradient(router)
                self.assert_task_gradient(frozen.encoder)
                self.assert_task_gradient(frozen.state_projection)

    def test_router_and_expert_state_policies_are_independent(self) -> None:
        x, mask, _ = make_inputs()
        for expert_state in ("initial", "dynamic"):
            with self.subTest(expert_state=expert_state):
                model = TemporalSpatialCoE(
                    2, dim=8, num_steps=3, routing_mode="soft",
                    router_state="initial", expert_state=expert_state,
                ).eval()
                output, captured = self.capture_inputs(model, x, mask)
                for features in captured["routers"][1:]:
                    torch.testing.assert_close(features, captured["routers"][0], rtol=0, atol=0)
                for name in ("T", "S", "shared"):
                    same = torch.equal(captured[name][-1], captured[name][0])
                    self.assertEqual(same, expert_state == "initial")
                self.assertEqual(output["coe"]["expert_state"], expert_state)

    def test_parallel_executes_each_expert_once_on_same_input_with_soft_weights(self) -> None:
        x, mask, target = make_inputs()
        parallel = TemporalSpatialCoE(2, dim=8, num_steps=1, routing_mode="parallel", temperature=0.7)
        soft = TemporalSpatialCoE(2, dim=8, num_steps=1, routing_mode="soft", temperature=0.7)
        soft.load_state_dict(parallel.state_dict(), strict=True)
        for training in (True, False):
            with self.subTest(training=training):
                parallel.train(training)
                soft.train(training)
                output, captured = self.capture_inputs(parallel, x, mask)
                for name in captured:
                    self.assertEqual(len(captured[name]), 1, name)
                for name in ("T", "S"):
                    torch.testing.assert_close(captured[name][0], captured["shared"][0], rtol=0, atol=0)
                reference = soft(x, mask)
                torch.testing.assert_close(output["x_hat_main"], reference["x_hat_main"], rtol=0, atol=0)
                self.assertEqual(output["coe"]["routing_mode"], "parallel")
                self.assertFalse(output["coe"]["paths_are_discrete"])
                weights = output["coe"]["route_weights"]
                self.assertEqual(weights.shape, (2, 1, 2))
                self.assertTrue(((weights > 0) & (weights < 1)).all())
                torch.testing.assert_close(weights.sum(-1), torch.ones(2, 1))
        missing = ~mask.bool()
        (output["x_hat_main"][missing] - target[missing]).square().mean().backward()
        for module in (*parallel.routers, *parallel.routed_experts(), parallel.shared_expert):
            self.assert_task_gradient(module)

    def test_controls_preserve_observations_and_ignore_missing_payloads(self) -> None:
        x, mask, _ = make_inputs()
        for kwargs in (
            {"num_steps": 3, "expert_state": "initial", "routing_mode": "soft"},
            {"num_steps": 1, "routing_mode": "parallel"},
        ):
            with self.subTest(kwargs=kwargs):
                model = TemporalSpatialCoE(2, dim=8, **kwargs).eval()
                output = model(x, mask)
                poisoned = x.clone()
                poisoned[~mask.bool()] = float("nan")
                actual = model(poisoned, mask)
                torch.testing.assert_close(output["x_hat_main"], actual["x_hat_main"], rtol=0, atol=0)
                for completion in [output["coe"]["initial_completion"], *output["coe"]["completions"]]:
                    torch.testing.assert_close(completion[mask.bool()], x[mask.bool()], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(output["x_hat_main"]).all())

    def test_default_state_policy_and_checkpoint_parameter_layout_are_unchanged(self) -> None:
        x, mask, _ = make_inputs()
        default = TemporalSpatialCoE(2, dim=8, routing_mode="soft").eval()
        explicit = TemporalSpatialCoE(2, dim=8, routing_mode="soft", expert_state="dynamic").eval()
        frozen = TemporalSpatialCoE(2, dim=8, routing_mode="soft", expert_state="initial").eval()
        explicit.load_state_dict(default.state_dict(), strict=True)
        frozen.load_state_dict(default.state_dict(), strict=True)
        self.assertEqual(tuple(default.state_dict()), tuple(frozen.state_dict()))
        for key, value in default.state_dict().items():
            self.assertEqual(value.shape, frozen.state_dict()[key].shape)
        torch.testing.assert_close(default(x, mask)["x_hat_main"], explicit(x, mask)["x_hat_main"], rtol=0, atol=0)
        config_model = TemporalSpatialCoE.from_config({"model": {"c_in": 2, "coe": {
            "dim": 8, "num_steps": 3, "expert_state": "initial", "router_state": "dynamic",
        }}})
        self.assertEqual(config_model.expert_state, "initial")
        self.assertEqual(config_model.router_state, "dynamic")

    def test_controls_reject_invalid_state_and_multiround_parallel(self) -> None:
        with self.assertRaisesRegex(ValueError, "expert_state"):
            TemporalSpatialCoE(2, dim=8, expert_state="stale")
        for steps in (2, 3, 4):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "num_steps=1"):
                TemporalSpatialCoE(2, dim=8, num_steps=steps, routing_mode="parallel")


if __name__ == "__main__":
    unittest.main()
