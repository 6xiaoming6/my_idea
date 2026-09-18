from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.temporal_spatial_coe import TemporalSpatialCoE
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator
from stmoe_imputer.utils.checkpoint import load_checkpoint, save_checkpoint


ROOT = Path(__file__).resolve().parents[1]
POOLS = {
    2: ("T", "S"),
    4: ("T", "S", "TD", "SD"),
    6: ("T", "S", "TD", "SD", "TA", "ST"),
}
CANDIDATES = ((3, 4), (4, 6))


def compact_config(steps: int = 4, experts: int = 6, **overrides) -> dict:
    cfg = json.loads((ROOT / "configs/v24/smoke.json").read_text(encoding="utf-8"))
    coe = {
        "num_steps": steps,
        "expert_pool": list(POOLS[experts]),
        "temporal_dilation": 2,
        "spatial_dilation": 2,
        "attention_heads": 4,
        **overrides,
    }
    # Intentionally keep smoke.json's old two-step fixed_path. Dynamic routing
    # must permit changing chain depth without also changing an unused path.
    return deep_update(cfg, {"model": {"main": {"dim": 8}, "coe": coe}})


def make_batch(shape=(2, 2, 5, 3, 5)) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    target = torch.randn(shape, generator=generator)
    mask = (torch.rand(shape, generator=generator) > 0.5).float()
    return {
        "x_f_gt": target,
        "x_f_obs": torch.where(mask.bool(), target, 0.0),
        "m_f": mask,
    }


def tensor_leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensor_leaves(child)


class V24DeeperCoETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        torch.manual_seed(11)

    def assert_module_has_task_gradient(self, module: torch.nn.Module) -> None:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        self.assertTrue(gradients, type(module).__name__)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0)

    def test_both_candidates_train_every_router_from_final_hidden_target_loss(self) -> None:
        for steps, count in CANDIDATES:
            with self.subTest(steps=steps, experts=count):
                cfg = compact_config(steps, count)
                cfg["loss"]["lambda_coe_balance"] = 0.0
                model = DualBranchSTImputer.from_config(cfg).train()
                batch = make_batch()
                output = model(batch)
                loss, _ = compute_main_stage_loss(output, batch, cfg)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for router in model.main_branch.routers:
                    self.assert_module_has_task_gradient(router)
                self.assert_module_has_task_gradient(model.main_branch.shared_expert)
                coe = output["coe"]
                self.assertEqual(tuple(coe["expert_names"]), POOLS[count])
                self.assertEqual(len(coe["predictions"]), steps)
                for name in ("route_logits", "route_probs", "route_weights"):
                    self.assertEqual(coe[name].shape, (2, steps, count))
                weights = coe["route_weights"].detach()
                self.assertTrue(((weights == 0) | (weights == 1)).all())
                torch.testing.assert_close(weights.sum(-1), torch.ones(2, steps))
                self.assertTrue(coe["paths_are_discrete"])

    def test_soft_candidates_train_all_expert_patterns(self) -> None:
        for steps, count in CANDIDATES:
            with self.subTest(steps=steps, experts=count):
                cfg = compact_config(steps, count, routing_mode="soft")
                cfg["loss"]["lambda_coe_balance"] = 0.0
                model = DualBranchSTImputer.from_config(cfg).train()
                batch = make_batch()
                output = model(batch)
                loss, _ = compute_main_stage_loss(output, batch, cfg)
                loss.backward()
                for module in (*model.main_branch.routers, *model.main_branch.routed_experts()):
                    self.assert_module_has_task_gradient(module)
                weights = output["coe"]["route_weights"].detach()
                self.assertTrue(((weights > 0) & (weights < 1)).all())
                torch.testing.assert_close(weights.sum(-1), torch.ones(2, steps))
                self.assertFalse(output["coe"]["paths_are_discrete"])

    def test_hard_eval_dispatches_only_selected_windows_for_all_six_patterns(self) -> None:
        model = TemporalSpatialCoE.from_config(compact_config()).eval()
        expected_paths = torch.tensor([[0, 2, 4, 5], [1, 3, 5, 4]])
        calls = {name: [] for name in POOLS[6]}
        handles = []
        for name, expert in zip(model.expert_names, model.routed_experts()):
            def capture(_module, args, name=name):
                calls[name].append(tuple(args[0].shape))
            handles.append(expert.register_forward_pre_hook(capture))
        for step, router in enumerate(model.routers):
            def force_path(_module, _args, logits, step=step):
                forced = torch.full_like(logits, -10.0)
                return forced.scatter(1, expected_paths[:, step, None], 10.0)
            handles.append(router.register_forward_hook(force_path))
        batch = make_batch()
        try:
            with torch.no_grad():
                result = model(batch["x_f_obs"], batch["m_f"])
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(result["coe"]["paths"], expected_paths)
        for name in POOLS[6]:
            # Dispatch retains the complete T/H/W context for each chosen sample.
            expected_calls = 2 if name in {"TA", "ST"} else 1
            self.assertEqual(calls[name], [(1, 8, 5, 3, 5)] * expected_calls)

    def test_fixed_paths_activate_each_named_expert_and_respect_reordered_pool(self) -> None:
        pool = ["ST", "TD", "T", "TA", "S", "SD"]
        batch = make_batch()
        for path in (["T", "S", "TD", "SD"], ["TA", "ST", "TA", "ST"]):
            with self.subTest(path=path):
                cfg = compact_config(routing_mode="fixed", fixed_path=path, expert_pool=pool)
                model = TemporalSpatialCoE.from_config(cfg).eval()
                with torch.no_grad():
                    baseline = model(batch["x_f_obs"], batch["m_f"])
                self.assertEqual(baseline["coe"]["paths"].tolist(), [[pool.index(x) for x in path]] * 2)
                experts = dict(zip(model.expert_names, model.routed_experts()))
                for name in set(path):
                    handle = experts[name].register_forward_hook(
                        lambda _module, _args, output: torch.zeros_like(output)
                    )
                    try:
                        with torch.no_grad():
                            ablated = model(batch["x_f_obs"], batch["m_f"])
                    finally:
                        handle.remove()
                    difference = (baseline["x_hat_main"] - ablated["x_hat_main"]).abs().max()
                    self.assertGreater(float(difference), 1e-7, name)

    def test_later_routers_consume_updated_states_in_both_candidates(self) -> None:
        batch = make_batch()
        for steps, count in CANDIDATES:
            with self.subTest(steps=steps, experts=count):
                dynamic = TemporalSpatialCoE.from_config(compact_config(steps, count, routing_mode="soft")).eval()
                initial = TemporalSpatialCoE.from_config(
                    compact_config(steps, count, routing_mode="soft", router_state="initial")
                ).eval()
                initial.load_state_dict(dynamic.state_dict(), strict=True)
                captured = {"dynamic": [], "initial": []}
                handles = []
                for name, model in (("dynamic", dynamic), ("initial", initial)):
                    for router in model.routers:
                        def capture(_module, args, name=name):
                            captured[name].append(args[0].detach().clone())
                        handles.append(router.register_forward_pre_hook(capture))
                try:
                    with torch.no_grad():
                        dynamic(batch["x_f_obs"], batch["m_f"])
                        initial(batch["x_f_obs"], batch["m_f"])
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(len(captured["dynamic"]), steps)
                torch.testing.assert_close(captured["dynamic"][0], captured["initial"][0], rtol=0, atol=0)
                for step in range(1, steps):
                    torch.testing.assert_close(captured["initial"][step], captured["initial"][0], rtol=0, atol=0)
                    self.assertFalse(torch.equal(captured["dynamic"][step], captured["dynamic"][step - 1]))
                    self.assertFalse(torch.equal(captured["dynamic"][step], captured["initial"][step]))

    def test_deeper_candidates_ignore_hidden_payload_and_preserve_all_observations(self) -> None:
        batch = make_batch()
        observed = batch["m_f"].bool()
        for steps, count in CANDIDATES:
            with self.subTest(steps=steps, experts=count):
                model = DualBranchSTImputer.from_config(compact_config(steps, count, routing_mode="soft")).eval()
                with torch.no_grad():
                    baseline = model(batch)
                    for payload in (100000.0, float("nan"), float("inf")):
                        changed = {key: value.clone() for key, value in batch.items()}
                        changed["x_f_obs"][~observed] = payload
                        changed["x_f_gt"][~observed] = payload
                        changed["target_mask"] = (~observed).float()
                        changed["available_mask"] = torch.zeros_like(batch["m_f"])
                        actual = model(changed)
                        for left, right in zip(tensor_leaves(baseline), tensor_leaves(actual)):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                completions = [baseline["coe"]["initial_completion"], *baseline["coe"]["completions"], baseline["x_comp"]]
                for completion in completions:
                    torch.testing.assert_close(completion[observed], batch["x_f_obs"][observed], rtol=0, atol=0)
                torch.testing.assert_close(baseline["coe"]["observation_mask"], batch["m_f"], rtol=0, atol=0)

    def test_all_experts_handle_extreme_masks_odd_grids_and_single_points(self) -> None:
        for steps, count in CANDIDATES:
            cfg = compact_config(steps, count, routing_mode="soft")
            model = DualBranchSTImputer.from_config(cfg).eval()
            for shape in ((2, 2, 5, 3, 5), (1, 2, 1, 1, 1)):
                for visible in (False, True):
                    with self.subTest(steps=steps, experts=count, shape=shape, visible=visible):
                        batch = make_batch(shape)
                        batch["m_f"].fill_(float(visible))
                        batch["x_f_obs"] = batch["x_f_gt"].clone() if visible else torch.full_like(batch["x_f_gt"], float("nan"))
                        with torch.no_grad():
                            result = model(batch)
                            loss, _ = compute_main_stage_loss(result, batch, cfg)
                        self.assertEqual(result["x_comp"].shape, shape)
                        self.assertTrue(all(torch.isfinite(t).all() for t in tensor_leaves(result)))
                        self.assertTrue(torch.isfinite(loss))
                        if visible:
                            self.assertEqual(float(loss), 0.0)
                            torch.testing.assert_close(result["x_comp"], batch["x_f_gt"], rtol=0, atol=0)

    def test_temporal_patterns_keep_locations_independent_and_cover_distinct_offsets(self) -> None:
        model = TemporalSpatialCoE.from_config(compact_config()).eval()
        experts = dict(zip(model.expert_names, model.routed_experts()))
        source = torch.randn(1, 8, 9, 5, 9)
        changed = source.clone()
        changed[:, 0, 4, 2, 4] += 3.0
        for name in ("TD", "TA"):
            with self.subTest(expert=name), torch.no_grad():
                difference = experts[name](changed) - experts[name](source)
                self.assertGreater(float(difference[:, :, :, 2, 4].abs().max()), 0.0)
                if name == "TD":
                    self.assertGreater(float(difference[:, :, 2, 2, 4].abs().max()), 0.0)
                    self.assertGreater(float(difference[:, :, 6, 2, 4].abs().max()), 0.0)
                    self.assertEqual(float(difference[:, :, 3, 2, 4].abs().max()), 0.0)
                    self.assertEqual(float(difference[:, :, 0, 2, 4].abs().max()), 0.0)
                else:
                    self.assertGreater(float(difference[:, :, 0, 2, 4].abs().max()), 0.0)
                    self.assertGreater(float(difference[:, :, 8, 2, 4].abs().max()), 0.0)
                difference[:, :, :, 2, 4] = 0
                self.assertEqual(float(difference.abs().max()), 0.0)

    def test_spatial_dilation_and_joint_expert_have_distinct_contexts(self) -> None:
        model = TemporalSpatialCoE.from_config(compact_config()).eval()
        experts = dict(zip(model.expert_names, model.routed_experts()))
        source = torch.randn(1, 8, 7, 5, 9)
        changed = source.clone()
        changed[:, 0, 3, 2, 4] += 3.0
        with torch.no_grad():
            spatial_difference = experts["SD"](changed) - experts["SD"](source)
            joint_difference = experts["ST"](changed) - experts["ST"](source)
        self.assertGreater(float(spatial_difference[:, :, 3, 0, 2].abs().max()), 0.0)
        self.assertEqual(float(spatial_difference[:, :, 3, 1, 3].abs().max()), 0.0)
        spatial_difference[:, :, 3] = 0
        self.assertEqual(float(spatial_difference.abs().max()), 0.0)
        # A single ST application reaches a different time AND grid cell.
        self.assertGreater(float(joint_difference[:, :, 2, 1, 3].abs().max()), 0.0)
        self.assertEqual(float(joint_difference[:, :, 0].abs().max()), 0.0)

    def test_candidate_checkpoints_reload_with_identical_predictions_and_routes(self) -> None:
        batch = make_batch()
        for steps, count in CANDIDATES:
            with self.subTest(steps=steps, experts=count), tempfile.TemporaryDirectory() as directory:
                cfg = compact_config(steps, count)
                model = DualBranchSTImputer.from_config(cfg).eval()
                with torch.no_grad():
                    expected = model(batch)
                path = Path(directory) / "candidate.pt"
                save_checkpoint(path, model, None, 3, {"mae": 0.5}, cfg)
                restored = DualBranchSTImputer.from_config(cfg).eval()
                checkpoint = load_checkpoint(path, restored)
                self.assertEqual(checkpoint["config"], cfg)
                self.assertEqual(checkpoint["epoch"], 3)
                with torch.no_grad():
                    actual = restored(batch)
                self.assertEqual(tuple(actual["coe"]["expert_names"]), POOLS[count])
                for left, right in zip(tensor_leaves(expected), tensor_leaves(actual)):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                reordered_cfg = deep_update(cfg, {"model": {"coe": {"expert_pool": list(reversed(POOLS[count]))}}})
                reordered = DualBranchSTImputer.from_config(reordered_cfg).eval()
                with self.assertRaisesRegex(ValueError, "expert_pool order"):
                    load_checkpoint(path, reordered)

    def test_default_pool_keeps_original_two_expert_parameter_names(self) -> None:
        cfg = compact_config(2, 2)
        del cfg["model"]["coe"]["expert_pool"]
        original = TemporalSpatialCoE.from_config(cfg).eval()
        explicit = TemporalSpatialCoE.from_config(compact_config(2, 2)).eval()
        state = original.state_dict()
        self.assertEqual(set(state), set(explicit.state_dict()))
        self.assertIn("temporal_expert.network.0.norm.weight", state)
        self.assertIn("spatial_expert.network.0.norm.weight", state)
        self.assertEqual(state["routers.0.3.weight"].shape[0], 2)
        self.assertFalse(any(name.startswith("pattern_experts.") for name in state))
        explicit.load_state_dict(state, strict=True)
        self.assertEqual(tuple(original.expert_names), POOLS[2])
        self.assertEqual(original.routed_experts(), (original.temporal_expert, original.spatial_expert))
        batch = make_batch()
        with torch.no_grad():
            expected = original(batch["x_f_obs"], batch["m_f"])
            actual = explicit(batch["x_f_obs"], batch["m_f"])
        torch.testing.assert_close(expected["x_hat_main"], actual["x_hat_main"], rtol=0, atol=0)

    def test_custom_pattern_subset_excludes_unused_experts_from_parameters(self) -> None:
        cfg = compact_config(expert_pool=["TA", "ST"], routing_mode="soft")
        model = DualBranchSTImputer.from_config(cfg).train()
        backbone = model.main_branch
        self.assertIsNone(backbone.temporal_expert)
        self.assertIsNone(backbone.spatial_expert)
        self.assertFalse(any(name.startswith(("temporal_expert.", "spatial_expert.")) for name in backbone.state_dict()))
        self.assertEqual(tuple(backbone.expert_names), ("TA", "ST"))
        self.assertEqual(len(backbone.routed_experts()), 2)
        batch = make_batch()
        output = model(batch)
        loss, _ = compute_main_stage_loss(output, batch, cfg)
        loss.backward()
        self.assertEqual(output["coe"]["route_weights"].shape, (2, 4, 2))
        for expert in backbone.routed_experts():
            self.assert_module_has_task_gradient(expert)

    def test_candidate_and_full_depth_pool_grid_configs_run_without_multiscale(self) -> None:
        base = compact_config(2, 2)
        root = ROOT / "configs/v24/candidates"
        expected = {"chain3.json": (3, 4), "chain4.json": (4, 6)}
        expected.update({f"ablation_grid/k{k}_e{e}.json": (k, e) for k in (2, 3, 4) for e in (2, 4, 6)})
        batch = make_batch()
        for filename, (steps, count) in expected.items():
            with self.subTest(config=filename):
                override = json.loads((root / filename).read_text(encoding="utf-8"))
                cfg = deep_update(base, override)
                self.assertEqual(cfg["model"]["coe"]["num_steps"], steps)
                self.assertEqual(tuple(cfg["model"]["coe"]["expert_pool"]), POOLS[count])
                self.assertFalse(cfg["data"]["multiscale"])
                model = DualBranchSTImputer.from_config(cfg).eval()
                self.assertFalse(model.main_branch.requires_multiscale)
                with torch.no_grad():
                    outputs = model(batch)
                self.assertEqual(outputs["coe"]["route_weights"].shape, (2, steps, count))
                self.assertTrue(torch.isfinite(outputs["x_hat_main"]).all())

    def test_invalid_pools_and_fixed_paths_fail_before_training(self) -> None:
        for pool in ([], ["T", "unknown"], ["T", "T"], "TS"):
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                TemporalSpatialCoE.from_config(compact_config(expert_pool=pool))
        for path in (["T", "S"], ["T", "S", "TA", "unknown"], ["T", "S", "TD", "TA"]):
            with self.subTest(path=path), self.assertRaises(ValueError):
                TemporalSpatialCoE.from_config(compact_config(4, 4, routing_mode="fixed", fixed_path=path))
        automatic = TemporalSpatialCoE.from_config(compact_config(4, 6, routing_mode="fixed", fixed_path=None)).eval()
        batch = make_batch()
        with torch.no_grad():
            output = automatic(batch["x_f_obs"], batch["m_f"])
        self.assertEqual(output["coe"]["paths"].shape, (2, 4))

    def test_multi_pattern_metrics_count_windows_and_name_paths_unambiguously(self) -> None:
        names = POOLS[6]
        paths = torch.tensor([[0, 2, 4, 5], [0, 2, 4, 5], [1, 3, 5, 4]])
        weights = torch.nn.functional.one_hot(paths, num_classes=len(names)).float()
        accumulator = CoERoutingMetricAccumulator()
        for indices in (slice(0, 2), slice(2, 3)):
            accumulator.update({
                "expert_names": names,
                "route_weights": weights[indices],
                "route_probs": weights[indices],
                "routing_mode": "hard",
                "paths": paths[indices],
                "paths_are_discrete": True,
            })
        metrics = accumulator.compute()
        self.assertEqual(metrics["coe_routing_sample_count"], 3)
        self.assertEqual(metrics["coe_num_experts"], 6)
        self.assertEqual(metrics["coe_num_steps"], 4)
        self.assertEqual(metrics["coe_path_unique_count"], 2)
        self.assertAlmostEqual(metrics["coe_path_max_fraction"], 2 / 3)
        self.assertAlmostEqual(metrics["coe_path_entropy"], -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)))
        self.assertAlmostEqual(metrics["coe_path_T__TD__TA__ST_fraction"], 2 / 3)
        self.assertAlmostEqual(metrics["coe_path_S__SD__ST__TA_fraction"], 1 / 3)
        for step in range(4):
            self.assertAlmostEqual(sum(metrics[f"coe_step{step + 1}_{name}_usage"] for name in names), 1.0)
            for expert, name in enumerate(names):
                fraction = float((paths[:, step] == expert).float().mean())
                self.assertAlmostEqual(metrics[f"coe_step{step + 1}_{name}_usage"], fraction, places=6)
                self.assertAlmostEqual(metrics[f"coe_step{step + 1}_{name}_prob"], fraction, places=6)
        soft = CoERoutingMetricAccumulator()
        mixture = 0.5 * weights + 0.5 / len(names)
        soft.update({
            "expert_names": names, "route_weights": mixture, "route_probs": mixture,
            "routing_mode": "soft", "paths": paths, "paths_are_discrete": False,
        })
        soft_metrics = soft.compute()
        self.assertFalse(any("_path_" in name or name.endswith("_usage") for name in soft_metrics))
        self.assertAlmostEqual(soft_metrics["coe_step3_TA_weight"], 0.5 * 2 / 3 + 0.5 / 6, places=6)

    def test_metrics_reject_mixing_candidate_metadata_within_one_evaluation(self) -> None:
        paths = torch.tensor([[0, 2, 4, 5]])
        weights = torch.nn.functional.one_hot(paths, num_classes=6).float()
        record = {
            "expert_names": POOLS[6], "route_weights": weights,
            "route_probs": weights, "routing_mode": "hard", "paths": paths,
        }
        accumulator = CoERoutingMetricAccumulator()
        accumulator.update(record)
        expected = accumulator.compute()
        for change in (
            {"expert_names": tuple(reversed(POOLS[6]))},
            {"routing_mode": "soft"},
            {"route_weights": weights[:, :3], "route_probs": weights[:, :3], "paths": paths[:, :3]},
        ):
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                accumulator.update({**record, **change})
            self.assertEqual(accumulator.compute(), expected)


if __name__ == "__main__":
    unittest.main()
