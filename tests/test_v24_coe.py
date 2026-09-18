from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.data import build_datasets, build_loader, build_test_dataset
from stmoe_imputer.data.transforms import prepare_single_scale
from stmoe_imputer.engine import build_optimizer, evaluate, train_one_epoch
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.models.registry import build_model_backbone
from stmoe_imputer.models.temporal_spatial_coe import (
    DirectionalExpert,
    SUPPORT_FEATURE_NAMES,
    TemporalSpatialCoE,
    compute_observation_support,
)
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator


ROOT = Path(__file__).resolve().parents[1]


def compact_config(**coe_overrides) -> dict:
    cfg = json.loads((ROOT / "configs/v24/smoke.json").read_text(encoding="utf-8"))
    return deep_update(cfg, {
        "model": {"main": {"dim": 8}, "coe": coe_overrides},
        "data": {"synthetic": {"num_train": 2, "num_val": 2, "t": 5, "h": 3, "w": 5}},
    })


def make_batch(channels: int = 2, seed: int = 17) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target = torch.randn(2, channels, 5, 3, 5, generator=generator)
    mask = (torch.rand(2, 1, 5, 3, 5, generator=generator) > 0.5).float()
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
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensor_leaves(child)


class V24CoETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_full_configs_select_single_scale_architecture(self) -> None:
        for name, channels in (("smoke", 2), ("taxibj", 2), ("bikenyc", 2), ("chap_beijing", 1)):
            with self.subTest(config=name):
                cfg = json.loads((ROOT / f"configs/v24/{name}.json").read_text())
                self.assertEqual(cfg["model"]["c_in"], channels)
                self.assertNotIn("scales", cfg["data"])
                self.assertFalse(cfg["data"]["multiscale"])
                self.assertFalse(cfg["model"]["main"]["use_multiscale"])
                self.assertFalse(cfg["model"]["aux"]["enabled"])
                self.assertEqual(cfg["loss"], {
                    "type": "l1", "lambda_coe_mid": 0.0, "lambda_coe_balance": 0.0,
                })
                model = build_model_backbone(cfg)
                self.assertIsInstance(model, TemporalSpatialCoE)
                self.assertFalse(model.requires_multiscale)

    def test_all_ablation_overrides_run_with_original_parameter_layout(self) -> None:
        cfg = compact_config()
        base = DualBranchSTImputer.from_config(cfg).eval()
        batch = make_batch()
        expected = {"fixed_ts", "fixed_st", "fixed_tt", "fixed_ss", "soft", "initial_router", "shared_only", "routed_only", "weak_mid", "no_route_balance"}
        paths = list((ROOT / "configs/v24/ablations").glob("*.json"))
        self.assertEqual({path.stem for path in paths}, expected)
        for path in paths:
            with self.subTest(ablation=path.stem):
                variant = deep_update(cfg, json.loads(path.read_text()))
                model = DualBranchSTImputer.from_config(variant).eval()
                model.load_state_dict(base.state_dict(), strict=True)
                with torch.no_grad():
                    outputs = model(batch)
                    loss, _ = compute_main_stage_loss(outputs, batch, variant)
                self.assertEqual(outputs["x_hat_main"].shape, batch["x_f_gt"].shape)
                self.assertTrue(torch.isfinite(loss))
                if path.stem.startswith("fixed_"):
                    route = [0 if name == "t" else 1 for name in path.stem.removeprefix("fixed_")]
                    self.assertEqual(outputs["coe"]["paths"].tolist(), [route, route])

    def test_hidden_labels_and_hidden_input_payload_do_not_change_forward(self) -> None:
        model = DualBranchSTImputer.from_config(compact_config()).eval()
        batch = make_batch()
        mask = batch["m_f"].bool().expand_as(batch["x_f_gt"])
        with torch.no_grad():
            baseline = model(batch)
            for hidden_value in (100000.0, float("nan"), float("inf")):
                changed = {key: value.clone() for key, value in batch.items()}
                changed["x_f_gt"][~mask] = hidden_value
                changed["x_f_obs"][~mask] = hidden_value
                changed["target_mask"] = (~mask).float()
                changed["available_mask"] = torch.zeros_like(mask)
                result = model(changed)
                for left, right in zip(tensor_leaves(baseline), tensor_leaves(result)):
                    torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
            inputs_only = model({"x_f_obs": batch["x_f_obs"], "m_f": batch["m_f"]})
        torch.testing.assert_close(baseline["x_hat_main"], inputs_only["x_hat_main"], rtol=0.0, atol=0.0)

    def test_every_completion_keeps_observations_and_original_mask(self) -> None:
        batch = make_batch()
        original_mask = batch["m_f"].clone()
        model = DualBranchSTImputer.from_config(compact_config()).eval()
        with torch.no_grad():
            output = model(batch)
        observed = original_mask.bool().expand_as(batch["x_f_obs"])
        completions = [output["coe"]["initial_completion"], *output["coe"]["completions"], output["x_comp"]]
        for completion in completions:
            torch.testing.assert_close(completion[observed], batch["x_f_obs"][observed], rtol=0.0, atol=0.0)
        torch.testing.assert_close(batch["m_f"], original_mask, rtol=0.0, atol=0.0)

    def test_variable_specific_masks_keep_each_channels_observations(self) -> None:
        batch = make_batch()
        batch["m_f"] = torch.cat([batch["m_f"], 1.0 - batch["m_f"]], dim=1)
        batch["x_f_obs"] = torch.where(batch["m_f"].bool(), batch["x_f_gt"], float("nan"))
        model = DualBranchSTImputer.from_config(compact_config()).eval()
        with torch.no_grad():
            output = model(batch)
        self.assertTrue(torch.isfinite(output["x_comp"]).all())
        observed = batch["m_f"].bool()
        torch.testing.assert_close(output["x_comp"][observed], batch["x_f_gt"][observed], rtol=0.0, atol=0.0)

    def test_all_missing_and_all_observed_are_finite(self) -> None:
        cfg = compact_config()
        model = DualBranchSTImputer.from_config(cfg).eval()
        for observed in (False, True):
            with self.subTest(observed=observed):
                batch = make_batch()
                batch["m_f"].fill_(float(observed))
                batch["x_f_obs"] = batch["x_f_gt"].clone() if observed else torch.full_like(batch["x_f_gt"], float("nan"))
                outputs = model(batch)
                loss, _ = compute_main_stage_loss(outputs, batch, cfg)
                self.assertTrue(all(torch.isfinite(tensor).all() for tensor in tensor_leaves(outputs)))
                self.assertTrue(torch.isfinite(loss))
                if observed:
                    self.assertEqual(loss.item(), 0.0)
                    torch.testing.assert_close(outputs["x_comp"], batch["x_f_gt"], rtol=0.0, atol=0.0)

    def test_final_task_loss_trains_both_routers_and_expert_pool(self) -> None:
        cfg = compact_config()
        cfg["loss"]["lambda_coe_balance"] = 0.0
        model = DualBranchSTImputer.from_config(cfg).train()
        optimizer = build_optimizer(model, cfg)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(make_batch())
        loss, _ = compute_main_stage_loss(outputs, make_batch(), cfg)
        loss.backward()
        modules = [*model.main_branch.routers, model.main_branch.temporal_expert,
                   model.main_branch.spatial_expert, model.main_branch.shared_expert]
        for module in modules:
            gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
            self.assertTrue(gradients, type(module).__name__)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)
        weights = outputs["coe"]["route_weights"].detach()
        self.assertTrue(((weights == 0.0) | (weights == 1.0)).all())
        torch.testing.assert_close(weights.sum(-1), torch.ones(weights.shape[:-1]))
        optimizer.step()

    def test_hard_eval_dispatches_only_selected_expert(self) -> None:
        model = TemporalSpatialCoE.from_config(compact_config()).eval()
        called = {"temporal": [], "spatial": []}
        handles = [
            model.temporal_expert.register_forward_pre_hook(lambda _module, args: called["temporal"].append(args[0].shape[0])),
            model.spatial_expert.register_forward_pre_hook(lambda _module, args: called["spatial"].append(args[0].shape[0])),
        ]

        def force_temporal(_module, _args, logits):
            forced = torch.full_like(logits, -10.0)
            forced[..., 0] = 10.0
            return forced

        handles.extend(router.register_forward_hook(force_temporal) for router in model.routers)
        batch = make_batch()
        try:
            with torch.no_grad():
                result = model(batch["x_f_obs"], batch["m_f"])
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(called, {"temporal": [2, 2], "spatial": []})
        self.assertEqual(result["coe"]["paths"].tolist(), [[0, 0], [0, 0]])

    def test_soft_routes_are_mixtures_and_are_not_discrete_paths(self) -> None:
        model = TemporalSpatialCoE.from_config(compact_config(routing_mode="soft")).eval()
        batch = make_batch()
        with torch.no_grad():
            result = model(batch["x_f_obs"], batch["m_f"])
        weights = result["coe"]["route_weights"]
        self.assertTrue(((weights > 0.0) & (weights < 1.0)).all())
        torch.testing.assert_close(weights.sum(-1), torch.ones(weights.shape[:-1]))
        self.assertFalse(result["coe"]["paths_are_discrete"])

    def test_fixed_temporal_spatial_order_is_not_commutative(self) -> None:
        first = TemporalSpatialCoE.from_config(compact_config(routing_mode="fixed", fixed_path=["T", "S"])).eval()
        second = TemporalSpatialCoE.from_config(compact_config(routing_mode="fixed", fixed_path=["S", "T"])).eval()
        second.load_state_dict(first.state_dict(), strict=True)
        batch = make_batch()
        with torch.no_grad():
            ts = first(batch["x_f_obs"], batch["m_f"])
            st = second(batch["x_f_obs"], batch["m_f"])
        self.assertGreater(float((ts["x_hat_main"] - st["x_hat_main"]).abs().max()), 1e-7)

    def test_temporal_expert_never_mixes_spatial_locations(self) -> None:
        expert = DirectionalExpert(dim=8, direction="temporal", kernel_size=3).eval()
        source = torch.randn(1, 8, 5, 3, 9)
        changed = source.clone()
        changed[:, :, 2, 1, 4] += torch.arange(1, 9).view(1, 8)
        with torch.no_grad():
            difference = expert(changed) - expert(source)
        self.assertGreater(float(difference[:, :, :, 1, 4].abs().max()), 0.0)
        difference[:, :, :, 1, 4] = 0
        self.assertEqual(float(difference.abs().max()), 0.0)

    def test_spatial_expert_never_mixes_time_or_wraps_grid_rows(self) -> None:
        expert = DirectionalExpert(dim=8, direction="spatial", kernel_size=3).eval()
        source = torch.randn(1, 8, 5, 3, 9)
        changed = source.clone()
        changed[:, :, 2, 0, 8] += torch.arange(1, 9).view(1, 8)
        with torch.no_grad():
            difference = expert(changed) - expert(source)
        self.assertGreater(float(difference[:, :, 2].abs().max()), 0.0)
        self.assertEqual(float(difference[:, :, 2, 1, 0].abs().max()), 0.0)
        difference[:, :, 2] = 0
        self.assertEqual(float(difference.abs().max()), 0.0)

    def test_dynamic_router_reads_updated_state_with_identical_parameter_budget(self) -> None:
        dynamic = TemporalSpatialCoE.from_config(compact_config(router_state="dynamic")).eval()
        initial = TemporalSpatialCoE.from_config(compact_config(router_state="initial")).eval()
        initial.load_state_dict(dynamic.state_dict(), strict=True)
        captured = {"dynamic": [], "initial": []}
        handles = []
        for name, model in (("dynamic", dynamic), ("initial", initial)):
            for router in model.routers:
                def capture(_module, args, name=name):
                    captured[name].append([tensor.detach().clone() for tensor in tensor_leaves(args)])
                handles.append(router.register_forward_pre_hook(capture))
        batch = make_batch()
        try:
            with torch.no_grad():
                dynamic(batch["x_f_obs"], batch["m_f"])
                initial(batch["x_f_obs"], batch["m_f"])
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(captured["dynamic"]), 2)
        self.assertEqual(len(captured["initial"]), 2)
        for left, right in zip(captured["dynamic"][0], captured["initial"][0]):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        differences = [not torch.equal(left, right) for left, right in zip(captured["dynamic"][1], captured["initial"][1])]
        self.assertTrue(any(differences), "The second dynamic router must consume an updated state.")

    def test_observation_support_respects_grid_edges_and_missing_runs(self) -> None:
        mask = torch.ones(1, 1, 5, 3, 5)
        full = compute_observation_support(mask)
        spatial_index = SUPPORT_FEATURE_NAMES.index("spatial_coverage")
        torch.testing.assert_close(full[:, spatial_index], torch.ones_like(full[:, spatial_index]))
        mask[:, :, 1:4, 1, 2] = 0
        support = compute_observation_support(mask)
        for name, expected in (("gap_length", 3 / 5), ("previous_distance", 2 / 5), ("next_distance", 2 / 5)):
            index = SUPPORT_FEATURE_NAMES.index(name)
            self.assertAlmostEqual(support[0, index, 2, 1, 2].item(), expected, places=6)
        empty = compute_observation_support(torch.zeros_like(mask))
        self.assertTrue(torch.isfinite(empty).all())
        index = SUPPORT_FEATURE_NAMES.index("no_temporal_observation")
        self.assertTrue((empty[:, index] == 1).all())

    def test_synthetic_loader_and_loss_do_not_call_multiscale_operations(self) -> None:
        cfg = compact_config()
        with patch("stmoe_imputer.data.synthetic.ensure_multiscale", side_effect=AssertionError("Unexpected multiscale construction")):
            train, val = build_datasets(cfg, synthetic=True)
            test = build_test_dataset(cfg, synthetic=True)
            for dataset in (train, val, test):
                sample = dataset[0]
                self.assertFalse(any(key in sample for key in ("x_m_obs", "x_c_obs", "m_m", "m_c", "r_m", "r_c")))
            batch = next(iter(build_loader(train, cfg, shuffle=False)))
        model = DualBranchSTImputer.from_config(cfg)
        with patch("stmoe_imputer.losses.cross_scale_loss", side_effect=AssertionError("Unexpected cross-scale loss")):
            loss, logs = compute_main_stage_loss(model(batch), batch, cfg)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("l_coe_mid", logs)

    def test_npz_preserves_available_and_holdout_masks_without_scales(self) -> None:
        cfg = compact_config()
        target = np.ones((2, 2, 5, 3, 5), dtype=np.float32)
        target[0, 0, 0, 0, 0] = np.nan
        available = np.ones_like(target)
        available[0, 1, 0, 0, 1] = 0
        holdout = np.zeros_like(target)
        holdout[:, :, 1, 1, 1] = 1
        with tempfile.TemporaryDirectory() as directory:
            npz = Path(directory) / "tiny.npz"
            csv = Path(directory) / "mask.csv"
            np.savez(npz, x_f_gt=target, available_mask=available, target_mask=holdout)
            np.savetxt(csv, np.ones((2, 15)), delimiter=",")
            cfg["data"]["mask"].update({split + "_csv": str(csv) for split in ("train", "val", "test")})
            with patch("stmoe_imputer.data.npz_dataset.ensure_multiscale", side_effect=AssertionError("Unexpected multiscale construction")):
                train, val = build_datasets(cfg, str(npz), str(npz))
                test = build_test_dataset(cfg, str(npz))
                for dataset in (train, val, test):
                    sample = dataset[0]
                    self.assertNotIn("x_m_obs", sample)
                    self.assertTrue(torch.isfinite(sample["x_f_obs"]).all())
                    self.assertEqual(sample["m_f"][0, 0, 0, 0].item(), 0)
                    self.assertEqual(sample["m_f"][1, 0, 0, 1].item(), 0)
                    self.assertEqual(sample["m_f"][:, 1, 1, 1].sum().item(), 0)
                    self.assertEqual(sample["target_mask"].sum().item(), 2)

    def test_loss_uses_only_finite_available_holdouts_and_optional_intermediate_step(self) -> None:
        cfg = compact_config()
        cfg["loss"]["lambda_coe_balance"] = 0.0
        target = torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0]).view(1, 1, 1, 1, 5)
        batch = prepare_single_scale({
            "x_f_gt": target,
            "m_f": torch.ones_like(target),
            "available_mask": torch.tensor([1, 1, 1, 0, 1]).view_as(target),
            "target_mask": torch.tensor([0, 1, 1, 1, 1]).view_as(target),
        })
        prediction = torch.tensor([1000.0, 3.0, 1000.0, 1000.0, 8.0], requires_grad=True).view_as(target)
        intermediate = prediction + 1.0
        outputs = {"x_hat_main": prediction, "x_hat_final": prediction, "coe": {"predictions": [intermediate, prediction]}}
        loss, logs = compute_main_stage_loss(outputs, batch, cfg)
        self.assertAlmostEqual(loss.item(), 2.0)
        self.assertEqual(logs["supervised_count"].item(), 2)
        weak = deep_update(cfg, {"loss": {"lambda_coe_mid": 0.1}})
        weak_loss, _ = compute_main_stage_loss(outputs, batch, weak)
        self.assertAlmostEqual(weak_loss.item(), 2.3, places=6)
        gradient, = torch.autograd.grad(loss, prediction)
        self.assertEqual(gradient.flatten().tolist(), [0.0, 0.5, 0.0, 0.0, 0.5])

    def test_no_available_targets_returns_differentiable_zero(self) -> None:
        cfg = compact_config()
        target = torch.full((1, 2, 3, 3, 5), float("nan"))
        batch = prepare_single_scale({"x_f_gt": target, "m_f": torch.zeros_like(target)})
        model = DualBranchSTImputer.from_config(cfg)
        outputs = model(batch)
        loss, _ = compute_main_stage_loss(outputs, batch, cfg)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(outputs["x_comp"]).all())

    def test_channel_last_npz_and_masks_use_the_same_layout(self) -> None:
        cfg = compact_config()
        target = np.ones((2, 5, 3, 5, 2), dtype=np.float32)
        observed = np.ones((2, 5, 3, 5, 1), dtype=np.float32)
        available = np.ones_like(target)
        available[0, 0, 0, 1, 1] = 0
        holdout = np.zeros_like(target)
        holdout[0, 1, 1, 1, 0] = 1
        holdout[0, 3, 2, 4, 1] = 1
        with tempfile.TemporaryDirectory() as directory:
            npz = Path(directory) / "channel_last.npz"
            csv = Path(directory) / "mask.csv"
            np.savez(npz, x_f_gt=target, m_f=observed, available_mask=available, target_mask=holdout)
            np.savetxt(csv, np.ones((2, 15)), delimiter=",")
            cfg["data"]["mask"].update({split + "_csv": str(csv) for split in ("train", "val", "test")})
            train, _ = build_datasets(cfg, str(npz), str(npz))
            sample = train[0]
        self.assertEqual(sample["x_f_gt"].shape, (2, 5, 3, 5))
        expected_available = torch.from_numpy(available[0]).permute(3, 0, 1, 2)
        expected_holdout = torch.from_numpy(holdout[0]).permute(3, 0, 1, 2)
        torch.testing.assert_close(sample["available_mask"], expected_available)
        torch.testing.assert_close(sample["target_mask"], expected_holdout)
        torch.testing.assert_close(sample["m_f"], expected_available * (1.0 - expected_holdout))

    def test_evaluation_without_supervision_does_not_report_zero_error(self) -> None:
        cfg = compact_config()
        model = DualBranchSTImputer.from_config(cfg)
        batch = make_batch()
        batch["m_f"].fill_(1.0)
        batch["x_f_obs"] = batch["x_f_gt"].clone()
        with self.assertRaisesRegex(ValueError, "no finite hidden supervision"):
            evaluate(model, [batch], torch.device("cpu"), cfg)

    def test_empty_training_batch_skips_adamw_update(self) -> None:
        cfg = compact_config()
        model = DualBranchSTImputer.from_config(cfg)
        optimizer = build_optimizer(model, cfg)
        batch = make_batch()
        loss, _ = compute_main_stage_loss(model(batch), batch, cfg)
        loss.backward()
        optimizer.step()
        parameters_before = {name: value.detach().clone() for name, value in model.named_parameters()}
        steps_before = [state["step"].item() for state in optimizer.state.values()]
        batch["m_f"].fill_(1.0)
        batch["x_f_obs"] = batch["x_f_gt"].clone()
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(ValueError, "no finite hidden supervision"):
                train_one_epoch(model, [batch], optimizer, torch.device("cpu"), cfg, epoch=1)
            step.assert_not_called()
        self.assertEqual(steps_before, [state["step"].item() for state in optimizer.state.values()])
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, parameters_before[name], rtol=0.0, atol=0.0)

    def test_training_budget_counts_exclude_empty_batches(self) -> None:
        cfg = compact_config()
        model = DualBranchSTImputer.from_config(cfg)
        optimizer = build_optimizer(model, cfg)
        batch = make_batch()
        empty = {key: value.clone() for key, value in batch.items()}
        empty["m_f"].fill_(1.0)
        empty["x_f_obs"] = empty["x_f_gt"].clone()
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            logs = train_one_epoch(model, [empty, batch, empty], optimizer, torch.device("cpu"), cfg, epoch=1)
            self.assertEqual(step.call_count, 1)
        self.assertEqual(logs["train_optimizer_steps"], 1)
        self.assertEqual(logs["train_seen_samples"], batch["x_f_gt"].shape[0])
        self.assertEqual(logs["train_skipped_empty_batches"], 2)
        self.assertEqual(logs["train_skipped_amp_steps"], 0)
        self.assertEqual(logs["l_coe_balance_weighted"], 0)
        self.assertAlmostEqual(logs["loss"], logs["l_main"], places=7)

    def test_chain_metrics_count_samples_and_never_label_soft_routes_as_paths(self) -> None:
        paths = torch.tensor([[0, 1], [0, 1], [1, 1]])
        weights = torch.nn.functional.one_hot(paths, num_classes=2).float()
        accumulator = CoERoutingMetricAccumulator()
        for indices in (slice(0, 2), slice(2, 3)):
            accumulator.update({
                "route_weights": weights[indices], "route_probs": weights[indices],
                "routing_mode": "hard", "paths": paths[indices],
            })
        metrics = accumulator.compute()
        self.assertEqual(metrics["coe_routing_sample_count"], 3)
        self.assertAlmostEqual(metrics["coe_path_TS_fraction"], 2 / 3)
        self.assertAlmostEqual(metrics["coe_path_SS_fraction"], 1 / 3)
        self.assertEqual(metrics["coe_path_TT_fraction"], 0)
        self.assertAlmostEqual(metrics["coe_step1_T_usage"], 2 / 3)
        soft = CoERoutingMetricAccumulator()
        soft_weights = 0.25 + 0.5 * weights
        soft.update({"route_weights": soft_weights, "route_probs": soft_weights, "routing_mode": "soft", "paths": paths})
        metrics = soft.compute()
        self.assertFalse(any("_path_" in name or name.endswith("_usage") for name in metrics))
        self.assertAlmostEqual(metrics["coe_step1_T_weight"], 7 / 12)


if __name__ == "__main__":
    unittest.main()
