from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v24_experiment_plan", ROOT / "scripts/v24/build_experiment_plan.py")
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(planner)


class V24ExperimentPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def args(self, *extra):
        return planner.parse_args([
            "--base-config", str(ROOT / "configs/v24/smoke.json"),
            "--output-dir", str(self.directory / "plan"), "--synthetic", *extra,
        ])

    def real_fixture(self, embedded=None):
        cfg = planner.load_config(ROOT / "configs/v24/smoke.json")
        cfg["model"]["main"].update(max_t=3, h=4, w=5)
        base = self.directory / "base.json"
        base.write_text(json.dumps(cfg), encoding="utf-8")
        metadata = {"schema_version": 1, "pattern": "random_point", "requested_missing_rate": 0.4, "mask_seed": 2026, "splits": {}}
        mask_directory = self.directory / "masks/random_point/0.4"
        mask_directory.mkdir(parents=True)
        paths = {}
        for split, samples in (("train", 5), ("val", 2), ("test", 3)):
            path = self.directory / f"{split}.npz"
            arrays = {"x_f_gt": np.zeros((samples, 2, 3, 4, 5), dtype=np.float32)}
            if embedded:
                arrays[embedded] = np.ones((samples, 1, 3, 4, 5), dtype=np.float32)
            np.savez(path, **arrays)
            csv = mask_directory / f"{split}.csv"
            mask = np.ones((samples, 60), dtype=np.uint8)
            mask[:, :24] = 0
            np.savetxt(csv, mask, delimiter=",", fmt="%d")
            metadata["splits"][split] = {
                "source_npz": str(path), "shape_ncthw": [samples, 2, 3, 4, 5],
                "rows": samples, "columns": 60, "csv": str(csv),
                "sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                "actual_missing_rate_min": 0.4, "actual_missing_rate_mean": 0.4,
                "actual_missing_rate_max": 0.4,
            }
            paths[split] = path
        metadata_path = mask_directory / "metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        args = planner.parse_args([
            "--base-config", str(base), "--output-dir", str(self.directory / "plan"),
            "--train-npz", str(paths["train"]), "--val-npz", str(paths["val"]),
            "--test-npz", str(paths["test"]), "--mask-root", str(self.directory / "masks"),
            "--patterns", "random_point", "--rates", "0.4", "--variants", "full", "fixed_ts",
        ])
        return args, metadata_path

    def test_core_defaults_use_single_seed_and_no_route_regularization(self):
        plan = planner.build_plan(self.args())
        self.assertEqual(plan["num_runs"], 11)
        self.assertEqual(plan["seeds"], [7])
        self.assertFalse(plan["paired_seeds"])
        self.assertEqual(plan["status"], "planned_not_executed")
        for run in plan["runs"]:
            cfg = run["config"]
            self.assertEqual(cfg["loss"]["lambda_coe_balance"], 0)
            self.assertEqual(cfg["loss"]["lambda_coe_mid"], 0)
            self.assertFalse(cfg["data"]["drop_last"])
            self.assertFalse(cfg["train"]["early_stopping"]["enabled"])
            self.assertEqual(run["budget"]["expected_update_slots"], 4)
        self.assertFalse((self.directory / "plan").exists())

    def test_mechanism_controls_change_the_intended_state_inputs(self):
        plan = planner.build_plan(self.args("--seeds", "7"))
        runs = {run["variant"]: run for run in plan["runs"]}
        def coe(name):
            return runs[name]["config"]["model"]["coe"]
        self.assertEqual((coe("initial_router")["router_state"], coe("initial_router")["expert_state"]), ("initial", "dynamic"))
        self.assertEqual((coe("no_expert_state_update")["router_state"], coe("no_expert_state_update")["expert_state"]), ("dynamic", "initial"))
        self.assertEqual((coe("parallel")["num_steps"], coe("parallel")["routing_mode"]), (1, "parallel"))
        self.assertEqual(coe("soft")["num_steps"], 2)
        self.assertEqual(coe("fixed_tt")["fixed_path"], ["T", "T"])
        self.assertEqual(runs["full"]["candidate_compute"]["train_routed_calls_per_window"], 4)
        self.assertEqual(runs["fixed_ts"]["candidate_compute"]["train_routed_calls_per_window"], 2)
        self.assertEqual(runs["parallel"]["candidate_compute"]["inference_routed_calls_per_window"], 2)

    def test_depth_factorial_keeps_data_and_losses_constant(self):
        plan = planner.build_plan(self.args("--stage", "depth"))
        self.assertEqual(plan["num_runs"], 9)
        cells = set()
        for run in plan["runs"]:
            cfg = run["config"]
            coe = cfg["model"]["coe"]
            cells.add((coe["num_steps"], len(coe["expert_pool"])))
            self.assertEqual(coe["expert_state"], "dynamic")
            self.assertIsNone(coe["fixed_path"])
            self.assertEqual(cfg["loss"]["lambda_coe_balance"], 0)
            self.assertFalse(cfg["data"]["multiscale"])
        self.assertEqual(cells, {(k, e) for k in (2, 3, 4) for e in (2, 4, 6)})

    def test_chain4_stage_enables_balance_without_expanding_depth_grid(self):
        plan = planner.build_plan(self.args("--stage", "chain4"))
        self.assertEqual(plan["num_runs"], 1)
        cfg = plan["runs"][0]["config"]
        self.assertEqual(cfg["model"]["coe"]["num_steps"], 4)
        self.assertEqual(cfg["model"]["coe"]["expert_pool"], ["T", "S", "TD", "SD", "TA", "ST"])
        self.assertEqual(cfg["loss"]["lambda_coe_balance"], 0.01)
        self.assertEqual(cfg["loss"]["lambda_coe_mid"], 0)
        self.assertTrue(plan["comparison_policy"]["route_balance_enabled_by_default"])
        self.assertEqual(plan["runs"][0]["candidate_compute"]["train_routed_calls_per_window"], 24)

    def test_weak_mid_is_an_explicit_optional_stage(self):
        plan = planner.build_plan(self.args("--stage", "optional"))
        self.assertEqual(plan["num_runs"], 1)
        for run in plan["runs"]:
            self.assertEqual(run["config"]["loss"], {"type": "l1", "lambda_coe_mid": 0.1, "lambda_coe_balance": 0.0})

    def test_compute_budget_uses_measured_cost_not_candidate_counts(self):
        profile = self.directory / "cost.json"
        profile.write_text(json.dumps({
            "reference_variant": "full", "seconds_per_epoch": {"full": 10.0, "fixed_ts": 4.0},
            "context": {"device": "test", "batch_size": 2, "dataset": "Synthetic"},
        }), encoding="utf-8")
        plan = planner.build_plan(self.args(
            "--variants", "full", "fixed_ts", "--seeds", "7", "--epochs", "5",
            "--budget-regime", "approx_compute", "--cost-profile", str(profile),
        ))
        budgets = {run["variant"]: run["budget"] for run in plan["runs"]}
        self.assertEqual(budgets["full"]["epochs"], 5)
        self.assertEqual(budgets["fixed_ts"]["epochs"], 12)
        self.assertEqual(budgets["fixed_ts"]["unallocated_seconds"], 2)
        with self.assertRaisesRegex(ValueError, "requires --cost-profile"):
            planner.build_plan(self.args("--budget-regime", "approx_compute"))

    def test_structured_masks_are_verified_shared_and_honestly_labeled(self):
        args, _ = self.real_fixture()
        plan = planner.build_plan(args)
        self.assertEqual(plan["num_runs"], 2)
        self.assertEqual(plan["protocols"][0]["kind"], "structured_csv")
        mask_configs = [run["config"]["data"]["mask"] for run in plan["runs"]]
        self.assertTrue(all(mask == mask_configs[0] for mask in mask_configs))
        self.assertEqual(mask_configs[0]["pattern"], "random")
        self.assertEqual(plan["protocols"][0]["pattern"], "random_point")
        self.assertEqual(plan["datasets"]["train"]["shape_ncthw"], [5, 2, 3, 4, 5])
        self.assertEqual(plan["runs"][0]["budget"]["expected_update_slots"], 3)

    def test_mask_metadata_cannot_silently_refer_to_other_data_or_modified_csv(self):
        args, metadata_path = self.real_fixture()
        metadata = json.loads(metadata_path.read_text())
        original = metadata["splits"]["val"]["source_npz"]
        metadata["splits"]["val"]["source_npz"] = metadata["splits"]["train"]["source_npz"]
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "source_npz"):
            planner.build_plan(args)
        metadata["splits"]["val"]["source_npz"] = original
        metadata_path.write_text(json.dumps(metadata))
        csv = Path(metadata["splits"]["test"]["csv"])
        csv.write_text(csv.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            planner.build_plan(args)

    def test_embedded_mask_conflict_and_ignored_synthetic_csv_are_rejected(self):
        args, _ = self.real_fixture(embedded="m_f")
        with self.assertRaisesRegex(ValueError, "embedded"):
            planner.build_plan(args)
        with self.assertRaisesRegex(ValueError, "synthetic loader ignores CSV"):
            planner.build_plan(self.args("--mask-root", str(self.directory)))

    def test_ambiguous_npz_layout_is_rejected_even_when_config_shape_matches(self):
        cfg = planner.load_config(ROOT / "configs/v24/smoke.json")
        cfg["model"]["main"].update(max_t=1, h=4, w=5)
        path = self.directory / "ambiguous.npz"
        np.savez(path, x_f_gt=np.zeros((3, 1, 4, 5, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "dataset loader infers"):
            planner.npz_shape(path, cfg)
        cfg["model"]["main"]["max_t"] = 3
        path = self.directory / "channels_last.npz"
        np.savez(path, x_f_gt=np.zeros((3, 3, 4, 5, 2), dtype=np.float32))
        self.assertEqual(planner.npz_shape(path, cfg)["shape_ncthw"], [3, 2, 3, 4, 5])

    def test_legacy_plan_rejects_embedded_masks_before_claiming_csv_protocol(self):
        args, _ = self.real_fixture(embedded="target_mask")
        args.mask_root = args.patterns = args.rates = None
        with self.assertRaisesRegex(ValueError, "embedded.*target_mask"):
            planner.build_plan(args)

    def test_limits_and_required_inputs_prevent_accidental_expansion(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            planner.build_plan(self.args("--max-plans", "10"))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            planner.build_plan(self.args("--seeds", "7", "7"))
        args = self.args()
        args.synthetic = False
        with self.assertRaisesRegex(ValueError, "Supply --train-npz"):
            planner.build_plan(args)
        with self.assertRaisesRegex(ValueError, "selected from"):
            planner.build_plan(self.args("--variants", "weak_mid"))

    def test_writer_only_emits_artifacts_and_preserves_existing_plan(self):
        executable = "python with spaces; $(unwanted)"
        args = self.args("--variants", "full", "--seeds", "7", "--python", executable)
        manifest = planner.build_plan(args)
        planner.write_plan(manifest, args.output_dir)
        saved = json.loads((args.output_dir / "manifest.json").read_text())
        self.assertEqual(saved["status"], "planned_not_executed")
        self.assertEqual(shlex.split(saved["runs"][0]["command"])[0], executable)
        self.assertTrue(Path(saved["runs"][0]["config_path"]).is_file())
        self.assertFalse((args.output_dir / "runs").exists())
        with self.assertRaises(FileExistsError):
            planner.write_plan(manifest, args.output_dir)


if __name__ == "__main__":
    unittest.main()
