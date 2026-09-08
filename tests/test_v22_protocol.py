import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v22_train", ROOT / "scripts/v22/train.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
spec = importlib.util.spec_from_file_location("v22_queue", ROOT / "scripts/v22/run_experiments.py")
queue = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"train": runner}):
    spec.loader.exec_module(queue)


class ProtocolTest(unittest.TestCase):
    def test_queue_skips_completed_by_default_and_can_explicitly_rerun(self):
        base = ["run_experiments.py", "--profile", "quick", "--gpus", "0", "1"]
        old = {"run": "/completed/fixture"}
        for flags, expected_calls in (([], 6), (["--skip-completed"], 6), (["--rerun-completed"], 8)):
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", base + flags), \
                 mock.patch.object(queue, "completed", side_effect=[old, old] + [None] * 6), \
                 mock.patch.object(queue.subprocess, "run") as launch, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                queue.main()
                self.assertEqual(launch.call_count, expected_calls)
                self.assertIn(f"pending={expected_calls}", output.getvalue())
                if expected_calls == 6:
                    self.assertIn("SKIP fixed_stats BikeNYC fixed@0.4", output.getvalue())
                    self.assertIn("uniform", launch.call_args_list[0].args[0])

    def test_queue_dry_run_and_all_completed_never_launch(self):
        base = ["run_experiments.py", "--profile", "quick", "--gpus", "0", "1"]
        for flags, old in ((["--dry-run"], None), ([], {"run": "/completed/fixture"})):
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", base + flags), \
                 mock.patch.object(queue, "completed", return_value=old), \
                 mock.patch.object(queue.subprocess, "run") as launch, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                queue.main()
                launch.assert_not_called()
                self.assertIn("WOULD RUN" if old is None else "pending=0", output.getvalue())

    def test_quick_profile_is_eight_short_jobs_without_changing_full(self):
        full = runner.load_suite()
        quick = runner.load_suite(profile="quick")
        self.assertEqual(len(quick["default_points"]) * len(quick["default_variants"]), 8)
        self.assertEqual(quick["default_variants"], ["fixed_stats", "single_local", "uniform", "moe"])
        for variant in quick["default_variants"]:
            for dataset, pattern, rate in quick["default_points"]:
                cfg, _ = runner.build_config(quick, variant, dataset, pattern, rate, 42, world_size=2)
                self.assertEqual(cfg["train"]["epochs"], 120)
                self.assertEqual(cfg["train"]["val_epoch"], 2)
                self.assertEqual(cfg["distributed"]["global_batch_size"], 8)
                self.assertEqual(cfg["output_dir"], "outputs/v22/quick")
        self.assertEqual(full["datasets"]["BikeNYC"]["train"]["epochs"], 140)
        self.assertNotIn("default_points", full)
        self.assertEqual(full, runner.load_suite())

    def test_all_configs_independent_of_v14_and_v21(self):
        suite = runner.load_suite()
        for variant in suite["variants"]:
            for dataset in suite["datasets"]:
                cfg, _ = runner.build_config(suite, variant, dataset, "fixed", "0.4", 42)
                self.assertEqual(cfg["model"]["architecture"], "v22_coarsening_moe")
                self.assertNotIn("v14", cfg["model"])
                self.assertNotIn("v21", cfg["model"])
                self.assertNotIn("lambda_cross", cfg["loss"])
                self.assertFalse(cfg["train"]["early_stopping"]["enabled"])
                self.assertGreater(cfg["train"]["epochs"], 5)

    def test_fingerprints_separate_protocols(self):
        suite = runner.load_suite()
        common = (suite, "moe", "TaxiBJ", "fixed", "0.4", 42)
        full = runner.build_config(*common)[0]
        short = runner.build_config(*common, epochs=1)[0]
        smoke = runner.build_config(*common, smoke=True)[0]
        fingerprints = [c["experiment_policy"]["fingerprint"] for c in (full, short, smoke)]
        self.assertEqual(len(set(fingerprints)), 3)
        self.assertEqual(short["train"]["val_epoch"], 1)
        self.assertEqual(smoke["output_dir"], "outputs/v22/smoke")

    def test_suite_not_mutated(self):
        suite = runner.load_suite()
        before = json.dumps(suite, sort_keys=True)
        runner.build_config(suite, "moe", "TaxiBJ", "random", "0.6", 42, smoke=True)
        self.assertEqual(before, json.dumps(suite, sort_keys=True))

    def test_ddp_batch_and_fingerprint(self):
        args = (runner.load_suite(), "moe", "BikeNYC", "random", "0.4", 42)
        single = runner.build_config(*args)[0]
        ddp = runner.build_config(*args, world_size=2)[0]
        self.assertEqual(ddp["distributed"]["per_rank_batch_size"], 4)
        self.assertEqual(ddp["distributed"]["global_batch_size"], single["data"]["batch_size"])
        self.assertNotEqual(ddp["experiment_policy"]["fingerprint"], single["experiment_policy"]["fingerprint"])

    def test_completion_requires_full_budget_matching_config_best_and_test(self):
        cfg, _ = runner.build_config(runner.load_suite(), "moe", "TaxiBJ", "fixed", "0.4", 42, epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            cfg["output_dir"] = directory
            run = Path(directory) / "TaxiBJ/ablation/v22_moe/fixed/rate0.4/run"
            (run / "logs").mkdir(parents=True)
            (run / "checkpoints").mkdir()
            (run / "config.json").write_text(json.dumps(cfg))
            (run / "checkpoints/best.pt").write_bytes(b"fixture")
            (run / "logs/train.log").write_text("Training finished normally:")
            (run / "logs/test.log").write_text("Testing finished:")
            records = [{"epoch": 1, "train": {}, "val": {"mae": 1.}, "is_best": True},
                       {"epoch": 2, "train": {}, "val": {"mae": 2.}, "is_best": False},
                       {"stage": "test", "metrics": {"mae": 1., "rmse": 2.}, "extra": {"best_epoch": 1}}]
            metrics = run / "logs/metrics.jsonl"
            metrics.write_text("\n".join(json.dumps(r) for r in records))
            self.assertIsNotNone(queue.completed(cfg, "moe"))
            metrics.write_text("\n".join(json.dumps(r) for r in [records[0], records[2]]))
            self.assertIsNone(queue.completed(cfg, "moe"))
            records[-1]["extra"]["best_epoch"] = 2
            metrics.write_text("\n".join(json.dumps(r) for r in records))
            self.assertIsNone(queue.completed(cfg, "moe"))
            records[-1]["extra"]["best_epoch"] = 1
            records[-1]["metrics"]["mae"] = float("nan")
            metrics.write_text("\n".join(json.dumps(r) for r in records))
            self.assertIsNone(queue.completed(cfg, "moe"))


if __name__ == "__main__":
    unittest.main()
