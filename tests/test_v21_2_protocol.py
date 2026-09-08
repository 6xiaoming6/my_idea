import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


train = module("v21_2_train_test", ROOT / "scripts/v21.2-single/train.py")
with patch.dict(sys.modules, {"train": train}):
    runner = module("v21_2_runner_test", ROOT / "scripts/v21.2-single/run_core6.py")


class ProtocolTest(unittest.TestCase):
    def test_full_epochs_and_separate_output(self):
        suite = train.load_suite(train.DEFAULT_CONFIG)
        for dataset, epochs in (("TaxiBJ", 160), ("BikeNYC", 140), ("CHAP", 150)):
            p = train.patch_for(suite, "risk", dataset)
            self.assertEqual(p["train"]["epochs"], epochs)
            self.assertFalse(p["train"]["early_stopping"]["enabled"])
            self.assertEqual(p["output_dir"], "outputs/v21.2-single")

    def test_ablation_and_epoch_change_identity(self):
        suite = train.load_suite(train.DEFAULT_CONFIG)
        fingerprints = [train.patch_for(suite, v, "TaxiBJ", e)["experiment_policy"]["fingerprint"]
                        for v, e in (("risk", None), ("risk", 1), ("constant", None), ("no_distortion", None))]
        self.assertEqual(len(set(fingerprints)), 4)
        self.assertEqual(train.patch_for(suite, "risk", "TaxiBJ", 1)["train"]["val_epoch"], 1)

    def test_skip_requires_full_finite_test_and_matching_config(self):
        suite = train.load_suite(train.DEFAULT_CONFIG)
        cfg = train.patch_for(suite, "risk", "TaxiBJ", 1)
        cfg["seed"] = 42
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "ROOT", Path(tmp)):
            run = Path(tmp) / "outputs/v21.2-single/TaxiBJ/ablation/v21_2_risk/random/rate0.6/run"
            (run / "logs").mkdir(parents=True)
            (run / "checkpoints").mkdir()
            (run / "config.json").write_text(json.dumps(cfg))
            (run / "checkpoints/best.pt").write_bytes(b"test fixture")
            args = (suite, "risk", "TaxiBJ", "random", "0.6", 42, 1)
            self.assertIsNone(runner.completed(*args))
            logs = [{"epoch": 1, "is_best": True, "val": {"mae": 1}}, {"stage": "test", "metrics": {"mae": 1, "rmse": 2}}]
            (run / "logs/metrics.jsonl").write_text("\n".join(json.dumps(r) for r in logs))
            (run / "logs/test.log").write_text("Testing finished:")
            self.assertEqual(runner.completed(*args), run)
            logs[-1]["metrics"]["mae"] = float("nan")
            (run / "logs/metrics.jsonl").write_text("\n".join(json.dumps(r) for r in logs))
            self.assertIsNone(runner.completed(*args))


if __name__ == "__main__":
    unittest.main()
