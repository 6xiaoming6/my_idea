"""Whole-window path oracle and intervention diagnostics, without training."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from stmoe_imputer.config import load_config
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import save_checkpoint


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v24_analyze_paths", ROOT / "scripts/v24/analyze_paths.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class PathAnalysisTests(unittest.TestCase):
    def tiny_model(self):
        cfg = deepcopy(load_config(ROOT / "configs/v24/smoke.json"))
        cfg["model"]["c_in"] = 1
        cfg["model"]["main"].update(dim=8, max_t=4, h=3, w=5)
        return DualBranchSTImputer.from_config(cfg), cfg

    def test_oracle_uses_equal_windows_and_correct_minimum_order(self) -> None:
        result = analysis.summarize_path_losses(
            np.array([[1, 5, 4, 6], [4, 1, 2, 6]]), np.array([1, 2]), split="val",
        )
        self.assertEqual(result["path_mean_window_mae"], {"TT": 2.5, "TS": 3., "ST": 3., "SS": 6.})
        self.assertEqual(result["best_fixed_mean_window_mae_posthoc"], 2.5)
        self.assertEqual(result["window_oracle_mae_posthoc"], 1.)
        self.assertEqual(result["oracle_gap_posthoc"], 1.5)
        self.assertEqual(result["policy_regret_vs_window_oracle_posthoc"], .5)
        self.assertEqual(result["selected_fixed_path"], "TT")
        self.assertEqual(result["selected_fixed_path_source"], "minimum_mean_window_mae_on_this_validation_subset")

    def test_test_split_cannot_select_deployable_path_from_targets(self) -> None:
        values, policy = np.array([[1, 2, 3, 4]]), np.array([1])
        result = analysis.summarize_path_losses(values, policy, split="test")
        self.assertEqual(result["best_fixed_path_posthoc"], "TT")
        self.assertIsNone(result["selected_fixed_path"])
        self.assertIsNone(result["selected_fixed_mean_window_mae"])
        result = analysis.summarize_path_losses(values, policy, split="test", selected_fixed_path="SS")
        self.assertEqual(result["selected_fixed_path"], "SS")
        self.assertEqual(result["selected_fixed_mean_window_mae"], 4.)
        self.assertEqual(result["selected_fixed_path_source"], "supplied_validation_choice")

    def test_empty_and_nonfinite_oracle_inputs_do_not_report_fake_zero_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "no finite hidden supervision"):
            analysis.summarize_path_losses(np.empty((0, 4)), np.empty(0), split="val")
        with self.assertRaisesRegex(ValueError, "finite"):
            analysis.summarize_path_losses(np.full((1, 4), np.nan), np.ones(1), split="test")
        with self.assertRaises(ValueError):
            analysis.summarize_path_losses(np.ones((1, 4)), np.ones(1), split="train")

    def test_forced_path_restores_router_even_if_intervention_fails(self) -> None:
        model, _ = self.tiny_model()
        backbone = model.main_branch
        saved_path = backbone.fixed_path
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with analysis.force_whole_window_path(backbone, "SS"):
                self.assertEqual(backbone.routing_mode, "fixed")
                self.assertEqual(tuple(backbone.fixed_path), ("S", "S"))
                raise RuntimeError("interrupted")
        self.assertEqual(backbone.routing_mode, "hard")
        self.assertEqual(backbone.fixed_path, saved_path)

    def test_training_path_context_distinguishes_unknown_from_zero(self) -> None:
        unknown = analysis.training_path_context(None)
        self.assertIsNone(unknown["source"])
        self.assertTrue(all(value is None for value in unknown["path_fractions"].values()))
        result = analysis.training_path_context({"train_coe_path_TT_fraction": 0., "train_coe_path_TS_fraction": .8})
        self.assertEqual(result["path_fractions"]["TT"], 0.)
        self.assertEqual(result["path_fractions"]["TS"], .8)
        self.assertIsNone(result["path_fractions"]["ST"])

    def test_checkpoint_npz_cli_helper_exports_actual_paths_and_preserves_scope(self) -> None:
        model, cfg = self.tiny_model()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, data_path, mask_path = root / "checkpoint.pt", root / "data.npz", root / "mask.csv"
            save_checkpoint(checkpoint, model, None, 0, {"train_coe_path_TS_fraction": .75}, cfg)
            data = np.random.default_rng(7).normal(size=(4, 1, 4, 3, 5)).astype(np.float32)
            data[2, 0, 0, 0, 0] = np.nan
            np.savez(data_path, x_f_gt=data)
            masks = np.ones((4, 60), dtype=np.float32)
            masks[1, 0] = 0
            masks[2] = 0
            masks[3, :30] = 0
            np.savetxt(mask_path, masks, delimiter=",")
            args = analysis.parse_args([
                "--checkpoint", str(checkpoint), "--data-npz", str(data_path),
                "--mask-csv", str(mask_path), "--output-dir", str(root / "analysis"),
                "--split", "test", "--device", "cpu", "--max-samples", "3",
                "--selected-fixed-path", "TS",
            ])
            result = analysis.run_analysis(args)
            self.assertEqual(result["scope"]["dataset_sample_count"], 4)
            self.assertEqual(result["scope"]["requested_subset_sample_count"], 3)
            self.assertEqual(result["scope"]["evaluated_sample_count"], 2)
            self.assertEqual(result["scope"]["skipped_zero_target_indices"], [0])
            self.assertEqual(result["selected_fixed_path"], "TS")
            self.assertEqual(result["training_path_context"]["path_fractions"]["TS"], .75)
            self.assertEqual(result["max_policy_forced_replay_prediction_linf"], 0.)
            rows = result["per_window_rows"]
            self.assertEqual([row["sample_index"] for row in rows], [1, 2])
            self.assertEqual([row["supervised_count"] for row in rows], [1, 59])
            for row in rows:
                self.assertEqual(row["policy_mae"], row[f"path_{row['policy_path']}_mae"])
                self.assertAlmostEqual(row["policy_regret_posthoc"], row["policy_mae"] - row["oracle_mae_posthoc"])
                for path in analysis.PATHS:
                    self.assertEqual(row[f"path_{path}_step2_mae"], row[f"path_{path}_mae"])
                    self.assertTrue(np.isfinite(row[f"path_{path}_step1_update_missing_mean"]))
            point_mae = sum(row["policy_mae"] * row["supervised_count"] for row in rows) / 60
            self.assertAlmostEqual(result["point_weighted_metrics"]["policy"]["mae"], point_mae, places=6)
            self.assertAlmostEqual(result["policy_mean_window_mae"], np.mean([row["policy_mae"] for row in rows]))
            self.assertTrue((root / "analysis/per_window.csv").is_file())
            with (root / "analysis/summary.json").open() as handle:
                self.assertEqual(json.load(handle), result)
            with self.assertRaises(FileExistsError):
                analysis.run_analysis(args)

    def test_analysis_preserves_weights_modes_and_original_observations(self) -> None:
        model, _ = self.tiny_model()
        target = torch.randn(1, 4, 3, 5)
        mask = torch.ones_like(target)
        mask[:, :2] = 0
        sample = {"x_f_gt": target, "m_f": mask, "x_f_obs": torch.where(mask.bool(), target, 0.)}
        before = {name: value.clone() for name, value in model.state_dict().items()}
        sample_before = {name: value.clone() for name, value in sample.items()}
        model.train()
        saved_path = model.main_branch.fixed_path
        result, _ = analysis.analyze_windows(model, [sample], split="val")
        self.assertIn(result["selected_fixed_path"], analysis.PATHS)
        self.assertTrue(model.training)
        self.assertEqual(model.main_branch.routing_mode, "hard")
        self.assertEqual(model.main_branch.fixed_path, saved_path)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for name, value in sample.items():
            torch.testing.assert_close(value, sample_before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_embedded_masks_and_unsupported_chains_are_rejected(self) -> None:
        model, _ = self.tiny_model()
        model.main_branch.routing_mode = "soft"
        with self.assertRaisesRegex(ValueError, "two-round hard"):
            analysis.analyze_windows(model, [], split="test")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for embedded_name in ("m_f", "target_mask"):
                data_path = root / f"{embedded_name}.npz"
                np.savez(data_path, x_f_gt=np.zeros((1, 1, 4, 3, 5)), **{embedded_name: np.zeros((1, 1, 4, 3, 5))})
                args = analysis.parse_args([
                    "--checkpoint", "unused.pt", "--data-npz", str(data_path),
                    "--mask-csv", "unused.csv", "--output-dir", str(root / "analysis"), "--split", "test",
                ])
                with self.subTest(mask=embedded_name), self.assertRaisesRegex(ValueError, "embedded masks"):
                    analysis.run_analysis(args)


if __name__ == "__main__":
    unittest.main()
