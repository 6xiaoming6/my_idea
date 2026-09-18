"""Offline whole-window path interventions for a trained two-round T/S CoE.

All four paths reuse the checkpoint's weights. This diagnostic does not train
fixed-path baselines, splice local predictions, or deploy a target-based oracle.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from stmoe_imputer.data import FlowNPZDataset
from stmoe_imputer.losses import supervision_mask
from stmoe_imputer.metrics import MaskedMetricAccumulator
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.routing_metrics import CoERoutingMetricAccumulator
from stmoe_imputer.utils import get_device


PATHS = ("TT", "TS", "ST", "SS")


def summarize_path_losses(
    path_losses: np.ndarray,
    policy_losses: np.ndarray,
    *,
    split: str,
    selected_fixed_path: str | None = None,
) -> dict:
    """Equal-window MAE oracle; test targets never select a deployable path."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if selected_fixed_path is not None and selected_fixed_path not in PATHS:
        raise ValueError(f"selected_fixed_path must be one of {PATHS}")
    values = np.asarray(path_losses, dtype=np.float64)
    policy = np.asarray(policy_losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(PATHS) or policy.shape != values.shape[:1]:
        raise ValueError("Expected path_losses [windows, 4] and policy_losses [windows]")
    if not len(values):
        raise ValueError("Path analysis has no finite hidden supervision in the requested subset")
    if not np.isfinite(values).all() or not np.isfinite(policy).all():
        raise ValueError("Path analysis losses must be finite")
    means = values.mean(axis=0)
    best_index = int(means.argmin())
    oracle_mae = float(values.min(axis=1).mean())
    if selected_fixed_path is not None:
        selection_source = "supplied_validation_choice"
    elif split == "val":
        selected_fixed_path = PATHS[best_index]
        selection_source = "minimum_mean_window_mae_on_this_validation_subset"
    else:
        selection_source = "not_selected_test_targets_are_diagnostic_only"
    return {
        "window_count": len(values),
        "path_order": list(PATHS),
        "path_mean_window_mae": dict(zip(PATHS, means.tolist())),
        "policy_mean_window_mae": float(policy.mean()),
        "best_fixed_path_posthoc": PATHS[best_index],
        "best_fixed_mean_window_mae_posthoc": float(means[best_index]),
        "window_oracle_mae_posthoc": oracle_mae,
        "oracle_gap_posthoc": float(means[best_index]) - oracle_mae,
        "policy_regret_vs_window_oracle_posthoc": float(policy.mean()) - oracle_mae,
        "selected_fixed_path": selected_fixed_path,
        "selected_fixed_path_source": selection_source,
        "selected_fixed_mean_window_mae": (
            float(means[PATHS.index(selected_fixed_path)]) if selected_fixed_path is not None else None
        ),
        "aggregation": "equal weight per window with at least one finite hidden target",
        "oracle_status": "posthoc diagnostic using targets; not deployable",
    }


@contextmanager
def force_whole_window_path(backbone: torch.nn.Module, path: str):
    if path not in PATHS:
        raise ValueError(f"Unsupported intervention path: {path}")
    saved_mode, saved_path = backbone.routing_mode, backbone.fixed_path
    try:
        backbone.routing_mode = "fixed"
        backbone.fixed_path = tuple(path)
        yield
    finally:
        backbone.routing_mode, backbone.fixed_path = saved_mode, saved_path


def training_path_context(checkpoint_metrics: dict | None) -> dict:
    metrics = checkpoint_metrics or {}
    fractions = {}
    for path in PATHS:
        value = metrics.get(f"train_coe_path_{path}_fraction")
        fractions[path] = (
            float(value)
            if isinstance(value, (int, float)) and np.isfinite(value) and 0 <= value <= 1
            else None
        )
    available = any(value is not None for value in fractions.values())
    return {
        "source": "checkpoint.metrics.train_coe_path_*_fraction" if available else None,
        "path_fractions": fractions,
        "scope": "checkpoint's recorded training epoch, not lifetime path coverage",
        "interpretation": "null means unknown; low-use forced paths may be outside learned execution patterns",
    }


def _validate_model(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "main_branch", model)
    if (
        tuple(getattr(backbone, "expert_names", ())) != ("T", "S")
        or getattr(backbone, "num_steps", None) != 2
        or getattr(backbone, "routing_mode", None) != "hard"
        or not getattr(backbone, "use_routed", False)
    ):
        raise ValueError("Path analysis requires a two-round hard CoE with expert_pool [T, S] and use_routed=true")
    return backbone


def _window_statistics(outputs: dict, batch: dict, selected: torch.Tensor) -> dict[str, float]:
    target = batch["x_f_gt"]
    observed = batch["m_f"].bool().expand_as(target)
    missing = ~observed
    values = {}
    for label, prediction in (
        ("", outputs["x_hat_main"]),
        ("initial_", outputs["coe"]["initial_prediction"]),
        *((f"step{index}_", prediction) for index, prediction in enumerate(outputs["coe"]["predictions"], 1)),
    ):
        difference = prediction[selected].float() - target[selected].float()
        if not bool(torch.isfinite(difference).all()):
            raise ValueError("Nonfinite predictions at finite hidden supervision positions")
        values[f"{label}mae"] = float(difference.abs().mean())
        if not label:
            values["rmse"] = float(difference.square().mean().sqrt())
    for index, (change, completion) in enumerate(zip(outputs["coe"]["changes"], outputs["coe"]["completions"]), 1):
        if not torch.equal(completion[observed], batch["x_f_obs"][observed]):
            raise ValueError("A path intervention changed original input observations")
        values[f"step{index}_update_missing_mean"] = float(change[missing].float().mean())
    return values


@torch.no_grad()
def analyze_windows(
    model: torch.nn.Module,
    dataset,
    *,
    split: str,
    max_samples: int = 128,
    device: torch.device | str = "cpu",
    selected_fixed_path: str | None = None,
    checkpoint_metrics: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Evaluate policy plus four actual whole-window executions on first N rows."""
    backbone = _validate_model(model)
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    if split not in {"val", "test"} or (selected_fixed_path is not None and selected_fixed_path not in PATHS):
        raise ValueError("Invalid split or selected_fixed_path")
    limit = min(max_samples, len(dataset))
    rows, path_losses, policy_losses, skipped_indices = [], [], [], []
    accumulators = {name: MaskedMetricAccumulator() for name in ("policy", *PATHS)}
    routing = CoERoutingMetricAccumulator()
    original_training = model.training
    model.eval()
    try:
        for sample_index in range(limit):
            batch = {name: value.unsqueeze(0).to(device) for name, value in dataset[sample_index].items()}
            selected = supervision_mask(batch["x_f_gt"], batch["m_f"], batch.get("target_mask"))
            count = int(selected.sum())
            if not count:
                skipped_indices.append(sample_index)
                continue
            policy = model(batch)
            policy_path = "".join(backbone.expert_names[index] for index in policy["coe"]["paths"][0].tolist())
            routing.update(policy["coe"])
            row = {
                "sample_index": sample_index,
                "supervised_count": count,
                "missing_fraction": float((~batch["m_f"].bool()).float().mean()),
                "policy_path": policy_path,
                **{f"policy_{name}": value for name, value in _window_statistics(policy, batch, selected).items()},
            }
            accumulators["policy"].update(policy["x_hat_main"], batch["x_f_gt"], batch["m_f"], batch.get("target_mask"))
            window_losses = []
            for path in PATHS:
                with force_whole_window_path(backbone, path):
                    intervention = model(batch)
                stats = _window_statistics(intervention, batch, selected)
                row.update({f"path_{path}_{name}": value for name, value in stats.items()})
                row[f"path_{path}_mae_delta_vs_policy"] = stats["mae"] - row["policy_mae"]
                prediction_change = (intervention["x_hat_main"][selected] - policy["x_hat_main"][selected]).abs()
                row[f"path_{path}_prediction_change_hidden_mean"] = float(prediction_change.float().mean())
                if path == policy_path:
                    row["policy_forced_replay_prediction_linf"] = float(prediction_change.max())
                accumulators[path].update(intervention["x_hat_main"], batch["x_f_gt"], batch["m_f"], batch.get("target_mask"))
                window_losses.append(stats["mae"])
            oracle_index = int(np.argmin(window_losses))
            row["oracle_path_posthoc"] = PATHS[oracle_index]
            row["oracle_mae_posthoc"] = window_losses[oracle_index]
            row["policy_regret_posthoc"] = row["policy_mae"] - window_losses[oracle_index]
            row["swapped_policy_path"] = policy_path[::-1]
            row["swapped_policy_mae_delta"] = row[f"path_{policy_path[::-1]}_mae_delta_vs_policy"]
            rows.append(row)
            path_losses.append(window_losses)
            policy_losses.append(row["policy_mae"])
    finally:
        model.train(original_training)
    summary = summarize_path_losses(
        np.asarray(path_losses, dtype=np.float64).reshape(-1, len(PATHS)),
        np.asarray(policy_losses, dtype=np.float64), split=split, selected_fixed_path=selected_fixed_path,
    )
    summary.update({
        "scope": {
            "split": split,
            "dataset_sample_count": len(dataset),
            "max_samples": max_samples,
            "requested_subset_sample_count": limit,
            "evaluated_sample_count": len(rows),
            "skipped_zero_target_sample_count": len(skipped_indices),
            "skipped_zero_target_indices": skipped_indices,
            "sampling": "first min(max_samples, dataset_sample_count) windows in stored order",
            "units": "NPZ stored-value units; no inverse normalization applied",
            "supervision": "finite hidden targets only; original unavailable entries excluded",
        },
        "comparison": "same checkpoint weights with whole-window path interventions; not independently trained fixed-path baselines",
        "point_weighted_metrics": {name: accumulator.compute() for name, accumulator in accumulators.items()},
        "policy_routing_on_evaluated_windows": routing.compute(),
        "training_path_context": training_path_context(checkpoint_metrics),
        "mean_swapped_policy_mae_delta": float(np.mean([row["swapped_policy_mae_delta"] for row in rows])),
        "max_policy_forced_replay_prediction_linf": max(row["policy_forced_replay_prediction_linf"] for row in rows),
        "per_window_rows": rows,
    })
    return summary, rows


def run_analysis(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    with np.load(args.data_npz, allow_pickle=False) as arrays:
        conflicts = {"m_f", "target_mask"}.intersection(arrays.files)
        if conflicts:
            raise ValueError(f"NPZ contains embedded masks {sorted(conflicts)}; remove them to make --mask-csv the sole holdout source")
        key = "x_f_gt" if "x_f_gt" in arrays.files else "x_f"
        if key not in arrays.files or not arrays[key].shape[0]:
            raise ValueError("NPZ must contain a nonempty x_f_gt or x_f array")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint.get("config")
    if not isinstance(cfg, dict) or cfg.get("model", {}).get("architecture") != "v24_ts_coe":
        raise ValueError("Checkpoint must contain the saved v24_ts_coe configuration")
    device = get_device(args.device)
    model = DualBranchSTImputer.from_config(cfg).to(device)
    _validate_model(model)
    model.load_state_dict(checkpoint["model"])
    dataset = FlowNPZDataset(
        args.data_npz, mask_cfg={"pattern": "random"}, mask_csv=args.mask_csv,
        multiscale=False, seed=cfg.get("seed", 42),
    )
    summary, rows = analyze_windows(
        model, dataset, split=args.split, max_samples=args.max_samples, device=device,
        selected_fixed_path=args.selected_fixed_path, checkpoint_metrics=checkpoint.get("metrics"),
    )
    summary["inputs"] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_npz": str(Path(args.data_npz).resolve()),
        "mask_csv": str(Path(args.mask_csv).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "model_config_source": "checkpoint.config",
        "supplied_selected_fixed_path": args.selected_fixed_path,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_window.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-npz", required=True)
    parser.add_argument("--mask-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "auto"), default="auto")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--selected-fixed-path", choices=PATHS)
    return parser.parse_args(argv)


def main() -> None:
    summary = run_analysis(parse_args())
    print(json.dumps({name: summary[name] for name in (
        "selected_fixed_path", "selected_fixed_path_source", "oracle_gap_posthoc",
        "policy_regret_vs_window_oracle_posthoc", "scope",
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
