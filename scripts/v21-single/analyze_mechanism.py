#!/usr/bin/env python3
"""Analyze evidence calibration and expert specialization for a trained V21 P2 run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stmoe_imputer.data.npz_dataset import FlowNPZDataset
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import load_checkpoint
from stmoe_imputer.utils.device import move_batch_to_device


DATA = {
    "TaxiBJ": {
        "val": "data/TaxiBJ/taxibj_val.npz",
        "test": "data/TaxiBJ/taxibj_test.npz",
    },
    "BikeNYC": {
        "val": "data/BikeNYC/bikenyc_val.npz",
        "test": "data/BikeNYC/bikenyc_test.npz",
    },
    "CHAP_Beijing": {
        "val": "data/CHAP/beijing/chap_beijing_val.npz",
        "test": "data/CHAP/beijing/chap_beijing_test.npz",
    },
}
EVIDENCE_NAMES = (
    "coverage",
    "relative_value_variance",
    "centroid_y",
    "centroid_x",
    "spread_y",
    "spread_x",
    "covariance_yx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if left.size < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def _sample_missing_mae(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    hidden = (1.0 - mask).expand_as(output)
    count = hidden.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    return ((output - target).abs() * hidden).sum(dim=(1, 2, 3, 4)) / count


def _dataset(cfg: dict, split: str) -> FlowNPZDataset:
    data_cfg = cfg["data"]
    dataset_name = data_cfg["dataset_name"]
    if dataset_name not in DATA:
        raise ValueError(f"Unsupported dataset_name={dataset_name!r}")
    mask_cfg = data_cfg["mask"]
    mask_csv = mask_cfg.get(f"{split}_csv")
    if mask_csv is None:
        raise ValueError(f"Saved config has no data.mask.{split}_csv")
    scale_cfg = data_cfg["scales"]
    return FlowNPZDataset(
        ROOT / DATA[dataset_name][split],
        mask_cfg=mask_cfg,
        fine_to_mid=scale_cfg["fine_to_mid"],
        fine_to_coarse=scale_cfg["fine_to_coarse"],
        pooling_mode=scale_cfg.get("pooling_mode", "avg"),
        pyramid_mode=scale_cfg.get("pyramid_mode", "legacy"),
        seed=int(cfg.get("seed", 42)) + (20000 if split == "val" else 30000),
        mask_csv=ROOT / mask_csv,
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoints/best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Expected config.json and checkpoints/best.pt under {run_dir}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_policy", {}).get("name") != "observation_moment_dual_state":
        raise ValueError("Mechanism analysis requires a completed P2 dual-state run")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = _dataset(cfg, args.split)
    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = DualBranchSTImputer.from_config(cfg).to(device).eval()
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)

    values: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            error = _sample_missing_mae(
                outputs["x_hat_final"], batch["x_f_gt"], batch["m_f"]
            )
            values["missing_mae"].append(error.cpu().numpy())
            for scale, evidence_key in (("mid", "e_m"), ("coarse", "e_c")):
                evidence = batch[evidence_key].float().mean(dim=(2, 3, 4))
                for index, name in enumerate(EVIDENCE_NAMES):
                    component = evidence[:, index]
                    if name in {"centroid_y", "centroid_x", "covariance_yx"}:
                        component = component.abs()
                    values[f"{scale}_{name}"].append(component.cpu().numpy())
                gate = outputs["gates"][scale].float()
                for expert in range(gate.shape[1]):
                    values[f"{scale}_expert_{expert}_gate"].append(gate[:, expert].cpu().numpy())
                values[f"{scale}_top1"].append(gate.argmax(dim=1).cpu().numpy())

    arrays = {key: np.concatenate(parts) for key, parts in values.items()}
    error = arrays["missing_mae"]
    evidence_error = {}
    gate_correlations = []
    specialization = []
    for scale in ("mid", "coarse"):
        for name in EVIDENCE_NAMES:
            evidence_key = f"{scale}_{name}"
            evidence_error[evidence_key] = _spearman(arrays[evidence_key], error)
            expert = 0
            while f"{scale}_expert_{expert}_gate" in arrays:
                gate_correlations.append({
                    "scale": scale,
                    "expert": expert,
                    "evidence": name,
                    "spearman": _spearman(
                        arrays[evidence_key], arrays[f"{scale}_expert_{expert}_gate"]
                    ),
                })
                expert += 1
        assignments = arrays[f"{scale}_top1"].astype(np.int64)
        for expert in sorted(set(assignments.tolist())):
            selected = assignments == expert
            row = {
                "scale": scale,
                "expert": int(expert),
                "samples": int(selected.sum()),
                "missing_mae": float(error[selected].mean()),
            }
            for name in EVIDENCE_NAMES:
                row[name] = float(arrays[f"{scale}_{name}"][selected].mean())
            specialization.append(row)

    finite_gate = [row for row in gate_correlations if math.isfinite(row["spearman"])]
    strongest = sorted(finite_gate, key=lambda row: abs(row["spearman"]), reverse=True)[:20]
    result = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "split": args.split,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "samples": int(error.size),
        "mean_missing_mae": float(error.mean()),
        "evidence_error_spearman": evidence_error,
        "strongest_evidence_gate_spearman": strongest,
        "expert_specialization": specialization,
    }

    output = args.output or run_dir / f"mechanism_analysis_{args.split}.md"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V21 Evidence Calibration and Expert Specialization",
        "",
        f"- Run: `{result['run_dir']}`",
        f"- Split: `{args.split}`",
        f"- Best checkpoint epoch: `{result['checkpoint_epoch']}`",
        f"- Samples: `{result['samples']}`",
        f"- Mean missing MAE: `{result['mean_missing_mae']:.6f}`",
        "",
        "## Evidence vs missing-value error (Spearman)",
        "",
        "| Evidence | Correlation |",
        "|---|---:|",
    ]
    for key, correlation in evidence_error.items():
        text = "nan" if not math.isfinite(correlation) else f"{correlation:+.4f}"
        lines.append(f"| {key} | {text} |")
    lines.extend((
        "",
        "## Strongest evidence–expert gate relationships",
        "",
        "| Scale | Expert | Evidence | Spearman |",
        "|---|---:|---|---:|",
    ))
    for row in strongest:
        lines.append(
            f"| {row['scale']} | {row['expert']} | {row['evidence']} | "
            f"{row['spearman']:+.4f} |"
        )
    lines.extend((
        "",
        "## Top-1 expert regimes",
        "",
        "| Scale | Expert | N | Missing MAE | Coverage | RelVar | |Cy| | |Cx| | SpreadY | SpreadX | |CovYX| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for row in specialization:
        lines.append(
            f"| {row['scale']} | {row['expert']} | {row['samples']} | "
            f"{row['missing_mae']:.4f} | {row['coverage']:.4f} | "
            f"{row['relative_value_variance']:.4f} | {row['centroid_y']:.4f} | "
            f"{row['centroid_x']:.4f} | {row['spread_y']:.4f} | "
            f"{row['spread_x']:.4f} | {row['covariance_yx']:.4f} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {output}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
