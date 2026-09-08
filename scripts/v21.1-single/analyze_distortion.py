#!/usr/bin/env python3
"""Measure whether V21.1 acceptance follows real scale distortion on validation data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stmoe_imputer.data.npz_dataset import FlowNPZDataset
from stmoe_imputer.data.transforms import masked_pool2d_spatial
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils.checkpoint import load_checkpoint
from stmoe_imputer.utils.device import move_batch_to_device


DATA = {
    "TaxiBJ": "data/TaxiBJ/taxibj_val.npz",
    "BikeNYC": "data/BikeNYC/bikenyc_val.npz",
    "CHAP_Beijing": "data/CHAP/beijing/chap_beijing_val.npz",
}


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if left.size < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rank(left), rank(right))[0, 1])


def sample_mae(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (value - target).abs().mean(dim=(1, 2, 3, 4))


def missing_mae(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    hidden = (1.0 - mask).expand_as(output)
    count = hidden.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    return ((output - target).abs() * hidden).sum(dim=(1, 2, 3, 4)) / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "checkpoints/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    dataset_name = cfg["data"]["dataset_name"]
    if dataset_name not in DATA:
        raise ValueError(f"Unsupported dataset_name={dataset_name!r}")
    scale_cfg = cfg["data"]["scales"]
    mask_cfg = cfg["data"]["mask"]
    dataset = FlowNPZDataset(
        ROOT / DATA[dataset_name],
        mask_cfg=mask_cfg,
        fine_to_mid=scale_cfg["fine_to_mid"],
        fine_to_coarse=scale_cfg["fine_to_coarse"],
        pooling_mode=scale_cfg.get("pooling_mode", "avg"),
        pyramid_mode=scale_cfg.get("pyramid_mode", "legacy"),
        seed=int(cfg.get("seed", 42)) + 20000,
        mask_csv=ROOT / mask_cfg["val_csv"],
    )
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = DualBranchSTImputer.from_config(cfg).to(device).eval()
    checkpoint_record = load_checkpoint(checkpoint, model, map_location=device)
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    # Fine→Mid is identical for both paths in the current hierarchy.  Analyze
    # only Coarse, where repeated Legacy pooling actually creates the tested
    # non-commutativity.
    kernels = {"coarse": scale_cfg["fine_to_coarse"]}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            features = output["features"]["v21_1"]
            values["missing_mae"].append(
                missing_mae(output["x_hat_final"], batch["x_f_gt"], batch["m_f"]).cpu().numpy()
            )
            full_mask = torch.ones_like(batch["m_f"])
            for scale, kernel in kernels.items():
                ideal, _, _ = masked_pool2d_spatial(
                    batch["x_f_gt"], full_mask, kernel_size=kernel, return_reliability=True
                )
                legacy_error = sample_mae(features[f"legacy_{scale}"], ideal)
                measure_error = sample_mae(features[f"measure_{scale}"], ideal)
                calibrated_error = sample_mae(features[f"calibrated_{scale}"], ideal)
                values[f"alpha_{scale}"].append(
                    features[f"alpha_{scale}"].mean(dim=(1, 2, 3, 4)).cpu().numpy()
                )
                values[f"distortion_{scale}"].append(
                    features[f"distortion_{scale}"].mean(dim=(1, 2, 3, 4)).cpu().numpy()
                )
                values[f"legacy_error_{scale}"].append(legacy_error.cpu().numpy())
                values[f"measure_error_{scale}"].append(measure_error.cpu().numpy())
                values[f"calibrated_error_{scale}"].append(calibrated_error.cpu().numpy())
                values[f"measure_advantage_{scale}"].append(
                    (legacy_error - measure_error).cpu().numpy()
                )
    arrays = {key: np.concatenate(parts) for key, parts in values.items()}
    result = {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "checkpoint_epoch": checkpoint_record.get("epoch"),
        "samples": int(arrays["missing_mae"].size),
        "scales": {},
    }
    for scale in ("coarse",):
        result["scales"][scale] = {
            "alpha_mean": float(arrays[f"alpha_{scale}"].mean()),
            "distortion_mean": float(arrays[f"distortion_{scale}"].mean()),
            "legacy_scale_mae": float(arrays[f"legacy_error_{scale}"].mean()),
            "measure_scale_mae": float(arrays[f"measure_error_{scale}"].mean()),
            "calibrated_scale_mae": float(arrays[f"calibrated_error_{scale}"].mean()),
            "distortion_error_spearman": spearman(
                arrays[f"distortion_{scale}"], arrays[f"legacy_error_{scale}"]
            ),
            "distortion_alpha_spearman": spearman(
                arrays[f"distortion_{scale}"], arrays[f"alpha_{scale}"]
            ),
            "advantage_alpha_spearman": spearman(
                arrays[f"measure_advantage_{scale}"], arrays[f"alpha_{scale}"]
            ),
        }
    combined_distortion = arrays["distortion_coarse"]
    quartiles = []
    for index, selected in enumerate(np.array_split(np.argsort(combined_distortion), 4), 1):
        quartiles.append({
            "quartile": index,
            "samples": int(selected.size),
            "distortion": float(combined_distortion[selected].mean()),
            "missing_mae": float(arrays["missing_mae"][selected].mean()),
            "alpha_coarse": float(arrays["alpha_coarse"][selected].mean()),
        })
    result["distortion_quartiles"] = quartiles
    output = run_dir / "distortion_analysis_val.md"
    lines = [
        "# V21.1 Validation Distortion Analysis",
        "",
        f"- Run: `{result['run_dir']}`",
        f"- Best checkpoint epoch: `{result['checkpoint_epoch']}`",
        f"- Samples: `{result['samples']}`",
        "",
        "## Scale calibration",
        "",
        "| Scale | Alpha | Distortion | Legacy MAE | Measure MAE | Calibrated MAE | ρ(D,error) | ρ(D,alpha) | ρ(advantage,alpha) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scale, value in result["scales"].items():
        def number(item: float) -> str:
            return "nan" if not math.isfinite(item) else f"{item:.5f}"
        lines.append(
            f"| {scale} | {value['alpha_mean']:.5f} | {value['distortion_mean']:.5f} | "
            f"{value['legacy_scale_mae']:.5f} | {value['measure_scale_mae']:.5f} | "
            f"{value['calibrated_scale_mae']:.5f} | {number(value['distortion_error_spearman'])} | "
            f"{number(value['distortion_alpha_spearman'])} | "
            f"{number(value['advantage_alpha_spearman'])} |"
        )
    lines.extend((
        "",
        "## Distortion quartiles",
        "",
        "| Quartile | N | Distortion | Missing MAE | Alpha coarse |",
        "|---:|---:|---:|---:|---:|",
    ))
    for value in quartiles:
        lines.append(
            f"| Q{value['quartile']} | {value['samples']} | {value['distortion']:.5f} | "
            f"{value['missing_mae']:.5f} | {value['alpha_coarse']:.5f} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote: {output}")
    print(f"Wrote: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
