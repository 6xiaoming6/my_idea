#!/usr/bin/env python3
"""Diagnose missingness-induced distortion in the current multiscale pipeline.

This script is deliberately model-free.  It compares the ideal scales built from
complete validation windows with two incomplete-data constructions:

1. the current hierarchical path (fine -> mid -> coarse), and
2. a path-consistent direct aggregation of observed sums and counts.

The output is intended to falsify or support the V21 research hypothesis before
any trainable architecture is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stmoe_imputer.data.npz_dataset import FlowNPZDataset
from src.stmoe_imputer.data.transforms import masked_pool2d_spatial


DATASETS = {
    "TaxiBJ": {
        "val": ROOT / "data/TaxiBJ/taxibj_val.npz",
        "mask_root": ROOT / "data/TaxiBJ",
        "fine_to_mid": 2,
        "fine_to_coarse": 4,
    },
    "BikeNYC": {
        "val": ROOT / "data/BikeNYC/bikenyc_val.npz",
        "mask_root": ROOT / "data/BikeNYC",
        "fine_to_mid": 2,
        "fine_to_coarse": 4,
    },
    "CHAP": {
        "val": ROOT / "data/CHAP/beijing/chap_beijing_val.npz",
        "mask_root": ROOT / "data/CHAP/beijing",
        "fine_to_mid": 2,
        "fine_to_coarse": 4,
    },
}


@dataclass
class ErrorAccumulator:
    absolute_error: float = 0.0
    squared_error: float = 0.0
    reference_absolute: float = 0.0
    count: int = 0

    def update(self, estimate: torch.Tensor, reference: torch.Tensor) -> None:
        error = (estimate.double() - reference.double()).reshape(-1)
        ref = reference.double().reshape(-1)
        self.absolute_error += float(error.abs().sum())
        self.squared_error += float(error.square().sum())
        self.reference_absolute += float(ref.abs().sum())
        self.count += error.numel()

    def values(self) -> dict[str, float]:
        eps = 1e-12
        return {
            "mae": self.absolute_error / max(self.count, 1),
            "rmse": math.sqrt(self.squared_error / max(self.count, 1)),
            "sdi": self.absolute_error / max(self.reference_absolute, eps),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    parser.add_argument(
        "--patterns", nargs="+", default=["fixed", "random"], choices=["fixed", "random"]
    )
    parser.add_argument(
        "--rates", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 uses the complete validation split; positive values provide a quick check.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/v21-single/diagnostics/scale_distortion",
    )
    return parser.parse_args()


def _full_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    ones = torch.ones_like(x[:, :1])
    result, _ = masked_pool2d_spatial(
        x, ones, kernel_size=kernel_size, mode="avg"
    )
    return result


def _direct_masked_average(
    x: torch.Tensor, mask: torch.Tensor, kernel_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return masked_pool2d_spatial(
        x,
        mask,
        kernel_size=kernel_size,
        mode="avg",
        return_reliability=True,
    )


def _hierarchical_measure_average(
    x: torch.Tensor,
    mask: torch.Tensor,
    fine_to_mid: int,
    fine_to_coarse: int,
) -> torch.Tensor:
    """Aggregate additive observed sums/counts through mid before normalizing."""
    b, c, t, h, w = x.shape
    x_2d = (x * mask).permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    m_2d = mask.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
    mid_area = float(fine_to_mid * fine_to_mid)
    mid_sum = F.avg_pool2d(x_2d, fine_to_mid, fine_to_mid) * mid_area
    mid_count = F.avg_pool2d(m_2d, fine_to_mid, fine_to_mid) * mid_area
    ratio = fine_to_coarse // fine_to_mid
    parent_area = float(ratio * ratio)
    coarse_sum = F.avg_pool2d(mid_sum, ratio, ratio) * parent_area
    coarse_count = F.avg_pool2d(mid_count, ratio, ratio) * parent_area
    coarse = coarse_sum / coarse_count.clamp_min(1e-6)
    coarse = coarse * (coarse_count > 0).to(dtype=coarse.dtype)
    h2, w2 = coarse.shape[-2:]
    return coarse.reshape(b, t, c, h2, w2).permute(0, 2, 1, 3, 4).contiguous()


def _parent_count_cv(mask: torch.Tensor, fine_to_mid: int, fine_to_coarse: int) -> torch.Tensor:
    """CV of observed fine-cell counts among the mid children of each coarse cell."""
    b, _, t, h, w = mask.shape
    fine = mask.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
    area_mid = float(fine_to_mid * fine_to_mid)
    mid_counts = F.avg_pool2d(fine, fine_to_mid, fine_to_mid) * area_mid
    ratio = max(1, fine_to_coarse // fine_to_mid)
    children = F.unfold(mid_counts, kernel_size=ratio, stride=ratio)
    children = children.transpose(1, 2).reshape(-1, ratio * ratio)
    mean = children.mean(dim=-1)
    std = children.std(dim=-1, unbiased=False)
    return std / mean.clamp_min(1e-6)


def _coarse_cell_error(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # Average channels so each item corresponds to one (sample,time,coarse-cell).
    return (left - right).abs().mean(dim=1).reshape(-1)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right)
    left = left[keep]
    right = right[keep]
    if left.size < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    # Average ranks for ties. Exact ties are common for mask-derived coverage.
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right)
    left = left[keep]
    right = right[keep]
    if left.size < 2:
        return float("nan")
    return _pearson(_rank(left), _rank(right))


def _coverage_bins(coverage: np.ndarray, error: np.ndarray) -> list[dict[str, float]]:
    edges = np.asarray([0.0, 0.25, 0.5, 0.75, 1.000001])
    rows: list[dict[str, float]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (coverage >= low) & (coverage < high)
        rows.append(
            {
                "low": float(low),
                "high": float(min(high, 1.0)),
                "count": int(selected.sum()),
                "mean_absolute_error": (
                    float(error[selected].mean()) if selected.any() else float("nan")
                ),
            }
        )
    return rows


def diagnose_one(
    dataset_name: str,
    pattern: str,
    rate: float,
    batch_size: int,
    max_samples: int,
) -> dict:
    spec = DATASETS[dataset_name]
    mask_csv = spec["mask_root"] / f"{pattern}_mask/{rate:.1f}/val.csv"
    dataset = FlowNPZDataset(
        spec["val"],
        mask_cfg={"pattern": pattern, "missing_rate": rate},
        fine_to_mid=spec["fine_to_mid"],
        fine_to_coarse=spec["fine_to_coarse"],
        pooling_mode="avg",
        mask_csv=mask_csv,
    )
    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    errors = {
        "mid_hierarchical": ErrorAccumulator(),
        "coarse_hierarchical": ErrorAccumulator(),
        "coarse_path_consistent": ErrorAccumulator(),
        "coarse_path_disagreement": ErrorAccumulator(),
        "measure_path_disagreement": ErrorAccumulator(),
    }
    coverages: list[np.ndarray] = []
    coarse_direct_errors: list[np.ndarray] = []
    count_cvs: list[np.ndarray] = []
    path_errors: list[np.ndarray] = []
    max_path_error = 0.0

    for batch in loader:
        x = batch["x_f_gt"].float()
        mask = batch["m_f"].float()
        ideal_mid = _full_average(x, spec["fine_to_mid"])
        ideal_coarse = _full_average(x, spec["fine_to_coarse"])
        direct_coarse, _, coarse_reliability = _direct_masked_average(
            x, mask, spec["fine_to_coarse"]
        )
        hierarchical_measure_coarse = _hierarchical_measure_average(
            x,
            mask,
            spec["fine_to_mid"],
            spec["fine_to_coarse"],
        )
        hierarchical_mid = batch["x_m_obs"].float()
        hierarchical_coarse = batch["x_c_obs"].float()

        errors["mid_hierarchical"].update(hierarchical_mid, ideal_mid)
        errors["coarse_hierarchical"].update(hierarchical_coarse, ideal_coarse)
        errors["coarse_path_consistent"].update(direct_coarse, ideal_coarse)
        errors["coarse_path_disagreement"].update(hierarchical_coarse, direct_coarse)
        errors["measure_path_disagreement"].update(
            hierarchical_measure_coarse, direct_coarse
        )

        cell_direct_error = _coarse_cell_error(direct_coarse, ideal_coarse)
        cell_path_error = _coarse_cell_error(hierarchical_coarse, direct_coarse)
        cell_cv = _parent_count_cv(mask, spec["fine_to_mid"], spec["fine_to_coarse"])
        cell_coverage = coarse_reliability.reshape(-1)
        coverages.append(cell_coverage.numpy())
        coarse_direct_errors.append(cell_direct_error.numpy())
        count_cvs.append(cell_cv.numpy())
        path_errors.append(cell_path_error.numpy())
        max_path_error = max(max_path_error, float(cell_path_error.max()))

    coverage = np.concatenate(coverages)
    direct_error = np.concatenate(coarse_direct_errors)
    count_cv = np.concatenate(count_cvs)
    path_error = np.concatenate(path_errors)
    metrics = {name: accumulator.values() for name, accumulator in errors.items()}
    old_sdi = metrics["coarse_hierarchical"]["sdi"]
    new_sdi = metrics["coarse_path_consistent"]["sdi"]
    reduction = 100.0 * (old_sdi - new_sdi) / max(old_sdi, 1e-12)

    return {
        "dataset": dataset_name,
        "pattern": pattern,
        "rate": rate,
        "samples": len(dataset),
        "metrics": metrics,
        "coarse_sdi_reduction_percent": reduction,
        "max_path_absolute_error": max_path_error,
        "correlations": {
            "coverage_vs_direct_error_pearson": _pearson(coverage, direct_error),
            "coverage_vs_direct_error_spearman": _spearman(coverage, direct_error),
            "child_count_cv_vs_path_error_pearson": _pearson(count_cv, path_error),
            "child_count_cv_vs_path_error_spearman": _spearman(count_cv, path_error),
        },
        "coverage_error_bins": _coverage_bins(coverage, direct_error),
    }


def _fmt(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.6f}"


def write_report(results: Iterable[dict], output_path: Path) -> None:
    results = list(results)
    lines = [
        "# Missingness-Induced Scale Distortion Diagnostic",
        "",
        "The current hierarchical coarse scale averages non-empty mid-cell means with equal ",
        "weights. The path-consistent construction instead aggregates fine-level observed ",
        "sums and counts, so it is identical whether computed directly or through levels.",
        "",
        "| Dataset | Pattern | Rate | Mid SDI | Current coarse SDI | Path-consistent coarse SDI | SDI reduction | Path disagreement | Count-CV/path Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        metrics = row["metrics"]
        corr = row["correlations"]
        lines.append(
            "| {dataset} | {pattern} | {rate:.1f} | {mid} | {old} | {new} | "
            "{reduction:+.2f}% | {path} | {corr} |".format(
                dataset=row["dataset"],
                pattern=row["pattern"],
                rate=row["rate"],
                mid=_fmt(metrics["mid_hierarchical"]["sdi"]),
                old=_fmt(metrics["coarse_hierarchical"]["sdi"]),
                new=_fmt(metrics["coarse_path_consistent"]["sdi"]),
                reduction=row["coarse_sdi_reduction_percent"],
                path=_fmt(metrics["coarse_path_disagreement"]["sdi"]),
                corr=_fmt(corr["child_count_cv_vs_path_error_spearman"]),
            )
        )

    old_mean = float(np.mean([r["metrics"]["coarse_hierarchical"]["sdi"] for r in results]))
    new_mean = float(np.mean([r["metrics"]["coarse_path_consistent"]["sdi"] for r in results]))
    lines.extend(
        [
            "",
            "## Aggregate conclusion",
            "",
            f"- Mean current coarse SDI: `{old_mean:.6f}`",
            f"- Mean path-consistent coarse SDI: `{new_mean:.6f}`",
            f"- Relative mean reduction: `{100.0 * (old_mean - new_mean) / max(old_mean, 1e-12):.2f}%`",
            "- A non-zero path disagreement directly demonstrates that missingness and the current hierarchical scale construction do not commute.",
            "- The additive sum/count hierarchy matches direct fine-to-coarse aggregation up to floating-point rounding; see `measure_path_disagreement` in the JSON artifact.",
            "- This diagnostic only validates representation distortion. It does not by itself prove that a trainable moment pyramid improves final imputation.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    total = len(args.datasets) * len(args.patterns) * len(args.rates)
    index = 0
    for dataset_name in args.datasets:
        for pattern in args.patterns:
            for rate in args.rates:
                index += 1
                print(f"[{index}/{total}] {dataset_name} {pattern}@{rate:.1f}", flush=True)
                result = diagnose_one(
                    dataset_name,
                    pattern,
                    rate,
                    args.batch_size,
                    args.max_samples,
                )
                results.append(result)
                metric = result["metrics"]
                print(
                    "  coarse SDI: "
                    f"{metric['coarse_hierarchical']['sdi']:.6f} -> "
                    f"{metric['coarse_path_consistent']['sdi']:.6f} "
                    f"({result['coarse_sdi_reduction_percent']:+.2f}%)",
                    flush=True,
                )

    json_path = args.output_dir / "scale_distortion_results.json"
    report_path = args.output_dir / "scale_distortion_report.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results, report_path)
    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
