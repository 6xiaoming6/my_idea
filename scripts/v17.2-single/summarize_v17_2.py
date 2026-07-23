#!/usr/bin/env python3
"""Summarize V17.2 results against the explicit paired Full manifest."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pipeline_common import (
    CORE_POINTS,
    DATASETS,
    OUTPUT_ROOT,
    RATES,
    ROOT,
    Job,
    baseline_key,
    completed_run,
    load_json,
    read_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("all", *DATASETS),
        default=["all"],
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=("fixed", "random"),
        default=["fixed", "random"],
    )
    parser.add_argument("--rates", nargs="+", choices=RATES, default=list(RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--points",
        nargs="+",
        choices=("core4", *CORE_POINTS),
        default=None,
        help="Use V17.1 core points instead of a dataset/pattern/rate grid.",
    )
    parser.add_argument(
        "--baseline-manifest",
        default="configs/v17.2-single/baseline_manifest.json",
    )
    parser.add_argument("--name", default="v17_2_results")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _manifest(path_text: str) -> dict:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return load_json(path).get("entries", {})


def _jobs(args: argparse.Namespace) -> list[Job]:
    if args.points:
        point_ids = list(CORE_POINTS) if "core4" in args.points else list(
            dict.fromkeys(args.points)
        )
        return [
            Job(*CORE_POINTS[point], seed)
            for seed in args.seeds
            for point in point_ids
        ]
    datasets = list(DATASETS) if "all" in args.datasets else list(
        dict.fromkeys(args.datasets)
    )
    patterns = [
        pattern for pattern in ("fixed", "random") if pattern in args.patterns
    ]
    return [
        Job(dataset, pattern, rate, seed)
        for seed in args.seeds
        for dataset in datasets
        for pattern in patterns
        for rate in args.rates
    ]


def collect(jobs: list[Job], manifest: dict) -> tuple[list[dict], list[str]]:
    rows = []
    missing = []
    for job in jobs:
        key = baseline_key(job)
        baseline = manifest.get(key)
        run_dir = completed_run(job)
        if baseline is None or run_dir is None:
            missing.append(
                f"{job.label}: "
                f"{'baseline ' if baseline is None else ''}"
                f"{'candidate' if run_dir is None else ''}".strip()
            )
            continue
        result = read_result(run_dir)
        if result is None:
            missing.append(f"{job.label}: non-finite candidate result")
            continue
        full_mae = float(baseline["test_mae"])
        full_rmse = float(baseline["test_rmse"])
        relative = (result["mae"] - full_mae) / full_mae * 100.0
        parameter_report = load_json(run_dir / "parameter_report.json")
        rows.append(
            {
                "dataset": job.dataset,
                "pattern": job.mask,
                "rate": float(job.rate),
                "seed": job.seed,
                "full_mae": full_mae,
                "v17_2_mae": result["mae"],
                "relative_mae_pct": relative,
                "win": result["mae"] < full_mae,
                "full_rmse": full_rmse,
                "v17_2_rmse": result["rmse"],
                "best_epoch": result["extra"].get("best_epoch"),
                "best_val_mae": result["extra"].get("best_val_mae"),
                "completed_epochs": result["completed_epochs"],
                "total_time_sec": result["total_time_sec"],
                "peak_memory_gb": result["peak_memory_gb"],
                "parameters": parameter_report.get("total_parameters"),
                "adapter_parameters": parameter_report.get(
                    "adapter_parameter_count"
                ),
                "baseline_run_dir": baseline["run_dir"],
                "run_dir": str(run_dir.relative_to(ROOT)),
            }
        )
    return rows, missing


def _mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def build_markdown(rows: list[dict], missing: list[str], expected: int) -> str:
    lines = [
        "# V17.2 Adapter-Free HSA-MoE 实验汇总",
        "",
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}  ",
        "> 主指标：原始数据范围、缺失位置 Test MAE/RMSE。  ",
        "> Relative 为相对显式配对 Full V17 的 MAE 变化，负数表示改善。",
        "",
        "## 逐点结果",
        "",
        "| Dataset | Pattern | Rate | Seed | Full MAE | V17.2 MAE | Relative | Win | Full RMSE | V17.2 RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            item["dataset"],
            item["pattern"],
            item["rate"],
            item["seed"],
        ),
    ):
        lines.append(
            f"| {row['dataset']} | {row['pattern']} | {row['rate']:.1f} | "
            f"{row['seed']} | {row['full_mae']:.6f} | "
            f"{row['v17_2_mae']:.6f} | {row['relative_mae_pct']:+.2f}% | "
            f"{'✓' if row['win'] else '✗'} | {row['full_rmse']:.6f} | "
            f"{row['v17_2_rmse']:.6f} |"
        )

    lines += [
        "",
        "## 数据集与 Pattern 平均",
        "",
        "| Dataset | Pattern | N | Full Avg MAE | V17.2 Avg MAE | Paired Relative Avg | Wins |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["pattern"])].append(row)
    for (dataset, pattern), group in sorted(groups.items()):
        lines.append(
            f"| {dataset} | {pattern} | {len(group)} | "
            f"{_mean(group, 'full_mae'):.6f} | "
            f"{_mean(group, 'v17_2_mae'):.6f} | "
            f"{_mean(group, 'relative_mae_pct'):+.2f}% | "
            f"{sum(bool(row['win']) for row in group)}/{len(group)} |"
        )

    if rows:
        lines += [
            "",
            "## 总体",
            "",
            f"- 配对平均 MAE 变化：{_mean(rows, 'relative_mae_pct'):+.2f}%",
            f"- 获胜数：{sum(bool(row['win']) for row in rows)}/{len(rows)}",
            f"- 最大单点改善：{min(row['relative_mae_pct'] for row in rows):+.2f}%",
            f"- 最大单点退化：{max(row['relative_mae_pct'] for row in rows):+.2f}%",
            f"- Adapter 参数检查："
            f"{sum(row['adapter_parameters'] != 0 for row in rows)} 个异常",
        ]

    core_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for point, spec in CORE_POINTS.items():
            if (
                row["dataset"],
                row["pattern"],
                f"{row['rate']:.1f}",
            ) == spec:
                core_rows[point].append(row)
    if any(len(group) > 1 for group in core_rows.values()):
        lines += [
            "",
            "## 核心点多 Seed 稳定性",
            "",
            "| Point | N | V17.2 MAE mean±std | Paired Relative Avg | Win Seeds |",
            "|---|---:|---:|---:|---:|",
        ]
        for point in CORE_POINTS:
            group = core_rows.get(point, [])
            if not group:
                continue
            values = [float(row["v17_2_mae"]) for row in group]
            lines.append(
                f"| {point} | {len(group)} | "
                f"{statistics.fmean(values):.6f}±{statistics.pstdev(values):.6f} | "
                f"{_mean(group, 'relative_mae_pct'):+.2f}% | "
                f"{sum(bool(row['win']) for row in group)}/{len(group)} |"
            )

    lines += [
        "",
        "## 完整性",
        "",
        f"- 计划：{expected}",
        f"- 完成：{len(rows)}",
        f"- 缺失：{expected - len(rows)}",
    ]
    if missing:
        lines += ["", "缺失项：", ""]
        lines.extend(f"- {item}" for item in missing)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    jobs = _jobs(args)
    rows, missing = collect(jobs, _manifest(args.baseline_manifest))
    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = summary_dir / f"{args.name}.md"
    csv_path = summary_dir / f"{args.name}.csv"
    json_path = summary_dir / f"{args.name}.json"

    markdown = build_markdown(rows, missing, len(jobs))
    markdown_path.write_text(markdown, encoding="utf-8")
    fields = (
        "dataset",
        "pattern",
        "rate",
        "seed",
        "full_mae",
        "v17_2_mae",
        "relative_mae_pct",
        "win",
        "full_rmse",
        "v17_2_rmse",
        "best_epoch",
        "best_val_mae",
        "completed_epochs",
        "total_time_sec",
        "peak_memory_gb",
        "parameters",
        "adapter_parameters",
        "baseline_run_dir",
        "run_dir",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(markdown)
    print(f"[saved] {markdown_path.relative_to(ROOT)}")
    print(f"[saved] {csv_path.relative_to(ROOT)}")
    print(f"[saved] {json_path.relative_to(ROOT)}")
    if args.require_complete and missing:
        raise SystemExit(
            f"Incomplete V17.2 summary: expected {len(jobs)}, found {len(rows)}"
        )


if __name__ == "__main__":
    main()
