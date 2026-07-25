#!/usr/bin/env python3
"""Summarize V18 results and pair them with matching V14 runs when available."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pipeline_common import (
    ABLATIONS,
    CORE_POINTS,
    DATASETS,
    OUTPUT_ROOT,
    PATTERNS,
    RATES,
    ROOT,
    SCREENING_POINTS,
    Job,
    build_resolved_config,
    compatible_v14_run,
    completed_run,
    read_result,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("screening", "core4", "full24"),
        default="full24",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=("all", *DATASETS), default=["all"]
    )
    parser.add_argument(
        "--patterns", nargs="+", choices=PATTERNS, default=list(PATTERNS)
    )
    parser.add_argument("--rates", nargs="+", choices=RATES, default=list(RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--ablation", choices=("none", *ABLATIONS), default="none")
    parser.add_argument("--training-policy", default=None)
    parser.add_argument("--name", default="v18_results")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _jobs(args: argparse.Namespace) -> list[Job]:
    if args.stage == "screening":
        points = SCREENING_POINTS
    elif args.stage == "core4":
        points = tuple(CORE_POINTS.values())
    else:
        datasets = list(DATASETS) if "all" in args.datasets else list(
            dict.fromkeys(args.datasets)
        )
        patterns = [
            pattern for pattern in PATTERNS if pattern in set(args.patterns)
        ]
        points = tuple(
            (dataset, pattern, rate)
            for dataset in datasets
            for pattern in patterns
            for rate in args.rates
        )
    return [
        Job(*point, seed, args.ablation)
        for seed in args.seeds
        for point in points
    ]


def collect(
    jobs: list[Job],
    training_policy: Path | None = None,
) -> tuple[list[dict], list[str], list[str], list[str]]:
    rows = []
    missing_candidate = []
    missing_baseline = []
    incompatible_baseline = []
    for job in jobs:
        expected_config = build_resolved_config(
            job.dataset,
            job.mask,
            job.rate,
            job.seed,
            ablation=job.ablation,
            training_policy=training_policy,
        )
        candidate_dir = completed_run(job, expected_config=expected_config)
        if candidate_dir is None:
            missing_candidate.append(job.label)
            continue
        candidate = read_result(candidate_dir)
        if candidate is None:
            missing_candidate.append(f"{job.label}: non-finite result")
            continue
        baseline_dir, incompatible = compatible_v14_run(
            job, candidate_config=expected_config
        )
        baseline = read_result(baseline_dir) if baseline_dir is not None else None
        if baseline is None:
            missing_baseline.append(job.label)
            if incompatible:
                incompatible_baseline.append(
                    f"{job.label}: {len(incompatible)} historical run(s) "
                    "found, but their paper protocol differs"
                )
        v14_mae = baseline["mae"] if baseline else None
        relative = (
            (candidate["mae"] - v14_mae) / v14_mae * 100.0
            if v14_mae is not None
            else None
        )
        rows.append(
            {
                "dataset": job.dataset,
                "pattern": job.mask,
                "rate": float(job.rate),
                "seed": job.seed,
                "ablation": job.ablation,
                "v14_mae": v14_mae,
                "v18_mae": candidate["mae"],
                "relative_mae_pct": relative,
                "win": relative is not None and relative < 0.0,
                "v14_rmse": baseline["rmse"] if baseline else None,
                "v18_rmse": candidate["rmse"],
                "best_epoch": candidate["extra"].get("best_epoch"),
                "best_val_mae": candidate["extra"].get("best_val_mae"),
                "completed_epochs": candidate["completed_epochs"],
                "total_time_sec": candidate["total_time_sec"],
                "peak_memory_gb": candidate["peak_memory_gb"],
                "baseline_run_dir": (
                    str(baseline_dir.relative_to(ROOT)) if baseline_dir else None
                ),
                "run_dir": str(candidate_dir.relative_to(ROOT)),
            }
        )
    return (
        rows,
        missing_candidate,
        missing_baseline,
        incompatible_baseline,
    )


def _mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _fmt(value: object, digits: int = 6) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def markdown(
    rows: list[dict],
    missing_candidate: list[str],
    missing_baseline: list[str],
    incompatible_baseline: list[str],
    expected: int,
) -> str:
    paired = [row for row in rows if row["relative_mae_pct"] is not None]
    lines = [
        "# V18 BARP-MoE 实验汇总",
        "",
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}  ",
        "> 指标：原始范围、缺失位置 Test MAE/RMSE。  ",
        "> Relative 为 V18 相对同点同 seed V14 的 MAE 变化，负数表示改善。",
        "",
        "## 逐点结果",
        "",
        "| Dataset | Pattern | Rate | Seed | V14 MAE | V18 MAE | Relative | Win | V14 RMSE | V18 RMSE |",
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
        relative = (
            "—"
            if row["relative_mae_pct"] is None
            else f"{row['relative_mae_pct']:+.2f}%"
        )
        win = "—" if row["relative_mae_pct"] is None else (
            "✓" if row["win"] else "✗"
        )
        lines.append(
            f"| {row['dataset']} | {row['pattern']} | {row['rate']:.1f} | "
            f"{row['seed']} | {_fmt(row['v14_mae'])} | "
            f"{_fmt(row['v18_mae'])} | {relative} | {win} | "
            f"{_fmt(row['v14_rmse'])} | {_fmt(row['v18_rmse'])} |"
        )

    lines += [
        "",
        "## 数据集与 Pattern 平均",
        "",
        "| Dataset | Pattern | N | V18 Avg MAE | Paired Relative Avg | Wins |",
        "|---|---|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["pattern"])].append(row)
    for (dataset, pattern), group in sorted(groups.items()):
        group_paired = [row for row in group if row["relative_mae_pct"] is not None]
        relative = (
            f"{_mean(group_paired, 'relative_mae_pct'):+.2f}%"
            if group_paired
            else "—"
        )
        lines.append(
            f"| {dataset} | {pattern} | {len(group)} | "
            f"{_mean(group, 'v18_mae'):.6f} | {relative} | "
            f"{sum(bool(row['win']) for row in group_paired)}/{len(group_paired)} |"
        )

    lines += [
        "",
        "## 总体与完整性",
        "",
        f"- 计划候选：{expected}",
        f"- 完成候选：{len(rows)}",
        f"- 成功配对 V14：{len(paired)}",
    ]
    if paired:
        lines += [
            f"- 配对平均 MAE 变化：{_mean(paired, 'relative_mae_pct'):+.2f}%",
            f"- 获胜：{sum(bool(row['win']) for row in paired)}/{len(paired)}",
            f"- 最大改善：{min(row['relative_mae_pct'] for row in paired):+.2f}%",
            f"- 最大退化：{max(row['relative_mae_pct'] for row in paired):+.2f}%",
        ]
    if missing_candidate:
        lines += ["", "缺失 V18 候选：", ""]
        lines.extend(f"- {item}" for item in missing_candidate)
    if missing_baseline:
        lines += ["", "缺失同 seed、同训练协议 V14 配对基线：", ""]
        lines.extend(f"- {item}" for item in missing_baseline)
    if incompatible_baseline:
        lines += [
            "",
            "发现但拒绝配对的历史 V14 结果：",
            "",
        ]
        lines.extend(f"- {item}" for item in incompatible_baseline)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    jobs = _jobs(args)
    training_policy = (
        resolve_path(args.training_policy)
        if args.training_policy
        else None
    )
    (
        rows,
        missing_candidate,
        missing_baseline,
        incompatible_baseline,
    ) = collect(jobs, training_policy=training_policy)
    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    md_path = summary_dir / f"{args.name}.md"
    csv_path = summary_dir / f"{args.name}.csv"
    json_path = summary_dir / f"{args.name}.json"
    text = markdown(
        rows,
        missing_candidate,
        missing_baseline,
        incompatible_baseline,
        len(jobs),
    )
    md_path.write_text(text, encoding="utf-8")
    fields = (
        "dataset", "pattern", "rate", "seed", "ablation",
        "v14_mae", "v18_mae", "relative_mae_pct", "win",
        "v14_rmse", "v18_rmse", "best_epoch", "best_val_mae",
        "completed_epochs", "total_time_sec", "peak_memory_gb",
        "baseline_run_dir", "run_dir",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    json_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "missing_candidate": missing_candidate,
                "missing_baseline": missing_baseline,
                "incompatible_baseline": incompatible_baseline,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(text)
    print(f"[saved] {md_path.relative_to(ROOT)}")
    print(f"[saved] {csv_path.relative_to(ROOT)}")
    print(f"[saved] {json_path.relative_to(ROOT)}")
    if args.require_complete and missing_candidate:
        raise SystemExit(
            f"Incomplete V18 summary: expected {len(jobs)}, found {len(rows)}"
        )


if __name__ == "__main__":
    main()
