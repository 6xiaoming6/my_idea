#!/usr/bin/env python3
"""Summarize three-model, four-point, three-seed V17.1 validation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime

from pipeline_common import (
    COMBINATION_VARIANTS,
    OUTPUT_ROOT,
    POINTS,
    ROOT,
    STAGE1_VARIANTS,
    combination_run,
    read_result,
    stage1_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-variant", required=True, choices=STAGE1_VARIANTS[1:])
    parser.add_argument("--combination-variant", required=True, choices=COMBINATION_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _specs(stage1_variant: str, combination_variant: str) -> list[tuple[str, str]]:
    return [
        ("full", "E0 Full"),
        (f"stage1:{stage1_variant}", f"Stage-1 {stage1_variant}"),
        (f"combination:{combination_variant}", f"Stage-2 {combination_variant}"),
    ]


def _run(model: str, point, seed: int):
    if model == "full":
        return stage1_run(point, seed, "full")
    kind, variant = model.split(":", 1)
    if kind == "stage1":
        return stage1_run(point, seed, variant)
    return combination_run(point, seed, variant)


def collect(stage1_variant: str, combination_variant: str, seeds: list[int]) -> list[dict]:
    rows = []
    for model, label in _specs(stage1_variant, combination_variant):
        for point in POINTS.values():
            for seed in seeds:
                run_dir = _run(model, point, seed)
                if run_dir is None:
                    continue
                result = read_result(run_dir)
                if result is None:
                    continue
                rows.append(
                    {
                        "model": model,
                        "label": label,
                        "point": point.point_id,
                        "seed": seed,
                        "run_dir": run_dir,
                        **result,
                    }
                )
    lookup = {(row["model"], row["point"], row["seed"]): row for row in rows}
    for row in rows:
        full = lookup.get(("full", row["point"], row["seed"]))
        row["relative_to_full_pct"] = (
            (row["mae"] - full["mae"]) / full["mae"] * 100.0 if full else None
        )
    return rows


def _metric_cell(rows: list[dict], key: str) -> str:
    if not rows:
        return "—"
    values = [row[key] for row in rows]
    return f"{statistics.fmean(values):.6f}±{statistics.pstdev(values):.6f}"


def build_markdown(
    rows: list[dict],
    stage1_variant: str,
    combination_variant: str,
    seeds: list[int],
) -> str:
    specs = _specs(stage1_variant, combination_variant)
    lines = [
        "# V17.1 Stage 3 多随机种子验证汇总",
        "",
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}  ",
        f"> Seeds：{', '.join(map(str, seeds))}  ",
        "> 稳定改善条件：相对 Full 平均 MAE 改善超过 1%，且至少 2/3 seeds 改善。",
        "",
        "## Test MAE（mean ± std）",
        "",
        "| Model | P1 | P2 | P3 | P4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, label in specs:
        model_rows = [row for row in rows if row["model"] == model]
        cells = [
            _metric_cell([row for row in model_rows if row["point"] == point], "mae")
            for point in POINTS
        ]
        lines.append(f"| {label} | {' | '.join(cells)} |")

    lines += [
        "",
        "## Test RMSE（mean ± std）",
        "",
        "| Model | P1 | P2 | P3 | P4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, label in specs:
        model_rows = [row for row in rows if row["model"] == model]
        cells = [
            _metric_cell([row for row in model_rows if row["point"] == point], "rmse")
            for point in POINTS
        ]
        lines.append(f"| {label} | {' | '.join(cells)} |")

    lines += [
        "",
        "## 配对稳定性判断",
        "",
        "| Model | Point | 平均相对 Full | 改善 seed 数 | 稳定改善 |",
        "|---|---|---:|---:|---|",
    ]
    for model, label in specs[1:]:
        for point in POINTS:
            point_rows = [
                row
                for row in rows
                if row["model"] == model
                and row["point"] == point
                and row["relative_to_full_pct"] is not None
            ]
            relative = [row["relative_to_full_pct"] for row in point_rows]
            mean_relative = statistics.fmean(relative) if relative else None
            improved = sum(value < 0 for value in relative)
            required_wins = 2 if len(seeds) >= 3 else max(1, len(seeds) - 1)
            stable = bool(
                mean_relative is not None
                and mean_relative < -1.0
                and improved >= required_wins
            )
            relative_text = f"{mean_relative:+.2f}%" if mean_relative is not None else "—"
            lines.append(
                f"| {label} | {point} | {relative_text} | "
                f"{improved}/{len(relative)} | {'是' if stable else '否'} |"
            )

    lines += [
        "",
        "## 总体配对结果",
        "",
        "| Model | 12 个 point-seed 平均变化 | 获胜数 |",
        "|---|---:|---:|",
    ]
    for model, label in specs[1:]:
        relative = [
            row["relative_to_full_pct"]
            for row in rows
            if row["model"] == model and row["relative_to_full_pct"] is not None
        ]
        mean_text = f"{statistics.fmean(relative):+.2f}%" if relative else "—"
        wins = sum(value < 0 for value in relative)
        lines.append(f"| {label} | {mean_text} | {wins}/{len(relative)} |")

    expected = len(specs) * len(POINTS) * len(seeds)
    lines += [
        "",
        "## 完整性",
        "",
        f"- 应有记录：{expected}",
        f"- 已完成：{len(rows)}",
        f"- 缺失：{expected - len(rows)}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if (
        args.stage1_variant == "independent_shared_scale"
        and args.combination_variant == "c1_independent_shared_scale"
    ):
        raise ValueError("Stage-1 candidate and C1 are identical; choose three distinct models.")
    rows = collect(args.stage1_variant, args.combination_variant, args.seeds)
    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.stage1_variant}__{args.combination_variant}"
    markdown_path = summary_dir / f"stage3_multiseed_{suffix}.md"
    csv_path = summary_dir / f"stage3_multiseed_{suffix}.csv"
    json_path = summary_dir / f"stage3_multiseed_{suffix}.json"
    markdown = build_markdown(
        rows, args.stage1_variant, args.combination_variant, args.seeds
    )
    markdown_path.write_text(markdown, encoding="utf-8")

    fields = (
        "model", "label", "point", "seed", "mae", "rmse", "loss",
        "relative_to_full_pct", "total_time_sec", "peak_memory_gb", "run_dir",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: str(row[key].relative_to(ROOT)) if key == "run_dir" else row.get(key)
                    for key in fields
                }
            )
    json_path.write_text(
        json.dumps(
            [
                {
                    **{key: value for key, value in row.items() if key != "run_dir"},
                    "run_dir": str(row["run_dir"].relative_to(ROOT)),
                }
                for row in rows
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(markdown)
    print(f"[saved] {markdown_path.relative_to(ROOT)}")
    print(f"[saved] {csv_path.relative_to(ROOT)}")
    print(f"[saved] {json_path.relative_to(ROOT)}")
    expected = 3 * len(POINTS) * len(args.seeds)
    if args.require_complete and len(rows) != expected:
        raise SystemExit(f"Incomplete Stage 3: expected {expected}, found {len(rows)}")


if __name__ == "__main__":
    main()
