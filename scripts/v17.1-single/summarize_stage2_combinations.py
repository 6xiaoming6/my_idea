#!/usr/bin/env python3
"""Summarize E0/C1/C2/C3 paired V17.1 combination results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from pipeline_common import (
    COMBINATION_VARIANTS,
    OUTPUT_ROOT,
    POINTS,
    ROOT,
    combination_run,
    read_result,
    stage1_run,
)


MODELS = ("full", *COMBINATION_VARIANTS)
LABELS = {
    "full": "E0 Full",
    "c1_independent_shared_scale": "C1 Independent Shared Scale (reuse E7)",
    "c2_independent_shared_hard_floor": "C2 C1 + Hard Fine Floor",
    "c3_independent_shared_hard_floor_global_gamma": "C3 C2 + Global Route Gamma",
}


def _run(model: str, point, seed: int) -> Path | None:
    if model == "full":
        return stage1_run(point, seed, "full")
    return combination_run(point, seed, model)


def collect(seeds: list[int]) -> list[dict]:
    rows = []
    for model in MODELS:
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
                        "point": point.point_id,
                        "seed": seed,
                        "run_dir": run_dir,
                        **result,
                    }
                )

    lookup = {(row["model"], row["point"], row["seed"]): row for row in rows}
    previous = {
        "c1_independent_shared_scale": "full",
        "c2_independent_shared_hard_floor": "c1_independent_shared_scale",
        "c3_independent_shared_hard_floor_global_gamma": "c2_independent_shared_hard_floor",
    }
    for row in rows:
        full = lookup.get(("full", row["point"], row["seed"]))
        prior = lookup.get((previous.get(row["model"], "full"), row["point"], row["seed"]))
        row["relative_to_full_pct"] = (
            (row["mae"] - full["mae"]) / full["mae"] * 100.0 if full else None
        )
        row["relative_to_previous_pct"] = (
            (row["mae"] - prior["mae"]) / prior["mae"] * 100.0
            if prior and row["model"] != "full"
            else None
        )
    return rows


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _cell(rows: list[dict]) -> str:
    if not rows:
        return "—"
    maes = [row["mae"] for row in rows]
    relative = [row["relative_to_full_pct"] for row in rows if row["relative_to_full_pct"] is not None]
    mae_text = (
        f"{maes[0]:.6f}"
        if len(maes) == 1
        else f"{_mean(maes):.6f}±{statistics.pstdev(maes):.6f}"
    )
    return f"{mae_text} ({_mean(relative):+.2f}%)" if relative else mae_text


def build_markdown(rows: list[dict], seeds: list[int]) -> str:
    lines = [
        "# V17.1 Stage 2 组合实验汇总",
        "",
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}  ",
        f"> Seeds：{', '.join(map(str, seeds))}  ",
        "> 括号内为相对同 seed E0 Full 的 Test MAE 变化，负数表示改善。",
        "",
        "| Model | P1 Taxi random0.4 | P2 Bike fixed0.8 | P3 Taxi fixed0.4 | P4 CHAP fixed0.4 | 四点平均 vs Full | 四点平均 vs 前序 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        cells = [
            _cell([row for row in model_rows if row["point"] == point])
            for point in POINTS
        ]
        vs_full = [row["relative_to_full_pct"] for row in model_rows if row["relative_to_full_pct"] is not None]
        vs_previous = [row["relative_to_previous_pct"] for row in model_rows if row["relative_to_previous_pct"] is not None]
        full_text = f"{_mean(vs_full):+.2f}%" if vs_full else "—"
        previous_text = f"{_mean(vs_previous):+.2f}%" if vs_previous else "—"
        lines.append(
            f"| {LABELS[model]} | {' | '.join(cells)} | "
            f"{full_text} | {previous_text} |"
        )

    lines += [
        "",
        "## 成功点与失败点",
        "",
        "| Model | P1/P2 平均 vs Full | P3/P4 平均 vs Full | 塌缩触发次数 |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        success = [row["relative_to_full_pct"] for row in model_rows if row["point"] in {"P1", "P2"} and row["relative_to_full_pct"] is not None]
        failure = [row["relative_to_full_pct"] for row in model_rows if row["point"] in {"P3", "P4"} and row["relative_to_full_pct"] is not None]
        collapse = sum(
            bool(row["collapse_flags"].get("scale"))
            or bool(row["collapse_flags"].get("route_gate"))
            or any(bool(value) for value in (row["collapse_flags"].get("expert") or {}).values())
            for row in model_rows
        )
        success_text = f"{_mean(success):+.2f}%" if success else "—"
        failure_text = f"{_mean(failure):+.2f}%" if failure else "—"
        lines.append(f"| {LABELS[model]} | {success_text} | {failure_text} | {collapse}/{len(model_rows)} |")

    expected = len(MODELS) * len(POINTS) * len(seeds)
    lines += [
        "",
        "## 完整性",
        "",
        f"- 应有记录：{expected}",
        f"- 已完成：{len(rows)}",
        f"- 缺失：{expected - len(rows)}",
        "- C1 直接复用 Stage-1 E7，不重复消耗训练资源。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect(args.seeds)
    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(map(str, args.seeds))
    markdown_path = summary_dir / f"stage2_combinations_seed{suffix}.md"
    csv_path = summary_dir / f"stage2_combinations_seed{suffix}.csv"
    json_path = summary_dir / f"stage2_combinations_seed{suffix}.json"
    markdown = build_markdown(rows, args.seeds)
    markdown_path.write_text(markdown, encoding="utf-8")

    fields = (
        "model", "point", "seed", "mae", "rmse", "loss",
        "relative_to_full_pct", "relative_to_previous_pct",
        "total_time_sec", "peak_memory_gb", "run_dir",
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
    expected = len(MODELS) * len(POINTS) * len(args.seeds)
    if args.require_complete and len(rows) != expected:
        raise SystemExit(f"Incomplete Stage 2: expected {expected}, found {len(rows)}")


if __name__ == "__main__":
    main()
