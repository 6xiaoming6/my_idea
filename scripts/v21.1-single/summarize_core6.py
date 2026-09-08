#!/usr/bin/env python3
"""Summarize V21 P0/P1/P2 and V21.1 P3/P4 on the same Core-6 points."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from v21_1_protocol import (
    CORE6,
    ROOT,
    VARIANTS,
    expected_epochs,
    find_completed_run,
    read_result,
)


def change(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def paired(rows: list[dict], left: str, right: str) -> dict | None:
    index = {(row["variant"], row["point"]): row for row in rows}
    pairs = []
    for point in CORE6:
        a = index.get((left, point.label))
        b = index.get((right, point.label))
        if a is not None and b is not None:
            pairs.append((a, b))
    if not pairs:
        return None
    val_mae = [change(a["val_mae"], b["val_mae"]) for a, b in pairs]
    val_rmse = [change(a["val_rmse"], b["val_rmse"]) for a, b in pairs]
    test_mae = [change(a["test_mae"], b["test_mae"]) for a, b in pairs]
    return {
        "left": left,
        "right": right,
        "count": len(pairs),
        "val_mae_macro": sum(val_mae) / len(val_mae),
        "val_rmse_macro": sum(val_rmse) / len(val_rmse),
        "test_mae_macro": sum(test_mae) / len(test_mae),
        "wins": sum(value < 0 for value in val_mae),
        "clear_wins": sum(value <= -0.5 for value in val_mae),
        "max_regression": max(val_mae),
    }


def fmt(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/v21.1-single/summary/core6_summary.md",
    )
    args = parser.parse_args()
    variants = tuple(dict.fromkeys(args.variants))
    rows = []
    missing = []
    for key in variants:
        variant = VARIANTS[key]
        for point in CORE6:
            epochs = expected_epochs(point, args.epochs)
            run_dir = find_completed_run(variant, point, args.seed, epochs)
            label = f"{key} {point.label} seed={args.seed} epochs={epochs}"
            if run_dir is None:
                missing.append(label)
                continue
            result = read_result(run_dir)
            rows.append({
                "variant": key,
                "point": point.label,
                "best_epoch": result["validation"]["epoch"],
                "val_mae": result["validation"]["mae"],
                "val_rmse": result["validation"]["rmse"],
                "test_mae": result["test"]["mae"],
                "test_rmse": result["test"]["rmse"],
                "test_wape": result["test"]["wape"],
                "run_dir": result["run_dir"],
            })
    expected = len(variants) * len(CORE6)
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"Core-6 incomplete: {len(missing)}/{expected} missing; use --allow-incomplete for progress"
        )

    comparisons = []
    for left, right in (("P3", "P0"), ("P3", "P1"), ("P3", "P4"), ("P4", "P0")):
        if left in variants and right in variants:
            value = paired(rows, left, right)
            if value is not None:
                comparisons.append(value)

    lines = [
        "# V21.1 Core-6汇总",
        "",
        "> 只用验证集选择模型；测试集仅描述冻结最佳checkpoint的最终表现。",
        "",
        f"- 完成：{len(rows)}/{expected}",
        f"- Seed：{args.seed}",
        "- 负变化表示左侧方案更好。",
        "",
        "## 单任务结果",
        "",
        "| 方案 | 数据点 | Best epoch | Val MAE | Val RMSE | Test MAE | Test RMSE | Test WAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["variant"], item["point"])):
        lines.append(
            f"| {row['variant']} | {row['point']} | {row['best_epoch']} | "
            f"{fmt(row['val_mae'])} | {fmt(row['val_rmse'])} | {fmt(row['test_mae'])} | "
            f"{fmt(row['test_rmse'])} | {fmt(row['test_wape'])} |"
        )
    lines.extend((
        "",
        "## 配对结果",
        "",
        "| 比较 | 匹配点 | Val MAE宏平均 | Val RMSE宏平均 | Test MAE宏平均 | Val胜点 | ≥0.5%胜点 | 最大退化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ))
    for value in comparisons:
        lines.append(
            f"| {value['left']} vs {value['right']} | {value['count']} | "
            f"{value['val_mae_macro']:+.3f}% | {value['val_rmse_macro']:+.3f}% | "
            f"{value['test_mae_macro']:+.3f}% | {value['wins']}/{value['count']} | "
            f"{value['clear_wins']}/{value['count']} | {value['max_regression']:+.3f}% |"
        )
    lines.extend((
        "",
        "## 解释规则",
        "",
        "- P3 vs P4隔离显式尺度失真输入的贡献，是核心创新判定。",
        "- P3 vs P1判断有界自适应接纳是否优于无条件Measure替换。",
        "- P3 vs P0判断性能竞争力，不作为创新存在与否的唯一门槛。",
        "- 最终结论还必须结合alpha–真实尺度误差相关性和失真分层结果。",
    ))
    if missing:
        lines.extend(("", "## 未完成", ""))
        lines.extend(f"- {item}" for item in missing)

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(
            {"rows": rows, "comparisons": comparisons, "missing": missing},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {output}")
    print(f"Wrote: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
