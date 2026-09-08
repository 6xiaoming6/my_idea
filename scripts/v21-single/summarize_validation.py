#!/usr/bin/env python3
"""Summarize V21 validation using validation metrics for decisions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from _protocol import (
    DATASETS,
    PATTERNS,
    RATES,
    ROOT,
    SEEDS,
    VARIANTS,
    expected_epochs,
    filtered_points,
    find_completed_run,
    read_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("core6", "multiseed", "all24"), required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=PATTERNS)
    parser.add_argument("--rates", nargs="+", choices=RATES, default=RATES)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _change(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def _fmt(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.4f}"


def _comparison(rows: list[dict], left: str, right: str) -> dict | None:
    by_key = {(row["variant"], row["seed"], row["point"]): row for row in rows}
    pairs = []
    for key, left_row in by_key.items():
        if key[0] != left:
            continue
        right_row = by_key.get((right, key[1], key[2]))
        if right_row is None:
            continue
        pairs.append((left_row, right_row))
    if not pairs:
        return None
    val_mae = [_change(a["val_mae"], b["val_mae"]) for a, b in pairs]
    val_rmse = [_change(a["val_rmse"], b["val_rmse"]) for a, b in pairs]
    test_mae = [_change(a["test_mae"], b["test_mae"]) for a, b in pairs]
    dataset_deltas: dict[str, list[float]] = defaultdict(list)
    seed_deltas: dict[int, list[float]] = defaultdict(list)
    for (a, _), delta in zip(pairs, val_mae):
        dataset_deltas[a["dataset"]].append(delta)
        seed_deltas[a["seed"]].append(delta)
    seed_macro = {
        str(key): sum(values) / len(values) for key, values in seed_deltas.items()
    }
    return {
        "left": left,
        "right": right,
        "count": len(pairs),
        "val_mae_macro": sum(val_mae) / len(val_mae),
        "val_rmse_macro": sum(val_rmse) / len(val_rmse),
        "test_mae_macro": sum(test_mae) / len(test_mae),
        "wins": sum(value < 0.0 for value in val_mae),
        "clear_wins": sum(value <= -0.5 for value in val_mae),
        "max_regression": max(val_mae),
        "dataset_macro": {
            key: sum(values) / len(values) for key, values in dataset_deltas.items()
        },
        "seed_macro": seed_macro,
        "seed_macro_std": (
            statistics.stdev(seed_macro.values()) if len(seed_macro) > 1 else float("nan")
        ),
    }


def _passes_core_rule(comparison: dict, expected: int, require_seed_stability: bool) -> bool:
    core_pass = (
        comparison["count"] == expected
        and comparison["clear_wins"] >= math.ceil(2 * expected / 3)
        and comparison["val_mae_macro"] <= -0.5
        and comparison["val_rmse_macro"] <= 0.5
        and comparison["max_regression"] <= 2.0
        and max(comparison["dataset_macro"].values(), default=float("inf")) <= 1.0
    )
    if not core_pass or not require_seed_stability:
        return core_pass
    seed_values = list(comparison["seed_macro"].values())
    return (
        len(seed_values) >= 3
        and sum(value < 0.0 for value in seed_values) >= math.ceil(2 * len(seed_values) / 3)
        and max(seed_values) <= 1.0
    )


def main() -> None:
    args = parse_args()
    variants = tuple(dict.fromkeys(args.variants))
    seeds = tuple(args.seeds) if args.seeds is not None else (
        SEEDS if args.phase == "multiseed" else (42,)
    )
    points = filtered_points(
        args.phase,
        tuple(args.datasets),
        tuple(args.patterns),
        tuple(args.rates),
    )
    rows: list[dict] = []
    missing: list[str] = []
    for variant_key in variants:
        variant = VARIANTS[variant_key]
        for seed in seeds:
            for point in points:
                epochs = expected_epochs(point, args.epochs)
                run_dir = find_completed_run(variant, point, seed, epochs)
                label = f"{variant_key} {point.label} seed={seed} epochs={epochs}"
                if run_dir is None:
                    missing.append(label)
                    continue
                result = read_result(run_dir)
                rows.append({
                    "variant": variant_key,
                    "seed": seed,
                    "point": point.label,
                    "dataset": point.dataset,
                    "pattern": point.pattern,
                    "rate": point.rate,
                    "epochs": epochs,
                    "best_epoch": result["validation"]["epoch"],
                    "val_mae": result["validation"]["mae"],
                    "val_rmse": result["validation"]["rmse"],
                    "test_mae": result["test"]["mae"],
                    "test_rmse": result["test"]["rmse"],
                    "test_wape": result["test"]["wape"],
                    "run_dir": result["run_dir"],
                })

    expected_jobs = len(variants) * len(seeds) * len(points)
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"V21 {args.phase} is incomplete: {len(missing)}/{expected_jobs} missing; "
            "pass --allow-incomplete only for progress inspection"
        )

    comparisons = []
    for left, right in (("P1", "P0"), ("P2", "P0"), ("P2", "P1")):
        if left in variants and right in variants:
            comparison = _comparison(rows, left, right)
            if comparison is not None:
                comparisons.append(comparison)

    lines = [
        f"# V21 {args.phase} 验证汇总",
        "",
        "> 候选选择只使用最佳验证集指标；测试集结果用于流程确认和最终描述，不参与调参。",
        "",
        f"- 完成：{len(rows)}/{expected_jobs}",
        f"- 缺失：{len(missing)}/{expected_jobs}",
        f"- Seeds：{', '.join(map(str, seeds))}",
        "- 变化率为负表示左侧方案更好。",
        "",
        "## 单任务结果",
        "",
        "| 方案 | Seed | 数据点 | Best epoch | Val MAE | Val RMSE | Test MAE | Test RMSE | Test WAPE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda x: (x["variant"], x["seed"], x["point"])):
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['point']} | {row['best_epoch']} | "
            f"{_fmt(row['val_mae'])} | {_fmt(row['val_rmse'])} | "
            f"{_fmt(row['test_mae'])} | {_fmt(row['test_rmse'])} | {_fmt(row['test_wape'])} |"
        )

    lines.extend(("", "## 配对比较", ""))
    if comparisons:
        lines.extend((
            "| 比较 | 匹配数 | Val MAE宏平均 | Val RMSE宏平均 | Test MAE宏平均 | Val胜点 | 清晰胜点 | 最大Val MAE退化 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ))
        for value in comparisons:
            lines.append(
                f"| {value['left']} vs {value['right']} | {value['count']} | "
                f"{value['val_mae_macro']:+.3f}% | {value['val_rmse_macro']:+.3f}% | "
                f"{value['test_mae_macro']:+.3f}% | {value['wins']}/{value['count']} | "
                f"{value['clear_wins']}/{value['count']} | {value['max_regression']:+.3f}% |"
            )
            dataset_text = ", ".join(
                f"{key}={delta:+.3f}%" for key, delta in value["dataset_macro"].items()
            )
            lines.append(f"- {value['left']} vs {value['right']} 数据集宏平均：{dataset_text}")
            if len(value["seed_macro"]) > 1:
                seed_text = ", ".join(
                    f"seed{key}={delta:+.3f}%" for key, delta in value["seed_macro"].items()
                )
                lines.append(f"- {value['left']} vs {value['right']} 种子宏平均：{seed_text}")
                lines.append(
                    f"- {value['left']} vs {value['right']} 种子宏平均样本标准差："
                    f"{value['seed_macro_std']:.3f} 个百分点。"
                )
    else:
        lines.append("缺少成对方案，暂时无法比较。")

    lines.extend(("", "## 预注册判定", ""))
    expected_pairs = len(seeds) * len(points)
    versus_p0 = {value["left"]: value for value in comparisons if value["right"] == "P0"}
    promoted = []
    for candidate in ("P1", "P2"):
        comparison = versus_p0.get(candidate)
        if comparison is None:
            continue
        passed = _passes_core_rule(
            comparison,
            expected_pairs,
            require_seed_stability=args.phase == "multiseed",
        )
        if passed:
            promoted.append(candidate)
        lines.append(
            f"- {candidate} 相对 P0：{'通过' if passed else '未通过'}。要求清晰胜点≥2/3、"
            "Val MAE宏平均≤-0.5%、Val RMSE宏平均≤+0.5%、最大单点退化≤2%、"
            "最差数据集MAE宏平均≤+1%。"
        )
        if args.phase == "multiseed":
            lines.append(
                f"  三种子附加条件：至少2/3个seed宏平均获胜且最差seed退化≤1%。"
            )
    evidence = next(
        (value for value in comparisons if value["left"] == "P2" and value["right"] == "P1"),
        None,
    )
    if evidence is not None:
        evidence_pass = (
            evidence["count"] == expected_pairs
            and evidence["val_mae_macro"] <= -0.3
            and evidence["wins"] >= math.ceil(2 * expected_pairs / 3)
            and evidence["max_regression"] <= 2.0
        )
        lines.append(
            f"- 7维 Evidence 独立贡献：{'通过' if evidence_pass else '未通过'}。"
            "P2 需相对 P1 的 Val MAE宏平均≤-0.3%、胜点≥2/3且最大退化≤2%。"
        )
    if not missing:
        lines.append(
            f"- 当前建议晋级：{', '.join(promoted) if promoted else '无；保留P0并停止该路线'}。"
        )
    else:
        lines.append("- 当前结果不完整，不作最终晋级判定。")

    if missing:
        lines.extend(("", "## 未完成任务", ""))
        lines.extend(f"- {item}" for item in missing)

    output = Path(args.output) if args.output else (
        ROOT / f"outputs/v21-single/summary/{args.phase}_validation_summary.md"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {"phase": args.phase, "rows": rows, "comparisons": comparisons, "missing": missing},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {output}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
