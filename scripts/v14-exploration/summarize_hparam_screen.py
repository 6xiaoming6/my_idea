#!/usr/bin/env python3
"""Summarize the final V14 hyperparameter Core-6 screen from validation only."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/v14-exploration/hparams"
CONFIGS = {
    path.name.split("_", 1)[0]: path
    for path in sorted(CONFIG_ROOT.glob("H*.json"))
}
POINTS = (
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("BikeNYC", "fixed", "0.6"),
    ("BikeNYC", "random", "0.8"),
    ("CHAP", "fixed", "0.2"),
    ("CHAP", "random", "0.4"),
)
OUTPUT_NAMES = {
    "TaxiBJ": "TaxiBJ",
    "BikeNYC": "BikeNYC",
    "CHAP": "CHAP_Beijing",
}
FULL_EPOCHS = {"TaxiBJ": 160, "BikeNYC": 140, "CHAP": 150}


def _best_validation(metrics_path: Path) -> dict | None:
    best = None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        validation = record.get("val")
        if validation is not None and record.get("is_best"):
            best = {
                "epoch": int(record["epoch"]),
                "mae": float(validation["mae"]),
                "rmse": float(validation["rmse"]),
            }
    if best is None or not all(math.isfinite(best[key]) for key in ("mae", "rmse")):
        return None
    return best


def _total_params(train_log: Path) -> int | None:
    match = re.search(
        r"^  total_params:\s+([0-9,]+)\s*$",
        train_log.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return int(match.group(1).replace(",", "")) if match else None


def _matching_run(
    root: Path,
    seed: int,
    epochs: int,
    expected_experiment: dict | None,
) -> dict | None:
    for config_path in sorted(root.glob("*/config.json"), reverse=True):
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if int(cfg.get("seed", -1)) != seed:
            continue
        if int(cfg.get("train", {}).get("epochs", -1)) != epochs:
            continue
        if expected_experiment is not None and cfg.get("experiment") != expected_experiment:
            continue
        run_dir = config_path.parent
        metrics_path = run_dir / "logs/metrics.jsonl"
        test_log = run_dir / "logs/test.log"
        train_log = run_dir / "logs/train.log"
        checkpoint = run_dir / "checkpoints/best.pt"
        if not all(path.is_file() for path in (metrics_path, test_log, train_log, checkpoint)):
            continue
        if "Testing finished:" not in test_log.read_text(encoding="utf-8"):
            continue
        best = _best_validation(metrics_path)
        if best is not None:
            best["params"] = _total_params(train_log)
            best["run_dir"] = str(run_dir.relative_to(ROOT))
            return best
    return None


def _base_result(dataset: str, pattern: str, rate: str) -> dict | None:
    root = (
        ROOT / "outputs/v14-single" / OUTPUT_NAMES[dataset] / "full/model"
        / pattern / f"rate{rate}"
    )
    return _matching_run(root, 42, FULL_EPOCHS[dataset], None)


def _candidate_result(
    candidate: str, dataset: str, pattern: str, rate: str
) -> dict | None:
    path = CONFIGS[candidate]
    experiment = json.loads(path.read_text(encoding="utf-8"))["experiment"]
    root = (
        ROOT / "outputs/v14-exploration" / OUTPUT_NAMES[dataset] / "ablation"
        / f"v14_exploration_{path.stem}" / pattern / f"rate{rate}"
    )
    return _matching_run(root, 42, FULL_EPOCHS[dataset], experiment)


def _change(value: float, base: float) -> float:
    return 100.0 * (value / base - 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/v14-exploration/summary/hparam_screen_validation_summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bases = {point: _base_result(*point) for point in POINTS}
    missing_base = [point for point, result in bases.items() if result is None]
    if missing_base:
        raise RuntimeError(f"Missing complete original V14 references: {missing_base}")

    rows, missing = [], []
    for candidate, config_path in CONFIGS.items():
        experiment = json.loads(config_path.read_text(encoding="utf-8"))["experiment"]
        for point in POINTS:
            result = _candidate_result(candidate, *point)
            if result is None:
                missing.append((candidate, *point))
                continue
            base = bases[point]
            parameter_change = None
            if result["params"] is not None and base["params"] is not None:
                parameter_change = _change(result["params"], base["params"])
            rows.append({
                "candidate": candidate,
                "field": experiment["field"],
                "value": experiment["value"],
                "dataset": point[0],
                "point": point,
                "mae_change": _change(result["mae"], base["mae"]),
                "rmse_change": _change(result["rmse"], base["rmse"]),
                "parameter_change": parameter_change,
                "best_epoch": result["epoch"],
            })
    expected = len(CONFIGS) * len(POINTS)
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"Hyperparameter screen incomplete: {len(missing)}/{expected} jobs missing; "
            "rerun run_hparam_exploration.py or pass --allow-incomplete only for diagnostics"
        )

    decisions = []
    for candidate in CONFIGS:
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        if len(candidate_rows) != len(POINTS):
            continue
        mae = [row["mae_change"] for row in candidate_rows]
        rmse = [row["rmse_change"] for row in candidate_rows]
        parameter_changes = [
            row["parameter_change"]
            for row in candidate_rows
            if row["parameter_change"] is not None
        ]
        dataset_means = {
            dataset: sum(
                row["mae_change"]
                for row in candidate_rows
                if row["dataset"] == dataset
            ) / 2.0
            for dataset in ("TaxiBJ", "BikeNYC", "CHAP")
        }
        first = candidate_rows[0]
        decision = {
            "candidate": candidate,
            "field": first["field"],
            "value": first["value"],
            "clear_wins": sum(value <= -0.5 for value in mae),
            "macro_mae": sum(mae) / len(mae),
            "macro_rmse": sum(rmse) / len(rmse),
            "max_mae": max(mae),
            "max_rmse": max(rmse),
            "max_dataset_mae": max(dataset_means.values()),
            "parameter_change": (
                sum(parameter_changes) / len(parameter_changes)
                if parameter_changes else None
            ),
        }
        decision["eligible"] = (
            decision["clear_wins"] >= 4
            and decision["macro_mae"] <= -1.0
            and decision["macro_rmse"] <= 0.5
            and decision["max_mae"] <= 3.0
            and decision["max_rmse"] <= 5.0
            and decision["max_dataset_mae"] <= 0.5
        )
        decisions.append(decision)
    decisions.sort(
        key=lambda row: (not row["eligible"], row["macro_mae"], row["macro_rmse"])
    )

    lines = [
        "# V14 最终模型超参数 Core-6 验证集汇总",
        "",
        "> 候选选择只读取最佳检查点的验证指标；测试集仅用于确认训练流程完整，未参与排序。",
        "",
        f"- 完成：{len(rows)}/{expected}",
        f"- 缺失：{len(missing)}/{expected}",
        "- 负的变化率表示优于原始V14。",
        "",
        "| 候选 | 单一改动 | 清晰MAE胜点 | MAE宏平均 | RMSE宏平均 | 最大MAE退化 | 最大RMSE退化 | 最差数据集MAE | 参数量变化 | 晋级 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in decisions:
        parameter_text = (
            f"{row['parameter_change']:+.2f}%"
            if row["parameter_change"] is not None else "n/a"
        )
        lines.append(
            f"| {row['candidate']} | `{row['field']}={row['value']}` | "
            f"{row['clear_wins']}/6 | {row['macro_mae']:+.3f}% | "
            f"{row['macro_rmse']:+.3f}% | {row['max_mae']:+.3f}% | "
            f"{row['max_rmse']:+.3f}% | {row['max_dataset_mae']:+.3f}% | "
            f"{parameter_text} | {'是' if row['eligible'] else '否'} |"
        )

    eligible = [row for row in decisions if row["eligible"]]
    lines.extend(("", "## 冻结判定", ""))
    if eligible:
        winner = eligible[0]
        close = [
            row for row in eligible
            if row["macro_mae"] <= winner["macro_mae"] + 0.2
        ]
        if len(close) > 1:
            close.sort(key=lambda row: (
                float("inf") if row["parameter_change"] is None else row["parameter_change"],
                row["macro_mae"],
            ))
            winner = close[0]
        lines.append(
            f"按验证指标与近似同分时优先较小模型的规则，唯一推荐候选为 "
            f"**{winner['candidate']}**（`{winner['field']}={winner['value']}`）。"
        )
        lines.append("下一步只对该候选执行三种子Core-6确认，不组合其他超参数。")
    elif len(rows) == expected:
        lines.append("没有候选满足全部预注册条件，结束超参数路线并保留原始V14。")
    else:
        lines.append("实验尚未完整，不作候选判定。")

    if missing:
        lines.extend(("", "## 未完成任务", ""))
        lines.extend(f"- {' / '.join(item)}" for item in missing)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote validation-only summary: {output}")


if __name__ == "__main__":
    main()
