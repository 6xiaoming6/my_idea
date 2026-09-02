#!/usr/bin/env python3
"""Summarize the BRLG Core-6 screen using validation metrics only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/v14-exploration/brlg"
CONFIGS = {
    path.name.split("_", 1)[0]: path
    for path in sorted(CONFIG_ROOT.glob("B*.json"))
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
                "modulation": float(
                    validation.get("v14_local_gate_modulation_mean", 1.0)
                ),
                "modulation_std": float(
                    validation.get("v14_local_gate_modulation_std", 0.0)
                ),
            }
    if best is None or not all(
        math.isfinite(best[key]) for key in ("mae", "rmse", "modulation")
    ):
        return None
    return best


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
        required = (
            run_dir / "logs/metrics.jsonl",
            run_dir / "logs/test.log",
            run_dir / "checkpoints/best.pt",
        )
        if not all(path.is_file() for path in required):
            continue
        if "Testing finished:" not in required[1].read_text(encoding="utf-8"):
            continue
        best = _best_validation(required[0])
        if best is not None:
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
        default="outputs/v14-exploration/summary/brlg_screen_validation_summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bases = {point: _base_result(*point) for point in POINTS}
    missing_base = [point for point, result in bases.items() if result is None]
    if missing_base:
        raise RuntimeError(f"Missing complete original V14 references: {missing_base}")

    rows, missing = [], []
    for candidate in CONFIGS:
        for point in POINTS:
            result = _candidate_result(candidate, *point)
            if result is None:
                missing.append((candidate, *point))
                continue
            base = bases[point]
            rows.append({
                "candidate": candidate,
                "dataset": point[0],
                "mae_change": _change(result["mae"], base["mae"]),
                "rmse_change": _change(result["rmse"], base["rmse"]),
                "modulation": result["modulation"],
            })
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"BRLG screen incomplete: {len(missing)}/36 jobs missing; rerun the "
            "runner or pass --allow-incomplete only for diagnostics"
        )

    decisions = []
    for candidate in CONFIGS:
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        if len(candidate_rows) != len(POINTS):
            continue
        mae = [row["mae_change"] for row in candidate_rows]
        rmse = [row["rmse_change"] for row in candidate_rows]
        dataset_means = {
            dataset: sum(
                row["mae_change"]
                for row in candidate_rows
                if row["dataset"] == dataset
            ) / 2.0
            for dataset in ("TaxiBJ", "BikeNYC", "CHAP")
        }
        decision = {
            "candidate": candidate,
            "clear_wins": sum(value <= -0.5 for value in mae),
            "macro_mae": sum(mae) / len(mae),
            "macro_rmse": sum(rmse) / len(rmse),
            "max_mae": max(mae),
            "max_rmse": max(rmse),
            "max_dataset_mae": max(dataset_means.values()),
            "mean_modulation": sum(row["modulation"] for row in candidate_rows)
            / len(candidate_rows),
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
        "# V14 BRLG Core-6 验证集汇总",
        "",
        "> 只读取最佳检查点对应的验证指标；候选冻结前不读取测试排名。",
        "",
        f"- 完成：{len(rows)}/36",
        f"- 缺失：{len(missing)}/36",
        "",
        "| 候选 | 清晰MAE胜点 | MAE宏平均变化 | RMSE宏平均变化 | 最大MAE退化 | 最大RMSE退化 | 最差数据集MAE均值 | 平均门控倍率 | 晋级 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['candidate']} | {row['clear_wins']}/6 | "
            f"{row['macro_mae']:+.3f}% | {row['macro_rmse']:+.3f}% | "
            f"{row['max_mae']:+.3f}% | {row['max_rmse']:+.3f}% | "
            f"{row['max_dataset_mae']:+.3f}% | {row['mean_modulation']:.4f} | "
            f"{'是' if row['eligible'] else '否'} |"
        )
    eligible = [row for row in decisions if row["eligible"]]
    lines.extend(("", "## 冻结判定", ""))
    if eligible:
        lines.append(
            f"按预注册排序，进入三种子确认的唯一候选为 **{eligible[0]['candidate']}**。"
        )
    elif len(rows) == 36:
        lines.append("没有候选满足全部预注册条件，关闭 BRLG 路线。")
    else:
        lines.append("实验尚未完整，不作候选判定。")

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote validation-only summary: {output}")


if __name__ == "__main__":
    main()
