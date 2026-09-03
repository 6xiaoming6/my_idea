#!/usr/bin/env python3
"""Summarize latest complete V20 runs without using test metrics for selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs/v20-single")
    parser.add_argument("--output", default="outputs/v20-single/summary/v20_results.md")
    return parser.parse_args()


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    args = parse_args()
    outputs = ROOT / args.outputs
    latest = {}
    for metrics_path in outputs.glob("**/logs/metrics.jsonl"):
        run_dir = metrics_path.parent.parent
        config_path = run_dir / "config.json"
        test_log = run_dir / "logs/test.log"
        if not config_path.is_file() or not test_log.is_file():
            continue
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if cfg.get("model", {}).get("architecture") != "v20_probe_validated_c2f_moe":
            continue
        records = _records(metrics_path)
        tests = [record for record in records if record.get("stage") == "test"]
        best = [record for record in records if record.get("is_best") and record.get("val")]
        if not tests or not best:
            continue
        key = (
            cfg["data"].get("dataset_name", "unknown"),
            cfg["data"]["mask"].get("pattern", "unknown"),
            float(cfg["data"]["mask"].get("missing_rate", 0.0)),
            int(cfg.get("seed", 42)),
        )
        item = (run_dir.stat().st_mtime, run_dir, best[-1], tests[-1])
        if key not in latest or item[0] > latest[key][0]:
            latest[key] = item
    lines = [
        "# V20 实验结果汇总",
        "",
        "> 候选选择应使用验证集；测试集仅报告最佳检查点的最终泛化结果。",
        "",
        "| 数据集 | 模式 | 缺失率 | Seed | Best epoch | Val MAE | Val RMSE | Test MAE | Test RMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(latest):
        _, run_dir, best, test = latest[key]
        validation = best["val"]
        metrics = test["metrics"]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]:.1f} | {key[3]} | "
            f"{best['epoch']} | {validation['mae']:.6f} | {validation['rmse']:.6f} | "
            f"{metrics['mae']:.6f} | {metrics['rmse']:.6f} |"
        )
    lines.extend(("", f"- 完整组合：{len(latest)}", ""))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
