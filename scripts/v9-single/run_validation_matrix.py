#!/usr/bin/env python3
"""Run the required five-point v9-single validation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_experiments.py"
MODEL_CONFIG_DIR = ROOT / "configs" / "v9-single"
QUICK_POLICY = ROOT / "configs" / "v9-single" / "policies" / "quick_5epoch.json"

MATRIX = (
    ("TaxiBJ", "fixed", "0.2", "低缺失下不能明显退化"),
    ("TaxiBJ", "random", "0.6", "复杂随机缺失下专家路由是否有效"),
    ("BikeNYC", "fixed", "0.6", "小数据集不能明显过拟合"),
    ("CHAP", "fixed", "0.4", "平滑环境场中多分辨率先验是否稳定"),
    ("CHAP", "random", "0.8", "高缺失下 MoE 专家是否能补偿"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v9-single five-point validation matrix.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for idx, (dataset, pattern, rate, purpose) in enumerate(MATRIX, 1):
        command = [
            sys.executable,
            str(RUNNER),
            "--dataset",
            dataset,
            "--gpu",
            str(args.gpu),
            "--mask-pattern",
            pattern,
            "--mask-rate",
            rate,
            "--experiments",
            "full",
            "--training-policy",
            str(QUICK_POLICY),
            "--model-config-dir",
            str(MODEL_CONFIG_DIR),
            "--conda-env",
            args.conda_env,
            "--cpu-threads",
            str(args.cpu_threads),
        ]
        if args.dry_run:
            command.append("--dry-run")
        print("\n" + "=" * 80, flush=True)
        print(f"[{idx}/{len(MATRIX)}] {dataset} {pattern}@{rate} | {purpose}", flush=True)
        print("[run]", shlex.join(command), flush=True)
        print("=" * 80, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
