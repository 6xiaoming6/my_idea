#!/usr/bin/env python3
"""Run all formal v9-single experiments on one GPU, fixed before random."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_experiments.py"
MODEL_CONFIG_DIR = ROOT / "configs" / "v9-single"
DEFAULT_POLICY = ROOT / "configs" / "policies" / "full_model_paper.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete v9-single full-model grid sequentially: fixed first, then random."
    )
    parser.add_argument("--dataset", choices=("TaxiBJ", "BikeNYC", "CHAP", "all"), default="all")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--mask-rate", choices=("0.2", "0.4", "0.6", "0.8", "all"), default="all")
    parser.add_argument("--fixed-seed", type=int, default=42)
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--training-policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, pattern: str) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--dataset",
        args.dataset,
        "--gpu",
        str(args.gpu),
        "--mask-pattern",
        pattern,
        "--mask-rate",
        args.mask_rate,
        "--experiments",
        "full",
        "--fixed-seed",
        str(args.fixed_seed),
        "--conda-env",
        args.conda_env,
        "--cpu-threads",
        str(args.cpu_threads),
        "--training-policy",
        args.training_policy,
        "--model-config-dir",
        str(MODEL_CONFIG_DIR),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    args = parse_args()
    for pattern, title in (
        ("fixed", "阶段 1/2：v9-single 完整 fixed 实验"),
        ("random", "阶段 2/2：v9-single 完整 random 实验"),
    ):
        command = build_command(args, pattern)
        print("\n" + "=" * 80, flush=True)
        print(title, flush=True)
        print(f"GPU: {args.gpu} | dataset: {args.dataset} | rates: {args.mask_rate}", flush=True)
        print("[run]", shlex.join(command), flush=True)
        print("=" * 80, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("\n全部 v9-single fixed 和 random 正式实验已完成。", flush=True)


if __name__ == "__main__":
    main()
