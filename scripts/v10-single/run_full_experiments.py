#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full v10-single experiments. By default it runs fixed first, then random."
    )
    parser.add_argument("--dataset", choices=("TaxiBJ", "BikeNYC", "CHAP", "all"), default="all")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--mask-pattern", choices=("fixed", "random", "all"), default="all")
    parser.add_argument("--mask-rate", choices=("0.2", "0.4", "0.6", "0.8", "all"), default="all")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--training-policy", default="configs/policies/full_model_paper.json")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _patterns(pattern: str) -> tuple[str, ...]:
    if pattern == "all":
        return ("fixed", "random")
    return (pattern,)


def main() -> None:
    args = parse_args()
    for pattern in _patterns(args.mask_pattern):
        print(f"\n[v10 full] pattern={pattern} dataset={args.dataset} rate={args.mask_rate}", flush=True)
        command = [
            sys.executable,
            "scripts/run_experiments.py",
            "--dataset",
            args.dataset,
            "--gpu",
            args.gpu,
            "--mask-pattern",
            pattern,
            "--mask-rate",
            args.mask_rate,
            "--experiments",
            "full",
            "--training-policy",
            args.training_policy,
            "--model-config-dir",
            "configs/v10-single",
            "--conda-env",
            args.conda_env,
            "--cpu-threads",
            str(args.cpu_threads),
        ]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
