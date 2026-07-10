#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_POINTS = (
    ("TaxiBJ", "fixed", "0.2"),
    ("TaxiBJ", "random", "0.6"),
    ("BikeNYC", "fixed", "0.6"),
    ("CHAP", "fixed", "0.4"),
    ("CHAP", "random", "0.8"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v10-single quick validation matrix.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--epochs", choices=("1", "5"), default="5")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = f"configs/v10-single/policies/quick_{args.epochs}epoch.json"
    for index, (dataset, pattern, rate) in enumerate(VALIDATION_POINTS, 1):
        print(f"\n[v10 validation {index}/{len(VALIDATION_POINTS)}] {dataset} {pattern}@{rate}", flush=True)
        command = [
            sys.executable,
            "scripts/run_experiments.py",
            "--dataset",
            dataset,
            "--gpu",
            args.gpu,
            "--mask-pattern",
            pattern,
            "--mask-rate",
            rate,
            "--experiments",
            "full",
            "--training-policy",
            policy,
            "--model-config-dir",
            "configs/v10-single",
            "--conda-env",
            args.conda_env,
        ]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
