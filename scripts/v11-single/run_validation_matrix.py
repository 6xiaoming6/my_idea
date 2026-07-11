#!/usr/bin/env python3
"""Run five representative v11 train/validation/test checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POINTS = (
    ("TaxiBJ", "fixed", "0.2"),
    ("TaxiBJ", "random", "0.8"),
    ("BikeNYC", "fixed", "0.6"),
    ("CHAP", "fixed", "0.4"),
    ("CHAP", "random", "0.8"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--epochs", type=int, choices=(1, 5), default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for index, (dataset, pattern, rate) in enumerate(POINTS, 1):
        print(f"[v11 validation {index}/{len(POINTS)}] {dataset} {pattern}@{rate}", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("train.py")),
            "--dataset",
            dataset,
            "--mask-pattern",
            pattern,
            "--mask-rate",
            rate,
            "--gpu",
            args.gpu,
            "--conda-env",
            args.conda_env,
            "--quick",
            str(args.epochs),
        ]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
