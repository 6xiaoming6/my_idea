#!/usr/bin/env python3
"""Run the complete V19 grid sequentially on one GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("TaxiBJ", "BikeNYC", "CHAP"),
        default=("TaxiBJ", "BikeNYC", "CHAP"),
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=("fixed", "random"),
        default=("fixed", "random"),
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        choices=("0.2", "0.4", "0.6", "0.8"),
        default=("0.2", "0.4", "0.6", "0.8"),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = [
        (dataset, pattern, rate)
        for pattern in args.patterns
        for dataset in args.datasets
        for rate in args.rates
    ]
    for index, (dataset, pattern, rate) in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] RUN {dataset} {pattern}@{rate}",
            flush=True,
        )
        command = [
            sys.executable,
            "scripts/v19-single/train.py",
            "--dataset",
            dataset,
            "--mask",
            pattern,
            "--rate",
            rate,
            "--gpu",
            args.gpu,
            "--conda-env",
            args.conda_env,
            "--cpu-threads",
            str(args.cpu_threads),
            "--seed",
            str(args.seed),
        ]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

