#!/usr/bin/env python3
"""Run the seven V17 screening points defined by the design document."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POINTS = (
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("TaxiBJ", "random", "0.8"),
    ("BikeNYC", "fixed", "0.6"),
    ("BikeNYC", "random", "0.6"),
    ("CHAP", "fixed", "0.4"),
    ("CHAP", "random", "0.8"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for index, (dataset, mask, rate) in enumerate(POINTS, start=1):
        command = [
            sys.executable,
            "scripts/v17-single/train.py",
            "--dataset", dataset,
            "--mask", mask,
            "--rate", rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(args.seed),
            "--run-name", "screening",
        ]
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(POINTS)}] {dataset} {mask}@{rate}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
