#!/usr/bin/env python3
"""Run the six representative V15 validation points from the design document."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POINTS = (
    ("TaxiBJ", "fixed", "0.2"),
    ("TaxiBJ", "fixed", "0.6"),
    ("TaxiBJ", "random", "0.4"),
    ("TaxiBJ", "random", "0.8"),
    ("BikeNYC", "fixed", "0.6"),
    ("CHAP", "random", "0.8"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for index, (dataset, mask, rate) in enumerate(POINTS, 1):
        command = [
            sys.executable,
            str(ROOT / "scripts" / "v15-single" / "train.py"),
            "--dataset", dataset,
            "--mask", mask,
            "--rate", rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(POINTS)}] {dataset} {mask}@{rate}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
