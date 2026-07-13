#!/usr/bin/env python3
"""Run five representative V13 train/validation/test checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("train.py")
MATRIX = (
    ("TaxiBJ", "fixed", "0.2"),
    ("TaxiBJ", "random", "0.8"),
    ("BikeNYC", "fixed", "0.6"),
    ("CHAP", "fixed", "0.4"),
    ("CHAP", "random", "0.8"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")

    for index, (dataset, mask, rate) in enumerate(MATRIX, 1):
        print(f"\n[{index}/{len(MATRIX)}] {dataset} {mask}@{rate}", flush=True)
        command = [
            sys.executable, str(SCRIPT),
            "--dataset", dataset,
            "--mask", mask,
            "--rate", rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--epochs", str(args.epochs),
        ]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
