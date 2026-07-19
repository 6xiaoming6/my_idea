#!/usr/bin/env python3
"""Run the documented six-point, three-seed V16 validation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
POINTS = (
    ("TaxiBJ", "random", "0.4"),
    ("TaxiBJ", "random", "0.8"),
    ("BikeNYC", "fixed", "0.6"),
    ("BikeNYC", "fixed", "0.8"),
    ("CHAP", "fixed", "0.4"),
    ("CHAP", "random", "0.8"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 2026, 3407))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = [(point, seed) for seed in args.seeds for point in POINTS]
    for index, ((dataset, mask, rate), seed) in enumerate(jobs, 1):
        command = [
            sys.executable,
            str(ROOT / "scripts" / "v16-single" / "train.py"),
            "--dataset", dataset,
            "--mask", mask,
            "--rate", rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(seed),
            "--teacher-seed", "42",
            "--run-name", "validation_matrix",
        ]
        if args.epochs is not None:
            command.extend(("--epochs", str(args.epochs)))
        if args.dry_run:
            command.append("--dry-run")
        print(
            f"[{index}/{len(jobs)}] {dataset} {mask}@{rate} seed={seed}",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
