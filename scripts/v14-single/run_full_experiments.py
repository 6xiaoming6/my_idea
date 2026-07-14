#!/usr/bin/env python3
"""Sequentially run all V14 datasets/rates: fixed first, then random."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("TaxiBJ", "BikeNYC", "CHAP")
RATES = ("0.2", "0.4", "0.6", "0.8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--mask", choices=("all", "fixed", "random"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    masks = ("fixed", "random") if args.mask == "all" else (args.mask,)
    jobs = [(dataset, mask, rate) for mask in masks for dataset in datasets for rate in RATES]
    for index, (dataset, mask, rate) in enumerate(jobs, 1):
        command = [
            sys.executable,
            str(ROOT / "scripts" / "v14-single" / "train.py"),
            "--dataset", dataset,
            "--mask", mask,
            "--rate", rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(f"\n[{index}/{len(jobs)}] {dataset} {mask}@{rate}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
