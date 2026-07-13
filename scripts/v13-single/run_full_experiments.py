#!/usr/bin/env python3
"""Run V13 full experiments sequentially: all fixed first, then all random."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("train.py")
DATASETS = ("TaxiBJ", "BikeNYC", "CHAP")
MASKS = ("fixed", "random")
RATES = ("0.2", "0.4", "0.6", "0.8")


def csv_values(text: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if text == "all":
        return allowed
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    invalid = sorted(set(values) - set(allowed))
    if not values or invalid:
        raise ValueError(f"Invalid selection {text!r}; allowed={allowed}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--mask", default="all")
    parser.add_argument("--rate", default="all")
    parser.add_argument("--training-policy", default="configs/policies/full_model_paper.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = csv_values(args.dataset, DATASETS)
    masks = csv_values(args.mask, MASKS)
    rates = csv_values(args.rate, RATES)
    jobs = [(dataset, mask, rate) for mask in masks for dataset in datasets for rate in rates]
    for index, (dataset, mask, rate) in enumerate(jobs, 1):
        print(f"\n[{index}/{len(jobs)}] {dataset} {mask}@{rate}", flush=True)
        command = [
            sys.executable, str(SCRIPT),
            "--dataset", dataset,
            "--mask", mask,
            "--rate", rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--training-policy", args.training_policy,
        ]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
