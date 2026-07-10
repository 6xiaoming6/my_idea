#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATASET_CONFIG = {
    "TaxiBJ": "configs/datasets/taxibj.json",
    "BikeNYC": "configs/datasets/bikenyc.json",
    "CHAP": "configs/datasets/chap_beijing.json",
}
DATASET_NPZ = {
    "TaxiBJ": (
        "data/TaxiBJ/taxibj_train.npz",
        "data/TaxiBJ/taxibj_val.npz",
        "data/TaxiBJ/taxibj_test.npz",
    ),
    "BikeNYC": (
        "data/BikeNYC/bikenyc_train.npz",
        "data/BikeNYC/bikenyc_val.npz",
        "data/BikeNYC/bikenyc_test.npz",
    ),
    "CHAP": (
        "data/CHAP/beijing/chap_beijing_train.npz",
        "data/CHAP/beijing/chap_beijing_val.npz",
        "data/CHAP/beijing/chap_beijing_test.npz",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one v10-single training job.")
    parser.add_argument("--dataset", choices=DATASET_CONFIG, required=True)
    parser.add_argument("--mask-pattern", choices=("fixed", "random"), required=True)
    parser.add_argument("--mask-rate", choices=("0.2", "0.4", "0.6", "0.8"), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--training-policy", default="configs/policies/full_model_paper.json")
    parser.add_argument("--quick", choices=("1", "5"), default=None, help="Use a quick v10 policy.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = args.training_policy
    if args.quick:
        policy = f"configs/v10-single/policies/quick_{args.quick}epoch.json"
    command = [
        sys.executable,
        "scripts/run_experiments.py",
        "--dataset",
        args.dataset,
        "--gpu",
        args.gpu,
        "--mask-pattern",
        args.mask_pattern,
        "--mask-rate",
        args.mask_rate,
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
    print("[v10-single]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
