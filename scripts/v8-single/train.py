#!/usr/bin/env python3
"""Dataset-aware training entry point for the v8-single model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[2]
SHARED_TRAIN = ROOT / "scripts" / "train.py"
DATASETS = {
    "TaxiBJ": ("taxibj.json", "taxibj.json"),
    "BikeNYC": ("bikenyc.json", "bikenyc.json"),
    "CHAP": ("chap_beijing.json", "chap.json"),
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train v8-single with the shared train/validation/test engine."
    )
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--name", default="full")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    dataset_name, override_name = DATASETS[args.dataset]
    command = [
        sys.executable,
        str(SHARED_TRAIN),
        "--config",
        str(ROOT / "configs" / "datasets" / dataset_name),
        "--override_config",
        str(ROOT / "configs" / "v8-single" / override_name),
        "--name",
        args.name,
        *remaining,
    ]
    if args.dry_run:
        print(shlex.join(command))
        return
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
