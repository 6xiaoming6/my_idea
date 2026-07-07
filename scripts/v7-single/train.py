#!/usr/bin/env python3
"""Dataset-aware training entry point for the v7-single model.

This wrapper only selects the stable dataset config and its v7-single model
override.  All training, validation, best-checkpoint, test, and logging logic
continues to live in the shared ``scripts/train.py`` implementation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[2]
SHARED_TRAIN = ROOT / "scripts" / "train.py"

DATASETS = {
    "TaxiBJ": (
        ROOT / "configs" / "datasets" / "taxibj.json",
        ROOT / "configs" / "v7-single" / "taxibj.json",
    ),
    "BikeNYC": (
        ROOT / "configs" / "datasets" / "bikenyc.json",
        ROOT / "configs" / "v7-single" / "bikenyc.json",
    ),
    "CHAP": (
        ROOT / "configs" / "datasets" / "chap_beijing.json",
        ROOT / "configs" / "v7-single" / "chap.json",
    ),
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train v7-single while reusing the shared training engine.",
    )
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument(
        "--name",
        default="full",
        help="Experiment name passed to the shared trainer (default: full).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved shared training command without executing it.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    dataset_config, model_config = DATASETS[args.dataset]
    command = [
        sys.executable,
        str(SHARED_TRAIN),
        "--config",
        str(dataset_config),
        "--override_config",
        str(model_config),
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
