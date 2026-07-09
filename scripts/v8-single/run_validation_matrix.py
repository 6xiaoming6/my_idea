#!/usr/bin/env python3
"""Run the five required v8-single validation points sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_experiments.py"
MODEL_CONFIG_DIR = ROOT / "configs" / "v8-single"
DEFAULT_POLICY = ROOT / "configs" / "policies" / "full_model_paper.json"
MATRIX = (
    ("TaxiBJ", "fixed", "0.2"),
    ("TaxiBJ", "random", "0.6"),
    ("BikeNYC", "fixed", "0.6"),
    ("CHAP", "fixed", "0.4"),
    ("CHAP", "random", "0.8"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--fixed-seed", type=int, default=42)
    parser.add_argument("--training-policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for index, (dataset, pattern, rate) in enumerate(MATRIX, 1):
        command = [
            sys.executable,
            str(RUNNER),
            "--dataset",
            dataset,
            "--gpu",
            str(args.gpu),
            "--mask-pattern",
            pattern,
            "--mask-rate",
            rate,
            "--experiments",
            "full",
            "--fixed-seed",
            str(args.fixed_seed),
            "--conda-env",
            args.conda_env,
            "--cpu-threads",
            str(args.cpu_threads),
            "--training-policy",
            args.training_policy,
            "--model-config-dir",
            str(MODEL_CONFIG_DIR),
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(
            f"\n[{index}/{len(MATRIX)}] v8-single {dataset} {pattern}@{rate}\n"
            f"[run] {shlex.join(command)}",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
