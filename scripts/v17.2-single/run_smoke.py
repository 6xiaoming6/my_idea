#!/usr/bin/env python3
"""Run one real-data train/validation/test epoch for each V17.2 dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys

from pipeline_common import CORE_POINTS, ROOT


SMOKE_POINTS = ("P1", "P2", "P4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for index, point in enumerate(SMOKE_POINTS, start=1):
        dataset, mask, rate = CORE_POINTS[point]
        command = [
            sys.executable,
            "scripts/v17.2-single/train.py",
            "--dataset",
            dataset,
            "--mask",
            mask,
            "--rate",
            rate,
            "--gpu",
            args.gpu,
            "--conda-env",
            args.conda_env,
            "--cpu-threads",
            str(args.cpu_threads),
            "--seed",
            str(args.seed),
            "--epochs",
            "1",
            "--run-name",
            f"debug_v17_2_smoke_{dataset.lower()}",
            "--allow-dirty",
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(SMOKE_POINTS)}] RUN {point} {dataset}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("[done] V17.2 real-data smoke completed.", flush=True)


if __name__ == "__main__":
    main()
