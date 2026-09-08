#!/usr/bin/env python3
"""Run only the new V21.1 P3/P4 Core-6 jobs; existing P0/P1/P2 are reused."""

from __future__ import annotations

import argparse
import subprocess
import sys

from v21_1_protocol import CORE6, ROOT, VARIANTS, expected_epochs, find_completed_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--variants", nargs="+", choices=("P3", "P4"), default=("P3", "P4"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    variants = tuple(dict.fromkeys(args.variants))
    jobs = [(key, point) for key in variants for point in CORE6]
    failures = []
    for index, (key, point) in enumerate(jobs, 1):
        variant = VARIANTS[key]
        epochs = expected_epochs(point, args.epochs)
        label = f"{key} {point.label} seed={args.seed} epochs={epochs}"
        if not args.rerun_completed:
            completed = find_completed_run(variant, point, args.seed, epochs)
            if completed is not None:
                print(f"[{index}/{len(jobs)}] SKIP complete {label}", flush=True)
                continue
        command = [
            sys.executable,
            "scripts/v21.1-single/train.py",
            "--variant", str(variant.train_variant),
            "--dataset", point.dataset,
            "--mask", point.pattern,
            "--rate", point.rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(args.seed),
            "--run-name", variant.run_name,
        ]
        if args.epochs is not None:
            command.extend(("--epochs", str(args.epochs)))
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] RUN {label}", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            failures.append(f"{label}: exit={error.returncode}")
            print(f"[{index}/{len(jobs)}] FAILED {label}", flush=True)
            if not args.continue_on_error:
                raise
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
