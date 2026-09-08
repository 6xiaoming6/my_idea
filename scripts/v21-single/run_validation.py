#!/usr/bin/env python3
"""Run the staged V21 P0/P1/P2 validation protocol on one GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys

from _protocol import (
    DATASETS,
    PATTERNS,
    RATES,
    ROOT,
    SEEDS,
    VARIANTS,
    expected_epochs,
    filtered_points,
    find_completed_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("core6", "multiseed", "all24"), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=PATTERNS)
    parser.add_argument("--rates", nargs="+", choices=RATES, default=RATES)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if args.variants is None:
        if args.phase == "core6":
            variants = tuple(VARIANTS)
        else:
            raise ValueError(
                f"--variants is required for phase={args.phase}; only promoted variants "
                "should enter expensive follow-up stages"
            )
    else:
        variants = tuple(dict.fromkeys(args.variants))

    if args.seeds is None:
        seeds = SEEDS if args.phase == "multiseed" else (42,)
    else:
        seeds = tuple(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates")
    if args.phase != "multiseed" and len(seeds) != 1:
        raise ValueError("core6/all24 accept exactly one seed; use phase=multiseed otherwise")

    points = filtered_points(
        args.phase,
        tuple(args.datasets),
        tuple(args.patterns),
        tuple(args.rates),
    )
    if not points:
        raise ValueError("The dataset/pattern/rate filters selected no protocol points")

    jobs = [
        (variant_key, seed, point)
        for variant_key in variants
        for seed in seeds
        for point in points
    ]
    failures: list[str] = []
    for index, (variant_key, seed, point) in enumerate(jobs, 1):
        variant = VARIANTS[variant_key]
        epochs = expected_epochs(point, args.epochs)
        label = f"{variant_key} {point.label} seed={seed} epochs={epochs}"
        if not args.rerun_completed:
            completed = find_completed_run(variant, point, seed, epochs)
            if completed is not None:
                print(f"[{index}/{len(jobs)}] SKIP complete {label}", flush=True)
                continue
        command = [
            sys.executable,
            "scripts/v21-single/train.py",
            "--variant", variant.train_variant,
            "--dataset", point.dataset,
            "--mask", point.pattern,
            "--rate", point.rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(seed),
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
