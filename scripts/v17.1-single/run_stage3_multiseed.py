#!/usr/bin/env python3
"""Run Full, the selected Stage-1 variant and selected combination on three seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys

from pipeline_common import (
    COMBINATION_VARIANTS,
    POINTS,
    ROOT,
    STAGE1_VARIANTS,
    combination_run,
    missing_stage1,
)


STAGE1_CANDIDATES = STAGE1_VARIANTS[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument(
        "--stage1-variant",
        required=True,
        choices=STAGE1_CANDIDATES,
        help="Best single-factor variant selected from the Stage-1 summary.",
    )
    parser.add_argument(
        "--combination-variant",
        required=True,
        choices=COMBINATION_VARIANTS,
        help="Best Stage-2 combination selected from the combination summary.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--skip-prerequisite-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _common_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--gpu", args.gpu,
        "--conda-env", args.conda_env,
        "--cpu-threads", str(args.cpu_threads),
        "--points", "core4",
        "--seeds", *[str(seed) for seed in args.seeds],
    ]
    if args.force_rerun:
        values.append("--force-rerun")
    if args.dry_run:
        values.append("--dry-run")
    return values


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if (
        args.stage1_variant == "independent_shared_scale"
        and args.combination_variant == "c1_independent_shared_scale"
    ):
        raise ValueError(
            "The selected Stage-1 variant and C1 are the same model. "
            "Choose C2/C3 or another Stage-1 candidate to keep three distinct models."
        )

    if not args.dry_run and not args.skip_prerequisite_check:
        missing = missing_stage1(seed=42)
        if missing:
            raise RuntimeError(
                f"Stage 1 is incomplete ({len(missing)}/32 missing). "
                "Do not start multi-seed validation yet."
            )
        missing_combo = [
            point.label
            for point in POINTS.values()
            if combination_run(point, 42, args.combination_variant) is None
        ]
        if missing_combo:
            raise RuntimeError(
                f"Selected Stage-2 combination lacks seed-42 results for: "
                f"{', '.join(missing_combo)}"
            )

    common = _common_args(args)
    stage1_script = "scripts/v17.1-single/run_exploratory_ablation.py"
    stage2_script = "scripts/v17.1-single/run_stage2_combinations.py"

    commands = [
        [
            sys.executable,
            stage1_script,
            *common,
            "--variants", "full",
            "--skip-reproduction-check",
        ],
        [
            sys.executable,
            stage1_script,
            *common,
            "--variants", args.stage1_variant,
            "--skip-reproduction-check",
        ],
    ]
    if args.combination_variant == "c1_independent_shared_scale":
        commands.append(
            [
                sys.executable,
                stage1_script,
                *common,
                "--variants", "independent_shared_scale",
                "--skip-reproduction-check",
            ]
        )
    else:
        commands.append(
            [
                sys.executable,
                stage2_script,
                *common,
                "--combinations", args.combination_variant,
                "--skip-stage1-check",
            ]
        )

    labels = (
        "Full V17.1",
        f"Stage-1 {args.stage1_variant}",
        f"Stage-2 {args.combination_variant}",
    )
    for index, (label, command) in enumerate(zip(labels, commands), start=1):
        print(f"[stage3 {index}/3] {label}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("[done] Stage 3 multi-seed jobs completed or skipped.", flush=True)


if __name__ == "__main__":
    main()
