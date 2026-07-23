#!/usr/bin/env python3
"""Run evidence-gated V17.1 C2/C3 combinations after Stage 1 is complete."""

from __future__ import annotations

import argparse
import subprocess
import sys

from pipeline_common import (
    COMBINATION_CODES,
    COMBINATION_VARIANTS,
    CONFIG_ROOT,
    POINTS,
    ROOT,
    combination_run,
    missing_stage1,
)


# C1 is exactly E7 and is deliberately reused instead of retrained.
RUNNABLE_COMBINATIONS = COMBINATION_VARIANTS[1:]


def _selection(values: list[str]) -> list[str]:
    if "all" in values:
        return list(RUNNABLE_COMBINATIONS)
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--points",
        nargs="+",
        choices=("core4", *POINTS),
        default=["core4"],
    )
    parser.add_argument(
        "--combinations",
        nargs="+",
        choices=("all", *RUNNABLE_COMBINATIONS),
        default=["all"],
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--skip-stage1-check",
        action="store_true",
        help="Bypass the requirement that all 32 seed-42 Stage-1 runs exist.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if not args.smoke and not args.dry_run and not args.skip_stage1_check:
        missing = missing_stage1(seed=42)
        if missing:
            preview = ", ".join(missing[:8])
            raise RuntimeError(
                f"Stage 1 is incomplete ({len(missing)}/32 missing): {preview}. "
                "Finish and summarize Stage 1 before running combinations."
            )

    combinations = _selection(args.combinations)
    point_ids = list(POINTS) if "core4" in args.points else list(dict.fromkeys(args.points))
    print(
        "[reuse] C1 c1_independent_shared_scale = Stage-1 E7 "
        "independent_shared_scale; no duplicate training.",
        flush=True,
    )
    jobs = [
        (combination, point, seed)
        for combination in combinations
        for point in (POINTS[point_id] for point_id in point_ids)
        for seed in args.seeds
    ]
    skipped = 0
    for index, (combination, point, seed) in enumerate(jobs, start=1):
        completed = None
        if not args.smoke and not args.force_rerun:
            completed = combination_run(point, seed, combination)
        if completed is not None:
            skipped += 1
            print(
                f"[{index}/{len(jobs)}] SKIP {COMBINATION_CODES[combination]} "
                f"{point.label} seed={seed}: {completed.relative_to(ROOT)}",
                flush=True,
            )
            continue

        config_path = CONFIG_ROOT / "combinations" / f"{combination}.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        run_name = (
            f"debug_v17_1_{combination}"
            if args.smoke
            else f"combination_{combination}"
        )
        command = [
            sys.executable,
            "scripts/v17-single/train.py",
            "--dataset", point.dataset,
            "--mask", point.mask,
            "--rate", point.rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(seed),
            "--variant-config", str(config_path.relative_to(ROOT)),
            "--output-dir", "outputs/v17.1-single",
            "--model-version", "v17.1-single",
            "--run-name", run_name,
        ]
        if args.smoke:
            command.extend(["--epochs", "1"])
        if args.dry_run:
            command.append("--dry-run")
        print(
            f"[{index}/{len(jobs)}] RUN {COMBINATION_CODES[combination]} "
            f"{point.label} seed={seed}",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)

    print(
        f"[done] stage=2 total={len(jobs)} skipped={skipped} "
        f"executed={len(jobs) - skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
