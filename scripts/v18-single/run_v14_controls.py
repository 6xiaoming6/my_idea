#!/usr/bin/env python3
"""Run clean V14 controls under the exact V18 comparison protocol."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from pipeline_common import (
    CORE_POINTS,
    DATASETS,
    PATTERNS,
    RATES,
    ROOT,
    SCREENING_POINTS,
    Job,
    build_resolved_config,
    compatible_v14_run,
)


def _git_dirty() -> bool | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [
                git,
                "status",
                "--porcelain",
                "--ignore-submodules=dirty",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _jobs(args: argparse.Namespace) -> list[Job]:
    if args.stage == "screening":
        points = SCREENING_POINTS
    elif args.stage == "core4":
        points = tuple(CORE_POINTS.values())
    else:
        datasets = (
            list(DATASETS)
            if "all" in args.datasets
            else list(dict.fromkeys(args.datasets))
        )
        patterns = [
            pattern for pattern in PATTERNS if pattern in set(args.patterns)
        ]
        points = tuple(
            (dataset, pattern, rate)
            for pattern in patterns
            for dataset in datasets
            for rate in args.rates
        )
    return [
        Job(*point, seed)
        for seed in args.seeds
        for point in points
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("screening", "core4", "full24"),
        default="screening",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--datasets", nargs="+", choices=("all", *DATASETS), default=["all"]
    )
    parser.add_argument(
        "--patterns", nargs="+", choices=PATTERNS, default=list(PATTERNS)
    )
    parser.add_argument("--rates", nargs="+", choices=RATES, default=list(RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    dirty = _git_dirty()
    if dirty is True and not (args.allow_dirty or args.dry_run):
        raise RuntimeError(
            "Formal matched V14 controls require a clean Git tree. "
            "Commit/stash changes or use --allow-dirty only for debugging."
        )
    if dirty is None:
        print("[warning] Git status is unavailable.", flush=True)

    jobs = _jobs(args)
    skipped = 0
    for index, job in enumerate(jobs, start=1):
        expected_v18 = build_resolved_config(
            job.dataset,
            job.mask,
            job.rate,
            job.seed,
        )
        previous, _ = compatible_v14_run(
            job, candidate_config=expected_v18
        )
        if previous is not None and not args.force_rerun:
            skipped += 1
            print(
                f"[{index}/{len(jobs)}] SKIP {job.label}: "
                f"{previous.relative_to(ROOT)}",
                flush=True,
            )
            continue

        policy = (
            "configs/v18-single/policies/v14_control_fixed.json"
            if job.mask == "fixed"
            else "configs/v18-single/policies/v14_control_random.json"
        )
        command = [
            sys.executable,
            "scripts/v14-single/train.py",
            "--dataset",
            job.dataset,
            "--mask",
            job.mask,
            "--rate",
            job.rate,
            "--gpu",
            args.gpu,
            "--conda-env",
            args.conda_env,
            "--cpu-threads",
            str(args.cpu_threads),
            "--seed",
            str(job.seed),
            "--training-policy",
            policy,
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(
            f"[{index}/{len(jobs)}] RUN matched V14 {job.label}",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)

    print(
        f"[done] total={len(jobs)} skipped={skipped} "
        f"executed={len(jobs) - skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
