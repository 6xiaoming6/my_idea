#!/usr/bin/env python3
"""Run the V17.2 clean core-point reproduction on multiple seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline_common import (
    CORE_POINTS,
    ROOT,
    Job,
    baseline_key,
    completed_run,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--points",
        nargs="+",
        choices=("all", *CORE_POINTS),
        default=None,
    )
    parser.add_argument(
        "--custom-point",
        nargs=3,
        action="append",
        metavar=("DATASET", "PATTERN", "RATE"),
        help="Add a non-core point; may be repeated.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument(
        "--baseline-manifest",
        default="configs/v17.2-single/baseline_manifest.json",
    )
    parser.add_argument("--skip-baseline-check", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.points or (["all"] if not args.custom_point else [])
    point_ids = list(CORE_POINTS) if "all" in selected else list(
        dict.fromkeys(selected)
    )
    point_specs = [CORE_POINTS[point] for point in point_ids]
    for custom in args.custom_point or []:
        dataset, pattern, rate = custom
        if dataset not in {"TaxiBJ", "BikeNYC", "CHAP"}:
            raise ValueError(f"Unsupported custom dataset: {dataset}")
        if pattern not in {"fixed", "random"}:
            raise ValueError(f"Unsupported custom pattern: {pattern}")
        if rate not in {"0.2", "0.4", "0.6", "0.8"}:
            raise ValueError(f"Unsupported custom rate: {rate}")
        point_specs.append((dataset, pattern, rate))
    point_specs = list(dict.fromkeys(point_specs))
    jobs = [
        Job(*point, seed)
        for seed in args.seeds
        for point in point_specs
    ]
    if not args.skip_baseline_check:
        path = Path(args.baseline_manifest).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        entries = load_json(path).get("entries", {})
        missing = [baseline_key(job) for job in jobs if baseline_key(job) not in entries]
        if missing:
            raise RuntimeError(
                "Explicit Full baseline manifest is missing: " + ", ".join(missing)
            )

    skipped = 0
    for index, job in enumerate(jobs, start=1):
        completed = None if args.force_rerun else completed_run(job)
        if completed is not None:
            skipped += 1
            print(
                f"[{index}/{len(jobs)}] SKIP {job.label}: "
                f"{completed.relative_to(ROOT)}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            "scripts/v17.2-single/train.py",
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
        ]
        if args.allow_dirty:
            command.append("--allow-dirty")
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] RUN {job.label}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(
        f"[done] total={len(jobs)} skipped={skipped} "
        f"executed={len(jobs) - skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
