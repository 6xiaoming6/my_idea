#!/usr/bin/env python3
"""Run the complete V17.2 3-dataset × 2-pattern × 4-rate grid."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline_common import (
    DATASETS,
    RATES,
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
        "--datasets",
        nargs="+",
        choices=("all", *DATASETS),
        default=["all"],
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=("fixed", "random"),
        default=["fixed", "random"],
    )
    parser.add_argument("--rates", nargs="+", choices=RATES, default=list(RATES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-policy", default=None)
    parser.add_argument(
        "--baseline-manifest",
        default="configs/v17.2-single/baseline_manifest.json",
    )
    parser.add_argument("--skip-baseline-check", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _datasets(values: list[str]) -> list[str]:
    return list(DATASETS) if "all" in values else list(dict.fromkeys(values))


def _manifest(path_text: str) -> dict:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return load_json(path).get("entries", {})


def main() -> None:
    args = parse_args()
    datasets = _datasets(args.datasets)
    ordered_patterns = [
        pattern for pattern in ("fixed", "random") if pattern in args.patterns
    ]
    jobs = [
        Job(dataset, pattern, rate, args.seed)
        for pattern in ordered_patterns
        for dataset in datasets
        for rate in args.rates
    ]
    if not args.skip_baseline_check:
        entries = _manifest(args.baseline_manifest)
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
        if args.training_policy:
            command.extend(["--training-policy", args.training_policy])
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
