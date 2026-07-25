#!/usr/bin/env python3
"""Run the V18 3-dataset × 2-pattern × 4-rate experiment grid."""

from __future__ import annotations

import argparse
import subprocess
import sys

from pipeline_common import (
    ABLATIONS,
    DATASETS,
    PATTERNS,
    RATES,
    ROOT,
    Job,
    build_resolved_config,
    completed_run,
    resolve_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", choices=("none", *ABLATIONS), default="none")
    parser.add_argument("--training-policy", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = list(DATASETS) if "all" in args.datasets else list(
        dict.fromkeys(args.datasets)
    )
    patterns = [
        pattern for pattern in PATTERNS if pattern in set(args.patterns)
    ]
    jobs = [
        Job(dataset, pattern, rate, args.seed, args.ablation)
        for pattern in patterns
        for dataset in datasets
        for rate in args.rates
    ]
    policy_path = (
        resolve_path(args.training_policy)
        if args.training_policy
        else None
    )

    skipped = 0
    for index, job in enumerate(jobs, start=1):
        expected_config = build_resolved_config(
            job.dataset,
            job.mask,
            job.rate,
            job.seed,
            ablation=job.ablation,
            training_policy=policy_path,
        )
        previous = (
            None
            if args.force_rerun
            else completed_run(job, expected_config=expected_config)
        )
        if previous is not None:
            skipped += 1
            print(
                f"[{index}/{len(jobs)}] SKIP {job.label}: "
                f"{previous.relative_to(ROOT)}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            "scripts/v18-single/train.py",
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
            "--ablation",
            job.ablation,
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
