#!/usr/bin/env python3
"""Run the six-point seed-42 V18 screening matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys

from pipeline_common import ABLATIONS, ROOT, SCREENING_POINTS, Job, completed_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", choices=("none", *ABLATIONS), default="none")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = [Job(*point, args.seed, args.ablation) for point in SCREENING_POINTS]
    skipped = 0
    for index, job in enumerate(jobs, start=1):
        previous = None if args.force_rerun else completed_run(job)
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
