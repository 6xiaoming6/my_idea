from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RATES = ("0.2", "0.4", "0.6", "0.8")
DATASETS = ("TaxiBJ", "BikeNYC", "CHAP")


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def run_points(
    args: argparse.Namespace,
    points: list[tuple[str, str, str]],
    *,
    seeds: list[int] | None = None,
    ablation: str = "none",
    run_name: str | None = None,
) -> None:
    jobs = [
        (dataset, pattern, rate, seed)
        for seed in (seeds or [args.seed])
        for dataset, pattern, rate in points
    ]
    failures = []
    for index, (dataset, pattern, rate, seed) in enumerate(jobs, start=1):
        label = f"{dataset} {pattern}@{rate} seed={seed}"
        command = [
            sys.executable,
            "scripts/v20-single/train.py",
            "--dataset", dataset,
            "--mask", pattern,
            "--rate", rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(seed),
            "--ablation", ablation,
        ]
        if args.epochs is not None:
            command.extend(("--epochs", str(args.epochs)))
        if run_name is not None:
            command.extend(("--run-name", run_name))
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] RUN {label}", flush=True)
        print(" ".join(command), flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            failures.append(f"{label}: {error}")
            print(f"[{index}/{len(jobs)}] FAILED {label}", flush=True)
            if args.stop_on_error:
                raise
    if failures:
        raise SystemExit("V20 jobs failed:\n  - " + "\n  - ".join(failures))
