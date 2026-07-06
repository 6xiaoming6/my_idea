#!/usr/bin/env python3
"""Run safe TaxiBJ baselines sequentially on one GPU.

All trainable models use full paper budgets. CSDI and PriSTI are always
excluded because their diffusion workloads are unsafe for this server. By
default all fixed jobs finish before any random job starts.
The final epoch is always validated and the best validation checkpoint is used
for one final test evaluation.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path


BENCH = Path(__file__).resolve().parents[2]
LAUNCHER = BENCH / "scripts" / "launch" / "run_all_baseline_train_2gpu.py"
POLICY = BENCH / "configs" / "policies" / "taxibj_full_no_diffusion.json"
MODELS = (
    "ImputeFormer", "BRITS", "GAIN", "LATC",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="Physical GPU ID, e.g. 0 or 1")
    parser.add_argument(
        "--mask", choices=("fixed", "random", "both"), default="both",
        help="Default 'both' runs all fixed jobs first, then all random jobs",
    )
    parser.add_argument("--rates", nargs="+", type=float, default=(0.2, 0.4, 0.6, 0.8))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--channel", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=0, help="Per-stage seconds; 0 means unlimited")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--resume-run", help="Resume an existing launcher run directory")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(not 0 < rate < 1 for rate in args.rates):
        raise ValueError("Every missing rate must be between 0 and 1")
    masks = ("fixed", "random") if args.mask == "both" else (args.mask,)
    command = [
        sys.executable,
        str(LAUNCHER),
        "--gpus", args.gpu,
        "--datasets", "TaxiBJ",
        "--masks", *masks,
        "--rates", *(format(rate, "g") for rate in args.rates),
        "--models", *args.models,
        "--channel", args.channel,
        "--seed", str(args.seed),
        "--timeout", str(args.timeout),
        "--output-root", args.output_root,
        "--policy-json", str(POLICY),
        "--run-root", str(BENCH / "artifacts" / "runs" / "taxibj"),
    ]
    if args.resume_run:
        command.extend(("--resume-run", args.resume_run))
    if args.skip_prepare:
        command.append("--skip-prepare")
    if args.rebuild_data:
        command.append("--rebuild-data")
    if args.dry_run:
        command.append("--dry-run")
    if args.no_checkpoint:
        command.append("--no-checkpoint")
    print("Launching:", " ".join(command), flush=True)
    if args.dry_run:
        raise SystemExit(subprocess.run(command, check=False).returncode)

    lock_path = BENCH / "artifacts" / "runs" / "taxibj" / ".single_gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                "Another TaxiBJ full-baseline launcher is already active. "
                "Do not run fixed/random in separate terminals."
            ) from exc
        lock.write(f"pid={os.getpid()} gpu={args.gpu} masks={','.join(masks)}\n")
        lock.flush()
        raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
