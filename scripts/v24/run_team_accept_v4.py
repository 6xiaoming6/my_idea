#!/usr/bin/env python3
"""Run six focused BikeNYC CoE versus layered MoE ablations sequentially."""
from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v24"))
import run_experiments as runner

POLICY = ROOT / "configs/v24/coe_focus_bikenyc_experiments.json"
STUDY = "coe_focus_bikenyc"


def check_gpu_idle(gpus: list[int]) -> None:
    visible = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    missing = set(map(str, gpus)) - set(visible.stdout.split())
    if missing:
        raise RuntimeError(f"GPU indices do not exist: {sorted(missing)}")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    if processes.stdout.strip():
        raise RuntimeError("A GPU already has a compute process; focused queue was not started:\n" + processes.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1", help="Two physical GPUs for one DDP experiment")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    parts = args.gpus.split(",")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        parser.error("--gpus must contain exactly two device indices, e.g. 0,1")
    gpu_ids = [int(part) for part in parts]
    if len(set(gpu_ids)) != 2:
        parser.error("--gpus must contain exactly two distinct device indices, e.g. 0,1")
    command = [sys.executable, "-u", str(ROOT / "scripts/v24/run_experiments.py"),
               "--config", str(POLICY), "--study", STUDY, "--gpus", args.gpus,
               "--epochs", str(args.epochs)]
    if args.dry_run or args.summary_only:
        command.append("--dry-run" if args.dry_run else "--summary-only")
        return subprocess.call(command, cwd=ROOT)
    output = runner.resolve(runner.load(POLICY)["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".ddp_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("focused queue already running") from None
        check_gpu_idle(gpu_ids)
        return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
