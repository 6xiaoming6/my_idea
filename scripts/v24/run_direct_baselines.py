#!/usr/bin/env python3
"""Run three direct-propagation MoE baselines sequentially."""
from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as runner
from run_team_accept_v4 import check_gpu_idle

POLICIES = {
    "taxibj": ROOT / "configs/v24/coe_direct_baselines_taxibj_experiments.json",
    "bikenyc": ROOT / "configs/v24/coe_direct_baselines_bikenyc_experiments.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=POLICIES, default="taxibj")
    parser.add_argument("--config", help="Custom policy; overrides --dataset")
    parser.add_argument("--study", help="Study in the chosen policy")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--gpu", help="One physical GPU for a single-process job")
    devices.add_argument("--gpus", help="Two physical GPUs for one DDP job (default: 0,1)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, help="Global batch size; default doubles the policy value for two GPUs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.gpu is not None:
        if not args.gpu.isdigit():
            parser.error("--gpu must be a nonnegative device index")
        selected = [int(args.gpu)]
        device_flag, device_value = "--gpu", args.gpu
    else:
        gpus = args.gpus or "0,1"
        parts = gpus.split(",")
        if len(parts) != 2 or len(set(parts)) != 2 or any(not item.isdigit() for item in parts):
            parser.error("--gpus must contain two distinct device indices, e.g. 0,1")
        selected = [int(item) for item in parts]
        device_flag, device_value = "--gpus", gpus
    policy = runner.resolve(args.config) if args.config else POLICIES[args.dataset]
    policy_config = runner.load(policy)
    study = args.study or policy_config["default_study"]
    batch_size = args.batch_size
    if batch_size is None and len(selected) == 2:
        configured = policy_config.get("batch_size")
        if type(configured) is not int or configured <= 0:
            parser.error("Two-GPU default requires a positive policy batch_size")
        batch_size = 2 * configured
    if batch_size is not None and batch_size % len(selected):
        parser.error("Global batch size must be divisible by the number of GPUs")
    command = [sys.executable, "-u", str(ROOT / "scripts/v24/run_experiments.py"),
               "--config", str(policy), "--study", study, device_flag, device_value,
               "--epochs", str(args.epochs)]
    if batch_size is not None:
        command.extend(("--batch-size", str(batch_size)))
    if args.dry_run or args.summary_only:
        command.append("--dry-run" if args.dry_run else "--summary-only")
        return subprocess.call(command, cwd=ROOT)
    output = runner.resolve(runner.load(policy)["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".ddp_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("v24 direct baseline queue already running") from None
        check_gpu_idle(selected)
        return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
