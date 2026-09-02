#!/usr/bin/env python3
"""Run the six E02 small-weight diagnoses on random@0.2 sequentially."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/v14-exploration/root_cause"
CONFIGS = {
    "2.5e-5": CONFIG_ROOT / "E02_lowrate_delta_scale_2.5e-5.json",
    "5e-5": CONFIG_ROOT / "E02_lowrate_delta_scale_5e-5.json",
}
POINTS = (
    ("TaxiBJ", "random", "0.2"),
    ("BikeNYC", "random", "0.2"),
    ("CHAP", "random", "0.2"),
)
OUTPUT_NAMES = {
    "TaxiBJ": "TaxiBJ",
    "BikeNYC": "BikeNYC",
    "CHAP": "CHAP_Beijing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights",
        nargs="+",
        choices=tuple(CONFIGS),
        default=("2.5e-5", "5e-5"),
        help="Small E02 weights to diagnose; both are run by default.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,
        help=(
            "Maximum seconds for one complete train/val/test job. "
            "Default: 14400 (4 hours); use 0 to disable."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Debug override only. Omit for the complete dataset-specific budget.",
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _test_completed(test_log: Path) -> bool:
    try:
        text = test_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    if "Testing finished:" not in text:
        return False
    values = []
    for metric in ("mae", "rmse"):
        match = re.search(rf"^  {metric}:\s+(\S+)\s*$", text, re.MULTILINE)
        if match is None:
            return False
        try:
            values.append(float(match.group(1)))
        except ValueError:
            return False
    return all(math.isfinite(value) for value in values)


def _completed(
    dataset: str,
    pattern: str,
    rate: str,
    config_path: Path,
    seed: int,
) -> bool:
    expected = json.loads(config_path.read_text(encoding="utf-8"))["experiment"]
    variant = f"v14_exploration_{config_path.stem}"
    root = (
        ROOT
        / "outputs/v14-exploration"
        / OUTPUT_NAMES[dataset]
        / "ablation"
        / variant
        / pattern
        / f"rate{rate}"
    )
    for test_log in sorted(root.glob("*/logs/test.log"), reverse=True):
        if not _test_completed(test_log):
            continue
        run_dir = test_log.parent.parent
        config_file = run_dir / "config.json"
        checkpoint = run_dir / "checkpoints/best.pt"
        metrics_file = run_dir / "logs/metrics.jsonl"
        if not all(path.is_file() for path in (config_file, checkpoint, metrics_file)):
            continue
        if checkpoint.stat().st_size == 0 or metrics_file.stat().st_size == 0:
            continue
        config = json.loads(config_file.read_text(encoding="utf-8"))
        if (
            int(config.get("seed", -1)) == seed
            and config.get("experiment") == expected
        ):
            return True
    return False


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _run(command: list[str], timeout: int) -> None:
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        process.wait(timeout=None if timeout == 0 else timeout)
    except KeyboardInterrupt:
        _terminate_process_group(process)
        raise
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise RuntimeError(
            f"Job exceeded --timeout={timeout}s and its process group was stopped. "
            "Rerun this script; completed jobs will be skipped."
        ) from exc
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def main() -> None:
    args = parse_args()
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.timeout < 0:
        raise ValueError("--timeout must be non-negative")
    if len(set(args.weights)) != len(args.weights):
        raise ValueError("--weights must not contain duplicates")

    jobs = [
        (dataset, pattern, rate, weight, CONFIGS[weight])
        for dataset, pattern, rate in POINTS
        for weight in args.weights
    ]
    for index, (dataset, pattern, rate, weight, config_path) in enumerate(jobs, 1):
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        label = (
            f"E02-LR weight={weight} {dataset} "
            f"{pattern}@{rate} seed={args.seed}"
        )
        if (
            not args.rerun_completed
            and _completed(dataset, pattern, rate, config_path, args.seed)
        ):
            print(f"[{index}/{len(jobs)}] SKIP complete {label}", flush=True)
            continue

        command = [
            sys.executable,
            "scripts/v14-single/train.py",
            "--dataset", dataset,
            "--mask", pattern,
            "--rate", rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(args.seed),
            "--experiment-config", str(config_path.relative_to(ROOT)),
        ]
        if args.epochs is not None:
            command.extend(("--epochs", str(args.epochs)))
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] RUN {label}", flush=True)
        _run(command, args.timeout)


if __name__ == "__main__":
    main()
