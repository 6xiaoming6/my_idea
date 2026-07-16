#!/usr/bin/env python3
"""Run the V15.1 full matrix, skipping only complete train/val/test jobs."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "v15.1-single"
RATES = ("0.2", "0.4", "0.6", "0.8")


@dataclass(frozen=True)
class DatasetSpec:
    cli_name: str
    output_name: str
    config_path: str

    @property
    def expected_epochs(self) -> int:
        return int(_load_json(ROOT / self.config_path)["train"]["epochs"])


DATASETS = {
    "TaxiBJ": DatasetSpec("TaxiBJ", "TaxiBJ", "configs/v15.1-single/taxibj.json"),
    "BikeNYC": DatasetSpec("BikeNYC", "BikeNYC", "configs/v15.1-single/bikenyc.json"),
    "CHAP": DatasetSpec("CHAP", "CHAP_Beijing", "configs/v15.1-single/chap.json"),
}


@dataclass(frozen=True)
class Job:
    dataset: DatasetSpec
    mask: str
    rate: str

    @property
    def label(self) -> str:
        return f"{self.dataset.cli_name} {self.mask}@{self.rate}"

    @property
    def run_root(self) -> Path:
        return (
            OUTPUT_ROOT
            / self.dataset.output_name
            / "full"
            / "model"
            / self.mask
            / f"rate{float(self.rate):g}"
        )


@dataclass(frozen=True)
class Completion:
    run_dir: Path
    completed_epochs: int
    test_mae: float


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _matches_config(config: dict, job: Job, seed: int) -> bool:
    data = config.get("data", {})
    mask = data.get("mask", {})
    model = config.get("model", {})
    try:
        rate_matches = math.isclose(
            float(mask.get("missing_rate")), float(job.rate), abs_tol=1e-9
        )
    except (TypeError, ValueError):
        return False
    return (
        data.get("dataset_name") == job.dataset.output_name
        and mask.get("pattern") == job.mask
        and rate_matches
        and int(config.get("seed", -1)) == seed
        and model.get("version") == "v15.1-single"
        and model.get("architecture") == "v15_1_scale_guided_residual_moe"
        and int(config.get("train", {}).get("epochs", -1))
        == job.dataset.expected_epochs
    )


def _read_metrics(path: Path) -> tuple[int, float | None]:
    max_epoch = 0
    test_mae: float | None = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record.get("epoch"), int):
                max_epoch = max(max_epoch, int(record["epoch"]))
            if record.get("stage") == "test" and isinstance(record.get("metrics"), dict):
                value = float(record["metrics"].get("mae"))
                if math.isfinite(value):
                    test_mae = value
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0, None
    return max_epoch, test_mae


def _completion(run_dir: Path, job: Job, seed: int) -> Completion | None:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    test_log = run_dir / "logs" / "test.log"
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if not all(path.is_file() for path in (config_path, metrics_path, test_log, checkpoint)):
        return None
    if test_log.stat().st_size == 0 or checkpoint.stat().st_size == 0:
        return None
    try:
        config = _load_json(config_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not _matches_config(config, job, seed):
        return None
    completed_epochs, test_mae = _read_metrics(metrics_path)
    if completed_epochs < job.dataset.expected_epochs or test_mae is None:
        return None
    return Completion(run_dir, completed_epochs, test_mae)


def find_completed_run(job: Job, seed: int) -> Completion | None:
    candidates = []
    if job.run_root.is_dir():
        for config_path in job.run_root.glob("*/config.json"):
            result = _completion(config_path.parent, job, seed)
            if result is not None:
                candidates.append(result)
    return max(candidates, key=lambda item: item.run_dir.stat().st_mtime) if candidates else None


def find_active_launcher(job: Job, seed: int) -> int | None:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
            continue
        try:
            parts = [
                item.decode(errors="replace")
                for item in (process_dir / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (OSError, PermissionError):
            continue
        if not any(item.endswith("scripts/v15.1-single/train.py") for item in parts):
            continue

        def value(name: str) -> str | None:
            try:
                index = parts.index(name)
            except ValueError:
                return None
            return parts[index + 1] if index + 1 < len(parts) else None

        if (
            value("--dataset") == job.dataset.cli_name
            and value("--mask") == job.mask
            and value("--rate") == job.rate
            and int(value("--seed") or 42) == seed
        ):
            return int(process_dir.name)
    return None


def _attempt_activity(
    job: Job,
    seed: int,
    started_at: float,
) -> tuple[float | None, Path | None]:
    if not job.run_root.is_dir():
        return None, None
    candidates: list[tuple[float, Path]] = []
    for config_path in job.run_root.glob("*/config.json"):
        try:
            if config_path.stat().st_mtime < started_at - 2.0:
                continue
            config = _load_json(config_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not _matches_config(config, job, seed):
            continue
        latest = config_path.stat().st_mtime
        for path in config_path.parent.rglob("*"):
            try:
                if path.is_file():
                    latest = max(latest, path.stat().st_mtime)
            except OSError:
                continue
        candidates.append((latest, config_path.parent))
    return max(candidates, key=lambda item: item[0]) if candidates else (None, None)


def _terminate_group(process: subprocess.Popen, grace: float) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(process.wait())
    try:
        return int(process.wait(timeout=grace))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(process.wait())


def _run_attempt(
    command: list[str],
    job: Job,
    seed: int,
    stall_timeout: float,
    poll_interval: float,
    terminate_grace: float,
) -> tuple[int, bool]:
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    started_at = time.time()
    last_activity = time.monotonic()
    last_mtime = 0.0
    run_dir: Path | None = None
    try:
        while process.poll() is None:
            latest, candidate = _attempt_activity(job, seed, started_at)
            if latest is not None and latest > last_mtime:
                last_mtime = latest
                last_activity = time.monotonic()
                run_dir = candidate
            idle = time.monotonic() - last_activity
            if idle >= stall_timeout:
                display = str(run_dir.relative_to(ROOT)) if run_dir else "run not created"
                print(
                    f"[watchdog] STALLED {job.label}: no file progress for "
                    f"{idle:.0f}s ({display}); terminating attempt.",
                    flush=True,
                )
                return _terminate_group(process, terminate_grace), True
            time.sleep(poll_interval)
        return int(process.returncode), False
    except BaseException:
        _terminate_group(process, terminate_grace)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--mask", choices=("all", "fixed", "random"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stall-timeout", type=float, default=900.0)
    parser.add_argument("--max-stall-retries", type=int, default=2)
    parser.add_argument("--watchdog-poll-interval", type=float, default=10.0)
    parser.add_argument("--terminate-grace", type=float, default=20.0)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stall_timeout <= 0 or args.watchdog_poll_interval <= 0:
        raise ValueError("watchdog intervals must be positive")
    if args.max_stall_retries < 0 or args.retry_delay < 0 or args.terminate_grace <= 0:
        raise ValueError("retry/grace settings are invalid")
    datasets = tuple(DATASETS.values()) if args.dataset == "all" else (DATASETS[args.dataset],)
    masks = ("fixed", "random") if args.mask == "all" else (args.mask,)
    jobs = [Job(dataset, mask, rate) for mask in masks for dataset in datasets for rate in RATES]

    skipped = 0
    executed = 0
    for index, job in enumerate(jobs, 1):
        prefix = f"[{index}/{len(jobs)}]"
        complete = None if args.force_rerun else find_completed_run(job, args.seed)
        if complete is not None:
            skipped += 1
            print(
                f"{prefix} SKIP {job.label}: epochs={complete.completed_epochs}, "
                f"test_mae={complete.test_mae:.6f}, "
                f"run={complete.run_dir.relative_to(ROOT)}",
                flush=True,
            )
            continue
        active_pid = find_active_launcher(job, args.seed)
        if active_pid is not None:
            message = f"{prefix} ACTIVE {job.label}: existing launcher pid={active_pid}"
            if args.dry_run:
                print(message, flush=True)
                continue
            raise RuntimeError(message)

        print(f"{prefix} RUN  {job.label}", flush=True)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "v15.1-single" / "train.py"),
            "--dataset", job.dataset.cli_name,
            "--mask", job.mask,
            "--rate", job.rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(args.seed),
        ]
        if args.dry_run:
            continue

        for attempt in range(1, args.max_stall_retries + 2):
            print(
                f"{prefix} ATTEMPT {attempt}/{args.max_stall_retries + 1} {job.label}",
                flush=True,
            )
            returncode, stalled = _run_attempt(
                command,
                job,
                args.seed,
                args.stall_timeout,
                args.watchdog_poll_interval,
                args.terminate_grace,
            )
            if not stalled:
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, command)
                break
            if attempt > args.max_stall_retries:
                raise RuntimeError(f"{job.label} stalled in all {attempt} attempts")
            print(f"{prefix} RETRY {job.label} after {args.retry_delay:.0f}s", flush=True)
            time.sleep(args.retry_delay)

        complete = find_completed_run(job, args.seed)
        if complete is None:
            raise RuntimeError(f"{job.label} exited without a complete formal test result")
        executed += 1
        print(f"{prefix} DONE {job.label}: test_mae={complete.test_mae:.6f}", flush=True)

    print(
        f"[summary] total={len(jobs)} skipped={skipped} executed={executed} "
        f"dry_run={args.dry_run}",
        flush=True,
    )


if __name__ == "__main__":
    main()

