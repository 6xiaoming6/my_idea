#!/usr/bin/env python3
"""Run the V15 full matrix and automatically skip fully completed jobs."""

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
OUTPUT_ROOT = ROOT / "outputs" / "v15-single"
RATES = ("0.2", "0.4", "0.6", "0.8")


@dataclass(frozen=True)
class DatasetSpec:
    cli_name: str
    output_name: str
    base_config_path: str
    config_path: str

    @property
    def expected_epochs(self) -> int:
        config = _load_json(ROOT / self.config_path)
        return int(config["train"]["epochs"])


DATASETS = {
    "TaxiBJ": DatasetSpec(
        cli_name="TaxiBJ",
        output_name="TaxiBJ",
        base_config_path="configs/datasets/taxibj.json",
        config_path="configs/v15-single/taxibj.json",
    ),
    "BikeNYC": DatasetSpec(
        cli_name="BikeNYC",
        output_name="BikeNYC",
        base_config_path="configs/datasets/bikenyc.json",
        config_path="configs/v15-single/bikenyc.json",
    ),
    "CHAP": DatasetSpec(
        cli_name="CHAP",
        output_name="CHAP_Beijing",
        base_config_path="configs/datasets/chap_beijing.json",
        config_path="configs/v15-single/chap.json",
    ),
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


@dataclass(frozen=True)
class AttemptResult:
    returncode: int
    stalled: bool
    run_dir: Path | None


def _load_json(path: Path) -> dict:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return content


def _matches_config(config: dict, job: Job, seed: int) -> bool:
    data_cfg = config.get("data", {})
    mask_cfg = data_cfg.get("mask", {})
    model_cfg = config.get("model", {})
    try:
        rate_matches = math.isclose(
            float(mask_cfg.get("missing_rate")),
            float(job.rate),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    except (TypeError, ValueError):
        return False
    return (
        str(data_cfg.get("dataset_name")) == job.dataset.output_name
        and str(mask_cfg.get("pattern")) == job.mask
        and rate_matches
        and int(config.get("seed", -1)) == seed
        and str(model_cfg.get("architecture")) == "v15_compact_residual_moe"
        and str(model_cfg.get("version")) == "v15-single"
        and int(config.get("train", {}).get("epochs", -1))
        == job.dataset.expected_epochs
    )


def _read_metrics(metrics_path: Path) -> tuple[int, float | None]:
    max_epoch = 0
    test_mae: float | None = None
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if isinstance(record.get("epoch"), int):
                    max_epoch = max(max_epoch, int(record["epoch"]))
                if record.get("stage") == "test":
                    metrics = record.get("metrics", {})
                    value = float(metrics.get("mae"))
                    if math.isfinite(value):
                        test_mae = value
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0, None
    return max_epoch, test_mae


def _completion_for_run(run_dir: Path, job: Job, seed: int) -> Completion | None:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    test_log = run_dir / "logs" / "test.log"
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if not all(path.is_file() for path in (config_path, metrics_path, test_log, checkpoint)):
        return None
    if checkpoint.stat().st_size == 0 or test_log.stat().st_size == 0:
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
    if not job.run_root.is_dir():
        return None
    candidates: list[Completion] = []
    for config_path in job.run_root.glob("*/config.json"):
        completion = _completion_for_run(config_path.parent, job, seed)
        if completion is not None:
            candidates.append(completion)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.run_dir.stat().st_mtime)


def _latest_attempt_activity(
    job: Job,
    seed: int,
    started_at: float,
) -> tuple[float | None, Path | None]:
    """Return the latest file activity for a matching run created by this attempt."""
    if not job.run_root.is_dir():
        return None, None
    candidates: list[tuple[float, Path]] = []
    for config_path in job.run_root.glob("*/config.json"):
        try:
            # Allow a small timestamp tolerance for filesystems with coarse mtimes.
            if config_path.stat().st_mtime < started_at - 2.0:
                continue
            config = _load_json(config_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not _matches_config(config, job, seed):
            continue
        run_dir = config_path.parent
        latest_mtime = config_path.stat().st_mtime
        for path in run_dir.rglob("*"):
            try:
                if path.is_file():
                    latest_mtime = max(latest_mtime, path.stat().st_mtime)
            except OSError:
                continue
        candidates.append((latest_mtime, run_dir))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])


def _terminate_process_group(process: subprocess.Popen, grace_seconds: float) -> int:
    """Terminate a per-job process tree without touching the unified runner."""
    if process.poll() is not None:
        return int(process.returncode)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(process.wait())
    try:
        return int(process.wait(timeout=grace_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(process.wait())


def _run_with_watchdog(
    command: list[str],
    job: Job,
    seed: int,
    stall_timeout: float,
    poll_interval: float,
    terminate_grace: float,
) -> AttemptResult:
    """Run one job and stop only its process group if logs cease progressing."""
    started_at = time.time()
    last_activity_at = time.monotonic()
    last_observed_mtime = 0.0
    active_run_dir: Path | None = None
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                return AttemptResult(int(returncode), False, active_run_dir)

            latest_mtime, run_dir = _latest_attempt_activity(job, seed, started_at)
            if latest_mtime is not None and latest_mtime > last_observed_mtime:
                last_observed_mtime = latest_mtime
                last_activity_at = time.monotonic()
                active_run_dir = run_dir

            idle_seconds = time.monotonic() - last_activity_at
            if idle_seconds >= stall_timeout:
                relative = (
                    str(active_run_dir.relative_to(ROOT))
                    if active_run_dir is not None
                    else "run directory not created"
                )
                print(
                    f"[watchdog] STALLED {job.label}: no output-file progress for "
                    f"{idle_seconds:.0f}s ({relative}); terminating this attempt.",
                    flush=True,
                )
                returncode = _terminate_process_group(process, terminate_grace)
                return AttemptResult(returncode, True, active_run_dir)
            time.sleep(poll_interval)
    except BaseException:
        _terminate_process_group(process, terminate_grace)
        raise


def _argument_value(arguments: list[str], name: str) -> str | None:
    try:
        index = arguments.index(name)
    except ValueError:
        return None
    return arguments[index + 1] if index + 1 < len(arguments) else None


def _actual_train_matches(
    arguments: list[str],
    job: Job,
    seed: int,
) -> bool:
    base_config = _argument_value(arguments, "-c") or _argument_value(
        arguments, "--config"
    )
    override_path = _argument_value(arguments, "--override_config")
    if base_config is None or override_path is None:
        return False
    try:
        base_matches = Path(base_config).as_posix().endswith(
            Path(job.dataset.base_config_path).as_posix()
        )
        override = _load_json(Path(override_path))
        mask_cfg = override.get("data", {}).get("mask", {})
        model_cfg = override.get("model", {})
        rate_matches = math.isclose(
            float(mask_cfg.get("missing_rate")),
            float(job.rate),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        return (
            base_matches
            and str(mask_cfg.get("pattern")) == job.mask
            and rate_matches
            and int(override.get("seed", -1)) == seed
            and str(model_cfg.get("architecture")) == "v15_compact_residual_moe"
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def find_active_job(job: Job, seed: int) -> int | None:
    """Find an already-running V15 per-job launcher to prevent duplicate training."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
            continue
        try:
            raw = (process_dir / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        arguments = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        is_per_job_launcher = any(
            part.endswith("scripts/v15-single/train.py") for part in arguments
        )
        launcher_matches = is_per_job_launcher and (
            _argument_value(arguments, "--dataset") == job.dataset.cli_name
            and _argument_value(arguments, "--mask") == job.mask
            and _argument_value(arguments, "--rate") == job.rate
            and int(_argument_value(arguments, "--seed") or 42) == seed
        )
        is_actual_train = any(part.endswith("scripts/train.py") for part in arguments)
        if launcher_matches or (
            is_actual_train and _actual_train_matches(arguments, job, seed)
        ):
            return int(process_dir.name)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--mask", choices=("all", "fixed", "random"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=900.0,
        help="Kill and retry a job after this many seconds without log/checkpoint progress.",
    )
    parser.add_argument(
        "--max-stall-retries",
        type=int,
        default=2,
        help="Maximum fresh reruns after a watchdog-detected stall.",
    )
    parser.add_argument(
        "--watchdog-poll-interval",
        type=float,
        default=10.0,
        help="Seconds between output progress checks.",
    )
    parser.add_argument(
        "--terminate-grace",
        type=float,
        default=20.0,
        help="Seconds to wait after SIGTERM before force-killing a stalled job tree.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=30.0,
        help="Seconds to wait for CUDA resources to be released before a stalled retry.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun jobs even when a complete formal result already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stall_timeout <= 0:
        raise ValueError("--stall-timeout must be positive")
    if args.max_stall_retries < 0:
        raise ValueError("--max-stall-retries cannot be negative")
    if args.watchdog_poll_interval <= 0:
        raise ValueError("--watchdog-poll-interval must be positive")
    if args.terminate_grace <= 0:
        raise ValueError("--terminate-grace must be positive")
    if args.retry_delay < 0:
        raise ValueError("--retry-delay cannot be negative")
    datasets = tuple(DATASETS.values()) if args.dataset == "all" else (DATASETS[args.dataset],)
    masks = ("fixed", "random") if args.mask == "all" else (args.mask,)
    jobs = [Job(dataset, mask, rate) for mask in masks for dataset in datasets for rate in RATES]

    skipped = 0
    executed = 0
    for index, job in enumerate(jobs, 1):
        prefix = f"[{index}/{len(jobs)}]"
        completion = None if args.force_rerun else find_completed_run(job, args.seed)
        if completion is not None:
            skipped += 1
            relative = completion.run_dir.relative_to(ROOT)
            print(
                f"{prefix} SKIP {job.label}: complete "
                f"(epochs={completion.completed_epochs}, test_mae={completion.test_mae:.6f}, "
                f"run={relative})",
                flush=True,
            )
            continue

        active_pid = find_active_job(job, args.seed)
        if active_pid is not None:
            message = (
                f"{prefix} ACTIVE {job.label}: an existing launcher is still running "
                f"(pid={active_pid}). Stop that process before retrying to avoid duplicate GPU jobs."
            )
            if args.dry_run:
                print(message, flush=True)
                continue
            raise RuntimeError(message)

        print(f"{prefix} RUN  {job.label}", flush=True)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "v15-single" / "train.py"),
            "--dataset", job.dataset.cli_name,
            "--mask", job.mask,
            "--rate", job.rate,
            "--gpu", args.gpu,
            "--conda-env", args.conda_env,
            "--cpu-threads", str(args.cpu_threads),
            "--seed", str(args.seed),
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.dry_run:
            continue

        for attempt in range(1, args.max_stall_retries + 2):
            print(
                f"{prefix} ATTEMPT {attempt}/{args.max_stall_retries + 1} "
                f"{job.label}",
                flush=True,
            )
            result = _run_with_watchdog(
                command,
                job,
                args.seed,
                args.stall_timeout,
                args.watchdog_poll_interval,
                args.terminate_grace,
            )
            if not result.stalled:
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, command)
                break
            if attempt > args.max_stall_retries:
                raise RuntimeError(
                    f"{job.label} stalled in all {attempt} attempts; refusing an "
                    "unbounded retry loop. Incomplete run directories were preserved."
                )
            print(
                f"{prefix} RETRY {job.label} after {args.retry_delay:.0f}s; "
                "the incomplete run directory is preserved.",
                flush=True,
            )
            time.sleep(args.retry_delay)

        completion = find_completed_run(job, args.seed)
        if completion is None:
            raise RuntimeError(
                f"{job.label} returned successfully but no complete formal test result was found."
            )
        executed += 1
        print(
            f"{prefix} DONE {job.label}: test_mae={completion.test_mae:.6f}",
            flush=True,
        )

    print(
        f"[summary] total={len(jobs)} skipped={skipped} executed={executed} "
        f"dry_run={args.dry_run}",
        flush=True,
    )


if __name__ == "__main__":
    main()
