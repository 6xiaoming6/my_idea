#!/usr/bin/env python3
"""Run the final V14 single-variable model-hyperparameter exploration."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs/v14-exploration/hparams"
CONFIGS = {
    path.name.split("_", 1)[0]: path
    for path in sorted(CONFIG_ROOT.glob("H*.json"))
}

# Run low-risk candidates first.  The two largest-capacity candidates are kept
# at the end so an unexpected OOM cannot delay all conservative candidates.
CANDIDATE_ORDER = (
    "H01", "H02", "H04", "H06", "H07",
    "H08", "H09", "H10", "H11", "H12",
    "H13", "H14", "H15", "H05", "H03",
)
CORE6 = (
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("BikeNYC", "fixed", "0.6"),
    ("BikeNYC", "random", "0.8"),
    ("CHAP", "fixed", "0.2"),
    ("CHAP", "random", "0.4"),
)
ALL24 = tuple(
    (dataset, pattern, rate)
    for dataset in ("TaxiBJ", "BikeNYC", "CHAP")
    for pattern in ("fixed", "random")
    for rate in ("0.2", "0.4", "0.6", "0.8")
)
OUTPUT_NAMES = {
    "TaxiBJ": "TaxiBJ",
    "BikeNYC": "BikeNYC",
    "CHAP": "CHAP_Beijing",
}
FULL_EPOCHS = {"TaxiBJ": 160, "BikeNYC": 140, "CHAP": 150}
MEASURED_HOURS = {"TaxiBJ": 1.364, "BikeNYC": 0.099, "CHAP": 0.847}
RUNTIME_FACTORS = {
    "H01": 0.68,
    "H02": 1.45,
    "H03": 2.00,
    "H04": 0.85,
    "H05": 1.35,
    "H06": 0.98,
    "H07": 1.02,
    "H08": 0.99,
    "H09": 1.01,
    "H10": 1.03,
    "H11": 0.99,
    "H12": 1.01,
    "H13": 1.00,
    "H14": 1.00,
    "H15": 1.01,
}


def _test_completed(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    if "Testing finished:" not in content:
        return False
    values = []
    for metric in ("mae", "rmse"):
        match = re.search(rf"^  {metric}:\s+(\S+)\s*$", content, re.MULTILINE)
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
    expected_epochs: int,
) -> bool:
    experiment = json.loads(config_path.read_text(encoding="utf-8"))["experiment"]
    root = (
        ROOT / "outputs/v14-exploration" / OUTPUT_NAMES[dataset] / "ablation"
        / f"v14_exploration_{config_path.stem}" / pattern / f"rate{rate}"
    )
    for test_log in sorted(root.glob("*/logs/test.log"), reverse=True):
        if not _test_completed(test_log):
            continue
        run_dir = test_log.parent.parent
        required = (
            run_dir / "config.json",
            run_dir / "logs/metrics.jsonl",
            run_dir / "checkpoints/best.pt",
        )
        if not all(path.is_file() for path in required):
            continue
        cfg = json.loads(required[0].read_text(encoding="utf-8"))
        if (
            int(cfg.get("seed", -1)) == seed
            and int(cfg.get("train", {}).get("epochs", -1)) == expected_epochs
            and cfg.get("experiment") == experiment
        ):
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("screen", "multiseed", "all24"), default="screen"
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=CANDIDATE_ORDER,
        default=CANDIDATE_ORDER,
        help="Screen defaults to all 15 candidates; later phases require one.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 2026, 3407))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    parser.add_argument("--max-jobs", type=int, default=None, help="Debug-only pending-job limit.")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _jobs(args: argparse.Namespace) -> list[tuple[str, Path, str, str, str, int]]:
    requested = tuple(dict.fromkeys(args.candidates))
    candidates = tuple(candidate for candidate in CANDIDATE_ORDER if candidate in requested)
    if args.phase != "screen" and len(candidates) != 1:
        raise ValueError(
            f"--phase {args.phase} requires exactly one validation-selected candidate"
        )
    if args.phase == "screen":
        points, seeds = CORE6, (args.seed,)
    elif args.phase == "multiseed":
        points, seeds = CORE6, tuple(dict.fromkeys(args.seeds))
    else:
        points, seeds = ALL24, (args.seed,)
    return [
        (candidate, CONFIGS[candidate], dataset, pattern, rate, seed)
        for candidate in candidates
        for seed in seeds
        for dataset, pattern, rate in points
    ]


def main() -> None:
    args = parse_args()
    if set(CONFIGS) != set(CANDIDATE_ORDER):
        raise RuntimeError(
            f"Expected exactly {list(CANDIDATE_ORDER)}, found {sorted(CONFIGS)}"
        )
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.timeout_hours <= 0:
        raise ValueError("--timeout-hours must be positive")
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError("--max-jobs must be at least 1")

    jobs = _jobs(args)
    pending = []
    for job in jobs:
        candidate, path, dataset, pattern, rate, seed = job
        epochs = args.epochs or FULL_EPOCHS[dataset]
        if args.rerun_completed or not _completed(
            dataset, pattern, rate, path, seed, epochs
        ):
            pending.append(job)
    if args.max_jobs is not None:
        pending = pending[:args.max_jobs]

    estimated_hours = sum(
        MEASURED_HOURS[job[2]]
        * RUNTIME_FACTORS[job[0]]
        * (
            min(args.epochs, FULL_EPOCHS[job[2]]) / FULL_EPOCHS[job[2]]
            if args.epochs is not None else 1.0
        )
        for job in pending
    )
    estimate_text = (
        f"{estimated_hours * 60.0:.0f} min"
        if estimated_hours < 1.0 else
        f"{estimated_hours:.1f} h ({estimated_hours / 24.0:.2f} days)"
    )
    print(
        f"V14-HPARAM {args.phase}: {len(pending)}/{len(jobs)} pending; "
        f"estimated single-GPU time {estimate_text}",
        flush=True,
    )
    pending_keys = {
        (candidate, dataset, pattern, rate, seed)
        for candidate, _, dataset, pattern, rate, seed in pending
    }
    failures: list[str] = []
    for index, (candidate, path, dataset, pattern, rate, seed) in enumerate(
        jobs, start=1
    ):
        key = (candidate, dataset, pattern, rate, seed)
        label = f"{candidate} {dataset} {pattern}@{rate} seed={seed}"
        if key not in pending_keys:
            if args.max_jobs is None:
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
            "--seed", str(seed),
            "--experiment-config", str(path.relative_to(ROOT)),
        ]
        if args.epochs is not None:
            command.extend(("--epochs", str(args.epochs)))
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] RUN {label}", flush=True)
        print(" ".join(command), flush=True)
        try:
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                timeout=args.timeout_hours * 3600.0,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            failures.append(f"{label}: {type(error).__name__}: {error}")
            print(f"[{index}/{len(jobs)}] FAILED {failures[-1]}", flush=True)
            if args.stop_on_error:
                raise

    if failures:
        print("\nCompleted with failed jobs:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        raise SystemExit(1)
    print("V14-HPARAM requested jobs completed.", flush=True)


if __name__ == "__main__":
    main()
