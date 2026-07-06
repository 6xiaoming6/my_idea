#!/usr/bin/env python3
"""Shared full-data preparation and process execution helpers."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from baseline_logger import BaselineLogAdapter, _records


BENCH = Path(__file__).resolve().parents[1]
PROJECT = BENCH.parent
DATASETS = ("TaxiBJ", "BikeNYC", "CHAP")
MASKS = ("fixed", "random", "SR-TR", "SR-TC", "SC-TR", "SC-TC")
WINDOW_LENGTHS = {"TaxiBJ": 12, "BikeNYC": 12, "CHAP": 7}
GRID_NODES = {"TaxiBJ": 1024, "BikeNYC": 288, "CHAP": 1024}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".params", ".pypots", ".safetensors"}


def rate_label(rate: float) -> str:
    return format(rate, "g")


def window_length(args: argparse.Namespace) -> int:
    return WINDOW_LENGTHS[args.dataset]


def recommended_batch(model: str, args: argparse.Namespace) -> int:
    large = GRID_NODES[args.dataset] * (2 if args.channel == "all" and args.dataset != "CHAP" else 1) >= 1000
    profiles = {"ImputeFormer": (8, 32)}
    large_batch, small_batch = profiles[model]
    return large_batch if large else small_batch


def parse_args(model: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train {model} on one full adapted grid dataset.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--mask", default="fixed", choices=MASKS)
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--channel", default="0", help="all or a zero-based channel index")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--seed", type=int, default=42, help="Shared paper-comparison seed")
    parser.add_argument("--timeout", type=int, default=0, help="Per stage timeout; 0 disables it")
    parser.add_argument("--output-root", default="outputs", help="Root for standardized baseline logs")
    parser.add_argument(
        "--policy-json", default=str(BENCH / "policies" / "baseline_paper.json"),
        help="Unified JSON training policy used when generating native configs",
    )
    parser.add_argument("--no-prepare", action="store_true", help="Reuse full adapted data and generated config")
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Delete this run's checkpoint after evaluation (default: retain exactly one best checkpoint)",
    )
    parser.add_argument("--keep-checkpoints", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def config(args: argparse.Namespace, model: str, suffix: str = "conf") -> str:
    name = f"{model}_{args.mask}_{rate_label(args.rate)}_channel_{args.channel}_train.{suffix}"
    return str(BENCH / "training_configs" / args.dataset / name)


def adapted_data(args: argparse.Namespace, split: bool = False) -> Path:
    path = BENCH / "data" / "adapted" / args.dataset / f"{args.mask}_{rate_label(args.rate)}" / f"channel_{args.channel}"
    return path / "split" if split else path


def prepare(args: argparse.Namespace) -> None:
    if not 0 < args.rate < 1:
        raise ValueError("--rate must be between 0 and 1")
    subprocess.run([
        args.python, str(BENCH / "prepare_grid_dataset.py"), "--dataset", args.dataset,
        "--mask", args.mask, "--rate", rate_label(args.rate), "--channel", args.channel,
        "--legacy-stream", "--output-root", str(BENCH / "data" / "adapted"),
    ], cwd=PROJECT, check=True)
    subprocess.run([
        args.python, str(BENCH / "generate_train_configs.py"), "--dataset", args.dataset,
        "--mask", args.mask, "--rate", rate_label(args.rate), "--channel", args.channel,
        "--seed", str(args.seed), "--policy-json", args.policy_json,
    ], cwd=PROJECT, check=True)


def _checkpoint_roots(model: str, cwd: Path) -> tuple[Path, ...]:
    """Return only the native output trees in which a baseline writes weights."""
    roots = {
        "AGCRN": (cwd / "experiments",),
        "ASTGNN": (BENCH / "experiments",),
        "BRITS": (cwd / "experiments",),
        "CSDI": (cwd / "experiments",),
        "E2GAN": (cwd / "experiments",),
        "GAIN": (cwd / "experiments",),
        "GCASTN": (cwd / "experiments",),
        "IGNNK": (cwd / "experiments",),
        "ImputeFormer": (cwd / "logs",),
        "mTAN": (cwd / "experiments",),
        "PriSTI": (cwd / "save",),
        "SSTBAN": (cwd / "experiments",),
        "SAITS": (cwd / "experiments",),
        "GRIN": (cwd / "experiments",),
        "STCPA": (cwd / "experiments",),
        "STAMImputer": (cwd / "experiments",),
        "PAST": (cwd / "experiments",),
    }
    return roots.get(model, ())


def _is_checkpoint(path: Path) -> bool:
    name = path.name.lower()
    return (
        (path.suffix.lower() in CHECKPOINT_SUFFIXES and not name.startswith("events.out.tfevents"))
        or "best_params" in name
        or name.startswith("best_model")
        or name.startswith("tmp_model")
    )


def _checkpoint_snapshot(roots: tuple[Path, ...]) -> dict[Path, tuple[int, int, int]]:
    snapshot: dict[Path, tuple[int, int, int]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not _is_checkpoint(path):
                continue
            stat = path.stat()
            snapshot[path.resolve()] = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    return snapshot


def _changed_checkpoints(
    roots: tuple[Path, ...], before: dict[Path, tuple[int, int, int]],
    args: argparse.Namespace,
) -> list[Path]:
    after = _checkpoint_snapshot(roots)
    dataset = args.dataset.lower()
    mask = args.mask.lower()
    return [path for path, signature in after.items()
            if before.get(path) != signature
            and dataset in str(path).lower() and mask in str(path).lower()]


def _best_checkpoint(candidates: list[Path], raw_text: str) -> Path | None:
    if not candidates:
        return None
    loaded = re.findall(r"load weight from:\s*(.+)", raw_text, re.I)
    if loaded:
        printed = loaded[-1].strip().strip("'\"")
        for candidate in candidates:
            if str(candidate) == printed or candidate.name == Path(printed).name:
                return candidate
    preferred = ("best_model.pth", "best_model.params", "imputeformer.pypots", "model.pth")
    for name in preferred:
        matches = [path for path in candidates if path.name.lower() == name]
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime_ns)
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _apply_checkpoint_policy(
    roots: tuple[Path, ...], before: dict[Path, tuple[int, int, int]],
    raw_text: str, retain_best: bool, args: argparse.Namespace,
) -> tuple[Path | None, list[Path], list[str]]:
    """Keep at most the best checkpoint written by this run and delete the rest."""
    candidates = _changed_checkpoints(roots, before, args)
    best = _best_checkpoint(candidates, raw_text) if retain_best else None
    removed: list[Path] = []
    errors: list[str] = []
    for path in candidates:
        if best is not None and path == best:
            continue
        try:
            path.unlink()
            removed.append(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    return best, removed, errors


def _store_best_checkpoint(checkpoint: Path, run_dir: Path) -> Path:
    suffix = checkpoint.suffix if checkpoint.suffix else ".pth"
    destination = run_dir / "checkpoints" / f"best_model{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=False)
    checkpoint.replace(destination)
    return destination


def run_stages(model: str, args: argparse.Namespace, cwd: Path, stages: list[list[str]],
               setup: Callable[[argparse.Namespace, Path], None] | None = None) -> int:
    if args.timeout < 0:
        raise ValueError("--timeout cannot be negative")
    if not args.no_prepare:
        prepare(args)
    if setup:
        setup(args, cwd)
    checkpoint_roots = _checkpoint_roots(model, cwd)
    checkpoint_before = _checkpoint_snapshot(checkpoint_roots)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONHASHSEED"] = str(args.seed)
    env.setdefault("PYTHONUNBUFFERED", "1")
    logger = BaselineLogAdapter(PROJECT, model, args, stages)
    total_start = time.monotonic()
    status, returncode = "finished", 0
    for index, command in enumerate(stages, 1):
        heading = f"\n=== {model} full training stage {index}/{len(stages)} ===\n$ {' '.join(command)}\n"
        print(heading, end="", flush=True)
        logger.write_raw(heading)
        stage_start = time.monotonic()
        try:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                logger.write_raw(line)
                if args.timeout and time.monotonic() - stage_start > args.timeout:
                    process.kill()
                    process.wait()
                    message = f"TIMEOUT: stage exceeded {args.timeout}s\n"
                    print(message, end="", flush=True)
                    logger.write_raw(message)
                    status, returncode = "timeout", 124
                    break
            else:
                returncode = process.wait()
                if returncode:
                    status = "failed"
        except KeyboardInterrupt:
            status, returncode = "interrupted", 130
            if 'process' in locals() and process.poll() is None:
                process.terminate()
            break
        except Exception as exc:
            status, returncode = "failed", 1
            message = f"LOGGER/PROCESS ERROR: {type(exc).__name__}: {exc}\n"
            print(message, end="", flush=True)
            logger.write_raw(message)
            break
        if returncode:
            break
    logger.raw.flush()
    raw_text = (logger.log_dir / "raw.log").read_text(encoding="utf-8", errors="replace")
    _, parsed_test = _records(model, raw_text)
    if returncode == 0 and (parsed_test["mae"] is None or parsed_test["rmse"] is None):
        status, returncode = "missing_test_metrics", 2
        message = (
            "METRIC ERROR: training process finished but no original-scale test "
            "MAE/RMSE could be parsed; refusing to mark this comparison run as passed\n"
        )
        print(message, end="", flush=True)
        logger.write_raw(message)
    best_checkpoint, removed, cleanup_errors = _apply_checkpoint_policy(
        checkpoint_roots, checkpoint_before, raw_text,
        retain_best=not args.no_checkpoint, args=args,
    )
    if best_checkpoint is not None and not args.no_checkpoint:
        try:
            best_checkpoint = _store_best_checkpoint(best_checkpoint, logger.run_dir)
        except OSError as exc:
            cleanup_errors.append(f"could not store best checkpoint {best_checkpoint}: {exc}")
    if args.no_checkpoint:
        message = f"[info] checkpoint disabled; removed {len(removed)} parameter file(s) from this run\n"
        print(message, end="", flush=True)
        logger.write_raw(message)
    elif best_checkpoint is not None:
        message = (f"[info] retained best checkpoint: {best_checkpoint}; "
                   f"removed {len(removed)} superseded checkpoint(s)\n")
        print(message, end="", flush=True)
        logger.write_raw(message)
    else:
        message = f"[info] {model} produced no trainable checkpoint for this run\n"
        print(message, end="", flush=True)
        logger.write_raw(message)
    for error in cleanup_errors:
        message = f"[warning] could not remove superseded checkpoint: {error}\n"
        print(message, end="", flush=True)
        logger.write_raw(message)
    if cleanup_errors and returncode == 0:
        status, returncode = "checkpoint_cleanup_failed", 1
    args.best_checkpoint = str(best_checkpoint) if best_checkpoint is not None else None
    logger.finalize(status, returncode, time.monotonic() - total_start)
    return returncode


def finish(code: int) -> None:
    raise SystemExit(code)
