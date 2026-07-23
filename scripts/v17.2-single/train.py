#!/usr/bin/env python3
"""Train one formally audited V17.2 dataset/mask/rate experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline_common import (
    DATASETS,
    OUTPUT_ROOT,
    RATES,
    ROOT,
    build_protocol_audit,
    build_resolved_config,
    config_sha256,
    deep_update,
    load_json,
)


def _environment(gpu: str, cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = str(cpu_threads)
    return env


def _python(conda_env: str, *arguments: str) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        *arguments,
    ]


def _ensure_masks(
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    spec = DATASETS[args.dataset]
    mask_dir = ROOT / str(spec["mask_root"]) / f"{args.mask}_mask" / args.rate
    expected = tuple(mask_dir / f"{split}.csv" for split in ("train", "val", "test"))
    if args.regenerate_masks or not all(path.is_file() for path in expected):
        command = _python(
            args.conda_env,
            "scripts/generate_fixed_masks.py",
            "--train_npz",
            str(spec["train"]),
            "--val_npz",
            str(spec["val"]),
            "--test_npz",
            str(spec["test"]),
            "--pattern",
            args.mask,
            "--mask_rate",
            args.rate,
            "--seed",
            str(args.seed),
            "--output_dir",
            str(mask_dir.relative_to(ROOT)),
        )
        print("[mask]", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


def _apply_training_policy(
    configs: tuple[dict, dict, dict],
    policy_path: Path,
    dataset: str,
) -> tuple[dict, dict, dict]:
    policy = load_json(policy_path)
    datasets = policy.get("datasets")
    if not isinstance(datasets, dict) or dataset not in datasets:
        raise ValueError(f"Policy {policy_path} has no datasets.{dataset} entry")
    metadata = {
        "experiment_policy": {
            "name": policy.get("name", policy_path.stem),
            "source": str(policy_path),
        }
    }
    return tuple(
        deep_update(deep_update(config, datasets[dataset]), metadata)
        for config in configs
    )


def _git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return bool(result.stdout.strip())


def _new_run(
    before: set[Path],
    candidate_sha: str,
) -> Path:
    matches = []
    for config_path in set(OUTPUT_ROOT.rglob("config.json")) - before:
        try:
            if config_sha256(load_json(config_path)) == candidate_sha:
                matches.append(config_path.parent)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not matches:
        raise RuntimeError("Training finished but the new V17.2 run directory was not found")
    return max(matches, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--mask", choices=("fixed", "random"), required=True)
    parser.add_argument("--rate", choices=RATES, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-policy", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--regenerate-masks", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow an uncommitted tree for smoke/debug only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    spec = DATASETS[args.dataset]
    for key in ("base", "v17", "v17_2", "train", "val", "test"):
        path = ROOT / str(spec[key])
        if not path.is_file():
            raise FileNotFoundError(path)

    if (
        not args.dry_run
        and args.run_name == "full"
        and not args.allow_dirty
        and _git_is_dirty()
    ):
        raise RuntimeError(
            "Formal V17.2 training requires a clean Git tree. Commit/stash changes "
            "or use --allow-dirty only for non-paper debugging."
        )

    common = (args.dataset, args.mask, args.rate, args.seed)
    configs = (
        build_resolved_config(*common, version="full", epochs=args.epochs),
        build_resolved_config(*common, version="e1", epochs=args.epochs),
        build_resolved_config(*common, version="v17_2", epochs=args.epochs),
    )
    if args.training_policy:
        path = Path(args.training_policy).expanduser()
        policy_path = path if path.is_absolute() else ROOT / path
        configs = _apply_training_policy(configs, policy_path, args.dataset)
    full, e1, candidate = configs
    audit = build_protocol_audit(full, e1, candidate)
    if not audit["passed"]:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        raise RuntimeError("V17.2 protocol audit failed before training")

    env = _environment(args.gpu, args.cpu_threads)
    _ensure_masks(args, env)
    before = set(OUTPUT_ROOT.rglob("config.json")) if OUTPUT_ROOT.exists() else set()
    candidate_sha = config_sha256(candidate)

    with tempfile.TemporaryDirectory(prefix="v17_2_single_") as directory:
        override = Path(directory) / "override.json"
        override.write_text(
            json.dumps(candidate, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        command = _python(
            args.conda_env,
            "scripts/train.py",
            "-c",
            str(spec["base"]),
            "--override_config",
            str(override),
            "--train_npz",
            str(spec["train"]),
            "--val_npz",
            str(spec["val"]),
            "--test_npz",
            str(spec["test"]),
            "--name",
            args.run_name,
            "--no_plot",
            "--quiet",
        )
        print(
            f"[audit] passed candidate_sha256={candidate_sha}",
            flush=True,
        )
        print("[train]", " ".join(command), flush=True)
        if args.dry_run:
            return
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    run_dir = _new_run(before, candidate_sha)
    (run_dir / "protocol_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] {run_dir.relative_to(ROOT) / 'protocol_audit.json'}")


if __name__ == "__main__":
    main()
