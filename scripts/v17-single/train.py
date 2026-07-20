#!/usr/bin/env python3
"""Train one V17 dataset/mask/rate experiment in the isolated V17 output tree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RATES = ("0.2", "0.4", "0.6", "0.8")
DATASETS = {
    "TaxiBJ": {
        "base": "configs/datasets/taxibj.json",
        "model": "configs/v17-single/taxibj.json",
        "train": "data/TaxiBJ/taxibj_train.npz",
        "val": "data/TaxiBJ/taxibj_val.npz",
        "test": "data/TaxiBJ/taxibj_test.npz",
        "mask_root": "data/TaxiBJ",
    },
    "BikeNYC": {
        "base": "configs/datasets/bikenyc.json",
        "model": "configs/v17-single/bikenyc.json",
        "train": "data/BikeNYC/bikenyc_train.npz",
        "val": "data/BikeNYC/bikenyc_val.npz",
        "test": "data/BikeNYC/bikenyc_test.npz",
        "mask_root": "data/BikeNYC",
    },
    "CHAP": {
        "base": "configs/datasets/chap_beijing.json",
        "model": "configs/v17-single/chap.json",
        "train": "data/CHAP/beijing/chap_beijing_train.npz",
        "val": "data/CHAP/beijing/chap_beijing_val.npz",
        "test": "data/CHAP/beijing/chap_beijing_test.npz",
        "mask_root": "data/CHAP/beijing",
    },
}
ABLATIONS = {
    "no_scale_adapter": "no_scale_adapter.json",
    "decoupled_router": "decoupled_router.json",
    "progressive_fusion": "progressive_fusion.json",
    "global_route_gamma": "global_route_gamma.json",
}


def _deep_update(base: dict, patch: dict) -> dict:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else ROOT / path


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
    return ["conda", "run", "--no-capture-output", "-n", conda_env, "python", *arguments]


def _ensure_masks(
    args: argparse.Namespace,
    spec: dict[str, str],
    env: dict[str, str],
) -> Path:
    mask_dir = ROOT / spec["mask_root"] / f"{args.mask}_mask" / args.rate
    expected = tuple(mask_dir / f"{split}.csv" for split in ("train", "val", "test"))
    if args.regenerate_masks or not all(path.is_file() for path in expected):
        command = _python(
            args.conda_env,
            "scripts/generate_fixed_masks.py",
            "--train_npz", spec["train"],
            "--val_npz", spec["val"],
            "--test_npz", spec["test"],
            "--pattern", args.mask,
            "--mask_rate", args.rate,
            "--seed", str(args.seed),
            "--output_dir", str(mask_dir.relative_to(ROOT)),
        )
        print("[mask]", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    return mask_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--mask", choices=("fixed", "random"), required=True)
    parser.add_argument("--rate", choices=RATES, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", choices=("none", *ABLATIONS), default="none")
    parser.add_argument("--training-policy", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--regenerate-masks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    spec = DATASETS[args.dataset]
    for key in ("base", "model", "train", "val", "test"):
        path = ROOT / spec[key]
        if not path.is_file() and not (args.dry_run and key in {"train", "val", "test"}):
            raise FileNotFoundError(path)

    patch = _load(ROOT / spec["model"])
    mask_policies = patch.pop("train_policy_by_mask", {})
    mask_train_patch = mask_policies.get(args.mask, {})
    if mask_train_patch:
        patch["train"] = _deep_update(patch.get("train", {}), mask_train_patch)

    if args.training_policy:
        policy_path = _resolve(args.training_policy)
        policy = _load(policy_path)
        dataset_patches = policy.get("datasets")
        if not isinstance(dataset_patches, dict) or args.dataset not in dataset_patches:
            raise ValueError(f"Policy {policy_path} has no datasets.{args.dataset} entry")
        patch = _deep_update(patch, dataset_patches[args.dataset])
        patch = _deep_update(
            patch,
            {
                "experiment_policy": {
                    "name": policy.get("name", policy_path.stem),
                    "source": str(policy_path),
                }
            },
        )
    if args.epochs is not None:
        patch = _deep_update(
            patch,
            {
                "train": {
                    "epochs": args.epochs,
                    "val_epoch": min(args.epochs, 5),
                    "early_stopping": {"enabled": False},
                }
            },
        )
    if args.ablation != "none":
        ablation_path = ROOT / "configs" / "v17-single" / "ablations" / ABLATIONS[args.ablation]
        patch = _deep_update(patch, _load(ablation_path))

    env = _environment(args.gpu, args.cpu_threads)
    mask_dir = _ensure_masks(args, spec, env)
    patch = _deep_update(
        patch,
        {
            "seed": args.seed,
            "data": {
                "mask": {
                    "pattern": args.mask,
                    "missing_rate": float(args.rate),
                    "train_csv": str((mask_dir / "train.csv").relative_to(ROOT)),
                    "val_csv": str((mask_dir / "val.csv").relative_to(ROOT)),
                    "test_csv": str((mask_dir / "test.csv").relative_to(ROOT)),
                }
            },
        },
    )

    with tempfile.TemporaryDirectory(prefix="v17_single_") as directory:
        override = Path(directory) / "override.json"
        override.write_text(json.dumps(patch, indent=2, ensure_ascii=False), encoding="utf-8")
        run_name = (
            f"ablation_{args.ablation}"
            if args.ablation != "none" and args.run_name == "full"
            else args.run_name
        )
        command = _python(
            args.conda_env,
            "scripts/train.py",
            "-c", spec["base"],
            "--override_config", str(override),
            "--train_npz", spec["train"],
            "--val_npz", spec["val"],
            "--test_npz", spec["test"],
            "--name", run_name,
            "--no_plot",
            "--quiet",
        )
        print("[train]", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
