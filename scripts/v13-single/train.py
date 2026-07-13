#!/usr/bin/env python3
"""Train one V13 dataset/mask/rate configuration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "TaxiBJ": {
        "base": "configs/datasets/taxibj.json",
        "model": "configs/v13-single/taxibj.json",
        "train": "data/TaxiBJ/taxibj_train.npz",
        "val": "data/TaxiBJ/taxibj_val.npz",
        "test": "data/TaxiBJ/taxibj_test.npz",
        "mask_root": "data/TaxiBJ",
    },
    "BikeNYC": {
        "base": "configs/datasets/bikenyc.json",
        "model": "configs/v13-single/bikenyc.json",
        "train": "data/BikeNYC/bikenyc_train.npz",
        "val": "data/BikeNYC/bikenyc_val.npz",
        "test": "data/BikeNYC/bikenyc_test.npz",
        "mask_root": "data/BikeNYC",
    },
    "CHAP": {
        "base": "configs/datasets/chap_beijing.json",
        "model": "configs/v13-single/chap.json",
        "train": "data/CHAP/beijing/chap_beijing_train.npz",
        "val": "data/CHAP/beijing/chap_beijing_val.npz",
        "test": "data/CHAP/beijing/chap_beijing_test.npz",
        "mask_root": "data/CHAP/beijing",
    },
}

ABLATIONS = {
    "main": None,
    "main_fallback": "configs/v13-single/ablations/main_fallback.json",
    "global_only": "configs/v13-single/ablations/global_only.json",
    "local_moe_only": "configs/v13-single/ablations/local_moe_only.json",
    "rank_8": "configs/v13-single/ablations/rank_8.json",
    "rank_32": "configs/v13-single/ablations/rank_32.json",
    "alpha_fixed_0.05": "configs/v13-single/ablations/alpha_fixed_0.05.json",
}


def deep_update(base: dict, patch: dict) -> dict:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_json(relative_path: str) -> dict:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def conda_python(env_name: str, *args: str) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", env_name, "python", *args]


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[run]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--mask", choices=("fixed", "random"), required=True)
    parser.add_argument("--rate", choices=("0.2", "0.4", "0.6", "0.8"), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--ablation", choices=ABLATIONS, default="main")
    parser.add_argument(
        "--training-policy",
        default="configs/policies/full_model_paper.json",
        help="Policy JSON containing a datasets object.",
    )
    parser.add_argument("--quick", choices=("1", "5"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--regenerate-masks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    spec = DATASETS[args.dataset]
    for key in ("base", "model", "train", "val", "test"):
        if not (ROOT / spec[key]).is_file():
            raise FileNotFoundError(ROOT / spec[key])

    policy_path = (
        f"configs/v13-single/policies/quick_{args.quick}epoch.json"
        if args.quick
        else args.training_policy
    )
    policy = load_json(policy_path)
    dataset_policies = policy.get("datasets", {})
    if args.dataset not in dataset_policies:
        raise ValueError(f"Policy {policy_path} has no {args.dataset} entry")

    override = dict(dataset_policies[args.dataset])
    override = deep_update(override, load_json(spec["model"]))
    ablation_path = ABLATIONS[args.ablation]
    if ablation_path:
        override = deep_update(override, load_json(ablation_path))
    mask_dir = Path(spec["mask_root"]) / f"{args.mask}_mask" / args.rate
    override = deep_update(override, {
        "data": {
            "mask": {
                "pattern": args.mask,
                "missing_rate": float(args.rate),
                "train_csv": str(mask_dir / "train.csv"),
                "val_csv": str(mask_dir / "val.csv"),
                "test_csv": str(mask_dir / "test.csv"),
            }
        },
        "experiment_policy": {"name": policy.get("name", Path(policy_path).stem)},
    })
    if args.epochs is not None:
        override = deep_update(override, {
            "train": {
                "epochs": args.epochs,
                "val_epoch": min(args.epochs, 1),
                "early_stopping": {"enabled": False},
            }
        })

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    mask_files = [ROOT / mask_dir / f"{split}.csv" for split in ("train", "val", "test")]
    if args.regenerate_masks or not all(path.is_file() for path in mask_files):
        run(
            conda_python(
                args.conda_env,
                "scripts/generate_fixed_masks.py",
                "--train_npz", spec["train"],
                "--val_npz", spec["val"],
                "--test_npz", spec["test"],
                "--pattern", args.mask,
                "--mask_rate", args.rate,
                "--seed", "42",
                "--output_dir", str(mask_dir),
            ),
            env,
            args.dry_run,
        )

    with tempfile.TemporaryDirectory(prefix="v13_single_") as directory:
        override_path = Path(directory) / "override.json"
        override_path.write_text(
            json.dumps(override, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        name = "full" if args.ablation == "main" else f"ablation_v13_{args.ablation}"
        run(
            conda_python(
                args.conda_env,
                "scripts/train.py",
                "-c", spec["base"],
                "--override_config", str(override_path),
                "--train_npz", spec["train"],
                "--val_npz", spec["val"],
                "--test_npz", spec["test"],
                "--name", name,
                "--no_plot",
                "--quiet",
            ),
            env,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
