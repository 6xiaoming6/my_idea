#!/usr/bin/env python3
"""Unified batch runner for all real-dataset training experiments.

Examples:
  python scripts/run_experiments.py --dataset CHAP --gpu 0 --mask-pattern all --mask-rate all
  python scripts/run_experiments.py --dataset TaxiBJ --gpu 0 --mask-pattern fixed --mask-rate 0.4
  python scripts/run_experiments.py --dataset all --gpu 0 --mask-pattern all --mask-rate all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MASK_RATES = ("0.2", "0.4", "0.6", "0.8")
MASK_PATTERNS = ("fixed", "random")

ALL_CORE_ABLATIONS = (
    "ablation_fine_only",
    "ablation_no_router",
    "ablation_fixed_scale_experts",
    "ablation_no_cross_scale",
    "ablation_routed_only",
    "ablation_shared_only",
)
CHAP_ABLATIONS = (
    "ablation_fine_only",
    "ablation_routed_only",
    "ablation_shared_only",
)

ABLATION_DESCRIPTIONS = {
    "ablation_fine_only": "仅细尺度",
    "ablation_no_router": "无动态路由",
    "ablation_fixed_scale_experts": "固定尺度专家",
    "ablation_no_cross_scale": "无跨尺度共享",
    "ablation_routed_only": "仅路由分支",
    "ablation_shared_only": "仅共享分支",
}


@dataclass(frozen=True)
class DatasetSpec:
    config: str
    train_npz: str
    val_npz: str
    test_npz: str
    mask_root: str
    default_ablations: tuple[str, ...]


DATASETS = {
    "TaxiBJ": DatasetSpec(
        config="configs/datasets/taxibj.json",
        train_npz="data/TaxiBJ/taxibj_train.npz",
        val_npz="data/TaxiBJ/taxibj_val.npz",
        test_npz="data/TaxiBJ/taxibj_test.npz",
        mask_root="data/TaxiBJ",
        default_ablations=ALL_CORE_ABLATIONS,
    ),
    "BikeNYC": DatasetSpec(
        config="configs/datasets/bikenyc.json",
        train_npz="data/BikeNYC/bikenyc_train.npz",
        val_npz="data/BikeNYC/bikenyc_val.npz",
        test_npz="data/BikeNYC/bikenyc_test.npz",
        mask_root="data/BikeNYC",
        default_ablations=ALL_CORE_ABLATIONS,
    ),
    "CHAP": DatasetSpec(
        config="configs/datasets/chap_beijing.json",
        train_npz="data/CHAP/beijing/chap_beijing_train.npz",
        val_npz="data/CHAP/beijing/chap_beijing_val.npz",
        test_npz="data/CHAP/beijing/chap_beijing_test.npz",
        mask_root="data/CHAP/beijing",
        default_ablations=CHAP_ABLATIONS,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-model and ablation experiments across datasets, masks, and rates."
    )
    parser.add_argument("--dataset", "-d", choices=(*DATASETS, "all"), required=True)
    parser.add_argument("--gpu", "-g", default="0", help="Physical CUDA GPU id exposed as cuda:0.")
    parser.add_argument(
        "--mask-pattern",
        "--mask_pattern",
        "-m",
        dest="mask_pattern",
        choices=(*MASK_PATTERNS, "all"),
        default="random",
    )
    parser.add_argument(
        "--mask-rate",
        "--mask_rate",
        "-r",
        dest="mask_rate",
        choices=(*MASK_RATES, "all"),
        default="0.4",
    )
    parser.add_argument("--fixed-seed", "--fixed_seed", type=int, default=42)
    parser.add_argument("--skip-full", "--skip_full", action="store_true")
    parser.add_argument(
        "--skip-ablations",
        "--skip_ablations",
        default="",
        help="Comma-separated ablation names, for example ablation_no_router,ablation_shared_only.",
    )
    parser.add_argument(
        "--experiments",
        default="default",
        help="'default' or a comma-separated list: full,ablation_fine_only,...",
    )
    parser.add_argument("--conda-env", "--conda_env", default="difftdi")
    parser.add_argument(
        "--cpu-threads",
        "--cpu_threads",
        type=int,
        default=4,
        help="OMP/MKL/OpenBLAS thread limit for each child training process.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the plan only.")
    return parser.parse_args()


def _deep_update(base: dict, patch: dict) -> dict:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _expand(value: str, all_values: tuple[str, ...]) -> tuple[str, ...]:
    return all_values if value == "all" else (value,)


def _parse_names(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _experiments_for(spec: DatasetSpec, requested: str, skip_full: bool, skip_ablations: set[str]) -> tuple[str, ...]:
    if requested == "default":
        selected = ("full", *spec.default_ablations)
    else:
        selected = _parse_names(requested)
        if not selected:
            raise ValueError("--experiments must not be empty.")

    allowed = {"full", *spec.default_ablations}
    unsupported = sorted(set(selected) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported experiments for this dataset: {', '.join(unsupported)}")
    return tuple(
        name
        for name in selected
        if not (name == "full" and skip_full) and name not in skip_ablations
    )


def _child_env(gpu: str, cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(cpu_threads)
    return env


def _conda_python(env_name: str, *args: str) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", env_name, "python", *args]


def _run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("[run]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def _validate_files(spec: DatasetSpec) -> None:
    for relative_path in (spec.config, spec.train_npz, spec.val_npz, spec.test_npz):
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")


def _mask_override(spec: DatasetSpec, pattern: str, rate: str) -> dict:
    mask_dir = Path(spec.mask_root) / f"{pattern}_mask" / rate
    return {
        "data": {
            "mask": {
                "pattern": pattern,
                "missing_rate": float(rate),
                "train_csv": str(mask_dir / "train.csv"),
                "val_csv": str(mask_dir / "val.csv"),
                "test_csv": str(mask_dir / "test.csv"),
            }
        }
    }


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def _generate_masks(args: argparse.Namespace, spec: DatasetSpec, pattern: str, rate: str, env: dict[str, str]) -> None:
    output_dir = Path(spec.mask_root) / f"{pattern}_mask" / rate
    print(f"[info] generating {pattern} masks: {output_dir}")
    _run(
        _conda_python(
            args.conda_env,
            "scripts/generate_fixed_masks.py",
            "--train_npz",
            spec.train_npz,
            "--val_npz",
            spec.val_npz,
            "--test_npz",
            spec.test_npz,
            "--pattern",
            pattern,
            "--mask_rate",
            rate,
            "--seed",
            str(args.fixed_seed),
            "--output_dir",
            str(output_dir),
        ),
        env,
        args.dry_run,
    )


def _run_dataset_combo(
    args: argparse.Namespace,
    dataset_name: str,
    spec: DatasetSpec,
    pattern: str,
    rate: str,
    env: dict[str, str],
    temp_dir: Path,
    run_index: int,
    run_total: int,
) -> None:
    print("\n" + "#" * 72)
    print(f"[{run_index}/{run_total}] {dataset_name} | pattern={pattern} | rate={rate}")
    print("#" * 72)
    _generate_masks(args, spec, pattern, rate, env)

    mask_override = _mask_override(spec, pattern, rate)
    mask_override_path = temp_dir / f"mask_{dataset_name}_{pattern}_{rate}.json"
    _write_json(mask_override_path, mask_override)

    skip_ablations = set(_parse_names(args.skip_ablations))
    experiments = _experiments_for(spec, args.experiments, args.skip_full, skip_ablations)
    if not experiments:
        print("[info] no experiments selected; mask generation completed only.")
        return

    for idx, experiment in enumerate(experiments, 1):
        if experiment == "full":
            override_path = mask_override_path
            display_name = "Full Model"
        else:
            ablation_path = ROOT / "configs" / "ablations" / f"{experiment}.json"
            if not ablation_path.is_file():
                raise FileNotFoundError(f"Ablation config not found: {ablation_path}")
            ablation_override = json.loads(ablation_path.read_text(encoding="utf-8"))
            combined = _deep_update(mask_override, ablation_override)
            override_path = temp_dir / f"{dataset_name}_{pattern}_{rate}_{experiment}.json"
            _write_json(override_path, combined)
            display_name = ABLATION_DESCRIPTIONS.get(experiment, experiment)

        print(f"[{datetime.now():%H:%M:%S}] [{idx}/{len(experiments)}] {dataset_name} / {display_name}")
        _run(
            _conda_python(
                args.conda_env,
                "scripts/train.py",
                "-c",
                spec.config,
                "--override_config",
                str(override_path),
                "--train_npz",
                spec.train_npz,
                "--val_npz",
                spec.val_npz,
                "-n",
                "full" if experiment == "full" else experiment,
                "--no_plot",
                "--quiet",
            ),
            env,
            args.dry_run,
        )


def main() -> None:
    args = parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be at least 1.")

    dataset_names = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    patterns = _expand(args.mask_pattern, MASK_PATTERNS)
    rates = _expand(args.mask_rate, MASK_RATES)
    for name in dataset_names:
        _validate_files(DATASETS[name])

    combinations = [(name, pattern, rate) for rate in rates for pattern in patterns for name in dataset_names]
    print("=" * 72)
    print("Unified experiment runner")
    print(f"datasets: {', '.join(dataset_names)}")
    print(f"patterns: {', '.join(patterns)}")
    print(f"rates: {', '.join(rates)}")
    print(f"GPU: cuda:{args.gpu} | child CPU threads: {args.cpu_threads}")
    print(f"dataset/mask combinations: {len(combinations)}")
    print("=" * 72)

    env = _child_env(args.gpu, args.cpu_threads)
    if not args.dry_run:
        _run(
            _conda_python(args.conda_env, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA not available'"),
            env,
            dry_run=False,
        )

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="stmoe_experiments_") as directory:
        temp_dir = Path(directory)
        for index, (name, pattern, rate) in enumerate(combinations, 1):
            _run_dataset_combo(
                args,
                name,
                DATASETS[name],
                pattern,
                rate,
                env,
                temp_dir,
                index,
                len(combinations),
            )

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 72)
    print(f"All selected experiments finished in {elapsed / 60:.1f} minutes.")
    print("=" * 72)


if __name__ == "__main__":
    main()
