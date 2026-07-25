#!/usr/bin/env python3
"""Run the three real-data two-epoch V18 smoke experiments."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys

from pipeline_common import (
    DATASETS,
    OUTPUT_ROOT,
    ROOT,
    load_json,
    read_records,
)


SMOKE_POINTS = (
    ("TaxiBJ", "fixed", "0.4"),
    ("BikeNYC", "random", "0.4"),
    ("CHAP", "fixed", "0.4"),
)


def _validate_run(
    dataset: str,
    pattern: str,
    rate: str,
    run_name: str,
) -> None:
    root = (
        OUTPUT_ROOT
        / DATASETS[dataset]["output_name"]
        / "debug"
        / run_name
        / pattern
        / f"rate{float(rate):g}"
    )
    configs = list(root.glob("*/config.json"))
    if not configs:
        raise RuntimeError(f"No V18 smoke output found under {root}")
    run_dir = max(configs, key=lambda path: path.stat().st_mtime).parent
    config = load_json(run_dir / "config.json")
    records = read_records(run_dir / "logs" / "metrics.jsonl")
    epochs = [record for record in records if "epoch" in record]
    test = next(
        (
            record
            for record in reversed(records)
            if record.get("stage") == "test"
        ),
        None,
    )
    if len(epochs) != 2 or test is None:
        raise RuntimeError(
            f"Smoke must contain 2 train/val epochs and one test: {run_dir}"
        )
    if any(record.get("val") is None for record in epochs):
        raise RuntimeError(f"Smoke validation is incomplete: {run_dir}")

    for record in records:
        containers = [
            record.get("train", {}),
            record.get("val", {}),
            record.get("metrics", {}),
        ]
        for container in containers:
            for key, value in container.items():
                if isinstance(value, (int, float)) and not math.isfinite(
                    float(value)
                ):
                    raise RuntimeError(
                        f"Non-finite smoke metric {key}={value}: {run_dir}"
                    )

    metrics = test["metrics"]
    required = (
        "v18_base_hidden_mae",
        "v18_probe_hidden_mae",
        "v18_final_hidden_mae",
        "v18_rho_c_min",
        "v18_rho_c_max",
        "v18_rho_m_min",
        "v18_rho_m_max",
        "v18_rho_f_min",
        "v18_rho_f_max",
        "v18_residual_bound_violation_rate",
        "v18_residual_bound_max_ratio",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(f"Missing V18 smoke diagnostics {missing}: {run_dir}")
    if metrics["v18_residual_bound_violation_rate"] != 0.0:
        raise RuntimeError(f"Residual bound violation detected: {run_dir}")
    if metrics["v18_residual_bound_max_ratio"] > 1.0 + 1e-5:
        raise RuntimeError(f"Residual hard bound exceeded: {run_dir}")
    maxima = {
        "c": config["model"]["v18"]["rho_coarse_max"],
        "m": config["model"]["v18"]["rho_mid_max"],
        "f": config["model"]["v18"]["rho_fine_max"],
    }
    for suffix, maximum in maxima.items():
        if metrics[f"v18_rho_{suffix}_min"] < 0.0:
            raise RuntimeError(f"Negative rho_{suffix}: {run_dir}")
        if metrics[f"v18_rho_{suffix}_max"] > maximum + 1e-6:
            raise RuntimeError(f"rho_{suffix} exceeds configured maximum: {run_dir}")

    checkpoints = list((run_dir / "checkpoints").glob("*.pt"))
    if [path.name for path in checkpoints] != ["best.pt"]:
        raise RuntimeError(
            f"Smoke must keep exactly one best.pt checkpoint: {run_dir}"
        )
    print(f"[verified] {run_dir.relative_to(ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="difftdi")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for index, (dataset, pattern, rate) in enumerate(SMOKE_POINTS, start=1):
        run_name = f"debug_v18_smoke_{dataset.lower()}"
        command = [
            sys.executable,
            "scripts/v18-single/train.py",
            "--dataset",
            dataset,
            "--mask",
            pattern,
            "--rate",
            rate,
            "--gpu",
            args.gpu,
            "--conda-env",
            args.conda_env,
            "--cpu-threads",
            str(args.cpu_threads),
            "--seed",
            str(args.seed),
            "--epochs",
            "2",
            "--smoke-config",
            "--run-name",
            run_name,
            "--allow-dirty",
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(
            f"[{index}/{len(SMOKE_POINTS)}] RUN {dataset} {pattern}@{rate}",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, check=True)
        if not args.dry_run:
            _validate_run(dataset, pattern, rate, run_name)
    print("[done] V18 real-data smoke completed.", flush=True)


if __name__ == "__main__":
    main()
