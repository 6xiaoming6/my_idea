#!/usr/bin/env python3
"""Build an explicit protocol-checked Full V17 baseline manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from pipeline_common import (
    CORE_POINTS,
    DATASETS,
    RATES,
    ROOT,
    Job,
    baseline_key,
    build_resolved_config,
    config_sha256,
    load_json,
    protocol_differences,
    read_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument(
        "--output",
        default="configs/v17.2-single/baseline_manifest.json",
    )
    return parser.parse_args()


def _candidate_roots(job: Job) -> list[Path]:
    dataset = job.output_dataset
    rate = f"rate{float(job.rate):g}"
    return [
        ROOT
        / "outputs/v17.1-single"
        / dataset
        / "ablation/full"
        / job.mask
        / rate,
        ROOT
        / "outputs/v17-single"
        / dataset
        / "full/model"
        / job.mask
        / rate,
    ]


def _compatible_run(job: Job) -> tuple[Path, dict, dict] | None:
    expected = build_resolved_config(
        job.dataset,
        job.mask,
        job.rate,
        job.seed,
        version="full",
    )
    matches = []
    for root in _candidate_roots(job):
        if not root.is_dir():
            continue
        for config_path in root.glob("*/config.json"):
            run_dir = config_path.parent
            try:
                config = load_json(config_path)
                mask = config.get("data", {}).get("mask", {})
                if (
                    config.get("model", {}).get("architecture")
                    != "v17_hierarchical_scale_moe"
                    or config.get("model", {})
                    .get("v17", {})
                    .get("adapter_enabled")
                    is not True
                    or int(config.get("seed", -1)) != job.seed
                    or mask.get("pattern") != job.mask
                    or abs(float(mask.get("missing_rate", -1)) - float(job.rate))
                    > 1e-9
                ):
                    continue
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            differences = protocol_differences(expected, config)
            unexpected = [item for item in differences if not item["allowed"]]
            result = read_result(run_dir)
            if unexpected or result is None:
                continue
            matches.append((run_dir, config, result))
    return max(matches, key=lambda item: item[0].stat().st_mtime) if matches else None


def _planned_jobs(seeds: list[int]) -> list[Job]:
    jobs = [
        Job(dataset, mask, rate, 42)
        for dataset in DATASETS
        for mask in ("fixed", "random")
        for rate in RATES
    ]
    for seed in seeds:
        if seed == 42:
            continue
        jobs.extend(
            Job(dataset, mask, rate, seed)
            for dataset, mask, rate in CORE_POINTS.values()
        )
    return list(dict.fromkeys(jobs))


def main() -> None:
    args = parse_args()
    entries = {}
    missing = []
    for job in _planned_jobs(args.seeds):
        match = _compatible_run(job)
        if match is None:
            missing.append(baseline_key(job))
            continue
        run_dir, config, result = match
        entries[baseline_key(job)] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "config_sha256": config_sha256(config),
            "test_mae": result["mae"],
            "test_rmse": result["rmse"],
            "test_loss": result["loss"],
            "protocol_audit_passed": True,
        }

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "description": (
            "Explicit Full V17/V17.1 baselines compatible with the V17.2 "
            "training protocol."
        ),
        "entries": entries,
        "missing": missing,
    }
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        display_path = output.relative_to(ROOT)
    except ValueError:
        display_path = output
    print(f"[saved] {display_path} entries={len(entries)} missing={len(missing)}")
    if missing:
        print("[missing]", ", ".join(missing))
        raise SystemExit(
            "Baseline manifest is incomplete; incompatible/missing Full runs must be rerun"
        )


if __name__ == "__main__":
    main()
