#!/usr/bin/env python3
"""Run six matched v24 arms, reusing verified completed three-arm suites."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as runner
from run_team_accept_v4 import check_gpu_idle

POLICIES = {
    "taxibj": ROOT / "configs/v24/coe_partner_residual4_taxibj_experiments.json",
    "bikenyc": ROOT / "configs/v24/coe_partner_residual4_bikenyc_experiments.json",
}
NATIVE_POLICIES = {
    "taxibj": ROOT / "configs/v24/coe_partner_native4_taxibj_experiments.json",
    "bikenyc": ROOT / "configs/v24/coe_partner_native4_bikenyc_experiments.json",
}


def _normalized_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("output_dir", None)
    result.pop("experiment_suite_fingerprint", None)
    return result


def _assert_matched_protocol(old_policy: Path, old_study: str,
                             native_policy: Path, native_study: str, epochs: int) -> None:
    old_plan, _, _, _ = runner.policy_plan(old_policy, old_study, epochs)
    native_plan, _, _, _ = runner.policy_plan(native_policy, native_study, epochs)
    if old_plan["datasets"] != native_plan["datasets"] or len(native_plan["runs"]) != 3:
        raise ValueError("Last three arms must use the same data and contain exactly three runs")
    def common(run: dict) -> dict:
        cfg = _normalized_config(run["config"])
        cfg["model"]["coe"].pop("expert_sharing", None)
        cfg["model"]["coe"].pop("pair_mode", None)
        cfg["model"]["coe"].pop("partner_fusion", None)
        cfg["model"]["coe"].pop("partner_aux_head_only", None)
        cfg["experiment_plan"].pop("stage", None)
        cfg["experiment_plan"].pop("variant", None)
        return cfg
    reference = common(native_plan["runs"][0])
    if any(common(run) != reference for run in old_plan["runs"] + native_plan["runs"]):
        raise ValueError("Comparison arms changed settings outside sharing and partner routing")


def _source_snapshot_intact(suite: Path) -> bool:
    snapshot = suite / "source_snapshot/manifest.json"
    if not snapshot.is_file():
        return False
    hashes = runner.load(snapshot)
    for name, expected in hashes.items():
        source = suite / "source_snapshot" / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            return False
    return True


def completed_suite(policy: Path, study: str, epochs: int, world_size: int):
    """Reuse a suite only when configs, receipts, data, and archived source match."""
    expected, output, _, _ = runner.policy_plan(policy, study, epochs)
    runner.validate_plan(expected)
    for plan_path in sorted(output.glob("*/plan.json"), reverse=True):
        suite = plan_path.parent
        try:
            plan = runner.load(plan_path)
            if (plan.get("stage") != expected["stage"] or
                    plan.get("world_size") != world_size or
                    plan.get("datasets") != expected["datasets"] or
                    len(plan.get("runs", [])) != len(expected["runs"]) or
                    not _source_snapshot_intact(suite)):
                continue
            matched = all(
                old["name"] == new["name"] and
                _normalized_config(old["config"]) == _normalized_config(new["config"])
                for old, new in zip(plan["runs"], expected["runs"])
            )
            if matched and all(runner.result_for(run, suite, plan) for run in plan["runs"]):
                return suite, plan
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None


def _command(policy: Path, study: str, device_flag: str, device_value: str,
             epochs: int, mode: str | None = None) -> list[str]:
    command = [sys.executable, "-u", str(ROOT / "scripts/v24/run_experiments.py"),
               "--config", str(policy), "--study", study, device_flag, device_value,
               "--epochs", str(epochs)]
    if mode:
        command.append(mode)
    return command


def _write_six_arm_summary(legacy, native, output: Path) -> Path:
    legacy_suite, legacy_plan = legacy
    native_suite, native_plan = native
    rows = []
    for suite, plan in ((legacy_suite, legacy_plan), (native_suite, native_plan)):
        for run in plan["runs"]:
            result = runner.result_for(run, suite, plan)
            rows.append({"variant": run["variant"], "seed": run["seed"],
                         "best_epoch": result["best_epoch"],
                         "best_val_mae": result["best_val_mae"],
                         "test_mae": result["test"]["mae"],
                         "test_rmse": result["test"]["rmse"],
                         "total_params": result.get("total_params"),
                         "total_time_sec": result.get("total_time_sec"),
                         "run_dir": result["run_dir"]})
    by_name = {row["variant"]: row for row in rows}
    native_row = by_name["residual4_shared_native"]
    comparisons = []
    for name in ("residual4_moe", "residual4_partner_legacy", "residual4_partner_residual",
                 "residual4_partner_fusion", "residual4_partner_fusion_headonly"):
        other = by_name[name]
        comparisons.append({"variant": name, "reference": "residual4_shared_native",
                            "test_mae_delta": other["test_mae"] - native_row["test_mae"],
                            "test_rmse_delta": other["test_rmse"] - native_row["test_rmse"]})
    for name, reference in (("residual4_partner_fusion", "residual4_partner_residual"),
                            ("residual4_partner_fusion_headonly", "residual4_partner_fusion")):
        comparisons.append({"variant": name, "reference": reference,
                            "test_mae_delta": by_name[name]["test_mae"] - by_name[reference]["test_mae"],
                            "test_rmse_delta": by_name[name]["test_rmse"] - by_name[reference]["test_rmse"]})
    path = output / f"six_arm_comparison_{native_suite.name}.json"
    runner.write_json(path, {"legacy_suite": str(legacy_suite), "native_suite": str(native_suite),
                             "rows": rows, "comparisons": comparisons,
                             "note": "Single-seed descriptive deltas; negative delta favors the listed variant."})
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=POLICIES, default="taxibj")
    parser.add_argument("--config", help="Custom first-three policy; overrides --dataset")
    parser.add_argument("--study", help="Study in the first-three policy")
    parser.add_argument("--native-config", help="Custom last-three-arm policy")
    parser.add_argument("--native-study", help="Study in the last-three-arm policy")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--gpu", help="One physical GPU for a single-process job")
    devices.add_argument("--gpus", help="Two physical GPUs for one DDP job (default: 0,1)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.gpu is not None:
        if not args.gpu.isdigit():
            parser.error("--gpu must be a nonnegative device index")
        selected = [int(args.gpu)]
        device_flag, device_value = "--gpu", args.gpu
    else:
        gpus = args.gpus or "0,1"
        parts = gpus.split(",")
        if len(parts) != 2 or len(set(parts)) != 2 or any(not item.isdigit() for item in parts):
            parser.error("--gpus must contain two distinct device indices, e.g. 0,1")
        selected = [int(item) for item in parts]
        device_flag, device_value = "--gpus", gpus
    policy = runner.resolve(args.config) if args.config else POLICIES[args.dataset]
    native_policy = runner.resolve(args.native_config) if args.native_config else NATIVE_POLICIES[args.dataset]
    study = args.study or runner.load(policy)["default_study"]
    native_study = args.native_study or runner.load(native_policy)["default_study"]
    output = runner.resolve(runner.load(policy)["output_dir"])
    native_output = runner.resolve(runner.load(native_policy)["output_dir"])
    if output != native_output:
        parser.error("Both policies must share the same output_dir for one six-arm queue")
    _assert_matched_protocol(policy, study, native_policy, native_study, args.epochs)
    legacy = completed_suite(policy, study, args.epochs, len(selected))
    if legacy:
        print(f"Skipping 3 verified completed runs: {legacy[0]}", flush=True)
    elif args.summary_only:
        raise FileNotFoundError("No matching completed first-three suite for this dataset, epochs and GPU count")
    old_command = _command(policy, study, device_flag, device_value, args.epochs)
    new_command = _command(native_policy, native_study, device_flag, device_value, args.epochs)
    if args.dry_run:
        if not legacy and subprocess.call([*old_command, "--dry-run"], cwd=ROOT):
            return 1
        return subprocess.call([*new_command, "--dry-run"], cwd=ROOT)
    if args.summary_only:
        native = completed_suite(native_policy, native_study, args.epochs, len(selected))
        if not native:
            raise FileNotFoundError("Last three results are not complete for this exact protocol")
        print(_write_six_arm_summary(legacy, native, output), flush=True)
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".ddp_queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("v24 six-arm comparison queue already running") from None
        check_gpu_idle(selected)
        if not legacy:
            code = subprocess.call(old_command, cwd=ROOT)
            if code:
                return code
            legacy = completed_suite(policy, study, args.epochs, len(selected))
            if not legacy:
                raise RuntimeError("First three jobs ended without complete verified receipts")
        code = subprocess.call(new_command, cwd=ROOT)
        if code:
            return code
        native = completed_suite(native_policy, native_study, args.epochs, len(selected))
        if not native:
            raise RuntimeError("Last three jobs ended without complete verified receipts")
        print(_write_six_arm_summary(legacy, native, output), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
