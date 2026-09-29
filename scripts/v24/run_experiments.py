#!/usr/bin/env python3
"""V24 experiments using the v23 lifecycle: paired jobs, fingerprints and restart.

Only complete, verified jobs are skipped. Interrupted jobs restart from epoch 1.
No formal training occurs with --dry-run or --summary-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_experiment_plan as planner


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def signature(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino


def source_files():
    return sorted(set((ROOT / "src/stmoe_imputer").rglob("*.py")) |
                  set((ROOT / "scripts/v24").glob("*.py")) | {ROOT / "scripts/train.py"})


def plan_inputs(manifest):
    paths = {resolve(item["path"]) for item in manifest["datasets"].values()}
    for run in manifest["runs"]:
        cfg = run["config"]
        if not manifest["datasets"]:
            continue
        mask = cfg["data"]["mask"]
        pattern = mask.get("pattern", "random")
        for split in ("train", "val", "test"):
            diversity_key = 'train_mask_diversity' if split == 'train' else 'eval_mask_diversity'
            if cfg['data'].get(diversity_key) is not None:
                continue  # Generator source and full config are already fingerprinted.
            value = mask.get(f"{split}_csv") or mask.get(f"{pattern}_{split}_csv")
            if value is None:
                raise ValueError(f"Missing {split} mask CSV")
            paths.add(resolve(value))
        metadata = cfg.get("experiment_plan", {}).get("mask_metadata")
        if metadata:
            paths.add(resolve(metadata))
    return paths


def identity(manifest, extra_files=()):
    files = set(source_files()) | plan_inputs(manifest) | {resolve(p) for p in extra_files}
    hashes, stamps = {}, {}
    for path in sorted(files):
        before = signature(path)
        hashes[str(path)] = planner._sha256(path)
        after = signature(path)
        stamps[str(path)] = (after, hashes[str(path)])
        if after != before:
            raise RuntimeError(f"Input changed while hashing: {path}")
    jobs = []
    for run in manifest["runs"]:
        cfg = copy.deepcopy(run["config"])
        cfg.pop("output_dir", None)
        jobs.append({"name": run["name"], "variant": run["variant"],
                     "protocol": run["protocol"], "config": cfg})
    versions = {}
    for package in ("torch", "numpy", "tqdm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return {"schema_version": 1, "jobs": jobs, "datasets": manifest["datasets"],
            "files_sha256": hashes, "python": sys.version, "executable": sys.executable,
            "packages": versions}, stamps


def snapshot_partner_sources(suite, extra_files):
    """Freeze executable code and small policy inputs, including dirty edits."""
    snapshot = suite / "source_snapshot"
    paths = set(source_files())
    for candidate in extra_files:
        path = resolve(candidate)
        if path.is_relative_to(ROOT) and path.suffix in {".py", ".json"} and not path.is_relative_to(ROOT / "data"):
            paths.add(path)
    manifest = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        content = path.read_bytes()
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError(f"Frozen source differs from active file: {path}")
        if not target.exists():
            target.write_bytes(content)
        manifest[str(relative)] = hashlib.sha256(content).hexdigest()
    write_json(snapshot / "manifest.json", manifest)


def assert_unchanged(stamps, code_paths):
    if set(source_files()) != code_paths:
        raise RuntimeError("Training source files were added or removed; restart in a new fingerprinted suite.")
    for path, expected in stamps.items():
        if (not Path(path).is_file() or signature(path) != expected[0]
                or planner._sha256(Path(path)) != expected[1]):
            raise RuntimeError(f"Code/data/config changed during the experiment: {path}. Restart to create a new suite.")


def policy_plan(path, study, epochs=None, batch_size=None):
    policy = load(path)
    if epochs is not None:
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("--epochs must be a positive integer")
        policy["training"]["epochs"] = epochs
        policy["training"].setdefault("scheduler", {})["total_epochs"] = epochs
    if batch_size is not None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("--batch-size must be a positive integer")
        policy["batch_size"] = batch_size
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported experiment policy schema")
    spec = policy["studies"][study]
    threads = policy.get("cpu_threads", 2)
    if type(threads) is not int or threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    args = ["--base-config", str(resolve(spec["base_config"])), "--output-dir",
            str(resolve(policy["output_dir"]) / study / "pending"),
            "--stage", spec["stage"], "--seeds", *map(str, policy["seeds"])]
    if spec.get("synthetic", False):
        args.append("--synthetic")
    else:
        for split in ("train", "val", "test"):
            args.extend([f"--{split}-npz", str(resolve(spec["sources"][split]))])
    for key in ("mask_root", "cost_profile"):
        if spec.get(key):
            args.extend(["--" + key.replace("_", "-"), str(resolve(spec[key]))])
    if spec.get("budget_regime"):
        args.extend(["--budget-regime", spec["budget_regime"]])
    for key in ("patterns", "rates", "variants"):
        if spec.get(key):
            args.extend(["--" + key, *map(str, spec[key])])
    patch = {"train": policy["training"], "data": {"drop_last": False}}
    if "batch_size" in policy:
        patch["data"]["batch_size"] = policy["batch_size"]
    manifest = planner.build_plan(planner.parse_args(args), base_patch=patch)
    extras = [path, resolve(spec["base_config"])]
    extras.extend(resolve(p) for p in policy.get("extra_inputs", []))
    extras.extend((ROOT / "configs/v24/experiments").glob("*.json"))
    if spec.get("cost_profile"):
        extras.append(resolve(spec["cost_profile"]))
    return manifest, resolve(policy["output_dir"]) / study, threads, extras


def validate_plan(manifest):
    if not manifest.get("runs") or manifest.get("schema_version") != 1:
        raise ValueError("Expected a nonempty v24 plan")
    for protocol in manifest.get("protocols", []):
        if protocol.get("kind") != "structured_csv":
            continue
        metadata = load(protocol["metadata_file"])
        if metadata != protocol["metadata"]:
            raise ValueError("Structured-mask metadata changed; regenerate the plan")
        splits = metadata.get("splits", metadata.get("split"))
        for split in manifest["datasets"]:
            path = resolve(protocol["mask_patch"][f"{split}_csv"])
            if planner._sha256(path) != splits[split]["sha256"]:
                raise ValueError("Structured mask content differs from its metadata")
    names = set()
    for run in manifest["runs"]:
        name = run["name"]
        if Path(name).name != name or name in names or name in (".", ".."):
            raise ValueError("Experiment names must be unique path components")
        names.add(name)
        cfg = run["config"]
        train = cfg["train"]
        if manifest.get("stage") == "coe_direct_baselines":
            coe = cfg["model"]["coe"]
            data = cfg["data"]
            expected = {
                "direct_moe_s1_top8": (1, 8, "per_step"),
                "direct_moe_s4_top2": (4, 2, "per_step"),
                "direct_shared_s4_top2": (4, 2, "shared"),
            }[run["variant"]]
            actual = tuple(coe.get(key) for key in ("num_steps", "top_k", "expert_sharing"))
            if (actual != expected or
                    coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST", "TL", "SL"] or
                    coe.get("fixed_expert_steps") != [None] * expected[0] or
                    coe.get("routing_mode") != "hard" or
                    coe.get("state_update_mode") != "direct" or
                    coe.get("use_shared") is not False or coe.get("use_routed") is not True or
                    coe.get("completion_feedback") is not False or
                    coe.get("pair_mode") != "native" or coe.get("fusion_mode") != "original" or
                    coe.get("acceptance") != "none" or
                    coe.get("router_state") != "dynamic" or coe.get("expert_state") != "dynamic" or
                    cfg["model"]["main"].get("dim") != 64 or
                    cfg["model"]["main"].get("use_multiscale") is not False or
                    cfg["loss"].get("balance_importance") != "candidate" or
                    cfg["loss"].get("lambda_coe_balance") != 0.01 or
                    data.get("mask", {}).get("pattern") != "random" or
                    data.get("mask", {}).get("missing_rate") != 0.4 or
                    data.get("train_mask_diversity", {}).get("rates") != [0.4] or
                    data.get("eval_mask_diversity", {}).get("rates") != [0.4] or
                    len(data.get("train_mask_diversity", {}).get("families", [])) != 9 or
                    data.get("train_mask_diversity", {}).get("families") !=
                    data.get("eval_mask_diversity", {}).get("families") or
                    data.get("train_mask_diversity", {}).get("resample_each_epoch") is not True or
                    data.get("eval_mask_diversity", {}).get("resample_each_epoch") is not False or
                    train.get("val_epoch") != 5 or
                    train.get("save_best_checkpoint") is not True or
                    train.get("best_checkpoint_name") != "best.pth" or
                    train.get("scheduler", {}).get("total_epochs") != train["epochs"]):
                raise ValueError(f"{name}: invalid direct MoE baseline settings")
        if manifest.get("stage") == "coe_depth_pair8":
            coe = cfg["model"]["coe"]
            data = cfg["data"]
            expected = {
                "depthpair_moe_s1_top8": (1, 8, "per_step", "native"),
                "depthpair_moe_s2_top4": (2, 4, "per_step", "native"),
                "depthpair_coe_s4_pair28": (4, 2, "shared", "interaction"),
            }[run["variant"]]
            actual = tuple(coe.get(key) for key in
                           ("num_steps", "top_k", "expert_sharing", "pair_mode"))
            if (actual != expected or
                    coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST", "TL", "SL"] or
                    coe.get("routing_mode") != "hard" or
                    coe.get("fixed_expert_steps") != [None] * expected[0] or
                    coe.get("completion_feedback") is not False or
                    coe.get("acceptance") != "none" or
                    cfg["model"]["main"].get("dim") != 64 or
                    cfg["model"]["main"].get("use_multiscale") is not False or
                    cfg["loss"].get("balance_importance") != "candidate" or
                    cfg["loss"].get("lambda_coe_balance") != 0.01 or
                    data.get("mask", {}).get("pattern") != "random" or
                    data.get("mask", {}).get("missing_rate") != 0.4 or
                    data.get("train_mask_diversity", {}).get("rates") != [0.4] or
                    data.get("eval_mask_diversity", {}).get("rates") != [0.4] or
                    len(data.get("train_mask_diversity", {}).get("families", [])) != 9 or
                    data.get("train_mask_diversity", {}).get("families") !=
                    data.get("eval_mask_diversity", {}).get("families") or
                    data.get("train_mask_diversity", {}).get("resample_each_epoch") is not True or
                    data.get("eval_mask_diversity", {}).get("resample_each_epoch") is not False or
                    train.get("partner_probe") is not None or
                    train.get("save_best_checkpoint") is not False or
                    train.get("scheduler", {}).get("total_epochs") != train["epochs"]):
                raise ValueError(f"{name}: invalid one/two/four-round eight-expert depth comparison")
        if manifest.get("stage") == "coe_fusion1":
            coe = cfg["model"]["coe"]
            data = cfg["data"]
            expected = {
                "pair28_warm": ("interaction", "original", 2),
                "top2_context": ("native", "context", 0),
                "top2_response": ("native", "response", 0),
                "top2_local": ("native", "local_response", 0),
                "top2_proposal": ("native", "proposal", 0),
            }[run["variant"]]
            actual = tuple(coe.get(key, "original" if key == "fusion_mode" else 0)
                           for key in ("pair_mode", "fusion_mode", "pair_dense_warmup_steps"))
            if (actual != expected or coe.get("num_steps") != 4 or coe.get("top_k") != 2 or
                    coe.get("expert_sharing") != "shared" or
                    coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST", "TL", "SL"] or
                    coe.get("routing_mode") != "hard" or
                    coe.get("fixed_expert_steps") != [None] * 4 or
                    coe.get("routing_warmup_epochs") != 3 or
                    coe.get("routing_transition_epochs") != 3 or
                    coe.get("completion_feedback") is not False or
                    coe.get("acceptance") != "none" or
                    cfg["model"]["main"].get("dim") != 64 or
                    cfg["model"]["main"].get("use_multiscale") is not False or
                    cfg["loss"].get("balance_importance") != "candidate" or
                    cfg["loss"].get("lambda_coe_balance") != 0.01 or
                    data.get("mask", {}).get("pattern") != "random" or
                    data.get("mask", {}).get("missing_rate") != 0.4 or
                    data.get("train_mask_diversity", {}).get("rates") != [0.4] or
                    data.get("eval_mask_diversity", {}).get("rates") != [0.4] or
                    len(data.get("train_mask_diversity", {}).get("families", [])) != 9 or
                    data.get("train_mask_diversity", {}).get("families") !=
                    data.get("eval_mask_diversity", {}).get("families") or
                    data.get("train_mask_diversity", {}).get("resample_each_epoch") is not True or
                    data.get("eval_mask_diversity", {}).get("resample_each_epoch") is not False or
                    train.get("partner_probe") is not None or
                    train.get("val_epoch") != 5 or
                    train.get("save_best_checkpoint") is not True or
                    train.get("best_checkpoint_name") != "best.pth" or
                    train.get("scheduler", {}).get("total_epochs") != train["epochs"]):
                raise ValueError(f"{name}: invalid fusion exploration settings")
        if manifest.get("stage") == "coe_main_s4_e8":
            coe = cfg["model"]["coe"]
            if (run["variant"] != "main_s4_e8_top2" or
                    coe.get("num_steps") != 4 or
                    coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST", "TL", "SL"] or
                    coe.get("top_k") != 2 or coe.get("routing_mode") != "hard" or
                    coe.get("expert_sharing") != "shared" or
                    coe.get("pair_mode") != "native" or
                    coe.get("completion_feedback") is not False or
                    coe.get("acceptance") != "none" or
                    coe.get("fixed_expert_steps") != [None] * 4 or
                    cfg["loss"].get("balance_importance") != "candidate" or
                    cfg["loss"].get("lambda_coe_balance") != 0.01 or
                    train.get("oracle_last_step_samples", 0) != 0 or
                    train.get("partner_probe") is not None or
                    train.get("save_best_checkpoint") is not False or
                    train.get("scheduler", {}).get("total_epochs") != train["epochs"] or
                    cfg["data"].get("mask", {}).get("pattern") != "random" or
                    cfg["data"].get("mask", {}).get("missing_rate") != 0.4 or
                    cfg["data"].get("train_mask_diversity", {}).get("rates") != [0.4] or
                    cfg["data"].get("eval_mask_diversity", {}).get("rates") != [0.4] or
                    len(cfg["data"].get("train_mask_diversity", {}).get("families", [])) != 9 or
                    cfg["data"].get("train_mask_diversity", {}).get("families") !=
                    cfg["data"].get("eval_mask_diversity", {}).get("families") or
                    cfg["data"].get("train_mask_diversity", {}).get("resample_each_epoch") is not True or
                    cfg["data"].get("eval_mask_diversity", {}).get("resample_each_epoch") is not False):
                raise ValueError(f"{name}: invalid four-step eight-expert main settings")
        if manifest.get("stage") == "coe_partner_native4":
            coe = cfg["model"]["coe"]
            diversity = cfg["data"]
            expected = {
                "residual4_shared_native": ("native", "individual", False),
                "residual4_partner_fusion": ("partner_residual", "corrected", False),
                "residual4_partner_fusion_headonly": ("partner_residual", "corrected", True),
            }[run["variant"]]
            actual = (coe.get("pair_mode"), coe.get("partner_fusion", "individual"),
                      coe.get("partner_aux_head_only", False))
            if (actual != expected or
                    coe.get("num_steps") != 4 or
                    coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST", "TL", "SL"] or
                    coe.get("top_k") != 2 or coe.get("routing_mode") != "hard" or
                    coe.get("expert_sharing") != "shared" or
                    coe.get("completion_feedback") is not False or
                    coe.get("fixed_expert_steps") != [None] * 4 or
                    coe.get("routing_warmup_epochs") != 3 or
                    coe.get("routing_transition_epochs") != 3 or
                    cfg["loss"].get("balance_importance") != "candidate" or
                    cfg["loss"].get("lambda_coe_balance") != 0.01 or
                    diversity.get("mask", {}).get("pattern") != "random" or
                    diversity.get("mask", {}).get("missing_rate") != 0.4 or
                    diversity.get("train_mask_diversity", {}).get("rates") != [0.4] or
                    diversity.get("eval_mask_diversity", {}).get("rates") != [0.4] or
                    len(diversity.get("train_mask_diversity", {}).get("families", [])) != 9 or
                    diversity.get("train_mask_diversity", {}).get("families") !=
                    diversity.get("eval_mask_diversity", {}).get("families") or
                    diversity.get("train_mask_diversity", {}).get("resample_each_epoch") is not True or
                    diversity.get("eval_mask_diversity", {}).get("resample_each_epoch") is not False or
                    train.get("save_best_checkpoint") is not False or
                    train.get("oracle_last_step_samples", 0) != 0 or
                    train.get("scheduler", {}).get("total_epochs") != train["epochs"]):
                raise ValueError(f"{name}: invalid four-round shared expert combination settings")
        if manifest.get("stage") == "coe_partner_residual4":
            coe = cfg["model"]["coe"]
            expected = {
                "residual4_moe": ("per_step", "native"),
                "residual4_partner_legacy": ("shared", "partner"),
                "residual4_partner_residual": ("shared", "partner_residual"),
            }[run["variant"]]
            diversity = cfg["data"]
            if (tuple(coe.get(key) for key in ("expert_sharing", "pair_mode")) != expected or
                    coe.get("num_steps") != 4 or
                    coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST", "TL", "SL"] or
                    coe.get("top_k") != 2 or coe.get("routing_mode") != "hard" or
                    coe.get("completion_feedback") is not False or
                    coe.get("fixed_expert_steps") != [None] * 4 or
                    coe.get("routing_warmup_epochs") != 3 or
                    coe.get("routing_transition_epochs") != 3 or
                    cfg["loss"].get("balance_importance") != "candidate" or
                    cfg["loss"].get("lambda_coe_balance") != 0.01 or
                    diversity.get("mask", {}).get("pattern") != "random" or
                    diversity.get("mask", {}).get("missing_rate") != 0.4 or
                    diversity.get("train_mask_diversity", {}).get("rates") != [0.4] or
                    diversity.get("eval_mask_diversity", {}).get("rates") != [0.4] or
                    len(diversity.get("train_mask_diversity", {}).get("families", [])) != 9 or
                    diversity.get("train_mask_diversity", {}).get("families") !=
                    diversity.get("eval_mask_diversity", {}).get("families") or
                    diversity.get("train_mask_diversity", {}).get("resample_each_epoch") is not True or
                    diversity.get("eval_mask_diversity", {}).get("resample_each_epoch") is not False or
                    train.get("save_best_checkpoint") is not False or
                    train.get("oracle_last_step_samples", 0) != 0 or
                    train.get("scheduler", {}).get("total_epochs") != train["epochs"] or
                    train.get("partner_probe", {}).get("interval_batches") != 16 or
                    train.get("partner_probe", {}).get("weight") != 0.1 or
                    train.get("partner_probe", {}).get("grad_diagnostic_interval_batches") != 64):
                raise ValueError(f"{name}: invalid four-round partner comparison settings")
        if manifest.get("stage") == "coe_focus":
            coe = cfg["model"]["coe"]
            expected = {
                "focus_moe": (False, "per_step", "native"),
                "focus_coe_independent": (True, "per_step", "native"),
                "focus_coe_shared": (True, "shared", "native"),
                "focus_pair_additive": (True, "shared", "additive"),
                "focus_pair_capacity": (True, "shared", "additive"),
                "focus_pair_interaction": (True, "shared", "interaction"),
            }[run["variant"]]
            actual = tuple(coe.get(key) for key in ("completion_feedback", "expert_sharing", "pair_mode"))
            expected_hidden = 68 if run["variant"] == "focus_pair_capacity" else 64
            mask = cfg["data"]["mask"]
            if (actual != expected or not cfg["data"].get("dataset_name") or
                    mask.get("pattern") != "random" or mask.get("missing_rate") != 0.4 or
                    coe.get("router_hidden_dim") != expected_hidden or
                    coe.get("num_steps") != 3 or coe.get("top_k") != 2 or
                    coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST"] or
                    coe.get("routing_mode") != "hard" or coe.get("router_state") != "dynamic" or
                    coe.get("expert_state") != "dynamic" or coe.get("acceptance") != "none" or
                    coe.get("fixed_expert_steps") != [None] * 3 or
                    coe.get("routing_warmup_epochs") != 3 or coe.get("routing_transition_epochs") != 3 or
                    cfg["loss"].get("lambda_coe_balance") != 0.01 or
                    cfg["data"].get("train_mask_diversity", {}).get("rates") != [0.4] or
                    cfg["data"].get("eval_mask_diversity", {}).get("rates") != [0.4] or
                    len(cfg["data"].get("train_mask_diversity", {}).get("families", [])) != 9 or
                    cfg["data"].get("train_mask_diversity", {}).get("resample_each_epoch") is not True or
                    cfg["data"].get("eval_mask_diversity", {}).get("resample_each_epoch") is not False or
                    train.get("save_best_checkpoint") is not False or
                    train.get("scheduler", {}).get("total_epochs") != train["epochs"]):
                raise ValueError(f"{name}: invalid focused comparison settings")
        if manifest.get("stage") == "coe_partner":
            coe = cfg["model"]["coe"]
            expected = {
                "focus_moe": (False, "per_step", "native"),
                "focus_coe_independent": (True, "per_step", "native"),
                "focus_shared_no_feedback": (False, "shared", "native"),
                "focus_coe_shared": (True, "shared", "native"),
                "focus_partner": (True, "shared", "partner"),
            }[run["variant"]]
            actual = tuple(coe.get(key) for key in
                           ("completion_feedback", "expert_sharing", "pair_mode"))
            mask = cfg["data"]["mask"]
            if (actual != expected or not cfg["data"].get("dataset_name") or
                    mask.get("pattern") != "random" or mask.get("missing_rate") != 0.4 or
                    coe.get("num_steps") != 3 or coe.get("top_k") != 2 or
                    coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST"] or
                    coe.get("routing_mode") != "hard" or coe.get("router_hidden_dim") != 64 or
                    coe.get("acceptance") != "none" or
                    coe.get("fixed_expert_steps") != [None] * 3 or
                    cfg["loss"].get("balance_importance") != "candidate" or
                    cfg["data"].get("train_mask_diversity", {}).get("rates") != [0.4] or
                    cfg["data"].get("eval_mask_diversity", {}).get("rates") != [0.4] or
                    len(cfg["data"].get("train_mask_diversity", {}).get("families", [])) != 9 or
                    cfg["data"].get("train_mask_diversity", {}).get("resample_each_epoch") is not True or
                    cfg["data"].get("eval_mask_diversity", {}).get("resample_each_epoch") is not False or
                    train.get("save_best_checkpoint") is not True or
                    train.get("scheduler", {}).get("total_epochs") != train["epochs"] or
                    train.get("oracle_last_step_samples", 0) < 1):
                raise ValueError(f"{name}: invalid five-arm partner comparison settings")
        if manifest.get("stage") == "coe_team_accept_v4":
            coe = cfg["model"]["coe"]
            expected = {
                "team_e0": ("native", "none", "shared", 64),
                "team_e1": ("additive", "none", "shared", 64),
                "team_e2": ("interaction", "none", "shared", 64),
                "team_e3": ("additive", "none", "shared", 68),
                "team_e4": ("additive", "point", "shared", 64),
                "team_e5": ("interaction", "point", "shared", 64),
                "team_e10": ("additive", "none", "per_step", 64),
            }[run["variant"]]
            actual = tuple(coe.get(key) for key in ("pair_mode", "acceptance", "expert_sharing", "router_hidden_dim"))
            if (actual != expected or coe.get("num_steps") != 3 or coe.get("top_k") != 2 or
                    coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST"] or
                    coe.get("fixed_expert_steps") != [None] * 3 or coe.get("routing_mode") != "hard" or
                    coe.get("routing_warmup_epochs") != 3 or coe.get("routing_transition_epochs") != 3 or
                    cfg["loss"].get("lambda_coe_balance") != 0.01 or
                    cfg["data"].get("train_mask_diversity", {}).get("rates") != [0.4] or
                    cfg["data"].get("eval_mask_diversity", {}).get("rates") != [0.4] or
                    len(cfg["data"].get("train_mask_diversity", {}).get("families", [])) != 9 or
                    cfg["data"].get("train_mask_diversity", {}).get("resample_each_epoch") is not True or
                    cfg["data"].get("eval_mask_diversity", {}).get("resample_each_epoch") is not False or
                    train.get("scheduler", {}).get("total_epochs") != train["epochs"] or
                    train.get("save_best_checkpoint") != (cfg["data"]["dataset_name"] == "TaxiBJ")):
                raise ValueError(f"{name}: invalid team/acceptance v4 comparison settings")
        if manifest.get("stage") == "coe_mechanism1":
            coe = cfg["model"]["coe"]
            if coe.get("num_steps") != 4 or coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST"]:
                raise ValueError(f"{name}: mechanism1 requires the four-step six-expert pool")
            if run["variant"] in {"coe_mech_main", "coe_mech_initial_router", "coe_mech_no_balance", "coe_mech_initial_expert"}:
                if (coe.get("routing_mode") != "hard" or coe.get("routing_warmup_epochs") != 3
                        or coe.get("routing_transition_epochs") != 3
                        or cfg["loss"].get("lambda_coe_balance") != (0.0 if run["variant"] == "coe_mech_no_balance" else 0.01)):
                    raise ValueError(f"{name}: warmup main configuration was overwritten")
            elif run["variant"] == "coe_mech_conditional_soft":
                if coe.get("routing_mode") != "soft" or coe.get("global_route_weights") or cfg["loss"].get("lambda_coe_balance") != 0.01:
                    raise ValueError(f"{name}: invalid conditional-soft configuration")
            elif run["variant"] == "coe_mech_global_soft":
                if coe.get("routing_mode") != "soft" or not coe.get("global_route_weights") or cfg["loss"].get("lambda_coe_balance") != 0.01:
                    raise ValueError(f"{name}: invalid global-soft configuration")
            else:
                if (coe.get("routing_mode") != "fixed" or len(coe.get("fixed_path") or []) != 4
                        or coe.get("routing_warmup_epochs", 0) or coe.get("routing_transition_epochs", 0)
                        or cfg["loss"].get("lambda_coe_balance") != 0.0):
                    raise ValueError(f"{name}: invalid fixed-chain configuration")
        if manifest.get("stage") == "coe_dual_mask":
            coe = cfg["model"]["coe"]
            if coe["num_steps"] != 4 or coe.get("expert_pool") != ["T", "S", "TD", "SD", "TA", "ST"]:
                raise ValueError(f"{name}: dual-mask experiments require four steps and the six-expert pool")
            if len(coe.get("fixed_expert_steps", [None] * 4)) != 4:
                raise ValueError(f"{name}: fixed_expert_steps must have four entries")
            if run["variant"] == "coe_main":
                if (coe["routing_mode"] != "hard" or coe.get("routing_warmup_epochs") != 3
                        or coe.get("routing_transition_epochs") != 3
                        or cfg["loss"].get("lambda_coe_balance") != 0.01):
                    raise ValueError(f"{name}: main warmup/balance settings were overwritten")
            elif (coe["routing_mode"] != "fixed" or len(coe.get("fixed_path") or []) != 4
                  or any(e not in coe["expert_pool"] for e in coe["fixed_path"])
                  or coe.get("routing_warmup_epochs", 0) or coe.get("routing_transition_epochs", 0)):
                raise ValueError(f"{name}: invalid fixed-chain configuration")
        for key in ("epochs", "val_epoch"):
            if type(train[key]) is not int or train[key] < 1:
                raise ValueError(f"train.{key} must be a positive integer")
        if type(train.get("save_best_checkpoint", True)) is not bool:
            raise ValueError("save_best_checkpoint must be boolean")
        if train.get("best_checkpoint_name", "best.pt") not in ("best.pt", "best.pth"):
            raise ValueError("best_checkpoint_name must be best.pt or best.pth")
        if train.get("early_stopping", {}).get("enabled", False) or cfg["data"].get("drop_last", False):
            raise ValueError("Matched experiments require full epochs and drop_last=false")
        if cfg["model"]["architecture"] != "v24_ts_coe":
            raise ValueError("Expected v24_ts_coe")
        if manifest["datasets"]:
            for split, item in manifest["datasets"].items():
                actual = planner.npz_shape(resolve(item["path"]), cfg)
                if actual["shape_ncthw"] != item["shape_ncthw"]:
                    raise ValueError(f"{split} dataset shape changed since planning")
                if set(actual["keys"]) & {"m_f", "target_mask"}:
                    raise ValueError("Embedded masks change the experiment supervision protocol")


    if manifest.get("stage") == "coe_focus":
        expected_order = ["focus_moe", "focus_coe_independent", "focus_coe_shared",
                          "focus_pair_additive", "focus_pair_capacity", "focus_pair_interaction"]
        by_seed = {}
        for run in manifest["runs"]:
            by_seed.setdefault(run["seed"], []).append(run)
        for seed, runs in by_seed.items():
            if [run["variant"] for run in runs] != expected_order:
                raise ValueError(f"Focused seed {seed} must contain the six planned variants in order")
            reference = None
            for run in runs:
                normalized = copy.deepcopy(run["config"])
                normalized["experiment_plan"]["variant"] = "reference"
                for key in ("completion_feedback", "expert_sharing", "pair_mode", "router_hidden_dim"):
                    normalized["model"]["coe"].pop(key, None)
                if reference is None:
                    reference = normalized
                elif normalized != reference:
                    raise ValueError(f"{run['variant']}: changed settings outside the planned intervention")
    if manifest.get("stage") == "coe_partner":
        expected_order = ["focus_moe", "focus_coe_independent", "focus_shared_no_feedback",
                          "focus_coe_shared", "focus_partner"]
        by_seed = {}
        for run in manifest["runs"]:
            by_seed.setdefault(run["seed"], []).append(run)
        for seed, runs in by_seed.items():
            if [run["variant"] for run in runs] != expected_order:
                raise ValueError(f"Partner seed {seed} must contain M0/M1/C0/C1/P1 in order")
            reference = None
            for run in runs:
                normalized = copy.deepcopy(run["config"])
                normalized["experiment_plan"]["variant"] = "reference"
                for key in ("completion_feedback", "expert_sharing", "pair_mode"):
                    normalized["model"]["coe"].pop(key, None)
                if reference is None:
                    reference = normalized
                elif normalized != reference:
                    raise ValueError(f"{run['variant']}: changed settings outside the planned intervention")

    if manifest.get("stage") == "coe_direct_baselines":
        expected_order = ["direct_moe_s1_top8", "direct_moe_s4_top2", "direct_shared_s4_top2"]
        by_seed = {}
        for run in manifest["runs"]:
            by_seed.setdefault(run["seed"], []).append(run)
        for seed, runs in by_seed.items():
            if [run["variant"] for run in runs] != expected_order:
                raise ValueError(f"Direct MoE seed {seed} must run all three baselines in order")
            reference = None
            for run in runs:
                normalized = copy.deepcopy(run["config"])
                normalized["experiment_plan"]["variant"] = "reference"
                for key in ("num_steps", "top_k", "expert_sharing", "fixed_expert_steps"):
                    normalized["model"]["coe"].pop(key, None)
                if reference is None:
                    reference = normalized
                elif normalized != reference:
                    raise ValueError(f"{run['variant']}: changed settings outside depth, Top-K, or sharing")

    if manifest.get("stage") == "coe_depth_pair8":
        expected_order = ["depthpair_moe_s1_top8", "depthpair_moe_s2_top4",
                          "depthpair_coe_s4_pair28"]
        by_seed = {}
        for run in manifest["runs"]:
            by_seed.setdefault(run["seed"], []).append(run)
        for seed, runs in by_seed.items():
            if [run["variant"] for run in runs] != expected_order:
                raise ValueError(f"Depth/pair seed {seed} must run all three arms in order")
            reference = None
            for run in runs:
                normalized = copy.deepcopy(run["config"])
                normalized["experiment_plan"]["variant"] = "reference"
                for key in ("num_steps", "top_k", "expert_sharing", "pair_mode",
                            "fixed_expert_steps"):
                    normalized["model"]["coe"].pop(key, None)
                if reference is None:
                    reference = normalized
                elif normalized != reference:
                    raise ValueError(f"{run['variant']}: changed settings outside depth/width/pair policy")

    if manifest.get("stage") == "coe_fusion1":
        expected_order = ["pair28_warm", "top2_context", "top2_response", "top2_local", "top2_proposal"]
        by_seed = {}
        for run in manifest["runs"]:
            by_seed.setdefault(run["seed"], []).append(run)
        for seed, runs in by_seed.items():
            if [run["variant"] for run in runs] != expected_order:
                raise ValueError(f"Fusion seed {seed} must run all five arms in order")
            reference = None
            for run in runs:
                normalized = copy.deepcopy(run["config"])
                normalized["experiment_plan"]["variant"] = "reference"
                for key in ("pair_mode", "fusion_mode", "pair_dense_warmup_steps"):
                    normalized["model"]["coe"].pop(key, None)
                if reference is None:
                    reference = normalized
                elif normalized != reference:
                    raise ValueError(f"{run['variant']}: changed settings outside fusion intervention")

    if manifest.get("stage") == "coe_partner_native4":
        expected_order = ["residual4_shared_native", "residual4_partner_fusion",
                          "residual4_partner_fusion_headonly"]
        by_seed = {}
        for run in manifest["runs"]:
            by_seed.setdefault(run["seed"], []).append(run)
        for seed, runs in by_seed.items():
            if [run["variant"] for run in runs] != expected_order:
                raise ValueError(f"Expert combination seed {seed} must run native/fusion/head-only in order")
            reference = None
            for run in runs:
                normalized = copy.deepcopy(run["config"])
                normalized["experiment_plan"]["variant"] = "reference"
                for key in ("pair_mode", "partner_fusion", "partner_aux_head_only"):
                    normalized["model"]["coe"].pop(key, None)
                if reference is None:
                    reference = normalized
                elif normalized != reference:
                    raise ValueError(f"{run['variant']}: changed settings outside expert combination")

    if manifest.get("stage") == "coe_partner_residual4":
        expected_order = ["residual4_moe", "residual4_partner_legacy", "residual4_partner_residual"]
        by_seed = {}
        for run in manifest["runs"]:
            by_seed.setdefault(run["seed"], []).append(run)
        for seed, runs in by_seed.items():
            if [run["variant"] for run in runs] != expected_order:
                raise ValueError(f"Partner residual seed {seed} must contain the three planned variants in order")
            reference = None
            for run in runs:
                normalized = copy.deepcopy(run["config"])
                normalized["experiment_plan"]["variant"] = "reference"
                for key in ("expert_sharing", "pair_mode"):
                    normalized["model"]["coe"].pop(key, None)
                if reference is None:
                    reference = normalized
                elif normalized != reference:
                    raise ValueError(f"{run['variant']}: changed settings outside sharing and partner score")

    if manifest.get("stage") == "coe_team_accept_v4":
        expected_order = ["team_e0", "team_e1", "team_e2", "team_e3", "team_e4", "team_e5", "team_e10"]
        if [run["variant"] for run in manifest["runs"]] != expected_order:
            raise ValueError("v4 first batch must contain E0/E1/E2/E3/E4/E5/E10 in order")
        reference = None
        for run in manifest["runs"]:
            normalized = copy.deepcopy(run["config"])
            normalized["experiment_plan"]["variant"] = "reference"
            for key in ("pair_mode", "acceptance", "expert_sharing", "router_hidden_dim"):
                normalized["model"]["coe"].pop(key, None)
            if reference is None:
                reference = normalized
            elif normalized != reference:
                raise ValueError(f"{run['variant']}: changed settings outside the planned v4 intervention")
        dataset_name = manifest["runs"][0]["config"]["data"]["dataset_name"]
        if dataset_name == "TaxiBJ":
            clean_manifest = load(ROOT / "data/TaxiBJ/v24_clean_abc_20260917/manifest.json")
            for split, dataset in manifest["datasets"].items():
                recorded = clean_manifest["splits"][split]
                if (recorded["shape"] != dataset["shape_ncthw"] or
                        recorded["npz_sha256"] != planner._sha256(Path(dataset["path"]))):
                    raise ValueError(f"{split} clean TaxiBJ NPZ differs from its preparation manifest")
        elif dataset_name != "BikeNYC":
            raise ValueError(f"Unsupported v4 dataset: {dataset_name}")


def materialize(manifest, suite, fingerprint, world_size=1):
    result = copy.deepcopy(manifest)
    result["suite_fingerprint"] = fingerprint
    result["world_size"] = world_size
    for run in result["runs"]:
        run["config"]["output_dir"] = (str(ROOT / "outputs/v24-COE")
            if result.get("stage") in {"coe_team_accept_v4", "coe_focus", "coe_partner", "coe_partner_residual4", "coe_partner_native4", "coe_depth_pair8", "coe_fusion1", "coe_direct_baselines"} else str(suite / "runs" / run["name"]))
        run["config"]["experiment_suite_fingerprint"] = fingerprint
        run["config_path"] = str(suite / "configs" / (run["name"] + ".json"))
        command = [sys.executable, "-u", str(ROOT / "scripts/train.py"), "-c",
                   run["config_path"], "--no_plot", "--name", run["name"]]
        if result["datasets"]:
            for split, item in result["datasets"].items():
                command.extend([f"--{split}_npz", str(resolve(item["path"]))])
        else:
            command.append("--synthetic")
        run["command_argv"] = command
        run.pop("command", None)
    return result


def expected_samples(manifest, cfg):
    if manifest["datasets"]:
        return {split: item["shape_ncthw"][0] for split, item in manifest["datasets"].items()}
    syn = cfg["data"]["synthetic"]
    return {"train": syn["num_train"], "val": syn["num_val"], "test": syn["num_val"]}


def check_complete(receipt_path, run, manifest):
    """Receipt alone is insufficient: verify full epoch/VAL/TEST records and files."""
    try:
        receipt = load(receipt_path)
        cfg = run["config"]
        if (receipt["status"] != "finished" or receipt["config_sha256"] != digest(cfg) or
                receipt.get("world_size", 1) != manifest.get("world_size", 1)):
            return False
        epochs, interval = cfg["train"]["epochs"], cfg["train"]["val_epoch"]
        if receipt["completed_epochs"] != epochs or receipt["samples"] != expected_samples(manifest, cfg):
            return False
        run_dir = Path(receipt["run_dir"])
        if not run_dir.resolve().is_relative_to(Path(cfg["output_dir"]).resolve()):
            return False
        if load(run_dir / "config.json") != cfg:
            return False
        log_dir = run_dir / "logs"
        if any(not (log_dir / name).is_file() for name in ("train.log", "val.log", "test.log", "metrics.jsonl")):
            return False
        rows = [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]
        history = [row for row in rows if "epoch" in row]
        tests = [row for row in rows if row.get("stage") == "test"]
        if [r["epoch"] for r in history] != list(range(1, epochs + 1)) or len(tests) != 1:
            return False
        vals = [row for row in history if row.get("val") is not None]
        expected = [e for e in range(1, epochs + 1) if e % interval == 0 or e == epochs]
        if [r["epoch"] for r in vals] != expected or receipt["validation_count"] != len(expected):
            return False
        for metrics in [r["train"] for r in history] + [r["val"] for r in vals] + [tests[0]["metrics"]]:
            if any(not math.isfinite(metrics[key]) for key in ("loss", "mae", "rmse")):
                return False
        best = min(vals, key=lambda r: r["val"]["mae"])
        if receipt["best_epoch"] != best["epoch"] or receipt["best_val_mae"] != best["val"]["mae"]:
            return False
        if tests[0]["extra"]["best_epoch"] != best["epoch"] or tests[0]["metrics"] != receipt["test"]:
            return False
        storage = "checkpoint" if cfg["train"].get("save_best_checkpoint", True) else "cpu_memory"
        if receipt["best_state_source"] != storage or tests[0]["extra"]["best_state_source"] != storage:
            return False
        best_name = cfg["train"].get("best_checkpoint_name", "best.pt")
        if storage == "checkpoint" and not (run_dir / "checkpoints" / best_name).is_file():
            return False
        if manifest.get("stage") == "coe_partner":
            oracle_path = log_dir / "oracle_last_step.json"
            if not oracle_path.is_file():
                return False
            oracle = load(oracle_path)
            if (oracle.get("status") != "label_only_diagnostic_not_deployable" or
                    oracle.get("best_epoch") != receipt["best_epoch"] or
                    oracle.get("evaluated_samples", 0) < 1 or
                    any(not math.isfinite(oracle[key]) for key in
                        ("top2_mae", "anchor_best_mae", "all_best_mae")) or
                    not oracle["all_best_mae"] <= oracle["anchor_best_mae"] <= oracle["top2_mae"]):
                return False
        if "Training finished normally" not in (log_dir / "train.log").read_text():
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def result_for(run, suite, manifest):
    attempts = sorted((suite / "results").glob(run["name"] + ".attempt*.json"))
    for path in reversed(attempts):
        if check_complete(path, run, manifest):
            return load(path)
    return None


def summarize(manifest, suite):
    rows, diagnostics = [], {}
    for run in manifest["runs"]:
        result = result_for(run, suite, manifest)
        row = {"name": run["name"], "dataset": run["config"]["data"]["dataset_name"],
               "protocol": run["protocol"], "variant": run["variant"], "seed": run["seed"],
               "evaluation_mask_source": run.get("config", {}).get("experiment_plan", {}).get(
                   "evaluation_mask_source",
                   "diverse_fixed_per_split" if "eval_mask_diversity" in run["config"].get("data", {}) else "protocol",
               ),
               "status": "complete" if result else "pending", "best_epoch": None,
               "best_val_mae": None, "test_mae": None, "test_rmse": None, "seconds": None, "run_dir": None}
        if result:
            row.update(best_epoch=result["best_epoch"], best_val_mae=result["best_val_mae"],
                       test_mae=result["test"]["mae"], test_rmse=result["test"]["rmse"],
                       seconds=result["total_time_sec"], run_dir=result["run_dir"])
            diagnostics[run["name"]] = {k: v for k, v in result["test"].items() if k.startswith("coe_")}
        if manifest.get("stage") in {"coe_partner", "coe_partner_residual4", "coe_partner_native4", "coe_depth_pair8", "coe_fusion1", "coe_direct_baselines"}:
            row.update(
                total_params=result.get("total_params") if result else None,
                peak_memory_gb=result.get("peak_memory_gb") if result else None,
                test_time_sec=result.get("test_time_sec") if result else None,
                forward_ms_per_batch_per_rank=result["test"].get("forward_ms_per_batch_per_rank") if result else None,
                partner_probe_forward_calls=result.get("partner_probe_forward_calls") if result else None,
            )
        rows.append(row)
    pairs, fixed = [], []
    groups = {(r["dataset"], r["protocol"], r["seed"]) for r in rows}
    for dataset, protocol, seed in sorted(groups):
        all_group = [r for r in rows if (r["dataset"], r["protocol"], r["seed"]) == (dataset, protocol, seed)]
        evaluation_sources = sorted({r["evaluation_mask_source"] for r in all_group})
        for evaluation_source in evaluation_sources:
            group = [r for r in all_group if r["evaluation_mask_source"] == evaluation_source]
            if manifest.get("stage") in {"abc", "abcd", "abcde"}:
                reference = "abc_a"
            elif manifest.get("stage") == "coe_team_accept_v4":
                reference = "team_e0"
            elif manifest.get("stage") == "coe_focus":
                reference = "focus_coe_shared"
            elif manifest.get("stage") == "coe_partner":
                reference = "focus_coe_shared"
            elif manifest.get("stage") == "coe_partner_residual4":
                reference = "residual4_partner_legacy"
            elif manifest.get("stage") == "coe_depth_pair8":
                reference = "depthpair_moe_s1_top8"
            elif manifest.get("stage") == "coe_direct_baselines":
                reference = "direct_moe_s4_top2"
            elif manifest.get("stage") == "coe_fusion1":
                reference = "top2_context"
            elif manifest.get("stage") in {"coe_validation", "coe_dual_mask", "coe_mechanism1"}:
                reference = "coe_main"
            elif manifest.get("stage") == "route20":
                reference = "route20_base"
            elif manifest.get("stage") == "followup":
                reference = "abc_d_eval_mixed" if evaluation_source == "diverse_fixed_per_split" else "abc_d_static"
            else:
                reference = "full"
            full = next((r for r in group if r["variant"] == reference and r["status"] == "complete"), None)
            if full and manifest.get("stage") not in {"coe_focus", "coe_partner", "coe_partner_residual4"}:
                for other in group:
                    if other["status"] == "complete" and other is not full:
                        pairs.append({"dataset": dataset, "protocol": protocol, "seed": seed,
                                      "evaluation_mask_source": evaluation_source,
                                      "reference": reference, "variant": other["variant"],
                                      f"test_mae_delta_vs_{reference}": other["test_mae"] - full["test_mae"],
                                      f"test_rmse_delta_vs_{reference}": other["test_rmse"] - full["test_rmse"]})
        if manifest.get('stage') == 'coe_partner_residual4':
            for reference, variant in [
                ('residual4_moe', 'residual4_partner_legacy'),
                ('residual4_partner_legacy', 'residual4_partner_residual'),
                ('residual4_moe', 'residual4_partner_residual'),
            ]:
                base = next((r for r in all_group if r['variant'] == reference and r['status'] == 'complete'), None)
                other = next((r for r in all_group if r['variant'] == variant and r['status'] == 'complete'), None)
                if base and other:
                    pairs.append({'dataset': dataset, 'protocol': protocol, 'seed': seed,
                                  'reference': reference, 'variant': variant,
                                  f'test_mae_delta_vs_{reference}': other['test_mae'] - base['test_mae'],
                                  f'test_rmse_delta_vs_{reference}': other['test_rmse'] - base['test_rmse']})
        if manifest.get('stage') == 'coe_partner':
            for reference, variant in [
                ('focus_moe', 'focus_coe_independent'),
                ('focus_moe', 'focus_shared_no_feedback'),
                ('focus_shared_no_feedback', 'focus_coe_shared'),
                ('focus_coe_independent', 'focus_coe_shared'),
                ('focus_coe_shared', 'focus_partner'),
            ]:
                base = next((r for r in all_group if r['variant'] == reference and r['status'] == 'complete'), None)
                other = next((r for r in all_group if r['variant'] == variant and r['status'] == 'complete'), None)
                if base and other:
                    pairs.append({'dataset': dataset, 'protocol': protocol, 'seed': seed,
                                  'reference': reference, 'variant': variant,
                                  f'test_mae_delta_vs_{reference}': other['test_mae'] - base['test_mae'],
                                  f'test_rmse_delta_vs_{reference}': other['test_rmse'] - base['test_rmse']})
        if manifest.get('stage') == 'coe_focus':
            for reference, variant in [
                ('focus_moe', 'focus_coe_independent'),
                ('focus_coe_independent', 'focus_coe_shared'),
                ('focus_moe', 'focus_coe_shared'),
                ('focus_coe_shared', 'focus_pair_additive'),
                ('focus_pair_additive', 'focus_pair_capacity'),
                ('focus_pair_additive', 'focus_pair_interaction'),
                ('focus_pair_capacity', 'focus_pair_interaction'),
                ('focus_coe_shared', 'focus_pair_interaction'),
            ]:
                base = next((r for r in all_group if r['variant'] == reference and r['status'] == 'complete'), None)
                other = next((r for r in all_group if r['variant'] == variant and r['status'] == 'complete'), None)
                if base and other:
                    pairs.append({'dataset': dataset, 'protocol': protocol, 'seed': seed,
                                  'reference': reference, 'variant': variant,
                                  f'test_mae_delta_vs_{reference}': other['test_mae'] - base['test_mae'],
                                  f'test_rmse_delta_vs_{reference}': other['test_rmse'] - base['test_rmse']})
        if manifest.get('stage') == 'coe_team_accept_v4':
            for reference, variant in [
                ('team_e0', 'team_e1'), ('team_e1', 'team_e2'),
                ('team_e1', 'team_e3'), ('team_e1', 'team_e4'),
                ('team_e2', 'team_e5'), ('team_e4', 'team_e5'),
                ('team_e1', 'team_e10'), ('team_e10', 'team_e5'),
            ]:
                base = next((r for r in all_group if r['variant'] == reference and r['status'] == 'complete'), None)
                other = next((r for r in all_group if r['variant'] == variant and r['status'] == 'complete'), None)
                if base and other:
                    pairs.append({'dataset': dataset, 'protocol': protocol, 'seed': seed,
                                  'reference': reference, 'variant': variant,
                                  f'test_mae_delta_vs_{reference}': other['test_mae'] - base['test_mae'],
                                  f'test_rmse_delta_vs_{reference}': other['test_rmse'] - base['test_rmse']})
        if manifest.get('stage') == 'abcde':
            for reference, variant in [('abc_b', 'abc_c'), ('abc_d', 'abc_e')]:
                base = next((r for r in group if r['variant'] == reference and r['status'] == 'complete'), None)
                other = next((r for r in group if r['variant'] == variant and r['status'] == 'complete'), None)
                if base and other:
                    pairs.append({'dataset': dataset, 'protocol': protocol, 'seed': seed,
                                  'reference': reference, 'variant': variant,
                                  f'test_mae_delta_vs_{reference}': other['test_mae'] - base['test_mae'],
                                  f'test_rmse_delta_vs_{reference}': other['test_rmse'] - base['test_rmse']})
        candidates = [r for r in group if r["variant"] in {"fixed_tt", "fixed_ts", "fixed_st", "fixed_ss"}]
        if len(candidates) == 4 and all(r["status"] == "complete" for r in candidates):
            chosen = min(candidates, key=lambda r: r["best_val_mae"])
            fixed.append({"dataset": dataset, "protocol": protocol, "seed": seed,
                          "selected_by": "validation_mae", "variant": chosen["variant"],
                          "test_mae": chosen["test_mae"], "test_rmse": chosen["test_rmse"]})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    temporary = suite / "summary.csv.tmp"
    temporary.write_text(stream.getvalue(), encoding="utf-8")
    os.replace(temporary, suite / "summary.csv")
    write_json(suite / "comparison.json", {"suite_fingerprint": manifest["suite_fingerprint"],
        "complete": sum(r["status"] == "complete" for r in rows), "total": len(rows),
        "note": "Matched single-seed descriptive comparisons; negative delta means variant is better. No test-based model selection.",
        "rows": rows, "paired": pairs, "validation_selected_fixed_path": fixed})
    write_json(suite / "diagnostics.json", diagnostics)
    return rows


def launch(run, suite, env, world_size=1):
    log_dir = suite / "launcher_logs"
    log_dir.mkdir(exist_ok=True)
    attempt = 1
    while (log_dir / f'{run["name"]}.attempt{attempt}.log').exists():
        attempt += 1
    receipt = suite / "results" / f'{run["name"]}.attempt{attempt}.json'
    command = run["command_argv"] + ["--result-file", str(receipt)]
    if world_size > 1:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={world_size}", *command[2:]]
    with (log_dir / f'{run["name"]}.attempt{attempt}.log').open("xb") as raw:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while chunk := process.stdout.read1(4096):
                raw.write(chunk); raw.flush()
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f'Experiment {run["name"]} failed (exit {code}); see {log_dir}. Completed jobs will be skipped on restart.')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v24/coe_focus_taxibj_experiments.json")
    parser.add_argument("--study", help="Study from the policy; defaults to its default_study, or pilot")
    parser.add_argument("--plan", type=Path, help="Run an existing immutable planner manifest using the same lifecycle")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", help="One physical GPU; omission preserves CUDA_VISIBLE_DEVICES")
    gpu_group.add_argument("--gpus", help="Comma-separated physical GPUs for one DDP job, e.g. 0,1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--epochs", type=int, help="Override all policy epochs and cosine scheduler period")
    parser.add_argument("--batch-size", type=int, help="Override global batch size; DDP splits it across ranks")
    args = parser.parse_args(argv)
    if args.plan and (args.epochs is not None or args.batch_size is not None):
        parser.error("--epochs and --batch-size cannot override an immutable --plan")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.gpu is not None and not args.gpu.isdigit():
        parser.error("--gpu must be a single nonnegative device index")
    gpu_ids = args.gpus.split(",") if args.gpus else None
    if gpu_ids and (len(gpu_ids) < 2 or len(set(gpu_ids)) != len(gpu_ids) or
                    any(not item.isdigit() for item in gpu_ids)):
        parser.error("--gpus requires two or more distinct nonnegative device indices")
    world_size = len(gpu_ids) if gpu_ids else 1
    if args.plan:
        plan_path = resolve(args.plan)
        manifest = load(plan_path)
        if manifest.get("status", "").startswith("superseded"):
            raise ValueError("This plan is superseded; use the current experiment config")
        for run in manifest["runs"]:
            if load(run["config_path"]) != run["config"]:
                raise ValueError("Plan configs differ from the manifest; generate a new plan")
        output, threads = plan_path.parent / "managed", 2
        extras = [plan_path, *(run["config_path"] for run in manifest["runs"])]
    else:
        policy_path = resolve(args.config)
        study = args.study or load(policy_path).get("default_study", "pilot")
        manifest, output, threads, extras = policy_plan(policy_path, study, args.epochs, args.batch_size)
    validate_plan(manifest)
    if any(run["config"]["data"]["batch_size"] % world_size for run in manifest["runs"]):
        parser.error("Global batch size must be divisible by DDP world size")
    record, stamps = identity(manifest, extras)
    record["cpu_threads"] = threads
    record["world_size"] = world_size
    fingerprint = digest(record)
    suite = output / fingerprint[:16]
    manifest = materialize(manifest, suite, fingerprint, world_size=world_size)
    if args.dry_run:
        print(json.dumps({"suite": str(suite), "world_size": world_size, "num_runs": len(manifest["runs"]),
                          "datasets": manifest["datasets"], "jobs": [
            {"name": r["name"], "train": r["config"]["train"],
             "batch_size": r["config"]["data"]["batch_size"]} for r in manifest["runs"]]}, indent=2))
        return
    if args.summary_only and not suite.exists():
        raise FileNotFoundError(f"No results for this exact code/data/config fingerprint: {suite}")
    suite.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    elif gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(threads)
    code_paths = set(source_files())
    with (suite / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"This suite is already running: {suite}") from None
        if (suite / "protocol.json").exists() and load(suite / "protocol.json") != record:
            raise RuntimeError("Suite fingerprint collision or modified protocol")
        assert_unchanged(stamps, code_paths)
        if manifest.get("stage") in {"coe_partner", "coe_main_s4_e8", "coe_partner_residual4", "coe_partner_native4", "coe_depth_pair8", "coe_fusion1", "coe_direct_baselines"}:
            snapshot_partner_sources(suite, extras)
        write_json(suite / "protocol.json", record)
        write_json(suite / "plan.json", manifest)
        for run in manifest["runs"]:
            path = Path(run["config_path"])
            if path.exists() and load(path) != run["config"]:
                raise RuntimeError(f"Generated config was changed: {path}")
            if not path.exists():
                write_json(path, run["config"])
            stamps[str(path)] = (signature(path), planner._sha256(path))
        summarize(manifest, suite)
        if args.summary_only:
            print(suite / "summary.csv")
            return
        for run in manifest["runs"]:
            assert_unchanged(stamps, code_paths)
            if result_for(run, suite, manifest):
                continue
            try:
                receipt = launch(run, suite, env, world_size=world_size)
                try:
                    assert_unchanged(stamps, code_paths)
                except RuntimeError:
                    if receipt.exists():
                        invalid = load(receipt); invalid["status"] = "invalidated_input_change"
                        write_json(receipt, invalid)
                    raise
                if not check_complete(receipt, run, manifest):
                    raise RuntimeError(f'Incomplete experiment {run["name"]}; inspect its logs before retrying.')
            finally:
                summarize(manifest, suite)


if __name__ == "__main__":
    main()
