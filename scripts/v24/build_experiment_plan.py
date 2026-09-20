#!/usr/bin/env python3
"""Compile v24 experiment configs and commands; never launch training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from stmoe_imputer.config import deep_update, load_config  # noqa: E402

CORE = (
    "full", "fixed_ts", "fixed_st", "fixed_tt", "fixed_ss", "initial_router",
    "no_expert_state_update", "parallel", "shared_only", "routed_only", "soft",
)
STAGES = {
    "coe_validation": ("coe_main", "coe_initial_router", "coe_initial_expert", "coe_fixed_chain", "coe_no_balance", "coe_original_mask"),
    "coe_dual_mask": ("coe_main", "coe_fixed_tast", "coe_fixed_tsts", "coe_fixed_stst", "coe_fixed_tata", "coe_fixed_ssss", "coe_fixed_tttt", "coe_fixed_stst_alt"),
    "route20": ("route20_base", "route20_warmup", "route20_grouped", "route20_previous", "route20_noise", "route20_fixed", "route20_small"),
    "route20_next": ("route20_next_noise", "route20_next_warmup", "route20_next_warmup_fixed2"),
    "pilot": ("full", "fixed_ts", "fixed_st", "fixed_tt", "fixed_ss", "parallel"),
    "core": CORE,
    "depth": tuple(f"k{k}_e{e}" for k in (2, 3, 4) for e in (2, 4, 6)),
    "optional": ("weak_mid",),
    "chain4": ("chain4_balance",),
    "abc": ("abc_a", "abc_b", "abc_c"),
    "abcd": ("abc_a", "abc_b", "abc_c", "abc_d"),
    "abcde": ("abc_a", "abc_b", "abc_c", "abc_d", "abc_e"),
    "followup": ("abc_d_eval_mixed", "fixed4_ta_st_s_ta", "abc_d_static", "abc_f"),
}
PATTERNS = (
    "random_point", "node_contiguous", "spatial_region", "spatiotemporal_block", "mixed",
)
QUESTIONS = {
    "coe_main": "主方案：四层六专家、软预热、当前状态重路由",
    "coe_initial_router": "主方案仅将各层路由输入固定为初始状态",
    "coe_initial_expert": "主方案仅将各层专家输入固定为初始状态",
    "coe_fixed_chain": "同深度固定 TA-ST-S-TA，对照动态路由方案",
    "coe_no_balance": "主方案仅关闭均衡辅助损失",
    "coe_original_mask": "主方案恢复原始 random_point mask，检验九类混合缺失协议的影响",
    "coe_fixed_tsts": "固定 T-S-T-S，时间/空间交替链",
    "coe_fixed_tast": "固定 TA-ST-S-TA，旧实验参照链",
    "coe_fixed_stst": "固定 S-T-S-T，空间/时间交替链",
    "coe_fixed_tata": "固定 TA-TA-TA-TA，重复时间注意力链",
    "coe_fixed_ssss": "固定 S-S-S-S，重复空间链",
    "coe_fixed_tttt": "固定 T-T-T-T，重复时间链",
    "coe_fixed_stst_alt": "固定 ST-ST-ST-ST，重复时空链",
    "route20_base": "R0：九类同分布、四层六专家硬路由基准",
    "route20_warmup": "R1：R0 + 3 epoch软预热、3 epoch过渡，随后完全硬路由",
    "route20_grouped": "R2：R0 + 分组归一化与观测模式特征",
    "route20_previous": "R3：R0 + 上一层选择的专家身份",
    "route20_noise": "R4：R0 + 训练阶段前两层路由输入相对高斯噪声",
    "route20_fixed": "R5：固定TA-ST-S-TA，检验动态选择收益",
    "route20_small": "R6：两层三专家T/S/ST容量候选",
    "route20_next_noise": "N1：四层六专家 R0 + 前两层训练期路由输入高斯噪声",
    "route20_next_warmup": "N2：复现实验 R1 软预热与过渡",
    "route20_next_warmup_fixed2": "N3：R1 + 第二层固定选择 TA",
    "full": "当前状态重路由的完整两轮模型",
    "fixed_ts": "固定时间后空间",
    "fixed_st": "固定空间后时间",
    "fixed_tt": "同深度仅时间方向功能消融",
    "fixed_ss": "同深度仅空间方向功能消融",
    "initial_router": "每轮 router 仅读初始状态；专家仍读当前状态",
    "no_expert_state_update": "共享和路由专家仅读初始输入；router 仍读当前状态",
    "parallel": "一次学习软融合 T/S，两个专家均读初始输入",
    "shared_only": "仅点级共享专家的功能消融",
    "routed_only": "仅可路由专家的功能消融",
    "soft": "两轮软混合链",
    "weak_mid": "完整模型增加 0.1 权重的中间监督",
    "chain4_balance": "四轮六专家，逐轮 batch 路由均衡辅助损失 0.01",
    "abc_a": "A：清洁数据，原始路由",
    "abc_b": "B：A + 分组路由输入与观测差分",
    "abc_c": "C：B + FP32/低学习率/软到硬预热/z-loss",
    "abc_d": "D：A + 每轮重采样九类训练缺失模式；验证/测试保持与 A 相同",
    "abc_e": "E：D + 向下一轮路由传递上一轮实际选择的专家身份",
    "fixed4_ta_st_s_ta": "四轮六专家固定路径 TA-ST-S-TA；使用 A 在验证集形成的路径作为公平参照",
    "abc_d_static": "D 静态版：九类训练 mask 只生成一次，区分形态多样性与逐轮重采样",
    "abc_f": "F：两轮、三专家 T/S/ST，检验更小链路候选",
    "abc_d_eval_mixed": "D-eval-mixed：训练、验证、测试都使用九类缺失模式，验证/测试固定各自的九类混合 mask",

}


def _unique(values, label: str):
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be nonempty and contain no duplicates")
    return values


def npz_shape(path: Path, cfg: dict) -> dict:
    """Read only the NPY header, avoiding materializing a complete dataset."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        keys = {name[:-4] for name in names if name.endswith(".npy")}
        key = "x_f_gt" if "x_f_gt" in keys else "x_f"
        if key not in keys:
            raise ValueError(f"{path}: missing x_f_gt/x_f")
        with archive.open(f"{key}.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"{path}: unsupported NPY header version {version}")
    if dtype.hasobject or len(shape) != 5 or shape[0] <= 0:
        raise ValueError(f"{path}: expected nonempty numeric 5-D array, got {shape}, {dtype}")
    main = cfg["model"]["main"]
    expected = tuple(int(value) for value in (
        cfg["model"]["c_in"], main["max_t"], main["h"], main["w"],
    ))
    if tuple(shape[1:]) == expected:
        layout = "NCTHW"
    elif tuple(shape[1:]) == (*expected[1:], expected[0]):
        layout = "NTHWC"
    else:
        raise ValueError(
            f"{path}: shape {shape} does not match configured C,T,H,W={expected}; "
            "use a base config matching the existing dataset protocol"
        )
    # Match to_bcthw's actual inference: matching a config alone is insufficient
    # for small NTHWC arrays whose time dimension resembles a channel count.
    if shape[1] <= 8 and (shape[-1] > 8 or shape[1] <= shape[-1]):
        loader_layout = "NCTHW"
    elif shape[-1] <= 8:
        loader_layout = "NTHWC"
    else:
        raise ValueError(f"{path}: the dataset loader cannot infer layout from {shape}")
    if layout != loader_layout:
        raise ValueError(
            f"{path}: config implies {layout}, but the dataset loader infers {loader_layout}; "
            "repack the source into an unambiguous supported layout first"
        )
    return {
        "path": str(path.resolve()), "shape_ncthw": [int(shape[0]), *expected],
        "stored_shape": list(shape), "layout": layout, "keys": sorted(keys),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocols(args: argparse.Namespace, cfg: dict, datasets: dict) -> list[dict]:
    if not args.synthetic:
        for dataset in datasets.values():
            conflicting = set(dataset["keys"]) & {"m_f", "target_mask"}
            if conflicting:
                raise ValueError(
                    f"{dataset['path']}: embedded {sorted(conflicting)} overrides/changes CSV supervision; "
                    "prepare an explicitly reviewed experiment NPZ without these fields first"
                )
    if args.mask_root is None:
        if cfg['data'].get('train_mask_diversity') and cfg['data'].get('eval_mask_diversity'):
            if args.synthetic or args.patterns is not None or args.rates is not None:
                raise ValueError('Generated mixed masks require real NPZs and base-config family/rate settings')
            mixed = {
                'name': 'mixed9_rate0.4' if args.stage in {'route20', 'route20_next', 'coe_validation', 'coe_dual_mask'} else 'generated_mixed',
                'kind': 'generated_diverse',
                'mask_patch': {},
                'description': 'Balanced mixed families; dynamic train masks and fixed independent val/test masks',
            }
            if args.stage == 'coe_dual_mask':
                original_cfg = deep_update(cfg, load_config(ROOT / 'configs/v24/experiments/coe_original_mask.json'))
                legacy = _protocols(args, original_cfg, datasets)[0]
                legacy['name'] = 'original_random_rate0.4'
                legacy['description'] = 'Original random_point CSV masks for train/val/test'
                legacy['data_patch'] = {'train_mask_diversity': None, 'eval_mask_diversity': None}
                mixed['data_patch'] = {}
                return [mixed, legacy]
            return [mixed]
        if args.patterns is not None or args.rates is not None:
            raise ValueError("--patterns/--rates require --mask-root")
        mask = cfg["data"]["mask"]
        csv_paths = {}
        if not args.synthetic:
            pattern = mask.get("pattern", "random")
            if pattern not in {"fixed", "random"}:
                raise ValueError("Legacy loader accepts data.mask.pattern fixed/random only")
            for split in ("train", "val", "test"):
                value = mask.get(f"{split}_csv") or mask.get(f"{pattern}_{split}_csv")
                if value is None:
                    raise ValueError(f"Base config needs an explicit {split} mask CSV")
                path = Path(value).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                csv_paths[f"{split}_csv"] = str(path)
        return [{
            "name": "legacy", "kind": "synthetic_smoke" if args.synthetic else "legacy_csv",
            "description": (
                "Synthetic smoke only; generated masks are node masks broadcast over time"
                if args.synthetic else
                "Existing CSV protocol; random/fixed are loader labels, not proof of random-point masking"
            ),
            "mask_patch": csv_paths,
            "requested_missing_rate": mask.get("missing_rate", mask.get("mask_rate")),
        }]
    if args.synthetic:
        raise ValueError("--mask-root cannot be used with --synthetic: synthetic loader ignores CSV masks")
    patterns = _unique(args.patterns or list(PATTERNS), "patterns")
    rates = _unique(args.rates or [0.4], "rates")
    if any(pattern not in PATTERNS for pattern in patterns):
        raise ValueError(f"patterns must be selected from {PATTERNS}")
    if any(not math.isfinite(rate) or not 0 < rate < 1 for rate in rates):
        raise ValueError("rates must be finite numbers strictly between 0 and 1")
    protocols = []
    for pattern in patterns:
        for rate in rates:
            directory = args.mask_root.resolve() / pattern / format(rate, "g")
            metadata_path = directory / "metadata.json"
            metadata = load_config(metadata_path)
            if metadata.get("schema_version") != 1 or metadata.get("pattern") != pattern:
                raise ValueError(f"{metadata_path}: incompatible metadata schema/pattern")
            recorded_rate = float(metadata.get("requested_missing_rate", float("nan")))
            if not math.isclose(recorded_rate, rate, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{metadata_path}: requested rate mismatch")
            split_metadata = metadata.get("splits", metadata.get("split"))
            if not isinstance(split_metadata, dict):
                raise ValueError(f"{metadata_path}: missing split metadata")
            mask_patch = {"pattern": "random", "missing_rate": rate}
            for split, dataset in datasets.items():
                record = split_metadata.get(split, {})
                shape = dataset["shape_ncthw"]
                csv = directory / f"{split}.csv"
                if Path(record.get("source_npz", "")).resolve() != Path(dataset["path"]):
                    raise ValueError(f"{metadata_path}: {split} source_npz does not match supplied dataset")
                if record.get("shape_ncthw") != shape:
                    raise ValueError(f"{metadata_path}: {split} shape mismatch")
                if record.get("rows") != shape[0] or record.get("columns") != math.prod(shape[2:]):
                    raise ValueError(f"{metadata_path}: {split} expected one row per sample and T*H*W columns")
                if not csv.is_file():
                    raise FileNotFoundError(csv)
                if record.get("sha256") != _sha256(csv):
                    raise ValueError(f"{metadata_path}: {split} CSV SHA-256 mismatch")
                mask_patch[f"{split}_csv"] = str(csv)
            protocols.append({
                "name": f"{pattern}_rate{format(rate, 'g')}", "kind": "structured_csv",
                "pattern": pattern, "requested_missing_rate": rate,
                "metadata_file": str(metadata_path), "metadata": metadata,
                "mask_patch": mask_patch,
            })
    return protocols


def _epochs(args: argparse.Namespace, cfg: dict, variants: list[str]) -> dict:
    reference_epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    if not isinstance(reference_epochs, int) or reference_epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if args.budget_regime == "updates":
        if args.cost_profile is not None:
            raise ValueError("--cost-profile is only used with --budget-regime approx_compute")
        return {name: {"epochs": reference_epochs} for name in variants}
    if args.cost_profile is None:
        raise ValueError("approx_compute requires --cost-profile with measured seconds_per_epoch")
    profile = load_config(args.cost_profile)
    costs = profile.get("seconds_per_epoch", {})
    reference = profile.get("reference_variant", "full")
    if not isinstance(profile.get("context"), dict) or not profile["context"]:
        raise ValueError("cost profile must describe measurement context (device, batch size, dataset)")
    for name in {reference, *variants}:
        value = costs.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"cost profile needs finite positive seconds_per_epoch for {name}")
    target_seconds = reference_epochs * costs[reference]
    budgets = {}
    for name in variants:
        epochs = int(math.floor(target_seconds / costs[name]))
        if epochs < 1:
            raise ValueError(f"Reference budget cannot fit one epoch of {name}; increase --epochs")
        budgets[name] = {
            "epochs": epochs, "reference_variant": reference, "reference_epochs": reference_epochs,
            "target_train_seconds": target_seconds, "measured_seconds_per_epoch": costs[name],
            "estimated_train_seconds": epochs * costs[name],
            "unallocated_seconds": target_seconds - epochs * costs[name],
            "cost_profile": str(args.cost_profile.resolve()), "measurement_context": profile["context"],
        }
    return budgets


def _candidate_calls(cfg: dict) -> dict:
    coe = cfg["model"]["coe"]
    steps, experts = coe["num_steps"], len(coe["expert_pool"])
    mode = coe["routing_mode"]
    routed = bool(coe["use_routed"])
    train_calls = steps * (1 if mode == "fixed" else experts) if routed else 0
    eval_calls = steps * (experts if mode in {"soft", "parallel"} else 1) if routed else 0
    return {
        "train_routed_calls_per_window": train_calls,
        "inference_routed_calls_per_window": eval_calls,
        "shared_calls_per_window": steps if coe["use_shared"] else 0,
        "interpretation": "Call counts only; heterogeneous operators and dispatch overhead make these unequal to FLOPs/time",
    }


def build_plan(args: argparse.Namespace, base_patch: dict | None = None) -> dict:
    cfg = deep_update(load_config(args.base_config), base_patch or {})
    if cfg.get("model", {}).get("architecture") != "v24_ts_coe":
        raise ValueError("base config must select architecture v24_ts_coe")
    if cfg.get("data", {}).get("multiscale", False) or cfg["model"].get("main", {}).get("use_multiscale", False):
        raise ValueError("This experiment plan requires a single-scale base config")
    seeds = _unique(args.seeds, "seeds")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative")
    variants = list(_unique(args.variants or list(STAGES[args.stage]), "variants"))
    if any(name not in STAGES[args.stage] for name in variants):
        raise ValueError(f"--stage {args.stage} variants must be selected from {STAGES[args.stage]}")
    if args.max_plans <= 0:
        raise ValueError("max-plans must be positive")
    datasets = {}
    supplied_paths = [getattr(args, f"{split}_npz") for split in ("train", "val", "test")]
    if args.synthetic:
        if any(path is not None for path in supplied_paths):
            raise ValueError("Choose --synthetic or the three NPZ paths, not both")
        samples = int(cfg["data"]["synthetic"]["num_train"])
    else:
        if any(path is None for path in supplied_paths):
            raise ValueError("Supply --train-npz, --val-npz, --test-npz, or use --synthetic")
        for split in ("train", "val", "test"):
            datasets[split] = npz_shape(getattr(args, f"{split}_npz").resolve(), cfg)
        samples = datasets["train"]["shape_ncthw"][0]
    if samples <= 0 or int(cfg["data"]["batch_size"]) <= 0:
        raise ValueError("Training dataset and batch size must be positive")
    protocols = _protocols(args, cfg, datasets)
    if args.stage == "coe_validation":
        original_cfg = deep_update(
            cfg, load_config(ROOT / "configs/v24/experiments/coe_original_mask.json")
        )
        original_protocols = _protocols(args, original_cfg, datasets)
        for protocol in protocols:
            protocol["variant_scope"] = "mixed"
        for protocol in original_protocols:
            protocol["variant_scope"] = "original"
        protocols.extend(original_protocols)
    count = len(seeds) * sum(
        len(variants) if protocol.get("variant_scope") is None else
        sum(variant != "coe_original_mask" if protocol["variant_scope"] == "mixed" else variant == "coe_original_mask" for variant in variants)
        for protocol in protocols
    )
    if count > args.max_plans:
        raise ValueError(f"Plan has {count} runs, exceeds --max-plans {args.max_plans}; filter explicitly or raise the cap")
    budgets = _epochs(args, cfg, variants)
    common = load_config(ROOT / "configs/v24/experiments/full.json")
    # This stage has a complete authoritative base: legacy two-expert defaults
    # must never overwrite its depth, expert pool, or auxiliary loss.
    if args.stage in {"coe_validation", "coe_dual_mask"}:
        common = {}
    common = deep_update(common, {"data": {"drop_last": False}, "train": {"early_stopping": {"enabled": False}}})
    output = args.output_dir.resolve()
    runs = []
    for protocol in protocols:
        for variant in variants:
            if protocol.get("variant_scope") == "mixed" and variant == "coe_original_mask":
                continue
            if protocol.get("variant_scope") == "original" and variant != "coe_original_mask":
                continue
            patch_path = (
                ROOT / "configs/v24/candidates/ablation_grid" / f"{variant}.json"
                if args.stage == "depth" else ROOT / "configs/v24/experiments" / f"{variant}.json"
            )
            variant_cfg = deep_update(deep_update(cfg, common), load_config(patch_path))
            for seed in seeds:
                name = f"v24_{args.stage}_{variant}_{protocol['name']}_{args.budget_regime}_seed{seed}"
                config = deep_update(variant_cfg, {
                    "seed": seed, "output_dir": str(output / "runs"),
                    "data": {"mask": protocol["mask_patch"]},
                    "train": {"epochs": budgets[variant]["epochs"]},
                })
                if protocol.get("data_patch"):
                    config["data"] = deep_update(config["data"], protocol["data_patch"])
                config["experiment_plan"] = {
                    "stage": args.stage, "variant": variant, "protocol": protocol["name"],
                    "protocol_kind": protocol["kind"], "budget_regime": args.budget_regime,
                    "mask_metadata": protocol.get("metadata_file"),
                    "training_mask_source": ("dynamic_diverse" if config['data'].get('train_mask_diversity') else "protocol"),
                    "evaluation_mask_source": ("diverse_fixed_per_split" if config['data'].get('eval_mask_diversity') else "protocol"),
                }
                path = output / "configs" / f"{name}.json"
                command = [args.python, str(ROOT / "scripts/train.py"), "-c", str(path), "--no_plot", "-n", name]
                if args.synthetic:
                    command.append("--synthetic")
                else:
                    for split in ("train", "val", "test"):
                        command.extend([f"--{split}_npz", datasets[split]["path"]])
                runs.append({
                    "name": name, "stage": args.stage, "variant": variant, "seed": seed,
                    "question": QUESTIONS.get(variant, "固定轮数比较专家池；固定专家池比较轮数"),
                    "protocol": protocol["name"], "config_path": str(path), "config": config,
                    "budget": {**budgets[variant], "regime": args.budget_regime,
                               "expected_update_slots": math.ceil(samples / int(config["data"]["batch_size"])) * budgets[variant]["epochs"],
                               "actual_updates_note": "Empty-supervision batches are skipped; record actual optimizer updates during training"},
                    "candidate_compute": _candidate_calls(config),
                    "command_argv": command, "command": shlex.join(command),
                })
    return {
        "schema_version": 1, "status": "planned_not_executed", "repo_root": str(ROOT),
        "base_config": str(args.base_config.resolve()), "stage": args.stage,
        "seeds": seeds, "paired_seeds": len(seeds) >= 3, "budget_regime": args.budget_regime,
        "num_runs": len(runs), "datasets": datasets, "protocols": protocols,
        "comparison_policy": {
            "same_dataset_and_masks_across_variants": len({json.dumps(r['config']['data'].get('train_mask_diversity'), sort_keys=True) for r in runs}) == 1 and len({json.dumps(r['config']['data'].get('eval_mask_diversity'), sort_keys=True) for r in runs}) == 1,
            "same_evaluation_masks_across_variants": len({
                json.dumps(r['config']['data'].get('eval_mask_diversity'), sort_keys=True)
                for r in runs
            }) == 1,
            "evaluation_mask_intervention": {
                r['variant']: r['config']['data']['eval_mask_diversity']
                for r in runs if 'eval_mask_diversity' in r['config']['data']
            },
            "training_mask_intervention": {
                r['variant']: r['config']['data']['train_mask_diversity'] for r in runs
                if 'train_mask_diversity' in r['config']['data']
            },
            "drop_last": False, "early_stopping": False,
            "route_balance_enabled_by_default": any(
                run["config"]["loss"].get("lambda_coe_balance", 0) > 0 for run in runs
            ),
            "mask_pairing_note": "Evaluation CSVs are shared unless evaluation_mask_intervention is present. Explicit training/evaluation mask interventions replace CSV masks only for that variant; metrics from different evaluation sources must not be compared as a single paired test.",
            "metrics": "MAE/RMSE over effective hidden targets; report original units only when scaling metadata is known",
            "selection": "Choose checkpoints and fixed-path baseline using validation data; test data is for final evaluation",
        },
        "runs": runs,
    }


def write_plan(manifest: dict, directory: Path) -> None:
    """Preflight all files before writing; do not overwrite a prior experiment plan."""
    directory = directory.resolve()
    targets = [directory / "manifest.json", directory / "commands.sh"]
    targets.extend(Path(run["config_path"]) for run in manifest["runs"])
    if any(path.exists() for path in targets):
        raise FileExistsError("Experiment plan files already exist; choose a new output directory")
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}", ""]
    for run in manifest["runs"]:
        path = Path(run["config_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run["config"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    commands.append(shlex.join([
        manifest["runs"][0]["command_argv"][0], str(ROOT / "scripts/v24/run_experiments.py"),
        "--plan", str(directory / "manifest.json"),
    ]))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="core")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--synthetic", action="store_true")
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-npz", type=Path)
    parser.add_argument("--mask-root", type=Path)
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS)
    parser.add_argument("--rates", nargs="+", type=float)
    parser.add_argument("--budget-regime", choices=("updates", "approx_compute"), default="updates")
    parser.add_argument("--cost-profile", type=Path)
    parser.add_argument("--epochs", type=int, help="Common epochs, or reference epochs for approx_compute")
    parser.add_argument("--max-plans", type=int, default=120)
    parser.add_argument("--python", default=sys.executable, help="Python executable written into commands; never launched by this script")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        manifest = build_plan(args)
        write_plan(manifest, args.output_dir)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        raise SystemExit(f"Experiment plan error: {error}") from error
    print(f"Generated {manifest['num_runs']} configs; training was not executed. Manifest: {args.output_dir.resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
