"""Shared helpers for the formal V17.2 experiment pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "v17.2-single"
CONFIG_ROOT = ROOT / "configs" / "v17.2-single"
RATES = ("0.2", "0.4", "0.6", "0.8")

DATASETS = {
    "TaxiBJ": {
        "base": "configs/datasets/taxibj.json",
        "v17": "configs/v17-single/taxibj.json",
        "v17_2": "configs/v17.2-single/taxibj.json",
        "train": "data/TaxiBJ/taxibj_train.npz",
        "val": "data/TaxiBJ/taxibj_val.npz",
        "test": "data/TaxiBJ/taxibj_test.npz",
        "mask_root": "data/TaxiBJ",
        "output_name": "TaxiBJ",
    },
    "BikeNYC": {
        "base": "configs/datasets/bikenyc.json",
        "v17": "configs/v17-single/bikenyc.json",
        "v17_2": "configs/v17.2-single/bikenyc.json",
        "train": "data/BikeNYC/bikenyc_train.npz",
        "val": "data/BikeNYC/bikenyc_val.npz",
        "test": "data/BikeNYC/bikenyc_test.npz",
        "mask_root": "data/BikeNYC",
        "output_name": "BikeNYC",
    },
    "CHAP": {
        "base": "configs/datasets/chap_beijing.json",
        "v17": "configs/v17-single/chap.json",
        "v17_2": "configs/v17.2-single/chap.json",
        "train": "data/CHAP/beijing/chap_beijing_train.npz",
        "val": "data/CHAP/beijing/chap_beijing_val.npz",
        "test": "data/CHAP/beijing/chap_beijing_test.npz",
        "mask_root": "data/CHAP/beijing",
        "output_name": "CHAP_Beijing",
    },
}


@dataclass(frozen=True)
class Job:
    dataset: str
    mask: str
    rate: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.dataset} {self.mask}@{self.rate} seed={self.seed}"

    @property
    def output_dataset(self) -> str:
        return str(DATASETS[self.dataset]["output_name"])

    @property
    def root(self) -> Path:
        return (
            OUTPUT_ROOT
            / self.output_dataset
            / "full"
            / "model"
            / self.mask
            / f"rate{float(self.rate):g}"
        )


CORE_POINTS = {
    "P1": ("TaxiBJ", "random", "0.4"),
    "P2": ("BikeNYC", "fixed", "0.8"),
    "P3": ("TaxiBJ", "fixed", "0.4"),
    "P4": ("CHAP", "fixed", "0.4"),
}


def baseline_key(job: Job) -> str:
    return f"{job.dataset}/{job.mask}/{float(job.rate):g}/seed{job.seed}"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def deep_update(base: dict, patch: dict) -> dict:
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def config_sha256(config: dict) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mask_paths(dataset: str, mask: str, rate: str) -> dict[str, str]:
    spec = DATASETS[dataset]
    root = ROOT / str(spec["mask_root"]) / f"{mask}_mask" / rate
    return {
        split: str((root / f"{split}.csv").relative_to(ROOT))
        for split in ("train", "val", "test")
    }


def _apply_mask_policy(patch: dict, mask: str) -> dict:
    resolved = deepcopy(patch)
    policies = resolved.pop("train_policy_by_mask", {})
    policy = policies.get(mask, {})
    if policy:
        resolved["train"] = deep_update(resolved.get("train", {}), policy)
    return resolved


def build_resolved_config(
    dataset: str,
    mask: str,
    rate: str,
    seed: int,
    *,
    version: str,
    epochs: int | None = None,
) -> dict:
    if version not in {"full", "e1", "v17_2"}:
        raise ValueError(f"Unsupported protocol version: {version}")
    spec = DATASETS[dataset]
    base = load_json(ROOT / str(spec["base"]))
    model_key = "v17_2" if version == "v17_2" else "v17"
    patch = _apply_mask_policy(load_json(ROOT / str(spec[model_key])), mask)

    if version == "full":
        patch = deep_update(
            patch,
            {
                "output_dir": "outputs/v17.1-single",
                "model": {"version": "v17.1-single"},
                "experiment": {
                    "group": "v17_exploratory_ablation",
                    "variant": "full",
                },
            },
        )
    elif version == "e1":
        patch = deep_update(
            patch,
            {
                "output_dir": "outputs/v17.1-single",
                "model": {
                    "version": "v17.1-single",
                    "v17": {"adapter_enabled": False},
                },
                "experiment": {
                    "group": "v17_exploratory_ablation",
                    "variant": "no_adapter",
                },
            },
        )
    else:
        patch = deep_update(
            patch,
            {
                "experiment": {
                    "group": "v17_2_formal",
                    "variant": "adapter_free_full",
                }
            },
        )

    if epochs is not None:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        patch = deep_update(
            patch,
            {
                "train": {
                    "epochs": epochs,
                    "val_epoch": min(epochs, 5),
                    "early_stopping": {"enabled": False},
                }
            },
        )

    paths = mask_paths(dataset, mask, rate)
    patch = deep_update(
        patch,
        {
            "seed": seed,
            "data": {
                "mask": {
                    "pattern": mask,
                    "missing_rate": float(rate),
                    "train_csv": paths["train"],
                    "val_csv": paths["val"],
                    "test_csv": paths["test"],
                }
            },
        },
    )
    return deep_update(base, patch)


_V17_DEFAULTS = {
    "expert_router_mode": "hierarchical_shared_head",
    "fine_floor_mode": "linear",
}


def _semantic_config(config: dict) -> dict:
    resolved = deepcopy(config)
    v17 = resolved.setdefault("model", {}).setdefault("v17", {})
    for key, value in _V17_DEFAULTS.items():
        v17.setdefault(key, value)
    return resolved


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(flatten(child, path))
    return flattened


ALLOWED_DIFF_PREFIXES = (
    "output_dir",
    "model.version",
    "model.architecture",
    "model.v17.adapter_enabled",
    "model.v17_2",
    "experiment",
    "run_name",
    "timestamp",
    "git_metadata",
)


def protocol_differences(left: dict, right: dict) -> list[dict[str, Any]]:
    left_flat = flatten(_semantic_config(left))
    right_flat = flatten(_semantic_config(right))
    differences = []
    for path in sorted(set(left_flat) | set(right_flat)):
        left_value = left_flat.get(path, "<missing>")
        right_value = right_flat.get(path, "<missing>")
        if left_value == right_value:
            continue
        allowed = any(
            path == prefix or path.startswith(prefix + ".")
            for prefix in ALLOWED_DIFF_PREFIXES
        )
        differences.append(
            {
                "path": path,
                "baseline": left_value,
                "candidate": right_value,
                "allowed": allowed,
            }
        )
    return differences


def build_protocol_audit(
    full: dict,
    e1: dict,
    candidate: dict,
) -> dict[str, Any]:
    full_e1 = protocol_differences(full, e1)
    e1_candidate = protocol_differences(e1, candidate)
    unexpected = [
        {"comparison": "full_vs_e1", **difference}
        for difference in full_e1
        if not difference["allowed"]
    ]
    unexpected.extend(
        {"comparison": "e1_vs_v17_2", **difference}
        for difference in e1_candidate
        if not difference["allowed"]
    )
    contract = {
        "full_adapter_enabled": full.get("model", {})
        .get("v17", {})
        .get("adapter_enabled", True),
        "e1_adapter_enabled": e1.get("model", {})
        .get("v17", {})
        .get("adapter_enabled", True),
        "v17_2_adapter_enabled": candidate.get("model", {})
        .get("v17", {})
        .get("adapter_enabled", True),
        "v17_2_architecture": candidate.get("model", {}).get("architecture"),
    }
    contract_passed = (
        contract["full_adapter_enabled"] is True
        and contract["e1_adapter_enabled"] is False
        and contract["v17_2_adapter_enabled"] is False
        and contract["v17_2_architecture"]
        == "v17_2_no_adapter_hierarchical_scale_moe"
    )
    return {
        "passed": not unexpected and contract_passed,
        "allowed_diff_prefixes": list(ALLOWED_DIFF_PREFIXES),
        "config_sha256": {
            "full": config_sha256(full),
            "e1": config_sha256(e1),
            "v17_2": config_sha256(candidate),
        },
        "contract": contract,
        "contract_passed": contract_passed,
        "full_vs_e1": full_e1,
        "e1_vs_v17_2": e1_candidate,
        "unexpected_differences": unexpected,
    }


def read_records(path: Path) -> list[dict]:
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    except (OSError, json.JSONDecodeError):
        return []
    return records


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_result(run_dir: Path) -> dict | None:
    records = read_records(run_dir / "logs" / "metrics.jsonl")
    test = next(
        (record for record in reversed(records) if record.get("stage") == "test"),
        None,
    )
    if test is None:
        return None
    metrics = test.get("metrics") or {}
    parsed = {key: finite(metrics.get(key)) for key in ("mae", "rmse", "loss")}
    if any(value is None for value in parsed.values()):
        return None
    epochs = [record for record in records if "epoch" in record]
    return {
        **parsed,
        "mape": finite(metrics.get("mape")),
        "extra": test.get("extra") or {},
        "completed_epochs": len(epochs),
        "total_time_sec": sum(
            finite(record.get("perf", {}).get("epoch_time_sec")) or 0.0
            for record in epochs
        ),
        "peak_memory_gb": max(
            (
                finite(record.get("perf", {}).get("peak_memory_gb")) or 0.0
                for record in epochs
            ),
            default=0.0,
        ),
    }


def completed_run(job: Job) -> Path | None:
    if not job.root.is_dir():
        return None
    completed = []
    for config_path in job.root.glob("*/config.json"):
        run_dir = config_path.parent
        try:
            config = load_json(config_path)
            mask = config.get("data", {}).get("mask", {})
            matches = (
                config.get("model", {}).get("version") == "v17.2-single"
                and config.get("model", {}).get("architecture")
                == "v17_2_no_adapter_hierarchical_scale_moe"
                and config.get("model", {})
                .get("v17", {})
                .get("adapter_enabled")
                is False
                and config.get("model", {})
                .get("v17_2", {})
                .get("remove_scale_adapter")
                is True
                and int(config.get("seed", -1)) == job.seed
                and mask.get("pattern") == job.mask
                and math.isclose(
                    float(mask.get("missing_rate", -1)),
                    float(job.rate),
                    abs_tol=1e-9,
                )
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            matches = False
        required = (
            run_dir / "checkpoints" / "best.pt",
            run_dir / "logs" / "train.log",
            run_dir / "logs" / "val.log",
            run_dir / "logs" / "test.log",
            run_dir / "logs" / "metrics.jsonl",
            run_dir / "router_diagnostics.json",
            run_dir / "parameter_report.json",
            run_dir / "protocol_audit.json",
            run_dir / "git_metadata.json",
        )
        if (
            matches
            and all(path.is_file() and path.stat().st_size > 0 for path in required)
            and load_json(run_dir / "protocol_audit.json").get("passed") is True
            and load_json(run_dir / "parameter_report.json").get(
                "adapter_parameter_count"
            )
            == 0
            and read_result(run_dir) is not None
        ):
            completed.append(run_dir)
    return max(completed, key=lambda path: path.stat().st_mtime) if completed else None
