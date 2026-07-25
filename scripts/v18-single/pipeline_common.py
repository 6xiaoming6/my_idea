"""Shared helpers for the V18 BARP-MoE experiment pipeline."""

from __future__ import annotations

import json
import hashlib
import math
import sys
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from experiment_fingerprint import source_tree_sha256


OUTPUT_ROOT = ROOT / "outputs" / "v18-single"
CONFIG_ROOT = ROOT / "configs" / "v18-single"
RATES = ("0.2", "0.4", "0.6", "0.8")
PATTERNS = ("fixed", "random")

DATASETS: dict[str, dict[str, str]] = {
    "TaxiBJ": {
        "base": "configs/datasets/taxibj.json",
        "model": "configs/v18-single/taxibj.json",
        "v14_model": "configs/v14-single/taxibj.json",
        "train": "data/TaxiBJ/taxibj_train.npz",
        "val": "data/TaxiBJ/taxibj_val.npz",
        "test": "data/TaxiBJ/taxibj_test.npz",
        "mask_root": "data/TaxiBJ",
        "output_name": "TaxiBJ",
    },
    "BikeNYC": {
        "base": "configs/datasets/bikenyc.json",
        "model": "configs/v18-single/bikenyc.json",
        "v14_model": "configs/v14-single/bikenyc.json",
        "train": "data/BikeNYC/bikenyc_train.npz",
        "val": "data/BikeNYC/bikenyc_val.npz",
        "test": "data/BikeNYC/bikenyc_test.npz",
        "mask_root": "data/BikeNYC",
        "output_name": "BikeNYC",
    },
    "CHAP": {
        "base": "configs/datasets/chap_beijing.json",
        "model": "configs/v18-single/chap.json",
        "v14_model": "configs/v14-single/chap.json",
        "train": "data/CHAP/beijing/chap_beijing_train.npz",
        "val": "data/CHAP/beijing/chap_beijing_val.npz",
        "test": "data/CHAP/beijing/chap_beijing_test.npz",
        "mask_root": "data/CHAP/beijing",
        "output_name": "CHAP_Beijing",
    },
}

SCREENING_POINTS = (
    ("TaxiBJ", "fixed", "0.2"),
    ("TaxiBJ", "fixed", "0.4"),
    ("TaxiBJ", "random", "0.4"),
    ("TaxiBJ", "random", "0.8"),
    ("BikeNYC", "random", "0.4"),
    ("CHAP", "fixed", "0.4"),
)

CORE_POINTS = {
    "P1": ("TaxiBJ", "fixed", "0.2"),
    "P2": ("TaxiBJ", "random", "0.4"),
    "P3": ("BikeNYC", "random", "0.4"),
    "P4": ("CHAP", "fixed", "0.4"),
}

ABLATIONS = {
    "absolute_c2f": "absolute_c2f.json",
    "unbounded_residual": "unbounded_residual.json",
    "no_observed_utility": "no_observed_utility.json",
    "no_reliability_filtering": "no_reliability_filtering.json",
    "fine_only_residual": "fine_only_residual.json",
    "fixed_budget": "fixed_budget.json",
    "no_sample_regret": "no_sample_regret.json",
}


@dataclass(frozen=True)
class Job:
    dataset: str
    mask: str
    rate: str
    seed: int
    ablation: str = "none"

    @property
    def label(self) -> str:
        suffix = "" if self.ablation == "none" else f" ablation={self.ablation}"
        return f"{self.dataset} {self.mask}@{self.rate} seed={self.seed}{suffix}"

    @property
    def output_dataset(self) -> str:
        return DATASETS[self.dataset]["output_name"]

    @property
    def experiment_parts(self) -> tuple[str, str]:
        if self.ablation == "none":
            return "full", "model"
        return "ablation", f"v18_{self.ablation}"

    @property
    def root(self) -> Path:
        experiment_type, variant = self.experiment_parts
        return (
            OUTPUT_ROOT
            / self.output_dataset
            / experiment_type
            / variant
            / self.mask
            / f"rate{float(self.rate):g}"
        )


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def deep_update(base: dict, patch: dict) -> dict:
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _apply_mask_policy(patch: dict, mask: str) -> dict:
    resolved = deepcopy(patch)
    policies = resolved.pop("train_policy_by_mask", {})
    policy = policies.get(mask, {}) if isinstance(policies, dict) else {}
    if policy:
        resolved["train"] = deep_update(resolved.get("train", {}), policy)
    return resolved


def mask_paths(dataset: str, mask: str, rate: str) -> dict[str, str]:
    spec = DATASETS[dataset]
    root = ROOT / spec["mask_root"] / f"{mask}_mask" / rate
    return {
        split: str((root / f"{split}.csv").relative_to(ROOT))
        for split in ("train", "val", "test")
    }


@lru_cache(maxsize=None)
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_resolved_config(
    dataset: str,
    mask: str,
    rate: str,
    seed: int,
    *,
    ablation: str = "none",
    smoke: bool = False,
    epochs: int | None = None,
    training_policy: Path | None = None,
) -> dict:
    spec = DATASETS[dataset]
    config = load_json(ROOT / spec["base"])
    patch = _apply_mask_policy(load_json(ROOT / spec["model"]), mask)
    config = deep_update(config, patch)
    if smoke:
        config = deep_update(config, load_json(CONFIG_ROOT / "smoke.json"))
        # Preserve the dataset's validated multiscale protocol.
        scale_mode = load_json(ROOT / spec["base"])["model"]["main"]["scale_mode"]
        config["model"]["main"]["scale_mode"] = scale_mode
    if ablation != "none":
        try:
            filename = ABLATIONS[ablation]
        except KeyError as error:
            raise ValueError(f"Unsupported V18 ablation: {ablation}") from error
        config = deep_update(config, load_json(CONFIG_ROOT / "ablations" / filename))
    if training_policy is not None:
        policy = load_json(training_policy)
        datasets = policy.get("datasets")
        if not isinstance(datasets, dict) or dataset not in datasets:
            raise ValueError(f"Policy {training_policy} has no datasets.{dataset}")
        config = deep_update(config, datasets[dataset])
        config["experiment_policy"] = {
            "name": policy.get("name", training_policy.stem),
            "source": str(training_policy),
        }
    if epochs is not None:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        config = deep_update(
            config,
            {
                "train": {
                    "epochs": epochs,
                    "val_epoch": 1 if epochs <= 2 else min(
                        epochs, int(config["train"].get("val_epoch", 1))
                    ),
                    "early_stopping": {"enabled": False},
                }
            },
        )
    config = deep_update(
        config,
        {
            "seed": seed,
            "data": {
                "mask": {
                    "pattern": mask,
                    "missing_rate": float(rate),
                    **mask_paths(dataset, mask, rate),
                }
            },
            "experiment": {
                "group": "v18_barp_moe",
                "variant": "full" if ablation == "none" else ablation,
            },
        },
    )
    # Public dataset config expects explicit CSV field names.
    paths = config["data"]["mask"]
    for split in ("train", "val", "test"):
        paths[f"{split}_csv"] = paths.pop(split)
    fingerprints = {
        split: file_sha256(ROOT / paths[f"{split}_csv"])
        for split in ("train", "val", "test")
        if (ROOT / paths[f"{split}_csv"]).is_file()
    }
    if len(fingerprints) == 3:
        paths["sha256"] = fingerprints
    data_fingerprints = {
        split: file_sha256(ROOT / spec[split])
        for split in ("train", "val", "test")
        if (ROOT / spec[split]).is_file()
    }
    if len(data_fingerprints) == 3:
        config["data"]["sha256"] = data_fingerprints
    config["reproducibility"] = {
        "source_sha256": source_tree_sha256(ROOT),
    }
    return config


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_records(path: Path) -> list[dict]:
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return records


def read_result(run_dir: Path) -> dict | None:
    records = read_records(run_dir / "logs" / "metrics.jsonl")
    test = next(
        (record for record in reversed(records) if record.get("stage") == "test"),
        None,
    )
    if test is None:
        return None
    metrics = test.get("metrics", {})
    parsed = {key: _finite(metrics.get(key)) for key in ("mae", "rmse", "loss")}
    if any(value is None for value in parsed.values()):
        return None
    epochs = [record for record in records if isinstance(record.get("epoch"), int)]
    return {
        **parsed,
        "mape": _finite(metrics.get("mape")),
        "extra": test.get("extra") or {},
        "completed_epochs": len(epochs),
        "total_time_sec": sum(
            _finite(record.get("perf", {}).get("epoch_time_sec")) or 0.0
            for record in epochs
        ),
        "peak_memory_gb": max(
            (
                _finite(record.get("perf", {}).get("peak_memory_gb")) or 0.0
                for record in epochs
            ),
            default=0.0,
        ),
    }


def _formal_signature(config: dict) -> str:
    """Canonical fields that determine whether a paper run is reusable."""
    data = config.get("data", {})
    model = config.get("model", {})
    payload = {
        "seed": config.get("seed"),
        "data": {
            "dataset_name": data.get("dataset_name"),
            "batch_size": data.get("batch_size"),
            "drop_last": data.get("drop_last"),
            "scales": data.get("scales"),
            "mask": data.get("mask"),
            "sha256": data.get("sha256"),
        },
        "model": model,
        "loss": config.get("loss"),
        "train": config.get("train"),
        "evaluation": config.get("evaluation"),
        "experiment": config.get("experiment"),
        "reproducibility": config.get("reproducibility"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def comparison_protocol_signature(config: dict) -> str:
    """Protocol fields that must match between V14 and V18 comparisons."""
    data = config.get("data", {})
    model = config.get("model", {})
    loss = config.get("loss", {})
    train = config.get("train", {})
    early = train.get("early_stopping", {})
    early_signature = {"enabled": bool(early.get("enabled", False))}
    if early_signature["enabled"]:
        early_signature.update(
            {
                "monitor": early.get("monitor", "val_mae"),
                "patience": int(early.get("patience", 20)),
                "mode": early.get("mode", "min"),
            }
        )
    payload = {
        "seed": config.get("seed"),
        "data": {
            "dataset_name": data.get("dataset_name"),
            "batch_size": data.get("batch_size"),
            "drop_last": data.get("drop_last"),
            "scales": data.get("scales"),
            "mask": data.get("mask"),
            "sha256": data.get("sha256"),
        },
        "main": model.get("main"),
        "aux": model.get("aux"),
        "main_loss": {
            key: value
            for key, value in loss.items()
            if not key.startswith("lambda_v14")
            and not key.startswith("lambda_v18")
        },
        "train": {
            key: train.get(key)
            for key in (
                "epochs",
                "val_epoch",
                "lr_main",
                "weight_decay",
                "grad_clip_norm",
                "amp",
                "aux_loss_warmup_epochs",
                "scheduler",
            )
        },
        "early_stopping": early_signature,
        "evaluation": config.get("evaluation"),
        "reproducibility": config.get("reproducibility"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _finished_normally(run_dir: Path) -> bool:
    try:
        text = (run_dir / "logs" / "train.log").read_text(
            encoding="utf-8"
        )
    except OSError:
        return False
    return "Training finished normally:" in text


def _checkpoint_is_readable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0 and zipfile.is_zipfile(
            path
        )
    except OSError:
        return False


def completed_run(
    job: Job,
    expected_config: dict | None = None,
) -> Path | None:
    if not job.root.is_dir():
        return None
    if expected_config is None:
        expected_config = build_resolved_config(
            job.dataset,
            job.mask,
            job.rate,
            job.seed,
            ablation=job.ablation,
        )
    expected_signature = _formal_signature(expected_config)
    completed = []
    for config_path in job.root.glob("*/config.json"):
        run_dir = config_path.parent
        try:
            config = load_json(config_path)
            mask = config.get("data", {}).get("mask", {})
            matches = (
                config.get("model", {}).get("version") == "v18-single"
                and config.get("model", {}).get("architecture")
                == "v18_base_anchored_residual_moe"
                and int(config.get("seed", -1)) == job.seed
                and mask.get("pattern") == job.mask
                and math.isclose(
                    float(mask.get("missing_rate", -1)),
                    float(job.rate),
                    abs_tol=1e-9,
                )
                and _formal_signature(config) == expected_signature
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            matches = False
        required = (
            run_dir / "checkpoints" / "best.pt",
            run_dir / "logs" / "train.log",
            run_dir / "logs" / "val.log",
            run_dir / "logs" / "test.log",
            run_dir / "logs" / "metrics.jsonl",
        )
        if (
            matches
            and all(path.is_file() and path.stat().st_size > 0 for path in required)
            and _checkpoint_is_readable(run_dir / "checkpoints" / "best.pt")
            and _finished_normally(run_dir)
            and read_result(run_dir) is not None
        ):
            completed.append(run_dir)
    return max(completed, key=lambda path: path.stat().st_mtime) if completed else None


def compatible_v14_run(
    job: Job,
    candidate_config: dict | None = None,
) -> tuple[Path | None, list[Path]]:
    """Find a finished V14 run with the exact V18 comparison protocol."""
    if candidate_config is None:
        candidate_config = build_resolved_config(
            job.dataset,
            job.mask,
            job.rate,
            job.seed,
            ablation=job.ablation,
        )
    expected_protocol = comparison_protocol_signature(candidate_config)
    root = (
        ROOT
        / "outputs"
        / "v14-single"
        / DATASETS[job.dataset]["output_name"]
        / "full"
        / "model"
        / job.mask
        / f"rate{float(job.rate):g}"
    )
    if not root.is_dir():
        return None, []
    compatible = []
    incompatible = []
    for config_path in root.glob("*/config.json"):
        run_dir = config_path.parent
        try:
            config = load_json(config_path)
            mask = config.get("data", {}).get("mask", {})
            matches_job = (
                config.get("model", {}).get("version") == "v14-single"
                and int(config.get("seed", -1)) == job.seed
                and mask.get("pattern") == job.mask
                and math.isclose(
                    float(mask.get("missing_rate", -1)),
                    float(job.rate),
                    abs_tol=1e-9,
                )
            )
            complete = (
                _finished_normally(run_dir)
                and read_result(run_dir) is not None
                and _checkpoint_is_readable(
                    run_dir / "checkpoints" / "best.pt"
                )
            )
            matches_protocol = (
                comparison_protocol_signature(config) == expected_protocol
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            matches_job = complete = matches_protocol = False
        if matches_job and complete and matches_protocol:
            compatible.append(run_dir)
        elif matches_job and complete:
            incompatible.append(run_dir)
    latest = (
        max(compatible, key=lambda path: path.stat().st_mtime)
        if compatible
        else None
    )
    return latest, incompatible


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else ROOT / path
