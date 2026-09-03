#!/usr/bin/env python3
"""Train one V20 dataset/mask/rate combination through the stable V14 launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/v14-single/train.py"
SPEC = importlib.util.spec_from_file_location("v20_shared_train_launcher", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load shared launcher: {SOURCE}")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)

for dataset, config_name in (
    ("TaxiBJ", "taxibj.json"),
    ("BikeNYC", "bikenyc.json"),
    ("CHAP", "chap.json"),
):
    LAUNCHER.DATASETS[dataset]["model"] = f"configs/v20-single/{config_name}"

LAUNCHER.ABLATIONS = {
    "random_exam_only": "random_exam_only.json",
    "geometry_exam_only": "geometry_exam_only.json",
    "random_hybrid": "random_hybrid.json",
    "geometry_hybrid": "geometry_hybrid.json",
    "geometry_prior_only": "geometry_prior_only.json",
    "legacy_geometry_hybrid": "legacy_geometry_hybrid.json",
}


if __name__ == "__main__":
    LAUNCHER.main()
