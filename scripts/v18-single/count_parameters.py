#!/usr/bin/env python3
"""Count V18 parameters by major module without loading experiment data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from pipeline_common import (
    DATASETS,
    ROOT,
    build_resolved_config,
    deep_update,
    load_json,
)


def _group(name: str) -> str:
    if name.startswith("main_branch.main_backbone."):
        return "main_backbone"
    if any(
        token in name
        for token in (
            ".residual_pyramid.",
            ".direction_pyramid.",
            ".refiner.",
        )
    ):
        return "residual_pyramid"
    if any(token in name for token in (".controller.", ".budget_controller.")):
        return "budget_controller"
    if ".condition_encoder." in name:
        return "difficulty_encoder"
    return "other"


def count(dataset: str) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from stmoe_imputer.models import DualBranchSTImputer

    config = build_resolved_config(dataset, "fixed", "0.4", 42)
    model = DualBranchSTImputer.from_config(config)
    spec = DATASETS[dataset]
    v14_config = deep_update(
        load_json(ROOT / spec["base"]),
        load_json(ROOT / spec["v14_model"]),
    )
    v14_model = DualBranchSTImputer.from_config(v14_config)
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "trainable": 0}
    )
    for name, parameter in model.named_parameters():
        group = _group(name)
        grouped[group]["total"] += parameter.numel()
        if parameter.requires_grad:
            grouped[group]["trainable"] += parameter.numel()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    v14_total = sum(parameter.numel() for parameter in v14_model.parameters())
    v18_main = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("main_branch.main_backbone.")
    )
    v14_main = sum(
        parameter.numel()
        for name, parameter in v14_model.named_parameters()
        if name.startswith("main_branch.main_backbone.")
    )
    return {
        "dataset": dataset,
        "architecture": config["model"]["architecture"],
        "total_parameters": total,
        "trainable_parameters": trainable,
        "v14_total_parameters": v14_total,
        "v18_to_v14_ratio": total / max(v14_total, 1),
        "main_backbone_parameters": v18_main,
        "v14_main_backbone_parameters": v14_main,
        "main_backbone_identical": v18_main == v14_main,
        "modules": dict(sorted(grouped.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=("all", *DATASETS), default=["all"]
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    datasets = list(DATASETS) if "all" in args.datasets else list(
        dict.fromkeys(args.datasets)
    )
    reports = [count(dataset) for dataset in datasets]
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return
    for report in reports:
        print(
            f"{report['dataset']}: total={report['total_parameters']:,} "
            f"trainable={report['trainable_parameters']:,} "
            f"v14_total={report['v14_total_parameters']:,} "
            f"ratio={report['v18_to_v14_ratio']:.4f} "
            f"main_identical={report['main_backbone_identical']}"
        )
        for name, values in report["modules"].items():
            print(
                f"  {name}: total={values['total']:,} "
                f"trainable={values['trainable']:,}"
            )


if __name__ == "__main__":
    main()
