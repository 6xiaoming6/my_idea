from __future__ import annotations

import json

from stmoe_imputer.config import deep_update

from _v14_utils import ROOT, compact_v14_config


def compact_v20_config(
    channels: int = 2,
    time_steps: int = 3,
    height: int = 16,
    width: int = 16,
) -> dict:
    cfg = compact_v14_config(channels, time_steps, height, width)
    patch = json.loads(
        (ROOT / "configs/v20-single/smoke.json").read_text(encoding="utf-8")
    )
    cfg = deep_update(cfg, patch)
    return deep_update(cfg, {
        "device": "cpu",
        "data": {
            "synthetic": {"t": time_steps, "h": height, "w": width},
            "batch_size": 1,
        },
        "model": {
            "c_in": channels,
            "main": {
                "dim": 8,
                "num_experts": 2,
                "top_k": 1,
                "max_t": time_steps,
                "h": height,
                "w": width,
                "num_groups": 2,
                "dropout": 0.0,
                "route_dropout": 0.0,
            },
            "v14": {
                "refiner_hidden": 8,
                "prediction_embed_dim": 4,
                "correction_hidden": 4,
                "refiner_dropout": 0.0,
                "difficulty_hidden": 8,
                "difficulty_out_dim": 8,
                "controller_hidden": 16,
                "controller_dropout": 0.0,
            },
            "v20": {
                "probe_min_count": 2,
                "probe_max_count": 16,
                "probe_min_remaining": 4,
            },
        },
        "train": {
            "amp": False,
            "lr_main": 1e-3,
            "lr_v14": 1e-2,
            "lr_v20": 1e-2,
        },
    })
