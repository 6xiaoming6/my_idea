from __future__ import annotations

import json
from pathlib import Path

from stmoe_imputer.config import deep_update

from _v14_utils import make_batch


ROOT = Path(__file__).resolve().parents[1]


def compact_v19_config(
    channels: int = 2,
    time_steps: int = 3,
    height: int = 16,
    width: int = 16,
) -> dict:
    base = json.loads(
        (ROOT / "configs/presets/smoke.json").read_text(encoding="utf-8")
    )
    v19 = json.loads(
        (ROOT / "configs/v19-single/smoke.json").read_text(encoding="utf-8")
    )
    cfg = deep_update(base, v19)
    return deep_update(
        cfg,
        {
            "device": "cpu",
            "data": {
                "synthetic": {
                    "t": time_steps,
                    "h": height,
                    "w": width,
                },
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
                "v19": {
                    "gain_hidden": 8,
                    "gain_dropout": 0.0,
                },
            },
            "train": {
                "amp": False,
                "lr_main": 1e-3,
                "lr_v14": 1e-2,
                "lr_v19_gain": 5e-3,
            },
        },
    )


__all__ = ["compact_v19_config", "make_batch"]
