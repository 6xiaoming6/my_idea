from __future__ import annotations

import json
from pathlib import Path

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.data.transforms import ensure_multiscale


ROOT = Path(__file__).resolve().parents[1]


def compact_v18_config(
    channels: int = 2,
    time_steps: int = 3,
    height: int = 16,
    width: int = 16,
) -> dict:
    base = json.loads(
        (ROOT / "configs/presets/smoke.json").read_text(encoding="utf-8")
    )
    v18 = json.loads(
        (ROOT / "configs/v18-single/smoke.json").read_text(encoding="utf-8")
    )
    cfg = deep_update(base, v18)
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
                "v18": {
                    "difficulty_hidden": 8,
                    "difficulty_out_dim": 8,
                    "residual_hidden": 8,
                    "anchor_embed_dim": 4,
                    "direction_embed_dim": 4,
                    "residual_dropout": 0.0,
                    "controller_hidden": 16,
                    "controller_dropout": 0.0,
                },
            },
            "train": {
                "amp": False,
                "lr_main": 1e-3,
                "lr_v18_refiner": 1e-2,
                "lr_v18_controller": 5e-3,
            },
        },
    )


def make_batch(cfg: dict, seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    data_cfg = cfg["data"]
    shape = (
        1,
        cfg["model"]["c_in"],
        data_cfg["synthetic"]["t"],
        data_cfg["synthetic"]["h"],
        data_cfg["synthetic"]["w"],
    )
    target = torch.randn(shape, generator=generator)
    mask = (
        torch.rand((shape[0], 1, *shape[2:]), generator=generator) > 0.45
    ).float()
    return ensure_multiscale(
        {"x_f_gt": target, "m_f": mask},
        fine_to_mid=data_cfg["scales"]["fine_to_mid"],
        fine_to_coarse=data_cfg["scales"]["fine_to_coarse"],
        pooling_mode=data_cfg["scales"].get("pooling_mode", "avg"),
    )


def backbone_kwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "x_f": batch["x_f_obs"],
        "m_f": batch["m_f"],
        "x_m": batch["x_m_obs"],
        "m_m": batch["m_m"],
        "x_c": batch["x_c_obs"],
        "m_c": batch["m_c"],
        "r_m": batch["r_m"],
        "r_c": batch["r_c"],
    }
