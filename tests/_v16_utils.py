from __future__ import annotations

import json
from pathlib import Path

import torch

from stmoe_imputer.config import deep_update
from stmoe_imputer.data.transforms import ensure_multiscale


ROOT = Path(__file__).resolve().parents[1]


def compact_v16_config(
    channels: int = 2,
    time_steps: int = 3,
    height: int = 16,
    width: int = 16,
    scale_mode: str = "fine_mid_coarse",
) -> dict:
    base = json.loads((ROOT / "configs/presets/smoke.json").read_text(encoding="utf-8"))
    patch = json.loads((ROOT / "configs/v16-single/smoke.json").read_text(encoding="utf-8"))
    cfg = deep_update(base, patch)
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
                "max_t": max(8, time_steps),
                "h": height,
                "w": width,
                "num_groups": 2,
                "dropout": 0.0,
                "route_dropout": 0.0,
                "scale_mode": scale_mode,
            },
            "v16": {
                "residual_dim": 8,
                "residual_dropout": 0.0,
                "calibration_hidden_dim": 8,
                "calibration_dropout": 0.0,
                "warmup_epochs": 1,
            },
        },
        "train": {"amp": False},
    })


def make_batch(cfg: dict, seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (
        1,
        cfg["model"]["c_in"],
        cfg["data"]["synthetic"]["t"],
        cfg["data"]["synthetic"]["h"],
        cfg["data"]["synthetic"]["w"],
    )
    target = torch.randn(shape, generator=generator)
    mask = (torch.rand((shape[0], 1, *shape[2:]), generator=generator) > 0.45).float()
    scales = cfg["data"]["scales"]
    return ensure_multiscale(
        {"x_f_gt": target, "m_f": mask},
        fine_to_mid=scales["fine_to_mid"],
        fine_to_coarse=scales["fine_to_coarse"],
        pooling_mode=scales.get("pooling_mode", "avg"),
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
