from __future__ import annotations

import copy
import os
from collections import OrderedDict
from pathlib import Path

import torch


def snapshot_model_state(model: torch.nn.Module) -> dict:
    """Independent CPU copy of parameters and buffers, as in the v23 runner."""
    state = model.state_dict()
    snapshot = OrderedDict(
        (key, value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value))
        for key, value in state.items()
    )
    if hasattr(state, "_metadata"):
        snapshot._metadata = copy.deepcopy(state._metadata)
    return snapshot


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict,
    cfg: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    backbone = getattr(model, "main_branch", model)
    current_experts = getattr(backbone, "expert_names", None)
    saved_config = checkpoint.get("config")
    if current_experts is not None and saved_config is not None:
        saved_pool = saved_config.get("model", {}).get("coe", {}).get("expert_pool")
        saved_experts = tuple(str(name).upper() for name in (saved_pool or ("T", "S")))
        if tuple(current_experts) != saved_experts:
            raise ValueError(
                "Checkpoint expert_pool order differs from the model; "
                "construct TS-CoE using the checkpoint's saved config"
            )
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
