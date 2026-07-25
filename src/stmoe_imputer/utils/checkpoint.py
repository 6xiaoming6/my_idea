from __future__ import annotations

import os
from pathlib import Path

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module for DP/DDP-safe checkpoint I/O."""
    wrappers = (
        torch.nn.DataParallel,
        torch.nn.parallel.DistributedDataParallel,
    )
    while isinstance(model, wrappers):
        model = model.module
    return model


def _normalize_model_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Accept legacy checkpoints saved directly from a DP/DDP wrapper."""
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


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
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg,
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
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
    unwrap_model(model).load_state_dict(
        _normalize_model_state(checkpoint["model"])
    )
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
