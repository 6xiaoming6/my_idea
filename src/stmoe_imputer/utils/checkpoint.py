from __future__ import annotations

from pathlib import Path
import os
import copy
from collections import OrderedDict

import torch


def snapshot_model_state(model: torch.nn.Module) -> dict:
    """Independent CPU copy of parameters/buffers, without optimizer or files."""
    state = model.state_dict()
    snapshot = OrderedDict((key, value.detach().cpu().clone() if torch.is_tensor(value)
                            else copy.deepcopy(value)) for key, value in state.items())
    if hasattr(state, '_metadata'):
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
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
