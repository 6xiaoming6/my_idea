from __future__ import annotations

import math

import numpy as np
import torch


def spatial_choice_mask_np(
    h: int,
    w: int,
    missing_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one HxW mask by choosing observed positions without replacement."""
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
    num_positions = h * w
    num_observed = int(math.ceil(num_positions * (1.0 - missing_rate)))
    mask = np.zeros(num_positions, dtype=np.int8)
    if num_observed > 0:
        chosen = rng.choice(num_positions, size=num_observed, replace=False)
        mask[chosen] = 1
    return mask.reshape(h, w)


def spatial_mask_to_bcthw(mask_hw: np.ndarray, t: int) -> torch.Tensor:
    mask = torch.from_numpy(mask_hw.astype(np.float32)).view(1, 1, 1, *mask_hw.shape)
    return mask.expand(1, 1, t, mask_hw.shape[0], mask_hw.shape[1]).squeeze(0).clone()


def make_mask(
    t: int,
    h: int,
    w: int,
    cfg: dict,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    pattern = cfg.get("pattern", "random")
    missing_rate = float(cfg.get("missing_rate", 0.4))
    if pattern not in {"fixed", "random"}:
        raise ValueError(f"Unsupported missing pattern: {pattern}. Use 'fixed' or 'random'.")
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()) if generator else None
    rng = np.random.default_rng(seed)
    mask_hw = spatial_choice_mask_np(h, w, missing_rate, rng)
    return spatial_mask_to_bcthw(mask_hw, t)
