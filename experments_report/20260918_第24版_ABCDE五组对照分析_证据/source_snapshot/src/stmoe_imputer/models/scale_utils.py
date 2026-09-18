from __future__ import annotations

import torch


def get_active_scales(scale_mode: str) -> list[str]:
    if scale_mode == "fine":
        return ["fine"]
    if scale_mode == "fine_mid":
        return ["fine", "mid"]
    if scale_mode == "fine_mid_coarse":
        return ["fine", "mid", "coarse"]
    raise ValueError(f"Unknown scale_mode: {scale_mode}")


def is_scale_active(scale_mode: str, scale: str) -> bool:
    return scale in get_active_scales(scale_mode)


def build_scale_active_mask(
    scale_mode: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if scale_mode == "fine":
        mask = torch.tensor([True, False, False], device=device)
    elif scale_mode == "fine_mid":
        mask = torch.tensor([True, True, False], device=device)
    elif scale_mode == "fine_mid_coarse":
        mask = torch.tensor([True, True, True], device=device)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")
    return mask.view(1, 3).expand(batch_size, 3)
