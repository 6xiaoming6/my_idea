from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import spatial_choice_mask_np, spatial_mask_to_bcthw
from .transforms import ensure_multiscale


class SyntheticFlowDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 64,
        t: int = 12,
        h: int = 32,
        w: int = 32,
        c_in: int = 2,
        mask_cfg: dict | None = None,
        fine_to_mid: int = 2,
        fine_to_coarse: int = 4,
        pooling_mode: str = "avg",
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.t = t
        self.h = h
        self.w = w
        self.c_in = c_in
        self.mask_cfg = mask_cfg or {"pattern": "random", "missing_rate": 0.4}
        self.fine_to_mid = fine_to_mid
        self.fine_to_coarse = fine_to_coarse
        self.pooling_mode = pooling_mode
        self.generator = torch.Generator().manual_seed(seed)

        pattern = self.mask_cfg.get("pattern", "random")
        if pattern not in {"fixed", "random"}:
            raise ValueError(f"Unsupported missing pattern: {pattern}. Use 'fixed' or 'random'.")

        missing_rate = float(self.mask_cfg.get("missing_rate", 0.4))
        if pattern == "fixed":
            rng = np.random.default_rng(seed)
            mask_hw = spatial_choice_mask_np(h, w, missing_rate, rng)
            self.masks = spatial_mask_to_bcthw(mask_hw, t).unsqueeze(0)
        else:
            masks = []
            for idx in range(num_samples):
                rng = np.random.default_rng(seed + idx)
                mask_hw = spatial_choice_mask_np(h, w, missing_rate, rng)
                masks.append(spatial_mask_to_bcthw(mask_hw, t))
            self.masks = torch.stack(masks, dim=0)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h),
            torch.linspace(-1.0, 1.0, w),
            indexing="ij",
        )
        tt = torch.linspace(0.0, 1.0, t).view(t, 1, 1)
        bases = [
            torch.sin(3.14159 * (xx + tt)) + torch.cos(3.14159 * (yy - 0.5 * tt)),
            torch.cos(3.14159 * (xx - tt)) - torch.sin(3.14159 * (yy + 0.25 * tt)),
        ]
        while len(bases) < c_in:
            bases.append(0.5 * bases[-1] + 0.25 * bases[0])
        self.base = torch.stack(bases[:c_in], dim=0)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        noise = 0.04 * torch.randn(self.c_in, self.t, self.h, self.w, generator=self.generator)
        trend = 0.2 * idx / max(1, self.num_samples - 1)
        x_f_gt = (self.base + noise + trend).float()
        mask_idx = 0 if self.masks.shape[0] == 1 else idx
        m_f = self.masks[mask_idx].clone()
        sample = {"x_f_gt": x_f_gt, "m_f": m_f}
        return ensure_multiscale(sample, self.fine_to_mid, self.fine_to_coarse, self.pooling_mode)


SyntheticSpatioTemporalDataset = SyntheticFlowDataset
