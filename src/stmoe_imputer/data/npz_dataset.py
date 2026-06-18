from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import ensure_multiscale, to_bcthw


class FlowNPZDataset(Dataset):
    """
    Generic NPZ dataset.

    Required key: ``x_f_gt`` or ``x_f``. Accepted layouts are [N,C,T,H,W] and
    [N,T,H,W,C]. Optional masks/scales use keys matching the model batch:
    ``m_f``, ``x_m_obs``, ``m_m``, ``x_c_obs``, ``m_c``.
    """

    def __init__(
        self,
        path: str | Path,
        mask_cfg: dict | None = None,
        fine_to_mid: int = 2,
        fine_to_coarse: int = 4,
        pooling_mode: str = "avg",
        seed: int = 42,
        mask_csv: str | Path | None = None,
        fixed_mask_csv: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=True) as arrays:
            self.arrays = {key: arrays[key] for key in arrays.files}
        self.x_key = "x_f_gt" if "x_f_gt" in self.arrays else "x_f"
        if self.x_key not in self.arrays:
            raise KeyError("NPZ must contain `x_f_gt` or `x_f`.")
        self.length = int(self.arrays[self.x_key].shape[0])
        self.mask_cfg = mask_cfg or {"pattern": "random", "missing_rate": 0.4}
        self.fine_to_mid = fine_to_mid
        self.fine_to_coarse = fine_to_coarse
        self.pooling_mode = pooling_mode
        self.seed = seed

        self.loaded_masks: torch.Tensor | None = None
        csv_path = mask_csv if mask_csv is not None else fixed_mask_csv
        if csv_path is not None:
            csv_file = Path(csv_path)
            if not csv_file.exists():
                raise FileNotFoundError(
                    f"Mask CSV not found: {csv_file}. "
                    f"Run scripts/generate_fixed_masks.py first."
                )
            print(f"[info] loading {self.mask_cfg.get('pattern', 'random')} masks from {csv_file} ...")
            masks_flat = np.loadtxt(str(csv_file), delimiter=",", dtype=np.float32)

            # Handle single-sample edge case: np.loadtxt returns 1D
            if masks_flat.ndim == 1:
                masks_flat = masks_flat.reshape(1, -1)

            # Infer expected shape from NPZ data
            x_shape = self.arrays[self.x_key].shape  # [N, C, T, H, W]
            _, _, t, h, w = x_shape
            spatial_cols = h * w
            st_cols = t * h * w
            pattern = self.mask_cfg.get("pattern", "random")

            if pattern == "fixed":
                if masks_flat.shape[0] != 1:
                    raise ValueError(
                        f"Fixed mask CSV must have exactly 1 row; got {masks_flat.shape[0]} rows."
                    )
            elif pattern == "random":
                if masks_flat.shape[0] != self.length:
                    raise ValueError(
                        f"Random mask CSV row count ({masks_flat.shape[0]}) "
                        f"does not match NPZ sample count ({self.length})."
                    )
            else:
                raise ValueError(f"Unsupported missing pattern: {pattern}. Use 'fixed' or 'random'.")

            if masks_flat.shape[1] == spatial_cols:
                masks = torch.from_numpy(masks_flat).reshape(masks_flat.shape[0], 1, 1, h, w)
                masks = masks.expand(masks.shape[0], 1, t, h, w).clone()
            elif masks_flat.shape[1] == st_cols:
                masks = torch.from_numpy(masks_flat).reshape(masks_flat.shape[0], 1, t, h, w)
            else:
                raise ValueError(
                    f"Mask CSV column count ({masks_flat.shape[1]}) does not match "
                    f"H*W ({h}*{w}={spatial_cols}) or T*H*W ({t}*{h}*{w}={st_cols})."
                )

            self.loaded_masks = masks.float()
            print(f"[info] masks loaded: {self.loaded_masks.shape}")

    def __len__(self) -> int:
        return self.length

    def _tensor_at(self, key: str, idx: int) -> torch.Tensor:
        return torch.as_tensor(self.arrays[key][idx]).float()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x_f_gt = to_bcthw(self._tensor_at(self.x_key, idx).unsqueeze(0)).squeeze(0)
        if "m_f" in self.arrays:
            m_f = to_bcthw(self._tensor_at("m_f", idx).unsqueeze(0), channels=1).squeeze(0)
        elif self.loaded_masks is not None:
            mask_idx = 0 if self.loaded_masks.shape[0] == 1 else idx
            m_f = self.loaded_masks[mask_idx].clone()
        else:
            raise RuntimeError(
                "Real NPZ training now requires offline mask CSVs for both fixed and random modes. "
                "Run scripts/generate_fixed_masks.py and set data.mask.train_csv / data.mask.val_csv."
            )

        sample: dict[str, torch.Tensor] = {"x_f_gt": x_f_gt, "m_f": m_f}
        for x_key, m_key in (("x_m_obs", "m_m"), ("x_c_obs", "m_c")):
            legacy_x_key = x_key.replace("_obs", "")
            if x_key in self.arrays and m_key in self.arrays:
                sample[x_key] = to_bcthw(self._tensor_at(x_key, idx).unsqueeze(0)).squeeze(0)
                sample[m_key] = to_bcthw(
                    self._tensor_at(m_key, idx).unsqueeze(0), channels=1
                ).squeeze(0)
            elif legacy_x_key in self.arrays and m_key in self.arrays:
                sample[x_key] = to_bcthw(self._tensor_at(legacy_x_key, idx).unsqueeze(0)).squeeze(0)
                sample[m_key] = to_bcthw(
                    self._tensor_at(m_key, idx).unsqueeze(0), channels=1
                ).squeeze(0)

        for r_key in ("r_m", "r_c"):
            if r_key in self.arrays:
                sample[r_key] = to_bcthw(
                    self._tensor_at(r_key, idx).unsqueeze(0), channels=1
                ).squeeze(0)

        return ensure_multiscale(sample, self.fine_to_mid, self.fine_to_coarse, self.pooling_mode)


NPZSpatioTemporalDataset = FlowNPZDataset
