from __future__ import annotations

from torch.utils.data import DataLoader

from .npz_dataset import FlowNPZDataset
from .synthetic import SyntheticFlowDataset


def build_datasets(
    cfg: dict,
    train_npz: str | None = None,
    val_npz: str | None = None,
    synthetic: bool = False,
):
    data_cfg = cfg["data"]
    scale_cfg = data_cfg["scales"]
    mask_cfg = data_cfg["mask"]
    if synthetic or train_npz is None:
        syn = data_cfg["synthetic"]
        train_ds = SyntheticFlowDataset(
            num_samples=syn["num_train"],
            t=syn["t"],
            h=syn["h"],
            w=syn["w"],
            c_in=cfg["model"]["c_in"],
            mask_cfg=mask_cfg,
            fine_to_mid=scale_cfg["fine_to_mid"],
            fine_to_coarse=scale_cfg["fine_to_coarse"],
            pooling_mode=scale_cfg.get("pooling_mode", "avg"),
            seed=cfg.get("seed", 42),
        )
        val_ds = SyntheticFlowDataset(
            num_samples=syn["num_val"],
            t=syn["t"],
            h=syn["h"],
            w=syn["w"],
            c_in=cfg["model"]["c_in"],
            mask_cfg=mask_cfg,
            fine_to_mid=scale_cfg["fine_to_mid"],
            fine_to_coarse=scale_cfg["fine_to_coarse"],
            pooling_mode=scale_cfg.get("pooling_mode", "avg"),
            seed=cfg.get("seed", 42) + 10000,
        )
        return train_ds, val_ds

    pattern = mask_cfg.get("pattern", "random")
    if pattern not in {"fixed", "random"}:
        raise ValueError(f"Unsupported data.mask.pattern={pattern!r}. Use 'fixed' or 'random'.")

    train_csv = (
        mask_cfg.get("train_csv")
        or mask_cfg.get(f"{pattern}_train_csv")
        or mask_cfg.get("fixed_train_csv")
    )
    val_csv = (
        mask_cfg.get("val_csv")
        or mask_cfg.get(f"{pattern}_val_csv")
        or mask_cfg.get("fixed_val_csv")
    )
    if train_csv is None:
        raise ValueError(
            f"data.mask.pattern='{pattern}' requires data.mask.train_csv "
            f"(or data.mask.{pattern}_train_csv)."
        )

    val_npz_path = val_npz or train_npz
    if val_csv is None and val_npz_path == train_npz:
        val_csv = train_csv
    if val_csv is None:
        raise ValueError(
            f"data.mask.pattern='{pattern}' requires data.mask.val_csv "
            f"(or data.mask.{pattern}_val_csv) when validation uses a separate NPZ."
        )

    train_ds = FlowNPZDataset(
        train_npz,
        mask_cfg=mask_cfg,
        fine_to_mid=scale_cfg["fine_to_mid"],
        fine_to_coarse=scale_cfg["fine_to_coarse"],
        pooling_mode=scale_cfg.get("pooling_mode", "avg"),
        seed=cfg.get("seed", 42),
        mask_csv=train_csv,
    )
    val_ds = FlowNPZDataset(
        val_npz_path,
        mask_cfg=mask_cfg,
        fine_to_mid=scale_cfg["fine_to_mid"],
        fine_to_coarse=scale_cfg["fine_to_coarse"],
        pooling_mode=scale_cfg.get("pooling_mode", "avg"),
        seed=cfg.get("seed", 42) + 20000,
        mask_csv=val_csv,
    )
    return train_ds, val_ds


def build_loader(dataset, cfg: dict, shuffle: bool) -> DataLoader:
    data_cfg = cfg["data"]
    return DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=data_cfg.get("drop_last", False) and shuffle,
    )
