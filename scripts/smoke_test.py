from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stmoe_imputer.config import load_config
from stmoe_imputer.data.synthetic import SyntheticFlowDataset
from stmoe_imputer.losses import compute_main_stage_loss
from stmoe_imputer.models import DualBranchSTImputer


def main() -> None:
    cfg = load_config(ROOT / "configs" / "smoke.json")
    scale_cfg = cfg["data"]["scales"]
    syn = cfg["data"]["synthetic"]
    dataset = SyntheticFlowDataset(
        num_samples=2,
        t=syn["t"],
        h=syn["h"],
        w=syn["w"],
        c_in=cfg["model"]["c_in"],
        mask_cfg=cfg["data"]["mask"],
        fine_to_mid=scale_cfg["fine_to_mid"],
        fine_to_coarse=scale_cfg["fine_to_coarse"],
        pooling_mode=scale_cfg.get("pooling_mode", "avg"),
        seed=cfg["seed"],
    )
    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
    model = DualBranchSTImputer.from_config(cfg)
    outputs = model(batch)
    loss, loss_dict = compute_main_stage_loss(outputs, batch, cfg)
    print("loss", float(loss.detach()))
    print({key: tuple(value.shape) for key, value in outputs.items() if torch.is_tensor(value)})
    print({key: float(value) for key, value in loss_dict.items()})
    print({key: tuple(value.shape) for key, value in outputs["gates"].items()})
    print({key: tuple(value.shape) for key, value in outputs["topk"].items()})


if __name__ == "__main__":
    main()
