from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stmoe_imputer.config import deep_update, load_config
from stmoe_imputer.data import FlowNPZDataset, build_datasets, build_loader
from stmoe_imputer.engine import evaluate
from stmoe_imputer.models import DualBranchSTImputer
from stmoe_imputer.utils import get_device
from stmoe_imputer.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--override_config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_npz", default=None)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.override_config:
        cfg = deep_update(cfg, load_config(args.override_config))
    device = get_device(cfg.get("device", "auto"))
    if args.synthetic:
        _, dataset = build_datasets(cfg, synthetic=True)
    else:
        if args.data_npz is None:
            raise ValueError("--data_npz is required unless --synthetic is set")
        scale_cfg = cfg["data"]["scales"]
        dataset = FlowNPZDataset(
            args.data_npz,
            mask_cfg=cfg["data"]["mask"],
            fine_to_mid=scale_cfg["fine_to_mid"],
            fine_to_coarse=scale_cfg["fine_to_coarse"],
            pooling_mode=scale_cfg.get("pooling_mode", "avg"),
            seed=cfg.get("seed", 42),
        )
    loader = build_loader(dataset, cfg, shuffle=False)
    model = DualBranchSTImputer.from_config(cfg).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, map_location=device)
    logs = evaluate(model, loader, device, cfg, desc=f"eval epoch {checkpoint.get('epoch', '?')}")
    print(logs)


if __name__ == "__main__":
    main()
