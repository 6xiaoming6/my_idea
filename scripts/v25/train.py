#!/usr/bin/env python3
"""V25-only training entrypoint using the shared trainer without editing v24 code."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from stmoe_imputer import engine, losses  # noqa: E402
from stmoe_imputer.models import registry  # noqa: E402
from v25_ras_coe.model import RASCoE  # noqa: E402
from v25_ras_coe.losses import compute_v25_loss  # noqa: E402
from v25_ras_coe.quality import RASQualityMetrics  # noqa: E402


def main() -> None:
    # These registrations exist only in this v25 process. Existing v24
    # entrypoints, source files, and experiment queues remain unchanged.
    registry.MODEL_REGISTRY['v25_ras_coe'] = RASCoE.from_config
    original_loss = losses.compute_main_stage_loss
    original_architecture = engine.resolve_architecture

    def v25_loss(outputs, batch, cfg, epoch=None):
        if registry.resolve_architecture(cfg) == 'v25_ras_coe':
            return compute_v25_loss(outputs, batch, cfg, epoch=epoch)
        return original_loss(outputs, batch, cfg, epoch=epoch)

    def coe_family_architecture(cfg):
        name = original_architecture(cfg)
        return 'v24_ts_coe' if name == 'v25_ras_coe' else name

    losses.compute_main_stage_loss = v25_loss
    engine.compute_main_stage_loss = v25_loss
    engine.resolve_architecture = coe_family_architecture
    engine._CoEQualityMetrics = RASQualityMetrics

    spec = importlib.util.spec_from_file_location('v25_shared_train', ROOT / 'scripts/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == '__main__':
    main()
