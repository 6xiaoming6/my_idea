from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .main_branch import MultiScaleMoEBackbone
from .v_single import V14SafeC2FMoE, V21DistortionCalibratedMoE
from .v_single.v21_2_paired_risk_moe import V21PairedRiskMoE
from .v_single.v22_coarsening_moe import V22CoarseningMoE


ModelBuilder = Callable[[dict], nn.Module]


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "main": MultiScaleMoEBackbone.from_config,
    "v14_safe_c2f_moe": V14SafeC2FMoE.from_config,
    "v21_distortion_calibrated_moe": V21DistortionCalibratedMoE.from_config,
    "v21_paired_risk_moe": V21PairedRiskMoE.from_config,
    "v22_coarsening_moe": V22CoarseningMoE.from_config,
}


def resolve_architecture(cfg: dict) -> str:
    model_cfg = cfg.get("model", {})
    main_cfg = model_cfg.get("main", {})
    return str(model_cfg.get("architecture", main_cfg.get("architecture", "main")))


def build_model_backbone(cfg: dict) -> nn.Module:
    architecture = resolve_architecture(cfg)
    try:
        builder = MODEL_REGISTRY[architecture]
    except KeyError as error:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model architecture {architecture!r}; supported: {supported}"
        ) from error
    return builder(cfg)
