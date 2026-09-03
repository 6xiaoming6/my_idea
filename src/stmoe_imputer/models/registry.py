from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .main_branch import MultiScaleMoEBackbone
from .v_single import V14SafeC2FMoE, V20ProbeValidatedC2FMoE


ModelBuilder = Callable[[dict], nn.Module]


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "main": MultiScaleMoEBackbone.from_config,
    "v14_safe_c2f_moe": V14SafeC2FMoE.from_config,
    "v20_probe_validated_c2f_moe": V20ProbeValidatedC2FMoE.from_config,
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
