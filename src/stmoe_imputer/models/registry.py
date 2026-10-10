from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .main_branch import MultiScaleMoEBackbone
from .temporal_spatial_coe import TemporalSpatialCoE
from .v_single import V14SafeC2FMoE


ModelBuilder = Callable[[dict], nn.Module]


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "main": MultiScaleMoEBackbone.from_config,
    "v14_safe_c2f_moe": V14SafeC2FMoE.from_config,
    "v24_ts_coe": TemporalSpatialCoE.from_config,
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
    if architecture == "v24_ts_coe" and cfg["model"].get("coe", {}).get("post_fusion_ffn", {}).get("enabled", False):
        from .post_fusion_ffn_coe import PostFusionFFNCoE
        return PostFusionFFNCoE.from_config(cfg)
    if architecture == "v24_ts_coe" and cfg["model"].get("coe", {}).get("core_validation", {}).get("enabled", False):
        from .core_validation_coe import CoreValidationCoE
        return CoreValidationCoE.from_config(cfg)
    if architecture == "v24_ts_coe" and cfg["model"].get("coe", {}).get("backbone_exploration", {}).get("enabled", False):
        from .backbone_exploration_coe import BackboneExplorationCoE
        return BackboneExplorationCoE.from_config(cfg)
    if architecture == "v24_ts_coe" and cfg["model"].get("coe", {}).get("coordination", {}).get("enabled", False):
        from .coordination_coe import CoordinationCoE
        return CoordinationCoE.from_config(cfg)
    if architecture == "v24_ts_coe" and cfg["model"].get("coe", {}).get("id_priority", {}).get("enabled", False):
        from .id_priority_coe import IDPriorityCoE
        return IDPriorityCoE.from_config(cfg)
    if architecture == "v24_ts_coe" and cfg["model"].get("coe", {}).get("four_direction", {}).get("enabled", False):
        from .four_direction_coe import FourDirectionCoE
        return FourDirectionCoE.from_config(cfg)
    if architecture == "v24_ts_coe" and cfg["model"].get("coe", {}).get("triscale", {}).get("enabled", False):
        from .triscale_coe import TriScaleCoE
        return TriScaleCoE.from_config(cfg)
    if architecture == "v24_ts_coe" and cfg["model"].get("coe", {}).get("local_routing", {}).get("enabled", False):
        from .local_support_coe import LocalSupportCoE
        builder = LocalSupportCoE.from_config
    if architecture == "v24_ts_coe" and cfg["model"].get("coe", {}).get("spatial_scale", {}).get("enabled", False):
        if cfg["model"]["coe"].get("local_routing", {}).get("enabled", False):
            raise ValueError("Spatial-scale baselines cannot enable C3 local routing")
        from .spatial_scale_coe import SpatialScaleCoE
        builder = SpatialScaleCoE.from_config
    if architecture == "v24_ts_coe" and cfg["model"].get("coe", {}).get("round_memory", {}).get("enabled", False):
        if any(cfg["model"]["coe"].get(name, {}).get("enabled", False) for name in ("local_routing", "spatial_scale")):
            raise ValueError("Round-memory baseline cannot combine local routing or spatial scales")
        from .round_memory_coe import RoundMemoryCoE
        builder = RoundMemoryCoE.from_config
    constraint=cfg.get('model',{}).get('coe',{}).get('expert_pair_constraint','none')
    if architecture == 'v24_ts_coe' and constraint != 'none':
        if constraint != 'cross_direction' or any(cfg['model']['coe'].get(k,{}).get('enabled',False)
                                                for k in ('local_routing','spatial_scale','round_memory')):
            raise ValueError('Direction constraint is a standalone baseline experiment')
        from .direction_pair_coe import DirectionPairCoE
        builder=DirectionPairCoE.from_config
    return builder(cfg)
