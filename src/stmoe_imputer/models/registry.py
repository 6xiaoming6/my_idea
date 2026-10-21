"""Release registry: only the frozen I37 architecture is supported."""
from .observed_consistency_coe import FreeScaleHistoryConsistencyCoE


def resolve_architecture(cfg):
    return cfg.get("model", {}).get("architecture", "v25_observed_multiscale_coe")


def build_model_backbone(cfg):
    if resolve_architecture(cfg) != "v25_observed_multiscale_coe":
        raise ValueError("This release supports the I37 architecture only")
    return FreeScaleHistoryConsistencyCoE.from_config(cfg)
