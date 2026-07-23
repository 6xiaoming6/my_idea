from __future__ import annotations

from copy import deepcopy
from typing import Any

from .v17_hierarchical_scale_moe import V17HierarchicalScaleMoEBackbone


class V17_2NoAdapterHierarchicalScaleMoEBackbone(
    V17HierarchicalScaleMoEBackbone
):
    """Formal V17.2 backbone selected from the V17.1 E1 ablation.

    V17.2 deliberately changes one architectural factor only: the three
    scale-specific adapters are replaced by parameter-free identities.  The
    guard in ``__init__`` makes this contract independent of later config
    merges.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["adapter_enabled"] = False
        # Keep adapter-only metadata deterministic even though these values are
        # ignored by the identity path.
        kwargs["adapter_dim"] = 16
        kwargs["adapter_dropout"] = 0.0
        kwargs["adapter_zero_init"] = True
        super().__init__(*args, **kwargs)

        self.adapter_parameter_count = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith(("adapter_f.", "adapter_m.", "adapter_c."))
        )
        if self.adapter_enabled or self.adapter_parameter_count != 0:
            raise RuntimeError(
                "V17.2 must use parameter-free IdentityScaleAdapter modules"
            )

    @classmethod
    def from_config(
        cls,
        cfg: dict,
    ) -> "V17_2NoAdapterHierarchicalScaleMoEBackbone":
        resolved = deepcopy(cfg)
        model_cfg = resolved.setdefault("model", {})
        v17_cfg = model_cfg.setdefault("v17", {})
        v17_cfg["adapter_enabled"] = False
        model = super().from_config(resolved)
        if model.adapter_enabled or model.adapter_parameter_count != 0:
            raise RuntimeError(
                "V17.2 construction unexpectedly enabled Scale Adapter"
            )
        return model

    def forward(self, *args: Any, **kwargs: Any) -> dict:
        outputs = super().forward(*args, **kwargs)
        outputs.update(
            {
                "v17_2_enabled": True,
                "v17_2_remove_adapter": True,
                "v17_2_source_ablation": "E1_no_adapter",
            }
        )
        diagnostics = outputs.setdefault("diagnostics", {})
        diagnostics["v17_2"] = {
            "adapter_enabled": False,
            "adapter_parameter_count": self.adapter_parameter_count,
        }
        return outputs
