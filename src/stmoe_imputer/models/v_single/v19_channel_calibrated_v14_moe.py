from __future__ import annotations

import torch
from torch import nn

from .channel_residual_gain import ChannelResidualGain
from .v14_safe_c2f_moe import V14SafeC2FMoE


class V19ChannelCalibratedV14MoE(nn.Module):
    """Single-stage V14 with a lightweight channel-wise residual gain."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        v14_cfg = model_cfg.get("v14", {})
        v19_cfg = model_cfg.get("v19", {})
        if not bool(v14_cfg.get("enabled", True)):
            raise ValueError("V19 requires model.v14.enabled=true")
        if not bool(v14_cfg.get("use_main_bypass", True)):
            raise ValueError("V19 requires V14's main bypass")
        if not bool(v19_cfg.get("enabled", True)):
            raise ValueError("V19 architecture requires model.v19.enabled=true")

        self.v14_model = V14SafeC2FMoE(cfg)
        self.gain_controller = ChannelResidualGain(
            hidden_dim=int(v19_cfg.get("gain_hidden", 32)),
            dropout=float(v19_cfg.get("gain_dropout", 0.1)),
            gain_range=float(v19_cfg.get("gain_range", 0.5)),
            scale_eps=float(v19_cfg.get("scale_eps", 1e-3)),
            zero_eps=float(v19_cfg.get("zero_eps", 1e-6)),
            zero_init=bool(v19_cfg.get("gain_zero_init", True)),
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "V19ChannelCalibratedV14MoE":
        return cls(cfg)

    def forward(
        self,
        x_f: torch.Tensor,
        m_f: torch.Tensor,
        x_m: torch.Tensor,
        m_m: torch.Tensor,
        x_c: torch.Tensor,
        m_c: torch.Tensor,
        r_m: torch.Tensor | None = None,
        r_c: torch.Tensor | None = None,
    ) -> dict:
        v14_outputs = self.v14_model(
            x_f=x_f,
            m_f=m_f,
            x_m=x_m,
            m_m=m_m,
            x_c=x_c,
            m_c=m_c,
            r_m=r_m,
            r_c=r_c,
        )
        x_base = v14_outputs["x_hat_base"]
        x_v14 = v14_outputs["x_hat_main"]
        effective_v14_residual = x_v14 - x_base
        gain, gain_diagnostics = self.gain_controller(
            x_base=x_base,
            x_v14=x_v14,
            x_obs=x_f,
            mask=m_f,
        )
        x_v19 = x_base + gain * effective_v14_residual

        outputs = dict(v14_outputs)
        features = dict(v14_outputs.get("features", {}))
        features.update(
            {
                "v19_effective_v14_residual": effective_v14_residual,
                "v19_calibrated_residual": gain * effective_v14_residual,
            }
        )
        diagnostics = dict(v14_outputs.get("diagnostics", {}))
        diagnostics["v19"] = gain_diagnostics
        outputs.update(
            {
                "x_hat_main": x_v19,
                "x_hat_v14": x_v14,
                "features": features,
                "diagnostics": diagnostics,
                "branch_mode": "v19_channel_calibrated_v14",
                "v19_enabled": True,
            }
        )
        return outputs

