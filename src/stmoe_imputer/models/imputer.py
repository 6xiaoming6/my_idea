from __future__ import annotations

import torch
from torch import nn

from .aux_branch import NullResidualBranch
from .main_branch import MultiScaleMoEBackbone


class DualBranchSTImputer(nn.Module):
    def __init__(
        self,
        main_branch: nn.Module,
        aux_branch: nn.Module,
        aux_enabled: bool = False,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.main_branch = main_branch
        self.aux_branch = aux_branch
        self.aux_enabled = aux_enabled
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @classmethod
    def from_config(cls, cfg: dict) -> "DualBranchSTImputer":
        main_branch = MultiScaleMoEBackbone.from_config(cfg)
        aux_branch = NullResidualBranch(c_out=cfg["model"]["c_in"])
        aux_cfg = cfg["model"].get("aux", {})
        return cls(
            main_branch=main_branch,
            aux_branch=aux_branch,
            aux_enabled=aux_cfg.get("enabled", False),
            alpha_init=aux_cfg.get("alpha_init", 0.0),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict:
        main_outputs = self.main_branch(
            x_f=batch["x_f_obs"],
            m_f=batch["m_f"],
            x_m=batch["x_m_obs"],
            m_m=batch["m_m"],
            x_c=batch["x_c_obs"],
            m_c=batch["m_c"],
            r_m=batch.get("r_m"),
            r_c=batch.get("r_c"),
        )
        x_hat_main = main_outputs["x_hat_main"]
        h_st_aux = main_outputs["h_st_aux"]
        if self.aux_enabled:
            delta_aux = self.aux_branch(h_st_aux=h_st_aux, mask_f=batch["m_f"])
            x_hat_final = x_hat_main + self.alpha * delta_aux
        else:
            delta_aux = torch.zeros_like(x_hat_main)
            x_hat_final = x_hat_main

        x_gt_or_obs = batch.get("x_f_gt", batch["x_f_obs"])
        x_comp = batch["m_f"] * x_gt_or_obs + (1.0 - batch["m_f"]) * x_hat_final
        outputs = {
            "x_hat_main": x_hat_main,
            "h_st_aux": h_st_aux,
            "delta_aux": delta_aux,
            "x_hat_final": x_hat_final,
            "x_comp": x_comp,
        }
        outputs.update(main_outputs)
        return outputs


ParallelTwoBranchImputer = DualBranchSTImputer
