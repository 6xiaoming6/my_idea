from __future__ import annotations

import torch
from torch import nn

from .low_rank_mixer import LowRankGlobalMixer


class GlobalLocalResidualFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        alpha_init: float = -3.0,
        alpha_trainable: bool = True,
        alpha_fixed: float = 0.05,
        local_proj_zero_init: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha_fixed <= 1.0:
            raise ValueError(f"local_alpha_fixed must be in [0,1], got {alpha_fixed}")
        self.local_proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout)
        self.alpha_trainable = alpha_trainable
        if alpha_trainable:
            self.local_alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("local_alpha_fixed", torch.tensor(float(alpha_fixed)))
        if local_proj_zero_init:
            nn.init.zeros_(self.local_proj.weight)
            nn.init.zeros_(self.local_proj.bias)
        else:
            nn.init.normal_(self.local_proj.weight, std=1e-3)
            nn.init.zeros_(self.local_proj.bias)

    def alpha_value(self) -> torch.Tensor:
        if self.alpha_trainable:
            return torch.sigmoid(self.local_alpha_logit)
        return self.local_alpha_fixed

    def forward(
        self,
        h_global: torch.Tensor,
        h_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_projected = self.dropout(self.local_proj(h_local))
        alpha = self.alpha_value().to(device=h_global.device, dtype=h_global.dtype)
        fused = h_global + alpha * local_projected
        return fused, local_projected, alpha


class V13LowRankGlobalLocalMoE(nn.Module):
    """Low-rank global modeling plus a sparse local MoE residual."""

    MODES = {"global_plus_local", "global_only", "local_only"}

    def __init__(
        self,
        dim: int,
        mode: str = "global_plus_local",
        use_lowrank_global: bool = True,
        use_sparse_local_moe: bool = True,
        lowrank_mode: str = "anchor_attention",
        rank: int = 16,
        num_heads: int = 4,
        lowrank_dropout: float = 0.1,
        global_gamma_init: float = -3.0,
        global_out_init_scale: float = 1e-3,
        local_alpha_init: float = -3.0,
        local_alpha_trainable: bool = True,
        local_alpha_fixed: float = 0.05,
        local_proj_zero_init: bool = True,
        local_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported global-local mode: {mode}")
        if lowrank_mode != "anchor_attention":
            raise ValueError(f"unsupported lowrank mode: {lowrank_mode}")
        if mode in {"global_plus_local", "global_only"} and not use_lowrank_global:
            raise ValueError(f"mode={mode} requires use_lowrank_global=true")
        if mode in {"global_plus_local", "local_only"} and not use_sparse_local_moe:
            raise ValueError(f"mode={mode} requires use_sparse_local_moe=true")

        self.mode = mode
        self.use_lowrank_global = use_lowrank_global
        self.use_sparse_local_moe = use_sparse_local_moe
        self.lowrank = (
            LowRankGlobalMixer(
                dim=dim,
                rank=rank,
                num_heads=num_heads,
                dropout=lowrank_dropout,
                gamma_init=global_gamma_init,
                out_init_scale=global_out_init_scale,
            )
            if use_lowrank_global
            else None
        )
        self.fusion = GlobalLocalResidualFusion(
            dim=dim,
            alpha_init=local_alpha_init,
            alpha_trainable=local_alpha_trainable,
            alpha_fixed=local_alpha_fixed,
            local_proj_zero_init=local_proj_zero_init,
            dropout=local_dropout,
        )

    def forward(
        self,
        h_f: torch.Tensor,
        h_local: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.lowrank is None:
            h_global = torch.zeros_like(h_f)
            lowrank_aux = {
                "rank": torch.zeros(1, device=h_f.device, dtype=h_f.dtype),
                "rank_attention_entropy": torch.zeros(
                    h_f.shape[0], device=h_f.device, dtype=h_f.dtype
                ),
                "global_gamma": torch.zeros(1, device=h_f.device, dtype=h_f.dtype),
                "global_update_norm": torch.zeros(
                    h_f.shape[0], device=h_f.device, dtype=h_f.dtype
                ),
            }
        else:
            h_global, lowrank_aux = self.lowrank(h_f)
        if self.mode == "global_only":
            fused = h_global
            local_projected = torch.zeros_like(h_local)
            alpha = torch.zeros((), device=h_f.device, dtype=h_f.dtype)
        elif self.mode == "local_only":
            fused = h_local
            h_global = torch.zeros_like(h_f)
            local_projected = h_local
            alpha = torch.ones((), device=h_f.device, dtype=h_f.dtype)
        else:
            fused, local_projected, alpha = self.fusion(h_global, h_local)

        reduce_dims = (1, 2, 3, 4)
        return fused, {
            **lowrank_aux,
            "alpha_local": alpha.reshape(1),
            "global_feature_norm": h_global.pow(2).mean(dim=reduce_dims).sqrt(),
            "local_feature_norm": h_local.pow(2).mean(dim=reduce_dims).sqrt(),
            "local_projected_norm": local_projected.pow(2).mean(dim=reduce_dims).sqrt(),
            "fused_feature_norm": fused.pow(2).mean(dim=reduce_dims).sqrt(),
            "h_global": h_global,
            "h_local": h_local,
            "h_local_projected": local_projected,
        }
