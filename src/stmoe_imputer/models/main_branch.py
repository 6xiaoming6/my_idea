from __future__ import annotations

import torch
from torch import nn

from .embedding import ScaleTokenEncoder
from .experts import CrossScaleSharedExpert, TopKRoutedExpertPool
from .fusion import ProgressiveScaleGatedFusion
from .router import QualityRouter, uniform_gate
from .stats import compute_observation_stats


class MultiScaleMoEBackbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        dim: int = 64,
        num_experts: int = 4,
        top_k: int = 2,
        max_t: int = 24,
        h: int = 32,
        w: int = 32,
        q_dim: int = 5,
        num_groups: int = 8,
        dropout: float = 0.0,
        use_multiscale: bool = True,
        use_router: bool = True,
        share_experts: bool = True,
        use_routed_branch: bool = True,
        use_shared_branch: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(max(1, top_k), num_experts)
        self.use_multiscale = use_multiscale
        self.use_router = use_router
        self.share_experts = share_experts
        self.use_routed_branch = use_routed_branch
        self.use_shared_branch = use_shared_branch

        self.embed_f = ScaleTokenEncoder(c_in, dim, max_t, h, w, num_groups=num_groups)
        self.embed_m = ScaleTokenEncoder(c_in, dim, max_t, h // 2, w // 2, num_groups=num_groups)
        self.embed_c = ScaleTokenEncoder(c_in, dim, max_t, h // 4, w // 4, num_groups=num_groups)

        self.router_f = QualityRouter(dim, q_dim, num_experts)
        self.router_m = QualityRouter(dim, q_dim, num_experts)
        self.router_c = QualityRouter(dim, q_dim, num_experts)

        self.routed_expert_pool = TopKRoutedExpertPool(
            dim, num_experts, top_k=self.top_k, num_groups=num_groups, dropout=dropout
        )
        if share_experts:
            self.routed_expert_pool_m = self.routed_expert_pool
            self.routed_expert_pool_c = self.routed_expert_pool
        else:
            self.routed_expert_pool_m = TopKRoutedExpertPool(
                dim, num_experts, top_k=self.top_k, num_groups=num_groups, dropout=dropout
            )
            self.routed_expert_pool_c = TopKRoutedExpertPool(
                dim, num_experts, top_k=self.top_k, num_groups=num_groups, dropout=dropout
            )

        self.cross_scale_shared_expert = CrossScaleSharedExpert(
            dim, num_groups=num_groups, dropout=dropout
        )
        self.progressive_fusion = ProgressiveScaleGatedFusion(
            dim, num_groups=num_groups, dropout=dropout
        )
        self.pred_head = nn.Sequential(
            nn.Conv3d(dim, max(1, dim // 2), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(max(1, dim // 2), c_in, kernel_size=1),
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "MultiScaleMoEBackbone":
        model_cfg = cfg["model"]
        main_cfg = model_cfg["main"]
        data_syn = cfg["data"]["synthetic"]
        return cls(
            c_in=model_cfg["c_in"],
            dim=main_cfg["dim"],
            num_experts=main_cfg["num_experts"],
            top_k=main_cfg.get("top_k", min(2, main_cfg["num_experts"])),
            max_t=main_cfg.get("max_t", data_syn["t"]),
            h=main_cfg.get("h", data_syn["h"]),
            w=main_cfg.get("w", data_syn["w"]),
            q_dim=main_cfg.get("q_dim", 5),
            num_groups=main_cfg.get("num_groups", 8),
            dropout=main_cfg.get("dropout", 0.0),
            use_multiscale=main_cfg.get("use_multiscale", True),
            use_router=main_cfg.get("use_router", True),
            share_experts=main_cfg.get("share_experts", True),
            use_routed_branch=main_cfg.get("use_routed_branch", True),
            use_shared_branch=main_cfg.get(
                "use_shared_branch", main_cfg.get("use_cross_scale_expert", True)
            ),
        )

    def get_scale_embed_vec(self, embed_module: ScaleTokenEncoder, batch_size: int) -> torch.Tensor:
        return embed_module.scale_embed.view(1, self.dim).expand(batch_size, self.dim)

    def _route(
        self,
        router: QualityRouter,
        h: torch.Tensor,
        mask: torch.Tensor,
        scale_embed_vec: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_router:
            return uniform_gate(h.shape[0], self.num_experts, h.device, h.dtype)
        return router(h, compute_observation_stats(mask), scale_embed_vec)

    def forward(
        self,
        x_f: torch.Tensor,
        m_f: torch.Tensor,
        x_m: torch.Tensor,
        m_m: torch.Tensor,
        x_c: torch.Tensor,
        m_c: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch_size = x_f.shape[0]
        h_f = self.embed_f(x_f, m_f)
        h_m = self.embed_m(x_m, m_m)
        h_c = self.embed_c(x_c, m_c)

        gate_f = self._route(self.router_f, h_f, m_f, self.get_scale_embed_vec(self.embed_f, batch_size))
        gate_m = self._route(self.router_m, h_m, m_m, self.get_scale_embed_vec(self.embed_m, batch_size))
        gate_c = self._route(self.router_c, h_c, m_c, self.get_scale_embed_vec(self.embed_c, batch_size))

        if self.use_routed_branch:
            z_f, top_idx_f, top_w_f = self.routed_expert_pool(h_f, gate_f)
            z_m, top_idx_m, top_w_m = self.routed_expert_pool_m(h_m, gate_m)
            z_c, top_idx_c, top_w_c = self.routed_expert_pool_c(h_c, gate_c)
        else:
            z_f = torch.zeros_like(h_f)
            z_m = torch.zeros_like(h_m)
            z_c = torch.zeros_like(h_c)
            top_idx_f = top_idx_m = top_idx_c = torch.zeros(
                (gate_f.shape[0], self.top_k), device=gate_f.device, dtype=torch.long
            )
            top_w_f = top_w_m = top_w_c = torch.zeros(
                (gate_f.shape[0], self.top_k), device=gate_f.device, dtype=gate_f.dtype
            )

        if self.use_shared_branch and self.use_multiscale:
            z_shared, h_m_up, h_c_up = self.cross_scale_shared_expert(h_f, h_m, h_c)
        else:
            z_shared = torch.zeros_like(z_f)
            h_m_up = torch.zeros_like(z_f)
            h_c_up = torch.zeros_like(z_f)

        if not self.use_multiscale:
            z_m = torch.zeros_like(h_m)
            z_c = torch.zeros_like(h_c)

        fusion_outputs = self.progressive_fusion(
            z_f=z_f,
            z_m=z_m,
            z_c=z_c,
            z_shared=z_shared,
        )
        h_main = fusion_outputs["h_main"]
        x_hat_main = self.pred_head(h_main)
        return {
            "x_hat_main": x_hat_main,
            "h_st_aux": h_main,
            "gates": {
                "fine": gate_f,
                "mid": gate_m,
                "coarse": gate_c,
                "fusion_16": fusion_outputs["gate_16"],
                "fusion_32": fusion_outputs["gate_32"],
            },
            "topk": {
                "fine_indices": top_idx_f,
                "fine_weights": top_w_f,
                "mid_indices": top_idx_m,
                "mid_weights": top_w_m,
                "coarse_indices": top_idx_c,
                "coarse_weights": top_w_c,
            },
            "features": {
                "z_f": z_f,
                "z_m": z_m,
                "z_c": z_c,
                "z_c_to_m": fusion_outputs["z_c_to_m"],
                "z_mc": fusion_outputs["z_mc"],
                "z_mc_to_f": fusion_outputs["z_mc_to_f"],
                "z_shared": z_shared,
                "h_m_up": h_m_up,
                "h_c_up": h_c_up,
                "h_main": h_main,
            },
        }


OAMSBackbone = MultiScaleMoEBackbone
ObservationAwareMultiScaleMoEImputer = MultiScaleMoEBackbone
