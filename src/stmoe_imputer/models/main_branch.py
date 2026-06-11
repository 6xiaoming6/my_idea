from __future__ import annotations

import torch
from torch import nn

from .embedding import ScaleTokenEncoder
from .experts import CrossScaleSharedExpert, TopKRoutedExpertPool
from .fusion import ProgressiveRouteFusion, SharedRoutedResidualFusion
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
        branch_fusion_mode: str = "shared_plus_routed_residual",
        route_gamma_init: float = -4.0,
        routing_mode: str = "topk",
        routing_mode_when_no_router: str = "dense",
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
        self.branch_fusion_mode = branch_fusion_mode
        self.route_gamma_init = route_gamma_init
        self.routing_mode = routing_mode
        self.routing_mode_when_no_router = routing_mode_when_no_router

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
        self.route_fusion = ProgressiveRouteFusion(
            dim, num_groups=num_groups, dropout=dropout
        )
        self.branch_fusion = SharedRoutedResidualFusion(
            dim,
            num_groups=num_groups,
            dropout=dropout,
            route_gamma_init=route_gamma_init,
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
            branch_fusion_mode=main_cfg.get(
                "branch_fusion_mode", "shared_plus_routed_residual"
            ),
            route_gamma_init=main_cfg.get("route_gamma_init", -4.0),
            routing_mode=main_cfg.get("routing_mode", "topk"),
            routing_mode_when_no_router=main_cfg.get(
                "routing_mode_when_no_router", "dense"
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

    def _effective_routing_mode(self) -> str:
        if self.use_router:
            return self.routing_mode
        return self.routing_mode_when_no_router

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

        routing_mode = self._effective_routing_mode()
        if self.use_routed_branch:
            z_f, top_idx_f, top_w_f, selected_f = self.routed_expert_pool(
                h_f, gate_f, routing_mode=routing_mode
            )
            z_m, top_idx_m, top_w_m, selected_m = self.routed_expert_pool_m(
                h_m, gate_m, routing_mode=routing_mode
            )
            z_c, top_idx_c, top_w_c, selected_c = self.routed_expert_pool_c(
                h_c, gate_c, routing_mode=routing_mode
            )
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
            selected_f = torch.zeros_like(gate_f)
            selected_m = torch.zeros_like(gate_m)
            selected_c = torch.zeros_like(gate_c)

        if self.use_shared_branch and self.use_multiscale:
            z_shared, h_m_up, h_c_up = self.cross_scale_shared_expert(h_f, h_m, h_c)
        else:
            z_shared = torch.zeros_like(z_f)
            h_m_up = torch.zeros_like(z_f)
            h_c_up = torch.zeros_like(z_f)

        if not self.use_multiscale:
            z_m = torch.zeros_like(h_m)
            z_c = torch.zeros_like(h_c)

        if not self.use_shared_branch and not self.use_routed_branch:
            raise ValueError("At least one of shared/routed branch must be enabled.")

        route_outputs = {
            "h_route": torch.zeros_like(z_f),
            "z_c_to_m": torch.zeros_like(z_m),
            "z_mc": torch.zeros_like(z_m),
            "z_mc_to_f": torch.zeros_like(z_f),
            "gate_16": torch.zeros(
                z_f.shape[0],
                2,
                z_m.shape[2],
                z_m.shape[3],
                z_m.shape[4],
                device=z_f.device,
                dtype=z_f.dtype,
            ),
            "gate_32_route": torch.zeros(
                z_f.shape[0],
                2,
                z_f.shape[2],
                z_f.shape[3],
                z_f.shape[4],
                device=z_f.device,
                dtype=z_f.dtype,
            ),
        }
        h_shared = torch.zeros_like(z_f)
        h_route_proj = torch.zeros_like(z_f)
        route_gamma = torch.zeros((), device=z_f.device, dtype=z_f.dtype)

        if self.use_routed_branch:
            route_outputs = self.route_fusion(z_f=z_f, z_m=z_m, z_c=z_c)

        if self.use_shared_branch and not self.use_routed_branch:
            h_shared = self.branch_fusion.refine_shared(z_shared)

        if self.use_shared_branch and not self.use_routed_branch:
            h_main = h_shared
            branch_mode = "shared_only"
        elif self.use_routed_branch and not self.use_shared_branch:
            h_main = route_outputs["h_route"]
            branch_mode = "routed_only"
        else:
            if self.branch_fusion_mode != "shared_plus_routed_residual":
                raise ValueError(f"Unsupported branch_fusion_mode: {self.branch_fusion_mode}")
            h_main, h_shared, h_route_proj = self.branch_fusion(
                z_shared=z_shared,
                h_route=route_outputs["h_route"],
            )
            route_gamma = torch.sigmoid(self.branch_fusion.route_gamma)
            branch_mode = "shared_plus_routed_residual"

        x_hat_main = self.pred_head(h_main)
        return {
            "x_hat_main": x_hat_main,
            "h_st_aux": h_main,
            "gates": {
                "fine": gate_f,
                "mid": gate_m,
                "coarse": gate_c,
                "route_fusion_16": route_outputs["gate_16"],
                "route_fusion_32": route_outputs["gate_32_route"],
            },
            "topk": {
                "fine_indices": top_idx_f,
                "fine_weights": top_w_f,
                "mid_indices": top_idx_m,
                "mid_weights": top_w_m,
                "coarse_indices": top_idx_c,
                "coarse_weights": top_w_c,
            },
            "selected_masks": {
                "fine": selected_f,
                "mid": selected_m,
                "coarse": selected_c,
            },
            "features": {
                "h_f": h_f,
                "h_m": h_m,
                "h_c": h_c,
                "z_f": z_f,
                "z_m": z_m,
                "z_c": z_c,
                "z_c_to_m": route_outputs["z_c_to_m"],
                "z_mc": route_outputs["z_mc"],
                "z_mc_to_f": route_outputs["z_mc_to_f"],
                "z_shared": z_shared,
                "h_shared": h_shared,
                "h_route": route_outputs["h_route"],
                "h_route_proj": h_route_proj,
                "h_m_up": h_m_up,
                "h_c_up": h_c_up,
                "h_main": h_main,
            },
            "routing_mode": routing_mode,
            "branch_mode": branch_mode,
            "route_gamma": route_gamma.detach(),
        }


OAMSBackbone = MultiScaleMoEBackbone
ObservationAwareMultiScaleMoEImputer = MultiScaleMoEBackbone
