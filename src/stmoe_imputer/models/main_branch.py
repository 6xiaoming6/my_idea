from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .difficulty import DifficultyDescriptor, aggregate_difficulty_score
from .embedding import ScaleTokenEncoder
from .experts import TopKRoutedExpertPool
from .fusion import (
    ExpertEnhancedSharedInput,
    GatedCrossScaleSharedExpert,
    ProgressiveRouteFusion,
    SharedRoutedResidualFusion,
)
from .router import DifficultyAwareRouter, QualityRouter, uniform_gate
from .scale_utils import build_scale_active_mask
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
        branch_fusion_mode: str = "residual",
        route_gamma_init: float = -3.0,
        routing_mode: str = "topk",
        routing_mode_when_no_router: str = "dense",
        scale_mode: str = "fine_mid_coarse",
        use_scale_gate: bool = True,
        use_reliability_gate: bool = True,
        shared_input_mode: str = "pre",
        shared_expert_beta_init: float = 0.1,
        detach_shared_expert_input: bool = False,
        branch_gate_init: str = "balanced",
        route_dropout: float = 0.0,
        enable_branch_aux: bool = True,
        enable_complementary_loss: bool = True,
        model_version: str = "main",
        use_difficulty_router: bool = False,
        difficulty_dim: int = 16,
        difficulty_hidden_dim: int = 32,
        difficulty_zero_init: bool = True,
        difficulty_descriptor_zero_init: bool = False,
        difficulty_dropout: float = 0.1,
        difficulty_router_mode: str = "hybrid",
        difficulty_use_spatial_block: bool = True,
        difficulty_use_cross_scale_consistency: bool = True,
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
        self.scale_mode = "fine" if not use_multiscale else scale_mode
        self.use_scale_gate = use_scale_gate
        self.use_reliability_gate = use_reliability_gate
        self.shared_input_mode = shared_input_mode
        self.shared_expert_beta_init = shared_expert_beta_init
        self.detach_shared_expert_input = detach_shared_expert_input
        self.branch_gate_init = branch_gate_init
        self.route_dropout = route_dropout
        self.enable_branch_aux = enable_branch_aux
        self.enable_complementary_loss = enable_complementary_loss
        self.model_version = model_version
        self.use_difficulty_router = use_difficulty_router
        self.difficulty_router_mode = difficulty_router_mode

        self.embed_f = ScaleTokenEncoder(c_in, dim, max_t, h, w, num_groups=num_groups)
        self.embed_m = ScaleTokenEncoder(c_in, dim, max_t, h // 2, w // 2, num_groups=num_groups)
        self.embed_c = ScaleTokenEncoder(c_in, dim, max_t, h // 4, w // 4, num_groups=num_groups)

        if use_difficulty_router:
            router_kwargs = {
                "dim": dim,
                "q_dim": q_dim,
                "num_experts": num_experts,
                "difficulty_dim": difficulty_dim,
                "dropout": difficulty_dropout,
                "zero_init": difficulty_zero_init,
                "mode": difficulty_router_mode,
            }
            self.router_f = DifficultyAwareRouter(**router_kwargs)
            self.router_m = DifficultyAwareRouter(**router_kwargs)
            self.router_c = DifficultyAwareRouter(**router_kwargs)
            self.difficulty_descriptor = DifficultyDescriptor(
                out_dim=difficulty_dim,
                hidden_dim=difficulty_hidden_dim,
                zero_init=difficulty_descriptor_zero_init,
                dropout=difficulty_dropout,
                use_spatial_block=difficulty_use_spatial_block,
                use_cross_scale_consistency=difficulty_use_cross_scale_consistency,
            )
        else:
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

        self.cross_scale_shared_expert = GatedCrossScaleSharedExpert(
            dim,
            stat_dim=q_dim,
            num_groups=num_groups,
            dropout=dropout,
            use_scale_gate=use_scale_gate,
        )
        self.shared_input_adapter = ExpertEnhancedSharedInput(
            dim=dim,
            mode=shared_input_mode,
            beta_init=shared_expert_beta_init,
        )
        self.route_fusion = ProgressiveRouteFusion(
            dim, num_groups=num_groups, dropout=dropout
        )
        self.branch_fusion = SharedRoutedResidualFusion(
            dim,
            num_groups=num_groups,
            dropout=dropout,
            route_gamma_init=route_gamma_init,
            branch_fusion_mode=branch_fusion_mode,
            branch_gate_init=branch_gate_init,
            q_dim=q_dim,
            route_dropout=route_dropout,
        )
        head_hidden = max(1, dim // 2)
        self.pred_head = nn.Sequential(
            nn.Conv3d(dim, max(1, dim // 2), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(max(1, dim // 2), c_in, kernel_size=1),
        )
        self.shared_aux_head = nn.Sequential(
            nn.Conv3d(dim, head_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(head_hidden, c_in, kernel_size=1),
        )
        self.route_aux_head = nn.Sequential(
            nn.Conv3d(dim, head_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(head_hidden, c_in, kernel_size=1),
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
                "branch_fusion_mode", "residual"
            ),
            route_gamma_init=main_cfg.get("route_gamma_init", -3.0),
            routing_mode=main_cfg.get("routing_mode", "topk"),
            routing_mode_when_no_router=main_cfg.get(
                "routing_mode_when_no_router", "dense"
            ),
            scale_mode=main_cfg.get("scale_mode", model_cfg.get("scale_mode", "fine_mid_coarse")),
            use_scale_gate=main_cfg.get("use_scale_gate", model_cfg.get("use_scale_gate", True)),
            use_reliability_gate=main_cfg.get(
                "use_reliability_gate", model_cfg.get("use_reliability_gate", True)
            ),
            shared_input_mode=main_cfg.get("shared_input_mode", "pre"),
            shared_expert_beta_init=main_cfg.get("shared_expert_beta_init", 0.1),
            detach_shared_expert_input=main_cfg.get("detach_shared_expert_input", False),
            branch_gate_init=main_cfg.get("branch_gate_init", "balanced"),
            route_dropout=main_cfg.get("route_dropout", 0.0),
            enable_branch_aux=main_cfg.get("enable_branch_aux", True),
            enable_complementary_loss=main_cfg.get("enable_complementary_loss", True),
            model_version=model_cfg.get("version", "main"),
            use_difficulty_router=main_cfg.get("use_difficulty_router", False),
            difficulty_dim=main_cfg.get("difficulty_dim", 16),
            difficulty_hidden_dim=main_cfg.get("difficulty_hidden_dim", 32),
            difficulty_zero_init=main_cfg.get("difficulty_zero_init", True),
            difficulty_descriptor_zero_init=main_cfg.get(
                "difficulty_descriptor_zero_init", False
            ),
            difficulty_dropout=main_cfg.get("difficulty_dropout", 0.1),
            difficulty_router_mode=main_cfg.get("difficulty_router_mode", "hybrid"),
            difficulty_use_spatial_block=main_cfg.get(
                "difficulty_use_spatial_block", True
            ),
            difficulty_use_cross_scale_consistency=main_cfg.get(
                "difficulty_use_cross_scale_consistency", True
            ),
        )

    def get_scale_embed_vec(self, embed_module: ScaleTokenEncoder, batch_size: int) -> torch.Tensor:
        return embed_module.scale_embed.view(1, self.dim).expand(batch_size, self.dim)

    def _route(
        self,
        router: QualityRouter,
        h: torch.Tensor,
        q: torch.Tensor,
        scale_embed_vec: torch.Tensor,
        difficulty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_router:
            return uniform_gate(h.shape[0], self.num_experts, h.device, h.dtype)
        return router(h, q, scale_embed_vec, difficulty=difficulty)

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
        r_m: torch.Tensor | None = None,
        r_c: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch_size = x_f.shape[0]
        if r_m is None:
            r_m = m_m.float()
        if r_c is None:
            r_c = m_c.float()

        h_f = self.embed_f(x_f, m_f)
        h_m = self.embed_m(x_m, m_m)
        h_c = self.embed_c(x_c, m_c)

        q_f = compute_observation_stats(m_f)
        q_m = compute_observation_stats(m_m)
        q_c = compute_observation_stats(m_c)

        if self.use_difficulty_router:
            target_f = x_f.shape[-3:]
            target_m = x_m.shape[-3:]
            target_c = x_c.shape[-3:]
            x_m_to_f = F.interpolate(x_m, size=target_f, mode="trilinear", align_corners=False)
            x_c_to_f = F.interpolate(x_c, size=target_f, mode="trilinear", align_corners=False)
            x_c_to_m = F.interpolate(x_c, size=target_m, mode="trilinear", align_corners=False)
            x_m_to_c = F.interpolate(x_m, size=target_c, mode="trilinear", align_corners=False)
            d_f, raw_d_f = self.difficulty_descriptor(
                x_f, m_f, h=h_f, reliability=None,
                cross_scale_reference=0.5 * (x_m_to_f + x_c_to_f),
            )
            d_m, raw_d_m = self.difficulty_descriptor(
                x_m, m_m, h=h_m, reliability=r_m, cross_scale_reference=x_c_to_m,
            )
            d_c, raw_d_c = self.difficulty_descriptor(
                x_c, m_c, h=h_c, reliability=r_c, cross_scale_reference=x_m_to_c,
            )
        else:
            d_f = d_m = d_c = None
            raw_d_f = x_f.new_zeros(batch_size, 9)
            raw_d_m = x_f.new_zeros(batch_size, 9)
            raw_d_c = x_f.new_zeros(batch_size, 9)

        difficulty_score_f = aggregate_difficulty_score(raw_d_f)
        difficulty_score_m = aggregate_difficulty_score(raw_d_m)
        difficulty_score_c = aggregate_difficulty_score(raw_d_c)

        gate_f = self._route(
            self.router_f, h_f, q_f, self.get_scale_embed_vec(self.embed_f, batch_size), d_f
        )
        gate_m = self._route(
            self.router_m, h_m, q_m, self.get_scale_embed_vec(self.embed_m, batch_size), d_m
        )
        gate_c = self._route(
            self.router_c, h_c, q_c, self.get_scale_embed_vec(self.embed_c, batch_size), d_c
        )

        routing_mode = self._effective_routing_mode()
        need_expert_features = self.use_routed_branch or (
            self.use_shared_branch and self.shared_input_mode != "pre"
        )
        if need_expert_features:
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

        active_mask = build_scale_active_mask(self.scale_mode, batch_size, x_f.device)
        scale_gate = active_mask.to(dtype=x_f.dtype)
        scale_gate = scale_gate / scale_gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.use_shared_branch:
            z_f_for_shared = z_f.detach() if self.detach_shared_expert_input else z_f
            z_m_for_shared = z_m.detach() if self.detach_shared_expert_input else z_m
            z_c_for_shared = z_c.detach() if self.detach_shared_expert_input else z_c
            h_f_shared, h_m_shared, h_c_shared = self.shared_input_adapter(
                h_f,
                h_m,
                h_c,
                z_f=z_f_for_shared,
                z_m=z_m_for_shared,
                z_c=z_c_for_shared,
            )
            z_shared, h_m_up, h_c_up, scale_gate = self.cross_scale_shared_expert(
                h_f=h_f_shared,
                h_m=h_m_shared,
                h_c=h_c_shared,
                q_f=q_f,
                q_m=q_m,
                q_c=q_c,
                r_m=r_m if self.use_reliability_gate else None,
                r_c=r_c if self.use_reliability_gate else None,
                active_mask=active_mask,
            )
        else:
            z_shared = torch.zeros_like(z_f)
            h_m_up = torch.zeros_like(z_f)
            h_c_up = torch.zeros_like(z_f)
            h_f_shared = torch.zeros_like(h_f)
            h_m_shared = torch.zeros_like(h_m)
            h_c_shared = torch.zeros_like(h_c)

        if not self.use_multiscale:
            z_m = torch.zeros_like(h_m)
            z_c = torch.zeros_like(h_c)

        if not self.use_shared_branch and not self.use_routed_branch:
            raise ValueError("At least one of shared/routed branch must be enabled.")

        route_outputs = {
            "h_route": torch.zeros_like(z_f),
            "z_c_to_m": torch.zeros_like(z_m),
            "z_mc": torch.zeros_like(z_m),
            "z_m_to_f": torch.zeros_like(z_f),
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
        branch_gate = torch.zeros(z_f.shape[0], 2, device=z_f.device, dtype=z_f.dtype)

        if self.use_routed_branch:
            route_outputs = self.route_fusion(
                z_f=z_f,
                z_m=z_m,
                z_c=z_c,
                scale_mode=self.scale_mode,
            )

        if self.use_shared_branch and not self.use_routed_branch:
            h_shared = self.branch_fusion.refine_shared(z_shared)

        if self.use_shared_branch and not self.use_routed_branch:
            h_main = h_shared
            branch_mode = "shared_only"
            branch_gate[:, 0] = 1.0
        elif self.use_routed_branch and not self.use_shared_branch:
            h_main = route_outputs["h_route"]
            branch_mode = "routed_only"
            branch_gate[:, 1] = 1.0
        else:
            h_main, h_shared, h_route_proj, branch_gate = self.branch_fusion(
                z_shared=z_shared,
                h_route=route_outputs["h_route"],
                q_f=q_f,
                scale_gate=scale_gate,
            )
            route_gamma = torch.sigmoid(self.branch_fusion.route_gamma)
            branch_mode = self.branch_fusion_mode

        x_hat_main = self.pred_head(h_main)
        is_full = self.use_shared_branch and self.use_routed_branch
        if is_full and self.enable_branch_aux:
            x_hat_shared = self.shared_aux_head(h_shared)
            x_hat_route = self.route_aux_head(h_route_proj)
        else:
            x_hat_shared = None
            x_hat_route = None
        return {
            "x_hat_main": x_hat_main,
            "x_hat_shared": x_hat_shared,
            "x_hat_route": x_hat_route,
            "h_st_aux": h_main,
            "gates": {
                "fine": gate_f,
                "mid": gate_m,
                "coarse": gate_c,
                "scale_gate": scale_gate,
                "branch_gate": branch_gate,
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
                "h_f_shared": h_f_shared,
                "h_m_shared": h_m_shared,
                "h_c_shared": h_c_shared,
                "z_f": z_f,
                "z_m": z_m,
                "z_c": z_c,
                "z_c_to_m": route_outputs["z_c_to_m"],
                "z_m_to_f": route_outputs["z_m_to_f"],
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
            "scale_mode": self.scale_mode,
            "use_scale_gate": self.use_scale_gate,
            "use_reliability_gate": self.use_reliability_gate,
            "shared_input_mode": self.shared_input_mode,
            "detach_shared_expert_input": self.detach_shared_expert_input,
            "enable_branch_aux": self.enable_branch_aux,
            "enable_complementary_loss": self.enable_complementary_loss,
            "model_version": self.model_version,
            "use_difficulty_router": self.use_difficulty_router,
            "difficulty_router_mode": self.difficulty_router_mode,
            "route_gamma": route_gamma.detach(),
            "diagnostics": {
                "shared_input_beta": self.shared_input_adapter.beta_values().detach(),
                "branch_gate": branch_gate.detach(),
                "difficulty_embeddings": {
                    "fine": d_f,
                    "mid": d_m,
                    "coarse": d_c,
                },
                "difficulty_stats": {
                    "fine": raw_d_f,
                    "mid": raw_d_m,
                    "coarse": raw_d_c,
                },
                "difficulty_scores": {
                    "fine": difficulty_score_f,
                    "mid": difficulty_score_m,
                    "coarse": difficulty_score_c,
                },
            },
        }


OAMSBackbone = MultiScaleMoEBackbone
ObservationAwareMultiScaleMoEImputer = MultiScaleMoEBackbone
