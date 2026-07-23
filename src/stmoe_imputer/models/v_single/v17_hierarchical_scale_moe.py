from __future__ import annotations

import torch
from torch import nn

from ..embedding import ScaleTokenEncoder
from ..experts import TopKRoutedExpertPool
from ..fusion import (
    GatedCrossScaleSharedExpert,
    ReliabilityAwareScaleGate,
    SharedRoutedResidualFusion,
)
from ..router import QualityRouter
from ..scale_utils import build_scale_active_mask
from ..stats import compute_observation_stats
from .fine_preserved_scale_fusion import (
    FinePreservedParallelRouteFusion,
    FinePreservedScaleWeight,
    ScaleWeightedProgressiveRouteFusion,
)
from .hierarchical_scale_expert_router import HierarchicalScaleExpertRouter
from .scale_specific_adapter import IdentityScaleAdapter, ScaleSpecificAdapter


def _mean_reliability(
    reliability: torch.Tensor | None,
    fallback_mask: torch.Tensor,
) -> torch.Tensor:
    source = fallback_mask.float() if reliability is None else reliability.float()
    return source.mean(dim=(1, 2, 3, 4), keepdim=False).view(source.shape[0], 1)


def _rms_per_sample(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().flatten(1).mean(dim=1).sqrt()


class V17HierarchicalScaleMoEBackbone(nn.Module):
    """V17 HSA-MoE backbone with unified scale/expert/branch routing."""

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
        share_experts: bool = True,
        routing_mode: str = "topk",
        scale_mode: str = "fine_mid_coarse",
        route_dropout: float = 0.0,
        enable_branch_aux: bool = True,
        enable_complementary_loss: bool = True,
        adapter_enabled: bool = True,
        adapter_dim: int = 16,
        adapter_dropout: float = 0.0,
        adapter_zero_init: bool = True,
        router_local_dim: int = 32,
        router_global_dim: int = 64,
        router_scale_embed_dim: int = 8,
        scale_temperature: float = 1.0,
        reliability_prior_enabled: bool = True,
        reliability_prior_init: float = 1.0,
        unified_scale_expert_router: bool = True,
        expert_router_mode: str | None = None,
        unified_scale_weight: bool = True,
        sample_route_gate: bool = True,
        route_gate_bias: float = -3.0,
        route_gate_zero_init: bool = True,
        fine_floor: float = 0.25,
        fine_floor_mode: str = "linear",
        mid_projection: bool = True,
        coarse_projection: bool = True,
        route_fusion: str = "fine_preserved_parallel",
    ) -> None:
        super().__init__()
        if q_dim != 5:
            raise ValueError(
                "V17 currently requires q_dim=5 to match compute_observation_stats"
            )
        if routing_mode not in {"topk", "soft_topk", "dense"}:
            raise ValueError(f"Unsupported routing_mode: {routing_mode}")

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(max(1, top_k), num_experts)
        self.routing_mode = routing_mode
        self.scale_mode = scale_mode
        self.enable_branch_aux = enable_branch_aux
        self.enable_complementary_loss = enable_complementary_loss
        self.adapter_enabled = adapter_enabled
        self.unified_scale_expert_router = unified_scale_expert_router
        if expert_router_mode is None:
            expert_router_mode = (
                "hierarchical_shared_head"
                if unified_scale_expert_router
                else "decoupled"
            )
        if expert_router_mode not in {"hierarchical_shared_head", "decoupled"}:
            raise ValueError(
                "expert_router_mode must be 'hierarchical_shared_head' or 'decoupled'"
            )
        self.expert_router_mode = expert_router_mode
        self.unified_scale_weight = bool(unified_scale_weight)
        self.sample_route_gate = sample_route_gate
        self.route_fusion_mode = route_fusion
        if route_fusion not in {"fine_preserved_parallel", "progressive"}:
            raise ValueError(
                "route_fusion must be 'fine_preserved_parallel' or 'progressive'"
            )

        self.embed_f = ScaleTokenEncoder(c_in, dim, max_t, h, w, num_groups=num_groups)
        self.embed_m = ScaleTokenEncoder(
            c_in, dim, max_t, h // 2, w // 2, num_groups=num_groups
        )
        self.embed_c = ScaleTokenEncoder(
            c_in, dim, max_t, h // 4, w // 4, num_groups=num_groups
        )

        adapter_type = ScaleSpecificAdapter if adapter_enabled else IdentityScaleAdapter
        if adapter_enabled:
            adapter_kwargs = {
                "dim": dim,
                "bottleneck_dim": adapter_dim,
                "dropout": adapter_dropout,
                "zero_init": adapter_zero_init,
            }
            self.adapter_f = adapter_type(**adapter_kwargs)
            self.adapter_m = adapter_type(**adapter_kwargs)
            self.adapter_c = adapter_type(**adapter_kwargs)
        else:
            self.adapter_f = adapter_type()
            self.adapter_m = adapter_type()
            self.adapter_c = adapter_type()

        self.hierarchical_router = HierarchicalScaleExpertRouter(
            dim=dim,
            q_dim=q_dim,
            num_experts=num_experts,
            local_dim=router_local_dim,
            global_dim=router_global_dim,
            scale_embed_dim=router_scale_embed_dim,
            scale_temperature=scale_temperature,
            reliability_prior_enabled=reliability_prior_enabled,
            reliability_prior_init=reliability_prior_init,
            route_gate_bias=route_gate_bias,
            route_gate_zero_init=route_gate_zero_init,
        )
        if self.expert_router_mode == "decoupled":
            self.router_f = QualityRouter(dim, q_dim, num_experts)
            self.router_m = QualityRouter(dim, q_dim, num_experts)
            self.router_c = QualityRouter(dim, q_dim, num_experts)
        if not unified_scale_expert_router:
            self.decoupled_scale_gate = ReliabilityAwareScaleGate(
                dim=dim,
                stat_dim=q_dim,
                hidden_dim=max(dim * 2, 32),
                dropout=dropout,
            )

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

        self.fine_preserve_weight = FinePreservedScaleWeight(
            fine_floor=fine_floor,
            mode=fine_floor_mode,
        )
        if route_fusion == "fine_preserved_parallel":
            self.route_fusion = FinePreservedParallelRouteFusion(
                dim=dim,
                num_groups=num_groups,
                dropout=dropout,
                mid_projection=mid_projection,
                coarse_projection=coarse_projection,
            )
        else:
            self.route_fusion = ScaleWeightedProgressiveRouteFusion(
                dim=dim, num_groups=num_groups, dropout=dropout
            )
        self.cross_scale_shared_expert = GatedCrossScaleSharedExpert(
            dim=dim,
            stat_dim=q_dim,
            num_groups=num_groups,
            dropout=dropout,
            use_scale_gate=not self.unified_scale_weight,
        )
        self.branch_fusion = SharedRoutedResidualFusion(
            dim=dim,
            num_groups=num_groups,
            dropout=dropout,
            route_gamma_init=route_gate_bias,
            branch_fusion_mode="residual",
            q_dim=q_dim,
            route_dropout=route_dropout,
        )

        head_hidden = max(1, dim // 2)
        self.pred_head = nn.Sequential(
            nn.Conv3d(dim, head_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(head_hidden, c_in, kernel_size=1),
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
    def from_config(cls, cfg: dict) -> "V17HierarchicalScaleMoEBackbone":
        model_cfg = cfg["model"]
        main_cfg = model_cfg["main"]
        v17_cfg = model_cfg.get("v17", {})
        data_syn = cfg["data"]["synthetic"]
        return cls(
            c_in=model_cfg["c_in"],
            dim=main_cfg.get("dim", 64),
            num_experts=main_cfg.get("num_experts", 4),
            top_k=main_cfg.get("top_k", 2),
            max_t=main_cfg.get("max_t", data_syn["t"]),
            h=main_cfg.get("h", data_syn["h"]),
            w=main_cfg.get("w", data_syn["w"]),
            q_dim=main_cfg.get("q_dim", 5),
            num_groups=main_cfg.get("num_groups", 8),
            dropout=main_cfg.get("dropout", 0.0),
            share_experts=main_cfg.get("share_experts", True),
            routing_mode=main_cfg.get("routing_mode", "topk"),
            scale_mode=main_cfg.get(
                "scale_mode", model_cfg.get("scale_mode", "fine_mid_coarse")
            ),
            route_dropout=main_cfg.get("route_dropout", 0.0),
            enable_branch_aux=main_cfg.get("enable_branch_aux", True),
            enable_complementary_loss=main_cfg.get("enable_complementary_loss", True),
            adapter_enabled=v17_cfg.get("adapter_enabled", True),
            adapter_dim=v17_cfg.get("adapter_dim", 16),
            adapter_dropout=v17_cfg.get("adapter_dropout", 0.0),
            adapter_zero_init=v17_cfg.get("adapter_zero_init", True),
            router_local_dim=v17_cfg.get("router_local_dim", 32),
            router_global_dim=v17_cfg.get("router_global_dim", 64),
            router_scale_embed_dim=v17_cfg.get("router_scale_embed_dim", 8),
            scale_temperature=v17_cfg.get("scale_temperature", 1.0),
            reliability_prior_enabled=v17_cfg.get("reliability_prior_enabled", True),
            reliability_prior_init=v17_cfg.get("reliability_prior_init", 1.0),
            unified_scale_expert_router=v17_cfg.get(
                "unified_scale_expert_router", True
            ),
            expert_router_mode=v17_cfg.get("expert_router_mode"),
            unified_scale_weight=v17_cfg.get("unified_scale_weight", True),
            sample_route_gate=v17_cfg.get("sample_route_gate", True),
            route_gate_bias=v17_cfg.get("route_gate_bias", -3.0),
            route_gate_zero_init=v17_cfg.get("route_gate_zero_init", True),
            fine_floor=v17_cfg.get("fine_floor", 0.25),
            fine_floor_mode=v17_cfg.get("fine_floor_mode", "linear"),
            mid_projection=v17_cfg.get("mid_projection", True),
            coarse_projection=v17_cfg.get("coarse_projection", True),
            route_fusion=v17_cfg.get("route_fusion", "fine_preserved_parallel"),
        )

    def _scale_embed_vec(
        self, embed_module: ScaleTokenEncoder, batch_size: int
    ) -> torch.Tensor:
        return embed_module.scale_embed.view(1, self.dim).expand(batch_size, self.dim)

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
        active_mask = build_scale_active_mask(self.scale_mode, batch_size, x_f.device)

        h_f_raw = self.embed_f(x_f, m_f)
        h_m_raw = self.embed_m(x_m, m_m)
        h_c_raw = self.embed_c(x_c, m_c)
        h_f = self.adapter_f(h_f_raw)
        h_m = self.adapter_m(h_m_raw)
        h_c = self.adapter_c(h_c_raw)

        q_f = compute_observation_stats(m_f)
        q_m = compute_observation_stats(m_m)
        q_c = compute_observation_stats(m_c)
        rel_f = q_f[:, 1:2]
        rel_m = _mean_reliability(r_m, m_m)
        rel_c = _mean_reliability(r_c, m_c)
        routing = self.hierarchical_router(
            h_f=h_f,
            h_m=h_m,
            h_c=h_c,
            q_f=q_f,
            q_m=q_m,
            q_c=q_c,
            rel_f=rel_f,
            rel_m=rel_m,
            rel_c=rel_c,
            active_scale_mask=active_mask,
        )

        if self.expert_router_mode == "hierarchical_shared_head":
            gate_f = routing["expert_gate_f"]
            gate_m = routing["expert_gate_m"]
            gate_c = routing["expert_gate_c"]
        else:
            gate_f = self.router_f(
                h_f, q_f, self._scale_embed_vec(self.embed_f, batch_size)
            )
            gate_m = self.router_m(
                h_m, q_m, self._scale_embed_vec(self.embed_m, batch_size)
            )
            gate_c = self.router_c(
                h_c, q_c, self._scale_embed_vec(self.embed_c, batch_size)
            )

        if self.unified_scale_expert_router:
            scale_weight_raw = routing["scale_weight"]
        else:
            active_f = active_mask[:, 0].to(h_f.dtype).view(batch_size, 1, 1, 1, 1)
            active_m = active_mask[:, 1].to(h_f.dtype).view(batch_size, 1, 1, 1, 1)
            active_c = active_mask[:, 2].to(h_f.dtype).view(batch_size, 1, 1, 1, 1)
            scale_weight_raw = self.decoupled_scale_gate(
                h_f=h_f * active_f,
                h_m=h_m * active_m,
                h_c=h_c * active_c,
                q_f=q_f * active_mask[:, 0:1].to(q_f.dtype),
                q_m=q_m * active_mask[:, 1:2].to(q_m.dtype),
                q_c=q_c * active_mask[:, 2:3].to(q_c.dtype),
                r_m=(m_m.float() if r_m is None else r_m) * active_m,
                r_c=(m_c.float() if r_c is None else r_c) * active_c,
                active_mask=active_mask,
            )
        z_f, top_idx_f, top_w_f, selected_f = self.routed_expert_pool(
            h_f, gate_f, routing_mode=self.routing_mode
        )
        z_m, top_idx_m, top_w_m, selected_m = self.routed_expert_pool_m(
            h_m, gate_m, routing_mode=self.routing_mode
        )
        z_c, top_idx_c, top_w_c, selected_c = self.routed_expert_pool_c(
            h_c, gate_c, routing_mode=self.routing_mode
        )

        scale_weight = self.fine_preserve_weight(scale_weight_raw, active_mask)
        if self.route_fusion_mode == "fine_preserved_parallel":
            route_outputs = self.route_fusion(
                z_f=z_f,
                z_m=z_m,
                z_c=z_c,
                scale_weight=scale_weight,
            )
            route_feature_outputs = {
                "z_m_to_f": route_outputs["z_m_to_f"],
                "z_c_to_f": route_outputs["z_c_to_f"],
                "h_route_mix": route_outputs["h_route_mix"],
            }
        else:
            route_outputs = self.route_fusion(
                z_f=z_f,
                z_m=z_m,
                z_c=z_c,
                scale_weight=scale_weight,
            )
            route_feature_outputs = {
                "z_m_to_f": route_outputs["z_m_to_f"],
                "z_c_to_f": route_outputs["z_c_to_f"],
                "h_route_mix": route_outputs["h_route_mix"],
            }
        shared_external_weight = scale_weight if self.unified_scale_weight else None
        z_shared, h_m_up, h_c_up, shared_scale_weight = self.cross_scale_shared_expert(
            h_f=h_f,
            h_m=h_m,
            h_c=h_c,
            q_f=q_f,
            q_m=q_m,
            q_c=q_c,
            r_m=r_m,
            r_c=r_c,
            active_mask=active_mask,
            external_scale_weight=shared_external_weight,
        )
        predicted_route_gate = routing["route_branch_gate"]
        external_route_gate = predicted_route_gate if self.sample_route_gate else None
        h_main, h_shared, h_route_proj, branch_gate = self.branch_fusion(
            z_shared=z_shared,
            h_route=route_outputs["h_route"],
            q_f=q_f,
            scale_gate=scale_weight,
            external_route_gate=external_route_gate,
        )
        if self.sample_route_gate:
            route_branch_gate = predicted_route_gate
        else:
            global_route_gate = torch.sigmoid(self.branch_fusion.route_gamma)
            route_branch_gate = global_route_gate.expand(batch_size).view(
                batch_size, 1, 1, 1, 1
            )

        x_hat_main = self.pred_head(h_main)
        x_hat_shared = self.shared_aux_head(h_shared) if self.enable_branch_aux else None
        x_hat_route = self.route_aux_head(h_route_proj) if self.enable_branch_aux else None

        adapter_delta = {
            "fine": h_f - h_f_raw,
            "mid": h_m - h_m_raw,
            "coarse": h_c - h_c_raw,
        }
        adapter_base = {"fine": h_f_raw, "mid": h_m_raw, "coarse": h_c_raw}
        adapter_delta_rms = {
            name: _rms_per_sample(value) for name, value in adapter_delta.items()
        }
        adapter_relative_rms = {
            name: adapter_delta_rms[name]
            / _rms_per_sample(adapter_base[name]).clamp_min(1e-6)
            for name in adapter_delta
        }
        scale_entropy = -(
            scale_weight * scale_weight.clamp_min(1e-8).log()
        ).sum(dim=1)
        shared_scale_entropy = -(
            shared_scale_weight * shared_scale_weight.clamp_min(1e-8).log()
        ).sum(dim=1)
        shared_routed_scale_l1 = (shared_scale_weight - scale_weight).abs().sum(dim=1)
        shared_routed_scale_cosine = torch.nn.functional.cosine_similarity(
            shared_scale_weight, scale_weight, dim=1, eps=1e-8
        )
        joint_scale_expert = torch.stack(
            [
                scale_weight[:, 0:1] * gate_f,
                scale_weight[:, 1:2] * gate_m,
                scale_weight[:, 2:3] * gate_c,
            ],
            dim=1,
        )

        return {
            "x_hat_main": x_hat_main,
            "x_hat_shared": x_hat_shared,
            "x_hat_route": x_hat_route,
            "h_st_aux": h_main,
            "gates": {
                "fine": gate_f,
                "mid": gate_m,
                "coarse": gate_c,
                "scale_gate": scale_weight,
                "scale_gate_raw": scale_weight_raw,
                "branch_gate": branch_gate,
                "route_branch_gate": route_branch_gate,
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
                "z_m_to_f": route_feature_outputs["z_m_to_f"],
                "z_c_to_f": route_feature_outputs["z_c_to_f"],
                "h_route_mix": route_feature_outputs["h_route_mix"],
                "z_shared": z_shared,
                "h_shared": h_shared,
                "h_route": route_outputs["h_route"],
                "h_route_proj": h_route_proj,
                "h_m_up": h_m_up,
                "h_c_up": h_c_up,
                "h_main": h_main,
            },
            "routing_mode": self.routing_mode,
            "branch_mode": "sample_residual" if self.sample_route_gate else "global_residual",
            "scale_mode": self.scale_mode,
            "enable_branch_aux": self.enable_branch_aux,
            "enable_complementary_loss": self.enable_complementary_loss,
            "route_gamma": route_branch_gate.detach().mean(),
            "v17_enabled": True,
            "v17_router_mode": (
                "hierarchical" if self.unified_scale_expert_router else "decoupled"
            ),
            "v17_scale_router_mode": (
                "hierarchical" if self.unified_scale_expert_router else "decoupled"
            ),
            "v17_expert_router_mode": self.expert_router_mode,
            "v17_fine_floor_mode": self.fine_preserve_weight.mode,
            "v17_unified_scale_weight": self.unified_scale_weight,
            "v17_route_fusion": self.route_fusion_mode,
            "diagnostics": {
                "v17": {
                    "active_scale_mask": active_mask,
                    "scale_weight_raw": scale_weight_raw,
                    "scale_weight": scale_weight,
                    "routed_scale_weight": scale_weight,
                    "shared_scale_weight": shared_scale_weight,
                    "scale_entropy": scale_entropy,
                    "shared_scale_entropy": shared_scale_entropy,
                    "scale_top1": scale_weight.argmax(dim=1),
                    "shared_scale_top1": shared_scale_weight.argmax(dim=1),
                    "shared_routed_scale_l1": shared_routed_scale_l1,
                    "shared_routed_scale_cosine": shared_routed_scale_cosine,
                    "fine_floor_adjustment_l1": (
                        scale_weight - scale_weight_raw
                    ).abs().sum(dim=1),
                    "route_branch_gate": route_branch_gate.flatten(1)[:, 0],
                    "missing_rate": q_f[:, 0],
                    "reliability_strength": routing["reliability_strength"],
                    "joint_scale_expert": joint_scale_expert,
                    "adapter_delta_rms": adapter_delta_rms,
                    "adapter_relative_rms": adapter_relative_rms,
                }
            },
        }
