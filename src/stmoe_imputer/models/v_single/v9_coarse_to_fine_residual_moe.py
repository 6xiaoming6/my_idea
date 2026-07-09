from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..blocks import valid_num_groups
from ..embedding import ScaleTokenEncoder
from ..experts import TopKRoutedExpertPool
from ..router import QualityRouter, uniform_gate
from ..scale_utils import build_scale_active_mask
from ..stats import compute_observation_stats


class _PredictionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        c_out: int,
        hidden_channels: int,
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        groups = valid_num_groups(hidden_channels, num_groups)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(hidden_channels, c_out, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoarseToFineResidualMoE(nn.Module):
    """Coarse-to-fine residual multi-resolution MoE.

    The model first predicts the low-resolution global structure at the coarse
    scale, then learns mid-scale and fine-scale residual corrections. It keeps
    the same public output contract as ``MultiScaleMoEBackbone`` so the existing
    trainer, logger, metrics, checkpointing and baseline scripts can be reused.
    """

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
        use_router: bool = True,
        share_experts: bool = False,
        routing_mode: str = "topk",
        routing_mode_when_no_router: str = "dense",
        scale_mode: str = "fine_mid_coarse",
        alpha_m_init: float = -3.0,
        alpha_f_init: float = -3.0,
        alpha_mode: str = "learnable",
        enable_mid_residual: bool = True,
        enable_fine_residual: bool = True,
        zero_init_residual_heads: bool = False,
    ) -> None:
        super().__init__()
        if scale_mode != "fine_mid_coarse":
            raise ValueError(
                "v9 CoarseToFineResidualMoE requires scale_mode='fine_mid_coarse' "
                f"because the coarse prediction is the global prior, got {scale_mode!r}."
            )
        if alpha_mode not in {"learnable", "fixed_1"}:
            raise ValueError(f"Unsupported alpha_mode={alpha_mode!r}.")
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(max(1, top_k), num_experts)
        self.use_router = use_router
        self.share_experts = share_experts
        self.routing_mode = routing_mode
        self.routing_mode_when_no_router = routing_mode_when_no_router
        self.scale_mode = scale_mode
        self.alpha_mode = alpha_mode
        self.enable_mid_residual = enable_mid_residual
        self.enable_fine_residual = enable_fine_residual

        self.embed_f = ScaleTokenEncoder(c_in, dim, max_t, h, w, num_groups=num_groups)
        self.embed_m = ScaleTokenEncoder(c_in, dim, max_t, h // 2, w // 2, num_groups=num_groups)
        self.embed_c = ScaleTokenEncoder(c_in, dim, max_t, h // 4, w // 4, num_groups=num_groups)

        self.router_f = QualityRouter(dim, q_dim, num_experts)
        self.router_m = QualityRouter(dim, q_dim, num_experts)
        self.router_c = QualityRouter(dim, q_dim, num_experts)

        self.expert_pool_c = TopKRoutedExpertPool(
            dim, num_experts, top_k=self.top_k, num_groups=num_groups, dropout=dropout
        )
        if share_experts:
            self.expert_pool_m = self.expert_pool_c
            self.expert_pool_f = self.expert_pool_c
        else:
            self.expert_pool_m = TopKRoutedExpertPool(
                dim, num_experts, top_k=self.top_k, num_groups=num_groups, dropout=dropout
            )
            self.expert_pool_f = TopKRoutedExpertPool(
                dim, num_experts, top_k=self.top_k, num_groups=num_groups, dropout=dropout
            )

        hidden = max(1, dim // 2)
        self.coarse_head = _PredictionHead(dim, c_in, hidden, num_groups=num_groups, dropout=dropout)
        self.mid_residual_head = _PredictionHead(
            dim + c_in, c_in, hidden, num_groups=num_groups, dropout=dropout
        )
        self.fine_residual_head = _PredictionHead(
            dim + c_in, c_in, hidden, num_groups=num_groups, dropout=dropout
        )
        if zero_init_residual_heads:
            self._zero_last_conv(self.mid_residual_head)
            self._zero_last_conv(self.fine_residual_head)

        if alpha_mode == "learnable":
            self.alpha_m_logit = nn.Parameter(torch.tensor(float(alpha_m_init)))
            self.alpha_f_logit = nn.Parameter(torch.tensor(float(alpha_f_init)))
        else:
            self.register_buffer("alpha_m_logit", torch.tensor(float("inf")))
            self.register_buffer("alpha_f_logit", torch.tensor(float("inf")))

    @staticmethod
    def _zero_last_conv(module: nn.Module) -> None:
        for layer in reversed(list(module.modules())):
            if isinstance(layer, nn.Conv3d):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                break

    @classmethod
    def from_config(cls, cfg: dict) -> "CoarseToFineResidualMoE":
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
            use_router=main_cfg.get("use_router", True),
            share_experts=main_cfg.get("share_experts", False),
            routing_mode=main_cfg.get("routing_mode", "topk"),
            routing_mode_when_no_router=main_cfg.get("routing_mode_when_no_router", "dense"),
            scale_mode=main_cfg.get("scale_mode", model_cfg.get("scale_mode", "fine_mid_coarse")),
            alpha_m_init=main_cfg.get("alpha_m_init", -3.0),
            alpha_f_init=main_cfg.get("alpha_f_init", -3.0),
            alpha_mode=main_cfg.get("alpha_mode", "learnable"),
            enable_mid_residual=main_cfg.get("enable_mid_residual", True),
            enable_fine_residual=main_cfg.get("enable_fine_residual", True),
            zero_init_residual_heads=main_cfg.get("zero_init_residual_heads", False),
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

    def _alpha_m(self) -> torch.Tensor:
        if self.alpha_mode == "fixed_1":
            return torch.ones((), device=self.alpha_m_logit.device, dtype=self.alpha_m_logit.dtype)
        return torch.sigmoid(self.alpha_m_logit)

    def _alpha_f(self) -> torch.Tensor:
        if self.alpha_mode == "fixed_1":
            return torch.ones((), device=self.alpha_f_logit.device, dtype=self.alpha_f_logit.dtype)
        return torch.sigmoid(self.alpha_f_logit)

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
        del r_m, r_c
        batch_size = x_f.shape[0]

        h_f = self.embed_f(x_f, m_f)
        h_m = self.embed_m(x_m, m_m)
        h_c = self.embed_c(x_c, m_c)

        gate_f = self._route(self.router_f, h_f, m_f, self.get_scale_embed_vec(self.embed_f, batch_size))
        gate_m = self._route(self.router_m, h_m, m_m, self.get_scale_embed_vec(self.embed_m, batch_size))
        gate_c = self._route(self.router_c, h_c, m_c, self.get_scale_embed_vec(self.embed_c, batch_size))

        routing_mode = self._effective_routing_mode()
        z_c, top_idx_c, top_w_c, selected_c = self.expert_pool_c(
            h_c, gate_c, routing_mode=routing_mode
        )
        z_m, top_idx_m, top_w_m, selected_m = self.expert_pool_m(
            h_m, gate_m, routing_mode=routing_mode
        )
        z_f, top_idx_f, top_w_f, selected_f = self.expert_pool_f(
            h_f, gate_f, routing_mode=routing_mode
        )

        x_hat_coarse = self.coarse_head(z_c)
        mid_size = x_m.shape[-3:]
        fine_size = x_f.shape[-3:]
        x_c_to_m = F.interpolate(x_hat_coarse, size=mid_size, mode="trilinear", align_corners=False)
        delta_m = self.mid_residual_head(torch.cat([z_m, x_c_to_m], dim=1))
        alpha_m = self._alpha_m()
        if self.enable_mid_residual:
            x_hat_mid = x_c_to_m + alpha_m * delta_m
        else:
            x_hat_mid = x_c_to_m
            delta_m = torch.zeros_like(delta_m)

        x_m_to_f = F.interpolate(x_hat_mid, size=fine_size, mode="trilinear", align_corners=False)
        delta_f = self.fine_residual_head(torch.cat([z_f, x_m_to_f], dim=1))
        alpha_f = self._alpha_f()
        if self.enable_fine_residual:
            x_hat_main = x_m_to_f + alpha_f * delta_f
        else:
            x_hat_main = x_m_to_f
            delta_f = torch.zeros_like(delta_f)

        active_mask = build_scale_active_mask(self.scale_mode, batch_size, x_f.device)
        scale_gate = active_mask.to(dtype=x_f.dtype)
        scale_gate = scale_gate / scale_gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        branch_gate = torch.zeros(batch_size, 2, device=x_f.device, dtype=x_f.dtype)
        branch_gate[:, 1] = 1.0
        zeros_f = torch.zeros_like(z_f)
        zeros_m = torch.zeros_like(z_m)

        return {
            "x_hat_main": x_hat_main,
            "x_hat_mid": x_hat_mid,
            "x_hat_coarse": x_hat_coarse,
            "x_hat_shared": None,
            "x_hat_route": None,
            "h_st_aux": z_f,
            "gates": {
                "fine": gate_f,
                "mid": gate_m,
                "coarse": gate_c,
                "scale_gate": scale_gate,
                "branch_gate": branch_gate,
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
                "z_c_to_m": zeros_m,
                "z_m_to_f": zeros_f,
                "z_mc": zeros_m,
                "z_mc_to_f": zeros_f,
                "z_shared": zeros_f,
                "h_shared": None,
                "h_route": z_f,
                "h_route_proj": z_f,
                "h_main": z_f,
            },
            "routing_mode": routing_mode,
            "branch_mode": "coarse_to_fine_residual",
            "scale_mode": self.scale_mode,
            "diagnostics": {
                "residual_alpha_m": alpha_m.detach(),
                "residual_alpha_f": alpha_f.detach(),
                "enable_mid_residual": torch.as_tensor(
                    float(self.enable_mid_residual), device=x_f.device, dtype=x_f.dtype
                ),
                "enable_fine_residual": torch.as_tensor(
                    float(self.enable_fine_residual), device=x_f.device, dtype=x_f.dtype
                ),
            },
        }


V9CoarseToFineResidualMoE = CoarseToFineResidualMoE
