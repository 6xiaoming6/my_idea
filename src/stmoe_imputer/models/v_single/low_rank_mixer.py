from __future__ import annotations

import math

import torch
from torch import nn


class LowRankGlobalMixer(nn.Module):
    """Input-dependent global mixing through a small latent bottleneck.

    Learnable anchors first collect information from all spatio-temporal tokens.
    The original tokens then read from these input-dependent latent tokens. Both
    attention operations are O(L * R), where R is much smaller than L.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
        gamma_init: float = -3.0,
        out_init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"lowrank rank must be positive, got {rank}")
        if num_heads < 1 or dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if out_init_scale < 0:
            raise ValueError(f"out_init_scale must be non-negative, got {out_init_scale}")

        self.rank = rank
        self.num_heads = num_heads
        self.anchors = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.input_norm = nn.LayerNorm(dim)
        self.anchor_norm = nn.LayerNorm(dim)
        self.latent_norm = nn.LayerNorm(dim)
        self.collect = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.broadcast = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.global_gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        if out_init_scale == 0:
            nn.init.zeros_(self.out.weight)
        else:
            nn.init.normal_(self.out.weight, std=out_init_scale)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if h.ndim != 5:
            raise ValueError(f"expected [B,D,T,H,W], got shape={tuple(h.shape)}")
        batch_size, dim, time, height, width = h.shape
        tokens = h.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, dim)
        tokens_norm = self.input_norm(tokens)
        anchors = self.anchors.unsqueeze(0).expand(batch_size, -1, -1)

        latents, _ = self.collect(
            query=self.anchor_norm(anchors),
            key=tokens_norm,
            value=tokens_norm,
            need_weights=False,
        )
        latents = anchors + self.dropout(latents)
        update, rank_attention = self.broadcast(
            query=tokens_norm,
            key=self.latent_norm(latents),
            value=self.latent_norm(latents),
            need_weights=True,
            average_attn_weights=True,
        )
        update = self.dropout(self.out(update))
        update_5d = update.reshape(batch_size, time, height, width, dim).permute(0, 4, 1, 2, 3)
        gamma = torch.sigmoid(self.global_gamma).to(dtype=h.dtype)
        h_global = h + gamma * update_5d

        probabilities = rank_attention.detach().clamp_min(1e-8)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        if self.rank > 1:
            entropy = entropy / math.log(self.rank)
        else:
            entropy = torch.zeros_like(entropy)
        return h_global, {
            "rank": torch.as_tensor(float(self.rank), device=h.device, dtype=h.dtype),
            "rank_attention_entropy": entropy.mean(dim=1),
            "global_gamma": gamma.reshape(1),
            "global_update_norm": update_5d.pow(2).mean(dim=(1, 2, 3, 4)).sqrt(),
        }
