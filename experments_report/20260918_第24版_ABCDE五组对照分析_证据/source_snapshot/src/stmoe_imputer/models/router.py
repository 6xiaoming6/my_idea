from __future__ import annotations

import torch
from torch import nn


class QualityRouter(nn.Module):
    def __init__(self, dim: int, q_dim: int, num_experts: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + q_dim + dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_experts),
        )

    def forward(
        self,
        h: torch.Tensor,
        q: torch.Tensor,
        scale_embed_vec: torch.Tensor,
    ) -> torch.Tensor:
        pooled = h.mean(dim=(2, 3, 4))
        logits = self.net(torch.cat([pooled, q, scale_embed_vec], dim=1))
        return torch.softmax(logits, dim=-1)


def uniform_gate(batch_size: int, num_experts: int, device, dtype) -> torch.Tensor:
    return torch.full((batch_size, num_experts), 1.0 / num_experts, device=device, dtype=dtype)


ObservationAwareRouter = QualityRouter
