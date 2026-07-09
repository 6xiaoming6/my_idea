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
        difficulty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del difficulty
        pooled = h.mean(dim=(2, 3, 4))
        logits = self.net(torch.cat([pooled, q, scale_embed_vec], dim=1))
        return torch.softmax(logits, dim=-1)


class DifficultyAwareRouter(QualityRouter):
    """QualityRouter plus a zero-initialized difficulty-conditioned residual."""

    def __init__(
        self,
        dim: int,
        q_dim: int,
        num_experts: int,
        difficulty_dim: int = 16,
        dropout: float = 0.1,
        zero_init: bool = True,
        mode: str = "hybrid",
    ) -> None:
        super().__init__(dim=dim, q_dim=q_dim, num_experts=num_experts)
        if mode not in {"hybrid", "difficulty_only"}:
            raise ValueError(f"Unsupported difficulty_router_mode: {mode}")
        self.mode = mode
        hidden_dim = max(dim // 2, 16)
        self.difficulty_net = nn.Sequential(
            nn.Linear(difficulty_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        if zero_init:
            nn.init.zeros_(self.difficulty_net[-1].weight)
            nn.init.zeros_(self.difficulty_net[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        q: torch.Tensor,
        scale_embed_vec: torch.Tensor,
        difficulty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = h.mean(dim=(2, 3, 4))
        base_logits = self.net(torch.cat([pooled, q, scale_embed_vec], dim=1))
        if difficulty is None:
            logits = base_logits
        else:
            difficulty_logits = self.difficulty_net(difficulty)
            logits = difficulty_logits if self.mode == "difficulty_only" else base_logits + difficulty_logits
        return torch.softmax(logits, dim=-1)


def uniform_gate(batch_size: int, num_experts: int, device, dtype) -> torch.Tensor:
    return torch.full((batch_size, num_experts), 1.0 / num_experts, device=device, dtype=dtype)


ObservationAwareRouter = QualityRouter
