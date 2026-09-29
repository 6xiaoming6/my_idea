"""Observable-only position-wise latent repair acceptance."""
from __future__ import annotations

import math

import torch
from torch import nn


class RepairAcceptanceGate(nn.Module):
    def __init__(self, dim: int, channels: int, init_prob: float = 0.99) -> None:
        super().__init__()
        if not 0.0 < float(init_prob) < 1.0 or not math.isfinite(float(init_prob)):
            raise ValueError("repair_accept_init_prob must be finite and in (0, 1)")
        hidden = max(16, dim // 2)
        self.net = nn.Sequential(
            nn.Conv3d(2 * dim + 16 * channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(init_prob / (1.0 - init_prob)))

    @staticmethod
    def features(
        hidden_before: torch.Tensor,
        update: torch.Tensor,
        old_completion: torch.Tensor,
        candidate_completion: torch.Tensor,
        previous_change: torch.Tensor,
        original_mask: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        proposal = candidate_completion - old_completion
        return torch.cat((
            hidden_before, update, old_completion, candidate_completion,
            proposal, proposal.abs(), previous_change, original_mask, support,
        ), dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
