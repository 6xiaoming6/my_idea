"""Optional grouped router inputs and observed-only local pattern summaries."""
from __future__ import annotations

import torch
from torch import nn


class GroupedRouter(nn.Module):
    def __init__(self, sizes, hidden_dim, experts, projection_dim=16):
        super().__init__()
        self.sizes = tuple(sizes)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(size, projection_dim), nn.LayerNorm(projection_dim), nn.GELU())
            for size in self.sizes
        ])
        self.head = nn.Sequential(nn.Linear(len(sizes) * projection_dim, hidden_dim),
                                  nn.GELU(), nn.Linear(hidden_dim, experts))

    def forward(self, features):
        return self.head(torch.cat([layer(part) for layer, part in
                                   zip(self.projections, features.split(self.sizes, dim=-1))], dim=-1))


class PreviousExpertRouter(nn.Sequential):
    """Keep the original MLP; condition its hidden activation on the last choice.

    Zero initialization preserves the base model's initial outputs and RNG.
    The discrete choice is detached; normal gradients through the evolving
    imputation state remain unchanged. No expert repeat/exclusion rule is added.
    """
    def __init__(self, original, experts):
        super().__init__(*original.children())
        self.previous_expert_embedding = nn.Parameter(torch.zeros(experts, self[1].out_features))

    def forward(self, features, previous_choice):
        hidden = self[1](self[0](features))
        context = previous_choice.detach().to(self.previous_expert_embedding.dtype) @ self.previous_expert_embedding
        return self[3](self[2](hidden + context.to(hidden.dtype)))


class PairRouter(nn.Module):
    """Legacy individual router plus an optional zero-initialized pair interaction.

    Keeping the same four-layer sequence preserves the initial individual
    logits of the native router under the same seed. The pair head lives under
    ``routers.*`` so optimizer and gradient diagnostics include it.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int,
                 num_pairs: int, interaction: bool) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, num_experts),
        )
        self.pair_head = None
        self._pair_head_shape = (hidden_dim, num_pairs)
        if interaction:
            self.enable_interaction()

    def enable_interaction(self) -> None:
        hidden_dim, num_pairs = self._pair_head_shape
        self.pair_head = nn.Linear(hidden_dim, num_pairs)
        nn.init.zeros_(self.pair_head.weight)
        nn.init.zeros_(self.pair_head.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.layers[2](self.layers[1](self.layers[0](features)))
        return self.layers[3](hidden), (self.pair_head(hidden) if self.pair_head is not None else None)


def observed_pattern_features(values, observed, value_std):
    """Abs differences, observed pair fraction and presence for T/H/W per channel.

    Only pairs with both endpoints observed contribute. Missing payloads never
    enter arithmetic, and no target, completion or cross-gap difference is used.
    """
    x = torch.where(observed.bool(), values, 0).float() / value_std.float()
    result = []
    for axis in (2, 3, 4):
        if x.shape[axis] < 2:
            result.extend([x.new_zeros(x.shape[:2])] * 3)
            continue
        left, right = [slice(None)] * 5, [slice(None)] * 5
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        valid = observed[left].bool() & observed[right].bool()
        count = valid.sum((2, 3, 4))
        total = torch.where(valid, (x[left] - x[right]).abs(), 0).sum((2, 3, 4))
        result.extend([total / count.clamp_min(1), valid.float().mean((2, 3, 4)),
                       (count > 0).float()])
    return torch.cat(result, dim=1)
