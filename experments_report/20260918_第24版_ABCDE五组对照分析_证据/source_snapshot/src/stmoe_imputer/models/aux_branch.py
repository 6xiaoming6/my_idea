from __future__ import annotations

import torch
from torch import nn


class NullResidualBranch(nn.Module):
    def __init__(self, c_out: int) -> None:
        super().__init__()
        self.c_out = c_out

    def forward(self, h_st_aux: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, _, t, h, w = h_st_aux.shape
        return h_st_aux.new_zeros(batch_size, self.c_out, t, h, w)


AuxiliaryPlaceholder = NullResidualBranch
