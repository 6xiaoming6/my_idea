from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FrequencyDecomposition(nn.Module):
    """Stable pseudo-frequency decomposition using trend and residual features."""

    def __init__(
        self,
        kernel_t: int = 3,
        kernel_s: int = 3,
        mode: str = "avg_residual",
        use_fft: bool = False,
    ) -> None:
        super().__init__()
        if mode != "avg_residual":
            raise ValueError(f"Unsupported frequency mode: {mode}")
        if use_fft:
            raise ValueError("FFT decomposition is reserved for a later version; v12 uses avg_residual.")
        if kernel_t < 1 or kernel_s < 1 or kernel_t % 2 == 0 or kernel_s % 2 == 0:
            raise ValueError(
                f"Frequency kernels must be positive odd integers, got {(kernel_t, kernel_s)}"
            )
        self.kernel_t = kernel_t
        self.kernel_s = kernel_s
        self.mode = mode

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h.ndim != 5:
            raise ValueError(f"Expected [B,D,T,H,W], got {tuple(h.shape)}")
        pad_t = self.kernel_t // 2
        pad_s = self.kernel_s // 2
        # Replicate padding avoids injecting artificial high-frequency edges.
        padded = F.pad(h, (pad_s, pad_s, pad_s, pad_s, pad_t, pad_t), mode="replicate")
        h_low = F.avg_pool3d(
            padded,
            kernel_size=(self.kernel_t, self.kernel_s, self.kernel_s),
            stride=1,
        )
        return h_low, h - h_low
