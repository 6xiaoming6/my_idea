"""Additional pattern operators for a chain on a single spatiotemporal grid.

Every expert returns an update with the input shape ``[B, D, T, H, W]``.
Temporal operators share their parameters across cells without mixing cells;
spatial operators share their parameters across time without mixing times.
No operator resamples the grid or constructs a scale hierarchy.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _odd_kernel(value: int) -> int:
    value = _positive_int("kernel_size", value)
    if value % 2 != 1:
        raise ValueError(f"kernel_size must be odd, got {value}")
    return value


class _PointwiseLayerNorm(nn.Module):
    """Normalize channels independently at each time and spatial position."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


class DilatedDirectionalExpert(nn.Module):
    """A depthwise dilated convolution along time or the spatial plane.

    Dilation extends the sampling offsets on the original grid.  The other
    direction remains pointwise, and every output keeps the original indices.
    """

    def __init__(
        self, dim: int, direction: str, kernel_size: int = 3, dilation: int = 2
    ) -> None:
        super().__init__()
        dim = _positive_int("dim", dim)
        kernel_size = _odd_kernel(kernel_size)
        dilation = _positive_int("dilation", dilation)
        if direction not in {"temporal", "spatial"}:
            raise ValueError("direction must be 'temporal' or 'spatial'")
        self.direction = direction
        if direction == "temporal":
            kernel = (kernel_size, 1, 1)
            dilations = (dilation, 1, 1)
        else:
            kernel = (1, kernel_size, kernel_size)
            dilations = (1, dilation, dilation)
        hidden_dim = dim * 2
        self.network = nn.Sequential(
            _PointwiseLayerNorm(dim),
            nn.Conv3d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel,
                padding=tuple((size // 2) * step for size, step in zip(kernel, dilations)),
                dilation=dilations,
                groups=hidden_dim,
            ),
            _PointwiseLayerNorm(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TemporalAttentionExpert(nn.Module):
    """Bidirectional full-window attention applied independently to each cell.

    Position features are supplied by the chain's common input projection.
    The returned update comprises attention and a pointwise nonlinear FFN;
    the chain owns the outer residual connection.  Explicit SDPA avoids
    MultiheadAttention's separate inference path under CPU autocast.
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.num_heads = _positive_int("num_heads", num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.head_dim = self.dim // self.num_heads
        self.input_norm = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.output_projection = nn.Linear(self.dim, self.dim)
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or any(size < 1 for size in x.shape):
            raise ValueError("x must have nonempty shape [B, D, T, H, W]")
        b, d, t, h, w = x.shape
        if d != self.dim:
            raise ValueError(f"x must have {self.dim} channels, got {d}")
        tokens = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, d)
        qkv = self.qkv(self.input_norm(tokens)).reshape(
            b * h * w, t, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(b * h * w, t, d)
        attended = self.output_projection(attended)
        update = attended + self.ffn(self.ffn_norm(tokens + attended))
        return update.reshape(b, h, w, t, d).permute(0, 4, 3, 1, 2).contiguous()


class JointSpatioTemporalExpert(nn.Module):
    """Local joint time-space interactions through a depthwise 3-D kernel."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        dim = _positive_int("dim", dim)
        kernel_size = _odd_kernel(kernel_size)
        hidden_dim = dim * 2
        self.network = nn.Sequential(
            _PointwiseLayerNorm(dim),
            nn.Conv3d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                (kernel_size, kernel_size, kernel_size),
                padding=kernel_size // 2,
                groups=hidden_dim,
            ),
            _PointwiseLayerNorm(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
