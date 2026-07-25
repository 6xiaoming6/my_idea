from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..blocks import ResidualSTBlock, valid_num_groups
from .safe_c2f_refiner import PredictionHead


class DirectionHead(nn.Module):
    """Lightweight bounded residual-direction predictor."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_groups: int = 8,
        dropout: float = 0.0,
        zero_init: bool = True,
        bounded_output: bool = True,
    ) -> None:
        super().__init__()
        groups = valid_num_groups(hidden_channels, num_groups)
        self.in_proj = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.block = ResidualSTBlock(
            hidden_channels,
            num_groups=num_groups,
            dropout=dropout,
        )
        self.out_proj = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)
        self.bounded_output = bool(bounded_output)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.block(self.in_proj(value))
        output = self.out_proj(hidden)
        return torch.tanh(output) if self.bounded_output else output


class BaseAnchoredResidualPyramid(nn.Module):
    """Predict bounded residual directions around the detached main prediction."""

    def __init__(
        self,
        dim: int,
        c_out: int,
        hidden: int = 32,
        anchor_embed_dim: int = 16,
        direction_embed_dim: int = 16,
        num_groups: int = 8,
        dropout: float = 0.0,
        zero_init: bool = True,
        bounded_directions: bool = True,
        use_reliability_filtered_propagation: bool = True,
        fine_only_residual: bool = False,
    ) -> None:
        super().__init__()
        if min(dim, c_out, hidden, anchor_embed_dim, direction_embed_dim) <= 0:
            raise ValueError("all channel dimensions must be positive")
        self.use_reliability_filtered_propagation = bool(
            use_reliability_filtered_propagation
        )
        self.fine_only_residual = bool(fine_only_residual)
        self.anchor_embed_f = nn.Conv3d(c_out, anchor_embed_dim, kernel_size=1)
        if self.fine_only_residual:
            self.anchor_embed_c = None
            self.anchor_embed_m = None
            self.coarse_to_mid_embed = None
            self.mid_to_fine_embed = None
            self.coarse_head = None
            self.mid_head = None
            fine_in_channels = dim + dim + anchor_embed_dim
        else:
            self.anchor_embed_c = nn.Conv3d(
                c_out, anchor_embed_dim, kernel_size=1
            )
            self.anchor_embed_m = nn.Conv3d(
                c_out, anchor_embed_dim, kernel_size=1
            )
            self.coarse_to_mid_embed = nn.Conv3d(
                c_out, direction_embed_dim, kernel_size=1
            )
            self.mid_to_fine_embed = nn.Conv3d(
                c_out, direction_embed_dim, kernel_size=1
            )
            self.coarse_head = DirectionHead(
                dim + anchor_embed_dim,
                hidden,
                c_out,
                num_groups=num_groups,
                dropout=dropout,
                zero_init=zero_init,
                bounded_output=bounded_directions,
            )
            self.mid_head = DirectionHead(
                dim + anchor_embed_dim + direction_embed_dim,
                hidden,
                c_out,
                num_groups=num_groups,
                dropout=dropout,
                zero_init=zero_init,
                bounded_output=bounded_directions,
            )
            fine_in_channels = (
                dim + dim + anchor_embed_dim + direction_embed_dim
            )
        self.fine_head = DirectionHead(
            fine_in_channels,
            hidden,
            c_out,
            num_groups=num_groups,
            dropout=dropout,
            zero_init=zero_init,
            bounded_output=bounded_directions,
        )

    @staticmethod
    def _resize(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            value,
            size=reference.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )

    @staticmethod
    def _validate_reliability(
        reliability: torch.Tensor,
        batch_size: int,
        name: str,
    ) -> None:
        if reliability.shape != (batch_size, 1, 1, 1, 1):
            raise ValueError(
                f"{name} must have shape [B,1,1,1,1], got {tuple(reliability.shape)}"
            )

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_c: torch.Tensor,
        h_main: torch.Tensor,
        x_base: torch.Tensor,
        reliability_m: torch.Tensor,
        reliability_c: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = x_base.shape[0]
        self._validate_reliability(reliability_m, batch_size, "reliability_m")
        self._validate_reliability(reliability_c, batch_size, "reliability_c")

        anchor_f = x_base.detach()
        anchor_m = self._resize(anchor_f, z_m)
        anchor_c = self._resize(anchor_f, z_c)
        if self.fine_only_residual:
            direction_c = anchor_c.new_zeros(anchor_c.shape)
            direction_m = anchor_m.new_zeros(anchor_m.shape)
            fine_inputs = (z_f, h_main, self.anchor_embed_f(anchor_f))
        else:
            assert self.coarse_head is not None
            assert self.mid_head is not None
            assert self.anchor_embed_c is not None
            assert self.anchor_embed_m is not None
            assert self.coarse_to_mid_embed is not None
            assert self.mid_to_fine_embed is not None
            direction_c = self.coarse_head(
                torch.cat((z_c, self.anchor_embed_c(anchor_c)), dim=1)
            )
            direction_c_up = self._resize(direction_c, z_m)
            if self.use_reliability_filtered_propagation:
                direction_c_up = (
                    direction_c_up
                    * reliability_c.to(dtype=direction_c.dtype)
                )
            direction_m = self.mid_head(
                torch.cat(
                    (
                        z_m,
                        self.anchor_embed_m(anchor_m),
                        self.coarse_to_mid_embed(direction_c_up),
                    ),
                    dim=1,
                )
            )
            direction_m_up = self._resize(direction_m, z_f)
            if self.use_reliability_filtered_propagation:
                direction_m_up = (
                    direction_m_up
                    * reliability_m.to(dtype=direction_m.dtype)
                )
            fine_inputs = (
                z_f,
                h_main,
                self.anchor_embed_f(anchor_f),
                self.mid_to_fine_embed(direction_m_up),
            )
        direction_f = self.fine_head(torch.cat(fine_inputs, dim=1))
        return {
            "anchor_f": anchor_f,
            "anchor_m": anchor_m,
            "anchor_c": anchor_c,
            "direction_f": direction_f,
            "direction_m": direction_m,
            "direction_c": direction_c,
        }


class AbsoluteCoarseToFinePyramid(nn.Module):
    """Absolute C2F candidate used only by the formal V18 ablation."""

    def __init__(
        self,
        dim: int,
        c_out: int,
        hidden: int = 32,
        prediction_embed_dim: int = 16,
        num_groups: int = 8,
        dropout: float = 0.0,
        use_reliability_filtered_propagation: bool = True,
    ) -> None:
        super().__init__()
        self.use_reliability_filtered_propagation = bool(
            use_reliability_filtered_propagation
        )
        self.coarse_head = PredictionHead(
            dim, hidden, c_out, num_groups, dropout
        )
        self.mid_pred_embed = nn.Conv3d(
            c_out, prediction_embed_dim, kernel_size=1
        )
        self.mid_residual_head = PredictionHead(
            dim + prediction_embed_dim,
            hidden,
            c_out,
            num_groups,
            dropout,
        )
        self.fine_pred_embed = nn.Conv3d(
            c_out, prediction_embed_dim, kernel_size=1
        )
        self.fine_residual_head = PredictionHead(
            dim + dim + prediction_embed_dim,
            hidden,
            c_out,
            num_groups,
            dropout,
        )

    @staticmethod
    def _resize(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            value,
            size=reference.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_c: torch.Tensor,
        h_main: torch.Tensor,
        x_base: torch.Tensor,
        reliability_m: torch.Tensor,
        reliability_c: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        anchor_f = x_base.detach()
        anchor_m = self._resize(anchor_f, z_m)
        anchor_c = self._resize(anchor_f, z_c)

        absolute_c = self.coarse_head(z_c)
        coarse_up = self._resize(absolute_c, z_m)
        if self.use_reliability_filtered_propagation:
            coarse_up = anchor_m + reliability_c.to(
                dtype=coarse_up.dtype
            ) * (coarse_up - anchor_m)
        absolute_m = coarse_up + self.mid_residual_head(
            torch.cat(
                (z_m, self.mid_pred_embed(coarse_up)),
                dim=1,
            )
        )
        mid_up = self._resize(absolute_m, z_f)
        if self.use_reliability_filtered_propagation:
            mid_up = anchor_f + reliability_m.to(
                dtype=mid_up.dtype
            ) * (mid_up - anchor_f)
        absolute_f = mid_up + self.fine_residual_head(
            torch.cat(
                (z_f, h_main, self.fine_pred_embed(mid_up)),
                dim=1,
            )
        )
        return {
            "anchor_f": anchor_f,
            "anchor_m": anchor_m,
            "anchor_c": anchor_c,
            "direction_f": absolute_f - anchor_f,
            "direction_m": absolute_m - anchor_m,
            "direction_c": absolute_c - anchor_c,
            "absolute_f": absolute_f,
            "absolute_m": absolute_m,
            "absolute_c": absolute_c,
        }
