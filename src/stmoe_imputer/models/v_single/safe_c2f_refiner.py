from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..blocks import valid_num_groups


class PredictionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
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
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class SafeCoarseToFineRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        c_out: int,
        hidden: int = 32,
        prediction_embed_dim: int = 16,
        num_groups: int = 8,
        dropout: float = 0.0,
        correction_hidden: int = 16,
        correction_zero_init: bool = True,
        fine_uses_main_feature: bool = True,
    ) -> None:
        super().__init__()
        self.fine_uses_main_feature = fine_uses_main_feature
        self.coarse_head = PredictionHead(dim, hidden, c_out, num_groups, dropout)
        self.mid_pred_embed = nn.Conv3d(c_out, prediction_embed_dim, kernel_size=1)
        self.mid_residual_head = PredictionHead(
            dim + prediction_embed_dim, hidden, c_out, num_groups, dropout
        )
        self.fine_pred_embed = nn.Conv3d(c_out, prediction_embed_dim, kernel_size=1)
        fine_channels = dim + prediction_embed_dim + (dim if fine_uses_main_feature else 0)
        self.fine_residual_head = PredictionHead(
            fine_channels, hidden, c_out, num_groups, dropout
        )
        self.correction_adapter = nn.Sequential(
            nn.Conv3d(c_out * 3, correction_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(correction_hidden, c_out, kernel_size=1),
        )
        if correction_zero_init:
            nn.init.zeros_(self.correction_adapter[-1].weight)
            nn.init.zeros_(self.correction_adapter[-1].bias)

    def forward(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        z_c: torch.Tensor,
        h_main: torch.Tensor,
        x_base: torch.Tensor,
        alpha_mid: torch.Tensor,
        alpha_fine: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_coarse = self.coarse_head(z_c)
        x_coarse_up = F.interpolate(
            x_coarse, size=z_m.shape[-3:], mode="trilinear", align_corners=False
        )
        mid_embed = self.mid_pred_embed(x_coarse_up)
        delta_mid = self.mid_residual_head(torch.cat([z_m, mid_embed], dim=1))
        x_mid = x_coarse_up + alpha_mid * delta_mid

        x_mid_up = F.interpolate(
            x_mid, size=z_f.shape[-3:], mode="trilinear", align_corners=False
        )
        fine_embed = self.fine_pred_embed(x_mid_up)
        fine_parts = [z_f]
        if self.fine_uses_main_feature:
            fine_parts.append(h_main)
        fine_parts.append(fine_embed)
        delta_fine = self.fine_residual_head(torch.cat(fine_parts, dim=1))
        x_ctf = x_mid_up + alpha_fine * delta_fine
        correction = self.correction_adapter(
            torch.cat([x_ctf - x_base, x_base, x_ctf], dim=1)
        )
        return {
            "x_hat_coarse": x_coarse,
            "x_hat_mid": x_mid,
            "x_hat_ctf": x_ctf,
            "delta_mid": delta_mid,
            "delta_fine": delta_fine,
            "delta_ctf": correction,
        }
