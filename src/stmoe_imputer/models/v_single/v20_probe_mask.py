from __future__ import annotations

import torch
import torch.nn.functional as F


class GeometryMatchedProbeBuilder:
    """Select observed probe positions that resemble the current missing geometry."""

    def __init__(
        self,
        probe_ratio: float = 0.08,
        min_count: int = 8,
        max_count: int = 128,
        min_remaining: int = 16,
        spatial_kernel_small: int = 3,
        spatial_kernel_large: int = 7,
        temporal_kernel: int = 3,
        reliability_kernel: int = 5,
        descriptor_weights: tuple[float, float, float, float] = (1.0, 1.0, 0.5, 1.0),
        selection_mode: str = "geometry_matched",
        eps: float = 1e-6,
    ) -> None:
        if not 0.0 < probe_ratio < 1.0:
            raise ValueError(f"probe_ratio must be in (0,1), got {probe_ratio}")
        if min_count < 1 or max_count < min_count:
            raise ValueError("probe counts must satisfy 1 <= min_count <= max_count")
        if min_remaining < 1:
            raise ValueError("min_remaining must be positive")
        kernels = (
            spatial_kernel_small,
            spatial_kernel_large,
            temporal_kernel,
            reliability_kernel,
        )
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError(f"Probe descriptor kernels must be positive odd values: {kernels}")
        if len(descriptor_weights) != 4 or any(weight < 0.0 for weight in descriptor_weights):
            raise ValueError("descriptor_weights must contain four non-negative values")
        if selection_mode not in {"geometry_matched", "random"}:
            raise ValueError("selection_mode must be geometry_matched or random")
        self.probe_ratio = float(probe_ratio)
        self.min_count = int(min_count)
        self.max_count = int(max_count)
        self.min_remaining = int(min_remaining)
        self.spatial_kernel_small = int(spatial_kernel_small)
        self.spatial_kernel_large = int(spatial_kernel_large)
        self.temporal_kernel = int(temporal_kernel)
        self.reliability_kernel = int(reliability_kernel)
        self.descriptor_weights = tuple(float(value) for value in descriptor_weights)
        self.selection_mode = selection_mode
        self.eps = float(eps)

    @staticmethod
    def _spatial_neighbor_mean(
        value: torch.Tensor,
        kernel: int,
        exclude_center: bool = True,
    ) -> torch.Tensor:
        b, _, t, h, w = value.shape
        value_2d = value.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        total = F.avg_pool2d(
            value_2d,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ) * float(kernel * kernel)
        if exclude_center:
            total = total - value_2d
            denominator = float(max(kernel * kernel - 1, 1))
        else:
            denominator = float(kernel * kernel)
        result = total / denominator
        return (
            result.reshape(b, t, 1, h, w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    @staticmethod
    def _temporal_neighbor_mean(value: torch.Tensor, kernel: int) -> torch.Tensor:
        b, _, t, h, w = value.shape
        value_1d = value.permute(0, 3, 4, 1, 2).reshape(b * h * w, 1, t)
        total = F.avg_pool1d(
            value_1d,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ) * float(kernel)
        result = (total - value_1d) / float(max(kernel - 1, 1))
        return (
            result.reshape(b, h, w, 1, t)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

    def descriptors(
        self,
        mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(mask, reliability)
        mask_f = mask.float()
        reliability_f = reliability.float().clamp(0.0, 1.0)
        return torch.cat(
            (
                self._spatial_neighbor_mean(mask_f, self.spatial_kernel_small),
                self._spatial_neighbor_mean(mask_f, self.spatial_kernel_large),
                self._temporal_neighbor_mean(mask_f, self.temporal_kernel),
                self._spatial_neighbor_mean(reliability_f, self.reliability_kernel),
            ),
            dim=1,
        )

    @staticmethod
    def _validate_inputs(mask: torch.Tensor, reliability: torch.Tensor) -> None:
        if mask.ndim != 5 or mask.shape[1] != 1:
            raise ValueError(f"mask must have shape [B,1,T,H,W], got {tuple(mask.shape)}")
        if reliability.shape != mask.shape:
            raise ValueError(
                f"reliability must match mask shape {tuple(mask.shape)}, got "
                f"{tuple(reliability.shape)}"
            )

    @staticmethod
    def _deterministic_random_score(indices: torch.Tensor) -> torch.Tensor:
        # Integer hash: deterministic without touching the global RNG state.
        hashed = (indices.to(torch.int64) * 1103515245 + 12345) & 0x7FFFFFFF
        return hashed.to(torch.float64) / float(0x7FFFFFFF)

    @torch.no_grad()
    def build(
        self,
        mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        descriptors = self.descriptors(mask, reliability)
        mask_f = mask.float().clamp(0.0, 1.0)
        reliability_f = reliability.float().clamp(0.0, 1.0)
        batch_size = mask.shape[0]
        flat_size = mask[0].numel()
        descriptor_flat = descriptors.reshape(batch_size, 4, flat_size)
        candidate_flat = (mask_f.reshape(batch_size, flat_size) > 0.5)
        target_weight = (1.0 - reliability_f).reshape(batch_size, flat_size)
        missing_fallback = (1.0 - mask_f).reshape(batch_size, flat_size)
        target_sum = target_weight.sum(dim=1)
        use_fallback = target_sum <= self.eps
        target_weight = torch.where(use_fallback[:, None], missing_fallback, target_weight)
        target_sum = target_weight.sum(dim=1)
        target_descriptor = (
            (descriptor_flat * target_weight[:, None, :]).sum(dim=-1)
            / target_sum.clamp_min(self.eps)[:, None]
        )
        weights = torch.tensor(
            self.descriptor_weights,
            device=mask.device,
            dtype=descriptor_flat.dtype,
        ).view(1, 4, 1)
        distances = (
            (descriptor_flat - target_descriptor[:, :, None]).abs() * weights
        ).sum(dim=1)

        probe_flat = torch.zeros_like(candidate_flat)
        valid = torch.zeros(batch_size, device=mask.device, dtype=torch.bool)
        match_distance = torch.zeros(batch_size, device=mask.device, dtype=torch.float32)
        realized_ratio = torch.zeros(batch_size, device=mask.device, dtype=torch.float32)
        probe_count = torch.zeros(batch_size, device=mask.device, dtype=torch.long)
        observed_count = candidate_flat.sum(dim=1)

        for sample in range(batch_size):
            observed = int(observed_count[sample].item())
            available_for_probe = observed - self.min_remaining
            if target_sum[sample] <= self.eps or available_for_probe < self.min_count:
                continue
            count = int(round(observed * self.probe_ratio))
            count = max(self.min_count, min(self.max_count, count, available_for_probe))
            if count < self.min_count:
                continue
            indices = torch.nonzero(candidate_flat[sample], as_tuple=False).flatten()
            if self.selection_mode == "geometry_matched":
                scores = distances[sample, indices]
            else:
                scores = self._deterministic_random_score(indices).to(distances.dtype)
            selected_order = torch.argsort(scores, stable=True)[:count]
            selected = indices[selected_order]
            probe_flat[sample, selected] = True
            valid[sample] = True
            probe_count[sample] = count
            match_distance[sample] = distances[sample, selected].mean().float()
            realized_ratio[sample] = float(count) / float(max(observed, 1))

        probe_mask = probe_flat.reshape_as(mask).to(dtype=mask.dtype)
        return {
            "probe_mask": probe_mask,
            "valid": valid,
            "match_distance": match_distance,
            "realized_ratio": realized_ratio,
            "probe_count": probe_count,
            "observed_count": observed_count,
            "target_descriptor": target_descriptor,
        }
