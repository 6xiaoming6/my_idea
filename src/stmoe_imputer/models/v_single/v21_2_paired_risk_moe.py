"""Observed-holdout paired scale-risk calibration; no hidden targets in the gate.

The auxiliary risk is a proxy under observed-support thinning, NOT an unbiased
estimate of the deployment risk under arbitrary structured/MNAR missingness.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ...data.transforms import ensure_multiscale, masked_pool2d_spatial
from .v14_safe_c2f_moe import V14SafeC2FMoE


class PairedScaleRisk(nn.Module):
    def __init__(self, settings: dict):
        super().__init__()
        grid = settings.get("candidate_alphas", [0.125, 0.25])
        if not grid or any(not 0 < a <= 0.5 for a in grid) or sorted(set(grid)) != grid:
            raise ValueError("candidate_alphas must be unique, increasing, in (0, 0.5]")
        self.register_buffer("candidates", torch.tensor(grid, dtype=torch.float32))
        self.margin = float(settings.get("risk_margin", 0.02))
        self.temperature = float(settings.get("acceptance_temperature", 0.1))
        if not 0 <= self.margin < 1 or self.temperature <= 0:
            raise ValueError("Require 0 <= risk_margin < 1 and positive temperature")
        self.use_distortion = bool(settings.get("use_distortion", True))
        hidden = int(settings.get("hidden", 16))
        self.net = nn.Sequential(nn.Conv3d(10, hidden, 1), nn.GELU(), nn.Conv3d(hidden, len(grid), 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.05)

    def forward(self, legacy, measure, r_legacy, r_measure, evidence):
        delta = measure - legacy
        denominator = (legacy.abs().mean(1, keepdim=True) + measure.abs().mean(1, keepdim=True)).clamp_min(1e-4)
        distortion = torch.cat((
            (delta.abs().mean(1, keepdim=True) / denominator).clamp(0, 1),
            (delta.mean(1, keepdim=True) / denominator).clamp(-1, 1),
            (r_measure - r_legacy).abs().clamp(0, 1),
        ), 1)
        if not self.use_distortion:
            distortion = torch.zeros_like(distortion)
        if evidence is None or evidence.shape[1] != 7:
            raise ValueError("V21.2 requires seven-channel observed-only coarse evidence")
        regret = torch.tanh(self.net(torch.cat((distortion, evidence), 1)))
        best_regret, index = regret.min(1, keepdim=True)
        # A positive predicted regret must not result in a correction.  This is
        # a decision margin, not a statistically calibrated confidence bound.
        strength = ((-best_regret - self.margin) / self.temperature).clamp(0, 1)
        alpha = self.candidates[index] * strength
        return alpha, regret


class V21PairedRiskMoE(V14SafeC2FMoE):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.settings = dict(cfg["model"].get("v21_2", {}))
        scales = cfg["data"]["scales"]
        if scales.get("pyramid_mode") != "dual_observation_moment":
            raise ValueError("V21.2 requires dual_observation_moment")
        if scales.get("pooling_mode", "avg") != "avg":
            raise ValueError("V21.2 paired risk currently supports mean pooling only")
        self.mid = int(scales["fine_to_mid"])
        self.coarse = int(scales["fine_to_coarse"])
        self.pattern = cfg["data"].get("mask", {}).get("pattern", "random")
        if self.pattern not in {"fixed", "random"}:
            raise ValueError("V21.2 supports fixed/random holdout geometry only")
        self.mode = self.settings.get("mode", "risk")
        if self.mode not in {"risk", "constant", "anchor"}:
            raise ValueError("V21.2 mode must be risk, constant or anchor")
        self.holdout_fraction = float(self.settings.get("holdout_fraction", 0.25))
        self.min_count = int(self.settings.get("min_holdout_count", 2))
        self.proxy_weight = float(self.settings.get("proxy_weight", 1.0))
        self.constant_alpha = float(self.settings.get("constant_alpha", 0.05))
        if not 0 < self.holdout_fraction < 1 or self.min_count < 1 or self.proxy_weight < 0:
            raise ValueError("Invalid holdout_fraction, min_holdout_count or proxy_weight")
        if not 0 <= self.constant_alpha <= 0.5:
            raise ValueError("constant_alpha must be in [0, 0.5]")
        self.proxy_seed = int(cfg.get("seed", 42)) + int(self.settings.get("module_seed", 2121))
        # Do not shift V14 initialization or its dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.settings.get("module_seed", 2121)))
            self.scale_risk = PairedScaleRisk(self.settings)
        self.register_buffer("proxy_step", torch.zeros((), dtype=torch.long))

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg)

    def proxy_batch(self, observed, mask):
        """Disjoint observed support and held-out labels; local RNG is resumable."""
        generator = torch.Generator(device=observed.device).manual_seed(self.proxy_seed + int(self.proxy_step))
        shape = list(mask.shape)
        if self.pattern == "fixed":
            shape[2] = 1  # Spatial-column holdout, shared across the whole window.
        drop = (torch.rand(shape, generator=generator, device=mask.device) < self.holdout_fraction).to(mask.dtype)
        held = mask * drop
        kept = mask - held
        # ensure_multiscale uses x_f_gt as a container here, but it receives ONLY
        # already-observed values and re-masks before computing any features.
        proxy = ensure_multiscale(
            {"x_f_gt": observed.detach(), "m_f": kept},
            fine_to_mid=self.mid, fine_to_coarse=self.coarse,
            pyramid_mode="dual_observation_moment",
        )
        mean, _, fraction = masked_pool2d_spatial(observed.detach(), held, self.coarse, return_reliability=True)
        second, _, _ = masked_pool2d_spatial(observed.detach().square(), held, self.coarse, return_reliability=True)
        valid = (fraction * self.coarse**2 >= self.min_count) & (proxy["r_c_measure"] > 0)
        return proxy, mean, second, valid

    def proxy_loss(self, observed, mask):
        proxy, mean, second, valid = self.proxy_batch(observed, mask)
        self.proxy_step.add_(1)
        _, prediction = self.scale_risk(proxy["x_c_obs"], proxy["x_c_measure"], proxy["r_c"], proxy["r_c_measure"], proxy["e_c"])
        legacy = proxy["x_c_obs"]
        delta = proxy["x_c_measure"] - legacy
        # Per-cell squared risk over held-out fine entries.  In the paired
        # numerator their shared second moment cancels; no full truth is used.
        error0 = (legacy.square() - 2 * legacy * mean + second).mean(1, keepdim=True).clamp_min(0)
        targets = []
        for alpha in self.scale_risk.candidates:
            candidate = legacy + alpha * delta
            error = (candidate.square() - 2 * candidate * mean + second).mean(1, keepdim=True).clamp_min(0)
            targets.append(((error - error0) / (error + error0).clamp_min(1e-4)).clamp(-1, 1))
        target = torch.cat(targets, 1).detach()
        if self.settings.get("shuffle_targets", False):
            # Mechanism-destruction control; preserve the valid target marginal.
            target = target.clone()
            generator = torch.Generator(device=target.device).manual_seed(self.proxy_seed + int(self.proxy_step) + 1000000)
            count = int(valid.sum())
            permutation = torch.randperm(count, device=target.device, generator=generator)
            for channel in range(target.shape[1]):
                selected = target[:, channel][valid[:, 0]]
                target[:, channel][valid[:, 0]] = selected[permutation]
        weights = valid.expand_as(target).to(prediction.dtype)
        loss = (F.smooth_l1_loss(prediction, target, reduction="none") * weights).sum() / weights.sum().clamp_min(1)
        sign_accuracy = (((prediction < 0) == (target < 0)).float() * weights).sum() / weights.sum().clamp_min(1)
        return loss, {
            "proxy_valid_fraction": valid.float().mean(),
            "proxy_sign_accuracy": sign_accuracy,
            "proxy_gain_fraction": ((target < 0).float() * weights).sum() / weights.sum().clamp_min(1),
            "proxy_regret_mae": ((prediction - target).abs() * weights).sum() / weights.sum().clamp_min(1),
        }

    def forward(self, x_f, m_f, x_m, m_m, x_c, m_c, r_m=None, r_c=None,
                e_f=None, e_m=None, e_c=None, x_m_measure=None, m_m_measure=None,
                r_m_measure=None, x_c_measure=None, m_c_measure=None, r_c_measure=None):
        if x_c_measure is None or r_c_measure is None or r_c is None:
            raise ValueError("V21.2 requires dual coarse values and reliability")
        # Keep risk arithmetic in float32 even with AMP. Main V14 still uses AMP.
        with torch.autocast(device_type=x_f.device.type, enabled=False):
            alpha, regret = self.scale_risk(x_c.float(), x_c_measure.float(), r_c.float(), r_c_measure.float(), e_c.float() if e_c is not None else None)
        if self.mode != "risk":
            alpha = torch.full_like(alpha, self.constant_alpha if self.mode == "constant" else 0.0)
        # Gate semantics come only from paired risk supervision, not from an
        # unrestricted imputation gradient that can turn it into a generic gate.
        alpha = alpha.detach().to(x_c.dtype) * (r_c_measure > 0).to(x_c.dtype)
        calibrated = x_c + alpha * (x_c_measure - x_c)
        output = super().forward(x_f=x_f, m_f=m_f, x_m=x_m, m_m=m_m,
                                 x_c=calibrated, m_c=m_c, r_m=r_m, r_c=r_c,
                                 e_f=None, e_m=None, e_c=None)
        diagnostics = {
            "alpha": alpha.detach(), "accept_fraction": (alpha > 0).float(),
            "predicted_regret": regret.detach(),
            "correction_rms": (calibrated - x_c).float().square().mean().sqrt().detach(),
        }
        if self.training and self.mode == "risk" and self.proxy_weight > 0:
            with torch.autocast(device_type=x_f.device.type, enabled=False):
                proxy_loss, proxy_logs = self.proxy_loss(x_f.float(), m_f.float())
            output["v21_2_proxy_loss"] = proxy_loss
            diagnostics.update(proxy_logs)
        output.setdefault("diagnostics", {})["v21_2"] = diagnostics
        output.setdefault("features", {})["v21_2"] = {
            "alpha": alpha, "regret": regret.detach(), "calibrated_coarse": calibrated,
            "legacy_coarse": x_c, "measure_coarse": x_c_measure,
        }
        output["branch_mode"] = "v21_2_paired_scale_risk"
        return output
