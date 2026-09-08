"""V22.1 experimental backbone extension. Historical V22 sources stay untouched.

Uniform-anchored, bounded routing with optional train-only fallback. This is a
testable optimization hypothesis, not a guarantee of lower imputation error.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from stmoe_imputer.models.v_single.v22_coarsening_moe import V22CoarseningMoE


class BoundedRouter(nn.Module):
    """Return log probabilities so the existing scale's softmax stays valid.

    g = (1-rho)/E + rho*softmax(logits), rho = max_strength*sigmoid(a(x)).
    A per-sample, per-scale train-only fallback sets rho=0 without rescaling.
    All experts still execute; this is NOT sparse expert dropout or a speedup.
    """

    def __init__(self, original, options):
        super().__init__()
        self.max_strength = float(options['max_strength'])
        initial = float(options['initial_strength'])
        self.fallback_probability = float(options['fallback_probability'])
        if not (0 < initial < self.max_strength < 1):
            raise ValueError('Require 0 < initial_strength < max_strength < 1')
        if not 0 <= self.fallback_probability < 1:
            raise ValueError('fallback_probability must be in [0, 1)')
        self.preference = original
        channels = original[0].in_channels
        self.strength = nn.Conv3d(channels, 1, 1)
        nn.init.zeros_(self.strength.weight)
        nn.init.constant_(self.strength.bias, math.log(initial / (self.max_strength - initial)))
        self.context_channels = channels - 7
        self.last_diagnostics = {}

    def forward(self, inputs):
        preference = self.preference(inputs).float().softmax(1)
        rho = self.max_strength * self.strength(inputs).float().sigmoid()
        if self.training and self.fallback_probability:
            keep = (torch.rand((inputs.shape[0], 1, 1, 1, 1), device=inputs.device)
                    >= self.fallback_probability).to(rho.dtype)
        else:
            keep = rho.new_ones(inputs.shape[0], 1, 1, 1, 1)
        effective_rho = rho * keep
        gate = (1 - effective_rho) / preference.shape[1] + effective_rho * preference
        # Detached diagnostics must not retain the just-completed autograd graph.
        with torch.no_grad():
            observed = inputs[:, self.context_channels:self.context_channels + 1].float()
            missing = 1 - observed
            self.last_diagnostics = {
                'mix_strength': rho.mean().detach(),
                'effective_mix_strength': effective_rho.mean().detach(),
                'mix_strength_missing': ((rho * missing).sum() / missing.sum().clamp_min(1)).detach(),
                'mix_strength_observed': ((rho * observed).sum() / observed.sum().clamp_min(1)).detach(),
                'fallback_fraction': (1 - keep).mean().detach(),
                'gate_deviation': (gate - 1 / preference.shape[1]).abs().mean().detach(),
            }
        return gate.log()


class RefinedScale(nn.Module):
    def __init__(self, original, options):
        super().__init__()
        self.base = original
        self.base.router = BoundedRouter(self.base.router, options)

    @property
    def coarseners(self):
        return self.base.coarseners

    def forward(self, features, evidence, mask):
        mixed, mass, balance, diagnostics = self.base(features, evidence, mask)
        diagnostics = dict(diagnostics, **self.base.router.last_diagnostics)
        return mixed, mass, balance, diagnostics


class V221CoarseningMoE(V22CoarseningMoE):
    def __init__(self, cfg):
        options = cfg['model']['v22'].get('refinement')
        if not options or cfg['model']['v22'].get('mode') != 'moe':
            raise ValueError('V22.1 requires mode=moe and explicit refinement options')
        if cfg['model']['v22'].get('geometry_control'):
            raise ValueError('Do not combine V22.1 with fixed-geometry control builders')
        # Initialize the ENTIRE original backbone first. Shared weights therefore
        # match original V22 under the same seed; only new strength heads are added.
        super().__init__(cfg)
        self.scales = nn.ModuleList([RefinedScale(scale, options) for scale in self.scales])


def install_refinement_builder():
    """Explicit worker-local extension of the existing V22 training entry point."""
    from stmoe_imputer.models.registry import MODEL_REGISTRY
    MODEL_REGISTRY['v22_coarsening_moe'] = V221CoarseningMoE.from_config
