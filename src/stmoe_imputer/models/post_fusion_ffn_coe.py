"""Opt-in, full-grid post-fusion FFNs on the fixed CMFF controlled-delta CoE."""
import copy

import torch
from torch import nn

from .core_validation_coe import CoreValidationCoE
from .temporal_spatial_coe import PointwiseLayerNorm


class PostFusionFFN(nn.Module):
    """Pre-LN residual FFN; 1x1x1 convolutions are position-wise linears."""

    def __init__(self, dim):
        super().__init__()
        self.norm = PointwiseLayerNorm(dim)
        self.up = nn.Conv3d(dim, 4 * dim, 1)
        self.activation = nn.GELU()
        self.down = nn.Conv3d(4 * dim, dim, 1)

    def forward(self, h):
        return h + self.down(self.activation(self.up(self.norm(h))))


class PostFusionFFNCoE(CoreValidationCoE):
    @classmethod
    def from_config(cls, cfg):
        m = super().from_config(cfg)
        spec = copy.deepcopy(cfg['model']['coe']['post_fusion_ffn'])
        expected = {'enabled': True, 'sharing': spec.get('sharing'),
                    'expansion': 4, 'pre_norm': True, 'residual': True,
                    'placement': 'after_fine_restore'}
        if spec != expected or spec['sharing'] not in ('shared', 'per_step'):
            raise ValueError('Expected shared/per_step 4x Pre-LN residual post-fusion FFN')
        if (m.core_spec != {'enabled': True, 'path': 'CMFF', 'communication': 'conditional'}
                or m.expert_sharing != 'shared'):
            raise ValueError('FFN experiment requires shared CMFF controlled-delta backbone')
        m.post_ffn_spec = spec
        # Preserve the backbone/global RNG stream. All four independent FFNs
        # begin with exactly the same weights as the shared FFN.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(cfg.get('seed', 7)) + 94001)
            first = PostFusionFFN(m.dim)
        m.post_ffns = nn.ModuleList([first])
        if spec['sharing'] == 'per_step':
            m.post_ffns.extend(copy.deepcopy(first) for _ in range(3))
        return m

    def _execute(self, c, corrected, weights, scale_weights, step):
        fused, message, bank = super()._execute(c, corrected, weights, scale_weights, step)
        index = 0 if self.post_ffn_spec['sharing'] == 'shared' else step
        transformed = self.post_ffns[index](fused)
        self._post_ffn_stats.append({
            'change_abs': (transformed - fused).detach().float().abs().mean(),
            'input_rms': fused.detach().float().square().mean().sqrt(),
            'output_rms': transformed.detach().float().square().mean().sqrt(),
        })
        # The inherited round decodes this transformed state and stores it as
        # next H; Hprev remains the previous round INPUT, as in K04.
        return transformed, message, bank

    def forward(self, *args, **kwargs):
        self._post_ffn_stats = []
        try:
            out = super().forward(*args, **kwargs)
            for step, stats in enumerate(self._post_ffn_stats, 1):
                for key, value in stats.items():
                    out['coe']['diagnostics'][f'post_ffn_step{step}_{key}'] = value
            return out
        finally:
            del self._post_ffn_stats
