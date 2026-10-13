"""v25 explicit F/M/C observations, fixed CMFF, native shared Top-2 CoE."""
from __future__ import annotations

import torch
from torch import nn

from .core_validation_coe import CoreValidationCoE
from .temporal_spatial_coe import TemporalSpatialCoE
from ..utils.deterministic import pool, resize


class ObservedMultiScaleCoE(CoreValidationCoE):
    requires_multiscale = True

    @classmethod
    def from_config(cls, cfg):
        m = super().from_config(cfg)
        if m.core_spec['path'] != 'CMFF' or m.core_spec['communication'] not in ('none','conditional'):
            raise ValueError('v25 initial validation is CMFF with/without controlled delta')
        m.observation_pooling = cfg['data'].get('scales', {}).get('pooling_mode')
        if m.observation_pooling not in ('sum', 'mean'):
            raise ValueError('Expected physical sum or mean observations')
        # Controls H0 only. data.input_scales continues to control observations
        # available throughout the run, including the native CMFF injections.
        m.initial_observation_scales = cfg['model']['coe'].get('initial_observation_scales', 'FMC')
        if m.initial_observation_scales not in ('F', 'FMC'):
            raise ValueError('initial_observation_scales must be F or FMC')
        # Identical additions and RNG consumption in all four factorial cells.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(cfg['seed']) + 25001)
            m.observation_fusion = nn.Conv3d(3*m.dim, m.dim, 1, bias=False)
        return m

    def forward(self, x_f, m_f, x_m, m_m, x_c, m_c, r_m=None, r_c=None, **kwargs):
        if hasattr(self, '_observations'):
            raise RuntimeError('Overlapping forward contexts')
        observations = []
        for x, mask, factor in ((x_f,m_f,1),(x_m,m_m,2),(x_c,m_c,4)):
            expected = (*x_f.shape[:3],x_f.shape[-2]//factor,x_f.shape[-1]//factor)
            if tuple(x.shape) != expected or mask.ndim != 5:
                raise ValueError('Observation scale shape mismatch')
            if not bool(((mask==0)|(mask==1)).all()):
                raise ValueError('Each observed-scale mask must be binary')
            obs = mask.bool().expand_as(x)
            if not bool((torch.isfinite(x)|~obs).all()):
                raise ValueError('Nonfinite observed value')
            # An invertible, known area conversion; stored inputs remain physical sums.
            # Do not equate partial-F means with independent coarse observations.
            divisor = factor*factor if self.observation_pooling == 'sum' else 1
            values = torch.where(obs,x,0.) / divisor
            observations.append((values,obs.to(x.dtype)))
        self._observations = observations
        try:
            out = super().forward(x_f=x_f,m_f=m_f,**kwargs)
            out['coe']['x_progress_diagnostics'] = True  # Diagnostics only; no round-wise supervision.
            comp = torch.where(m_f.bool(),x_f,out['x_hat_main']).detach().float()
            for name, (values,mask), factor in zip(('F','M','C'),observations,(1,2,4)):
                out['coe']['diagnostics']['v25_observed_fraction_'+name] = mask.float().mean()
                if factor>1:
                    residual = (pool(comp,factor)-values.detach().float()).abs()*mask
                    multiplier = factor**2 if self.observation_pooling == 'sum' else 1
                    key = 'v25_observed_'+self.observation_pooling+'_residual_'+name
                    out['coe']['diagnostics'][key] = residual.sum()*multiplier/mask.sum().clamp_min(1)
            return out
        finally:
            del self._observations

    def _initialize(self, x, mask):
        c = super()._initialize(x,mask)
        pyramid = []
        encodings = [c['h']]
        for sid, ((values, native_mask),factor) in enumerate(zip(self._observations,(1,2,4))):
            # Pool normalized physical coordinates to retain region-center geometry.
            pos = pool(c['pos'],factor) if factor>1 else c['pos']
            support = self._observation_support(native_mask)
            pyramid.append((values,native_mask,support,pos))
            if sid and self.initial_observation_scales == 'FMC':
                encoded = self.encoder(torch.cat((values,native_mask,support,pos),1))
                # No pseudo-observations when this scale is unavailable.
                encoded = encoded * native_mask.mean(1,keepdim=True)
                encodings.append(resize(encoded,x.shape[-2:]))
            elif sid:
                # Keep the same projection and state_dict as the FMC baseline.
                # Real M/C observations remain in pyramid for their own rounds.
                encodings.append(torch.zeros_like(c['h']))
        h = self.observation_fusion(torch.cat(encodings,1))
        pred = self.decoder(h)
        c.update(h=h,h0=h,prev=h,initial_prediction=pred,last_prediction=pred,
                 completion=torch.where(c['mask'].bool(),c['x'],pred),observed_pyramid=pyramid)
        return c

    def _execute(self,c,corrected,weights,scale_weights,step):
        sid = (2,1,0,0)[step]
        factor = (1,2,4)[sid]
        if not bool((scale_weights[:,sid]==1).all()):
            raise ValueError('v25 first batch must retain CMFF')
        values,mask,support,pos = c['observed_pyramid'][sid]
        h = pool(corrected,factor) if factor>1 else corrected
        fallback = pool(c['completion'],factor) if factor>1 else c['completion']
        completion = torch.where(mask.bool(),values,fallback)
        z = self.state_norm(self.state_projection(torch.cat((h,completion,mask,support,pos),1)))
        c['w_stats']['adapter_abs'] = z.new_zeros(())
        update = TemporalSpatialCoE._dispatch_weighted(self,z,weights,step)
        return resize(update,c['h'].shape[-2:]) if factor>1 else update,None,[]
