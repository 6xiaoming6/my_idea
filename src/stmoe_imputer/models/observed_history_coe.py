"""v25 observations with the existing X05 prediction-history reader only."""
import torch

from .observed_multiscale_coe import ObservedMultiScaleCoE
from .history_plane_interaction_coe import HistoryReader


class ObservedHistoryCoE(ObservedMultiScaleCoE):
    @classmethod
    def from_config(cls, cfg):
        spec = cfg['model']['coe'].get('observed_history', {})
        if spec != {'enabled': True, 'mode': 'adaptive'}:
            raise ValueError('Expected isolated adaptive observed-history experiment')
        if cfg['model']['coe'].get('scale_delta_exploration', {}).get('enabled'):
            raise ValueError('History and free scales are separate experiments')
        model = super().from_config(cfg)
        if model.core_spec['communication'] != 'conditional':
            raise ValueError('This experiment retains controlled difference')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(cfg['seed']) + 105001)
            model.x_history = HistoryReader(model.c_in, model.dim, 'adaptive')
        return model

    def _initialize(self, x, mask):
        c = super()._initialize(x, mask)
        c['x_pool'] = []
        observed = torch.where(c['mask'].bool(), c['x'].float().square(), 0.)
        c['x_scale'] = (observed.sum((2, 3, 4), keepdim=True) /
                        c['mask'].sum((2, 3, 4), keepdim=True).clamp_min(1)).sqrt().clamp_min(1).detach()
        return c

    def _memory(self, c):
        corrected, route, gate = super()._memory(c)
        stats = {'pool_size': c['h'].new_tensor(len(c['x_pool'])),
                 'pool_delta_abs': c['h'].new_zeros(())}
        if c['x_pool']:
            delta, weights = self.x_history(c['h'], c['x_pool'], c['mask'], c['x_scale'])
            corrected = corrected + delta
            stats['pool_delta_abs'] = delta.detach().float().abs().mean()
            for i in range(len(c['x_pool'])):
                stats[f'pool_weight_P{i+1}'] = weights[:, i].detach().mean()
        c['x_stats'] = stats
        return corrected, route, gate

    def _round(self, c, step, *args, **kwargs):
        new, row = super()._round(c, step, *args, **kwargs)
        self._x_stats.append(c['x_stats'])
        pred = torch.where(c['mask'].bool(), c['x'], row['prediction']).detach()
        new['x_pool'] = c['x_pool'] + [pred]
        return new, row

    def forward(self, *args, **kwargs):
        if hasattr(self, '_x_stats'):
            raise RuntimeError('Overlapping history forward contexts')
        self._x_stats = []
        try:
            out = super().forward(*args, **kwargs)
            for step, stats in enumerate(self._x_stats, 1):
                for key, value in stats.items():
                    out['coe']['diagnostics'][f'x_step{step}_{key}'] = value.detach()
            return out
        finally:
            del self._x_stats
