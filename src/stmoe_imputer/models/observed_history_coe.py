"""v25 observations with the existing X05 prediction-history reader only."""
import math
import torch
from ..performance import enabled as perf, diagnostics as collect_diagnostics

from .observed_multiscale_coe import ObservedMultiScaleCoE
from .history_plane_interaction_coe import HistoryReader


class ObservedHistoryCoE(ObservedMultiScaleCoE):
    @classmethod
    def from_config(cls, cfg):
        spec = cfg['model']['coe'].get('observed_history', {})
        if (set(spec) - {'enabled', 'mode', 'gain'} or not spec.get('enabled')
                or spec.get('mode') != 'adaptive'):
            raise ValueError('Expected isolated adaptive observed-history experiment')
        gain = float(spec.get('gain', 1.0))
        if not math.isfinite(gain) or not 0 <= gain <= 1:
            raise ValueError('observed_history.gain must be finite in [0, 1]')
        if (cfg['model']['coe'].get('scale_delta_exploration', {}).get('enabled')
                and not getattr(cls, 'with_free_scales', False)):
            raise ValueError('History and free scales are separate experiments')
        model = super().from_config(cfg)
        if model.core_spec['communication'] not in ('conditional', 'none'):
            raise ValueError('Expected controlled difference or sequential state only')
        model.history_gain = gain
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
            if self.history_gain != 1.0:
                delta = delta * self.history_gain
            corrected = corrected + delta
            if collect_diagnostics(self):stats['pool_delta_abs'] = delta.detach().float().abs().mean()
            for i in (range(len(c['x_pool'])) if collect_diagnostics(self) else ()):
                stats[f'pool_weight_P{i+1}'] = weights[:, i].detach().mean()
        c['x_stats'] = stats
        return corrected, route, gate

    def _round(self, c, step, *args, **kwargs):
        new, row = super()._round(c, step, *args, **kwargs)
        self._x_stats.append(c['x_stats'])
        if not (perf(self,"skip_last_history") and step==3):
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


from .scale_delta_exploration_coe import ScaleDeltaExplorationCoE


class FreeScaleObservedHistoryCoE(ObservedHistoryCoE, ScaleDeltaExplorationCoE):
    """I26: the same causal history/free-scale backbone, without projection."""

    with_free_scales = True

    @classmethod
    def from_config(cls, cfg):
        coe = cfg['model']['coe']
        if coe.get('observation_consistency', {}).get('enabled', False):
            raise ValueError('Unprojected history backbone excludes output projection')
        if cfg['data'].get('return_multiscale_targets', False):
            raise ValueError('Observed-only I26 excludes full coarse targets')
        if cfg.get('loss', {}).get('learnable_f_mix', {}).get('enabled', False):
            raise ValueError('Unprojected prediction has no distinct pre/post F losses')
        return super().from_config(cfg)
