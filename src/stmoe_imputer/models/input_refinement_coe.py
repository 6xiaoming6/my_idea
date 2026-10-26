"""I54-I63: isolate explicit observation, estimate and metadata input paths.

The original I37 round, routers, communication, history and projection are
inherited unchanged. Only explicit router value statistics and native expert
input construction are opt-in interventions. No new trainable parameters.
"""
import copy
import math

import torch

from .observed_consistency_coe import FreeScaleHistoryConsistencyCoE
from .observation_regions import ObservationRegions
from .temporal_spatial_coe import TemporalSpatialCoE
from ..utils.deterministic import pool, resize


DEFAULTS = dict(expert_fill='initial', router_values='initial',
                repeated_metadata=True, coarse_values='absolute', feedback_gain=0.1,
                support_gain=1.0, position_gain=1.0, prediction_gain=1.0,
                component_diagnostics=False)


class InputRefinementCoE(FreeScaleHistoryConsistencyCoE):
    @classmethod
    def from_config(cls, cfg):
        requested = copy.deepcopy(cfg['model']['coe']['input_refinement'])
        if not requested.pop('enabled', False) or set(requested) - set(DEFAULTS):
            raise ValueError('Invalid input_refinement configuration')
        spec = {**DEFAULTS, **requested}
        if spec['expert_fill'] not in ('initial', 'zero', 'bounded_current', 'visible_initial'):
            raise ValueError('Unknown expert fill source')
        if spec['router_values'] not in ('initial', 'observed'):
            raise ValueError('Unknown router value source')
        if spec['coarse_values'] not in ('absolute', 'remaining_budget'):
            raise ValueError('Unknown coarse observation source')
        if type(spec['repeated_metadata']) is not bool or spec['feedback_gain'] != 0.1:
            raise ValueError('Metadata flag must be boolean; feedback gain is frozen at 0.1')
        for key in ('support_gain', 'position_gain', 'prediction_gain'):
            value = spec[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(key + ' must be finite and in [0, 1]')
        if not spec['repeated_metadata'] and any(spec[k] != 1 for k in ('support_gain', 'position_gain')):
            raise ValueError('Do not mix the legacy metadata switch with explicit gains')
        if spec['expert_fill'] != 'visible_initial' and spec['prediction_gain'] != 1:
            raise ValueError('Prediction gain applies only to visible_initial fallback')
        if type(spec['component_diagnostics']) is not bool:
            raise ValueError('component_diagnostics must be boolean')
        if cfg['model']['coe'].get('observation_iteration', {}).get('enabled'):
            raise ValueError('Input cleanup cannot enable the previous iteration suite')
        if cfg['data'].get('return_multiscale_targets', False):
            raise ValueError('Only visible observations may enter input refinement')
        model = super().from_config(cfg)
        model.input_refinement = spec
        return model

    def _initialize(self, x, mask):
        c = super()._initialize(x, mask)
        if self.input_refinement['component_diagnostics']:
            visible = c['mask'].bool()
            known = torch.where(visible, c['completion'].detach(), 0.)
            prediction = torch.where(visible, 0., c['completion'].detach())
            gain = self.input_refinement['prediction_gain']
            if self.input_refinement['expert_fill'] == 'zero':
                known, prediction = torch.zeros_like(known), torch.zeros_like(prediction)
            elif self.input_refinement['expert_fill'] == 'visible_initial':
                prediction = prediction * gain
            elif self.input_refinement['expert_fill'] != 'initial':
                raise ValueError('Component diagnostics require static decomposable fallback')
            c['input_components'] = [(pool(known, f) if f > 1 else known,
                                      pool(prediction, f) if f > 1 else prediction) for f in (1, 2, 4)]
        if self.input_refinement['coarse_values'] == 'remaining_budget':
            c['input_regions'] = ObservationRegions(self._observations)
        if self.input_refinement['expert_fill'] == 'bounded_current':
            # Per sample/channel physical density scale; never read hidden slots.
            numerator = sum(torch.where(m.bool(), v.float().square(), 0.).sum((2, 3, 4), keepdim=True)
                            for v, m in self._observations)
            denominator = sum(m.sum((2, 3, 4), keepdim=True) for _, m in self._observations)
            c['input_sigma'] = (numerator / denominator.clamp_min(1)).sqrt().clamp_min(1).detach()
        return c

    def _router_features(self, hidden, values, support_summary, change, missing, pattern_summary=None):
        result = super()._router_features(hidden, values, support_summary, change, missing, pattern_summary)
        if self.input_refinement['router_values'] == 'observed':
            observed = (1 - missing).bool().expand_as(values)
            denominator = observed.sum((2, 3, 4))
            mean = torch.where(observed, values, 0.).sum((2, 3, 4)) / denominator.clamp_min(1)
            # Preserve every other feature and every parameter name. The second
            # C-sized value block no longer exposes missing-slot predictions.
            result = torch.cat((result[:, :2*self.dim], mean, torch.zeros_like(mean),
                                result[:, 2*self.dim + 2*self.c_in:]), 1)
        return result

    def _native_inputs(self, c, corrected, sid, ids):
        factor = (1, 2, 4)[sid]
        values, mask, support, pos = [v.index_select(0, ids) for v in c['observed_pyramid'][sid]]
        if sid and self.input_refinement['coarse_values'] == 'remaining_budget':
            ledger = c['input_regions']
            count = ledger.counts[sid].index_select(0, ids)
            budget = ledger.budgets[sid].index_select(0, ids)
            mask = (count > 0).to(mask.dtype)
            values = torch.where(mask.bool(), budget / count.clamp_min(1), 0.)
            support = self._observation_support(mask)
        h = corrected.index_select(0, ids)
        if factor > 1:
            h = pool(h, factor)
        fill_source = self.input_refinement['expert_fill']
        if fill_source in ('initial', 'visible_initial'):
            fallback = c['completion'].index_select(0, ids)
            if fill_source == 'visible_initial' and self.input_refinement['prediction_gain'] != 1:
                # c.completion already contains visible F and initial P0 in its
                # missing slots. Scale only the latter before spatial pooling.
                fine_mask = c['mask'].index_select(0, ids).bool()
                fallback = torch.where(fine_mask, fallback,
                                       self.input_refinement['prediction_gain'] * fallback)
            if factor > 1:
                fallback = pool(fallback, factor)
        elif fill_source == 'zero':
            fallback = torch.zeros_like(values)
        else:
            fallback = c['last_prediction'].detach().index_select(0, ids)
            if factor > 1:
                fallback = pool(fallback, factor)
            sigma = c['input_sigma'].index_select(0, ids)
            fallback = self.input_refinement['feedback_gain'] * sigma * (fallback.float()/sigma).tanh()
        completion = torch.where(mask.bool(), values, fallback)
        if not self.input_refinement['repeated_metadata']:
            support, pos = torch.zeros_like(support), torch.zeros_like(pos)
        else:
            for key in ('support_gain', 'position_gain'):
                gain = self.input_refinement[key]
                if gain != 1:
                    if key == 'support_gain': support = support * gain
                    else: pos = pos * gain
        return h, completion, mask, support, pos, fallback

    def _execute(self, c, corrected, weights, scale_weights, step):
        out = torch.zeros_like(c['h'])
        estimate_sum = out.new_zeros((), dtype=torch.float32)
        estimate_count = estimate_sum.clone()
        observed_sum = estimate_sum.clone()
        observed_count = estimate_sum.clone()
        component_sum = {key: estimate_sum.clone() for key in
                         ('fallback_known_f_abs', 'fallback_prediction_abs', 'support_square', 'position_square',
                          'support_count', 'position_count')}
        for sid in range(3):
            ids = (scale_weights[:, sid].detach() != 0).nonzero().flatten()
            if not ids.numel():
                continue
            h, values, mask, support, pos, fallback = self._native_inputs(c, corrected, sid, ids)
            z = self.state_norm(self.state_projection(torch.cat((h, values, mask, support, pos), 1)))
            update = TemporalSpatialCoE._dispatch_weighted(self, z, weights.index_select(0, ids), step)
            if sid:
                update = resize(update, c['h'].shape[-2:])
            update = update * scale_weights[ids, sid, None, None, None, None].to(update.dtype)
            out = out.index_add(0, ids, update.to(out.dtype))
            missing = ~mask.bool()
            estimate_sum += torch.where(missing, fallback.detach().float().abs(), 0.).sum()
            estimate_count += missing.sum()
            observed_sum += torch.where(mask.bool(), values.detach().float().abs(), 0.).sum()
            observed_count += mask.sum()
            if 'input_components' in c:
                for key, value in zip(('fallback_known_f_abs', 'fallback_prediction_abs'), c['input_components'][sid]):
                    value = value.index_select(0, ids)
                    component_sum[key] += torch.where(missing, value.float().abs(), 0.).sum()
                for key, value in [('support', support), ('position', pos)]:
                    component_sum[key+'_square'] += value.detach().float().square().sum()
                    component_sum[key+'_count'] += value.numel()
        row = dict(estimate_abs=estimate_sum/estimate_count.clamp_min(1), estimate_count=estimate_count,
                   observed_abs=observed_sum/observed_count.clamp_min(1), observed_count=observed_count)
        if 'input_components' in c:
            for key in ('fallback_known_f_abs', 'fallback_prediction_abs'):
                row[key] = component_sum[key]/estimate_count.clamp_min(1)
            for key in ('support', 'position'):
                row[key+'_rms'] = (component_sum[key+'_square']/component_sum[key+'_count'].clamp_min(1)).sqrt()
        self._input_rows.append(row)
        c['w_stats']['adapter_abs'] = out.new_zeros(())
        return out, None, []

    def forward(self, *args, **kwargs):
        if hasattr(self, '_input_rows'):
            raise RuntimeError('Overlapping input refinement contexts')
        self._input_rows = []
        try:
            out = super().forward(*args, **kwargs)
            for i, row in enumerate(self._input_rows, 1):
                for key, value in row.items():
                    out['coe']['diagnostics'][f'input_step{i}_{key}'] = value.detach()
            return out
        finally:
            del self._input_rows
