"""Y09-Y12: shared input planes and isolated history/interaction combinations."""
import copy

import torch
from torch import nn

from .history_plane_interaction_coe import HistoryPlaneInteractionCoE, TriPlaneBranch, zero_last
from ..utils.deterministic import pool, resize


class HistoryFollowupCoE(HistoryPlaneInteractionCoE):
    @classmethod
    def from_config(cls, cfg):
        spec = cfg['model']['coe']['history_followup']
        if set(spec) != {'enabled', 'global_plane', 'history_pair'} or spec['enabled'] is not True:
            raise ValueError('Invalid history followup config')
        if spec['global_plane'] not in ('none', 'uniform', 'coverage') or type(spec['history_pair']) is not bool:
            raise ValueError('Invalid followup mode')
        if (spec['global_plane'] != 'none') == spec['history_pair']:
            raise ValueError('Select exactly one followup extension')
        original = cfg['model']['coe']['history_plane_interaction']
        if original['plane'] != 'none' or original['interaction'] != 'none' or original['residual']:
            raise ValueError('Followup requires unmodified ST and no preexisting interaction')
        if spec['history_pair'] and original['history'] != 'adaptive':
            raise ValueError('History/pair combination requires adaptive history')
        model = super().from_config(cfg)
        if model.core_spec['communication'] != 'conditional':
            raise ValueError('Exploratory Y extensions require controlled difference')
        model.y_spec = copy.deepcopy(spec)
        with torch.random.fork_rng(devices=[]):
            if spec['global_plane'] != 'none':
                torch.random.default_generator.manual_seed(int(cfg['seed']) + 106001)
                model.x_global_plane = TriPlaneBranch(model.dim, spec['global_plane'])
            if spec['history_pair']:
                # Exactly the X10 module/initial values; add only the X05 history reader.
                torch.random.default_generator.manual_seed(int(cfg['seed']) + 105003)
                model.x_interaction = zero_last(nn.Sequential(
                    nn.Conv3d(2 * model.dim, 32, 1), nn.GELU(), nn.Conv3d(32, model.dim, 1)))
                model.x_spec['interaction'] = 'product'
        return model

    def _execute(self, c, corrected, weights, scale_weights, step):
        if self.y_spec['global_plane'] == 'none':
            return super()._execute(c, corrected, weights, scale_weights, step)
        sid = (2, 1, 0, 0)[step]
        factor = (1, 2, 4)[sid]
        if not bool((scale_weights[:, sid] == 1).all()):
            raise ValueError('CMFF path changed')
        h, mask, completion, support, pos = corrected, c['mask'], c['completion'], c['support'], c['pos']
        if factor > 1:
            coverage = pool(mask.float(), factor)
            values = pool(torch.where(mask.bool(), completion, 0.).float(), factor) / coverage.clamp_min(1e-8)
            completion = torch.where(coverage > 0, values, pool(completion, factor))
            mask = coverage
            h, support, pos = pool(h, factor), pool(support, factor), pool(pos, factor)
        z = self.state_norm(self.state_projection(torch.cat((h, completion, mask, support, pos), 1)))
        c['w_stats']['adapter_abs'] = z.new_zeros(())
        delta, stats = self.x_global_plane(z, mask)
        c['x_stats'].update({'global_' + key: value for key, value in stats.items()})
        c['x_stats']['global_plane_executions'] = z.new_tensor(1.)
        # One shared view computation per round. Only the two selected experts execute.
        out = self._native_dispatch(z + delta.to(z.dtype), weights, mask, step, c['x_stats'])
        if factor > 1:
            out = resize(out, c['h'].shape[-2:])
        return out, None, []
