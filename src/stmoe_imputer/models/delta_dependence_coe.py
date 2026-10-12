"""Explicit, from-initialization no-delta controls for Y12/Z06/X10/X09.

Reuse the parent's construction (and RNG stream), then disable communication
before any forward pass or optimizer construction. No trained weights are read.
The historical Y/Z constructors and their validation rules remain unchanged.
"""
import copy

from .history_followup_coe import HistoryFollowupCoE
from .history_mechanism_coe import HistoryMechanismCoE
from .history_plane_interaction_coe import HistoryPlaneInteractionCoE


class DeltaDependenceCoE:
    @classmethod
    def from_config(cls, cfg):
        parent_cfg = copy.deepcopy(cfg)
        coe = parent_cfg['model']['coe']
        spec = coe.pop('delta_dependence')
        if set(spec) != {'enabled', 'parent'} or spec['enabled'] is not True:
            raise ValueError('Invalid delta dependence specification')
        parent = spec['parent']
        if parent not in ('Y12', 'Z06', 'X10', 'X09'):
            raise ValueError('Unsupported delta dependence parent')
        if coe['core_validation']['communication'] != 'none':
            raise ValueError('Delta dependence controls require communication=none')
        if coe['core_validation']['path'] != 'CMFF' or cfg['loss'].get('lambda_coe_mid', 0) != 0:
            raise ValueError('Delta controls require CMFF without middle supervision')
        history = parent in ('Y12', 'Z06')
        expected = dict(enabled=True, history='adaptive' if history else 'none',
                        residual=False, plane='none',
                        interaction='none' if history else 'product' if parent == 'X10' else 'concat')
        if coe.get('history_plane_interaction') != expected:
            raise ValueError('Parent history/interaction configuration changed')
        if history:
            if coe.get('history_followup') != dict(enabled=True, global_plane='none', history_pair=True):
                raise ValueError('Parent history pair configuration changed')
        elif 'history_followup' in coe:
            raise ValueError('No-history controls cannot enable followup modules')
        if parent == 'Z06':
            if coe.get('history_mechanism') != dict(enabled=True, interaction='product_difference',
                                                   history='adaptive', representation='innovation'):
                raise ValueError('Z06 requires relative history')
        elif 'history_mechanism' in coe:
            raise ValueError('Unexpected history representation override')
        coe['core_validation']['communication'] = 'conditional'
        builder = (HistoryMechanismCoE if parent == 'Z06' else
                   HistoryFollowupCoE if parent == 'Y12' else HistoryPlaneInteractionCoE)
        model = builder.from_config(parent_cfg)
        model.core_spec['communication'] = 'none'
        model.communication = 'none'
        model.history_gate.requires_grad_(False)
        model.delta_dependence_spec = copy.deepcopy(spec)
        return model
