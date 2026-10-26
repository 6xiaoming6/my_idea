"""Observed-only, output-level consistency for the v25 CMFF backbone.

The aligned 1:2:4 hierarchy is partitioned into disjoint residual regions:
observed M cells first, then each observed C cell excluding observed M cells.
Only missing F entries in these regions are projected. No target tensor is
read here; coarse targets used by the separate I04 loss are not observations.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from ..performance import diagnostics as collect_diagnostics

from .observed_multiscale_coe import ObservedMultiScaleCoE
from .observed_history_coe import ObservedHistoryCoE


def _blocks(x: torch.Tensor, factor: int) -> torch.Tensor:
    """B,C,T,H,W -> B,C,T,H/f,W/f,f*f (row-major inside each block)."""
    b, c, t, h, w = x.shape
    if h % factor or w % factor:
        raise ValueError('Consistency requires aligned spatial dimensions divisible by 4')
    return x.reshape(b, c, t, h // factor, factor, w // factor, factor).permute(
        0, 1, 2, 3, 5, 4, 6).reshape(b, c, t, h // factor, w // factor, factor * factor)


def _unblocks(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, t, h, w, _ = x.shape
    return x.reshape(b, c, t, h, w, factor, factor).permute(
        0, 1, 2, 3, 5, 4, 6).reshape(b, c, t, h * factor, w * factor)


def _expand_cells(x: torch.Tensor, factor: int) -> torch.Tensor:
    return x.repeat_interleave(factor, -2).repeat_interleave(factor, -1)


def weighted_simplex_projection(values: torch.Tensor, weights: torch.Tensor,
                                eligible: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    """Minimize sum((x-z)^2/w) with x>=0 and sum(x)=budget on eligible entries.

Inputs have shape (..., N), except budget (...). Ineligible output entries are
zero. Sorting z/w identifies the exact active set; gradients flow through the
active affine solution, including when raw predictions are negative. FP16 and
BF16 inputs are promoted to FP32; FP64 is preserved for numerical tests.
"""
    if values.shape != weights.shape or values.shape != eligible.shape or values.shape[:-1] != budget.shape:
        raise ValueError('Weighted simplex shape mismatch')
    dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    z, w, b = values.to(dtype), weights.to(dtype), budget.to(dtype)
    allowed = eligible.bool()
    if not bool(torch.isfinite(z).all() and torch.isfinite(w).all() and torch.isfinite(b).all()):
        raise ValueError('Nonfinite simplex input')
    if not bool((w > 0).all()) or not bool((b >= 0).all()):
        raise ValueError('Simplex weights must be positive and budgets nonnegative')
    if bool(((~allowed.any(-1)) & (b > 0)).any()):
        raise ValueError('Positive budget without any eligible position')
    # For x_i=max(z_i-lambda*w_i,0), active coordinates are a prefix ordered
    # by z_i/w_i. Padding contributes neither z nor w to cumulative sums.
    ratios = torch.where(allowed, z / w, z.new_full((), -torch.inf))
    ordered, indices = ratios.sort(dim=-1, descending=True)
    # Subtract a common multiple of w before solving. This leaves the optimum
    # unchanged because sum(x) is fixed, and avoids cancellation when a large
    # raw prediction must be projected onto a very small positive budget.
    anchor = torch.where(allowed.any(-1), ordered[..., 0], 0.).unsqueeze(-1)
    shifted_ratios = torch.where(allowed, z / w - anchor, 0.)
    shifted_z = shifted_ratios * w
    az = shifted_z.gather(-1, indices)
    aw = torch.where(allowed, w, 0.).gather(-1, indices)
    sums_w = aw.cumsum(-1)
    threshold = (az.cumsum(-1) - b.unsqueeze(-1)) / sums_w.clamp_min(torch.finfo(dtype).tiny)
    active = ((ordered - anchor) > threshold) & torch.isfinite(ordered)
    last = (active.sum(-1) - 1).clamp_min(0)
    lam = threshold.gather(-1, last.unsqueeze(-1))
    projected = torch.where(allowed, (shifted_z - lam * w).clamp_min(0), 0.)
    return torch.where((b > 0).unsqueeze(-1), projected, projected * 0.)


def _validated_budget(residual: torch.Tensor, total: torch.Tensor,
                      deducted: torch.Tensor, count: torch.Tensor,
                      observed: torch.Tensor, name: str) -> torch.Tensor:
    # Stored hierarchy values are float32. Only errors within this relative
    # reduction tolerance may be rounded to zero; impossible observations fail.
    tolerance = 2e-5 + 2e-6 * (total.abs() + deducted.abs())
    impossible = observed & ((residual < -tolerance) | ((count == 0) & (residual.abs() > tolerance)))
    if bool(impossible.any()):
        raise ValueError(f'Infeasible observed {name} constraint: negative remaining mass or inconsistent fully observed region')
    return torch.where(observed & (count > 0), residual.clamp_min(0), 0.)


def project_observed_residual_regions(raw: torch.Tensor, weights: torch.Tensor,
                                      x_f: torch.Tensor, m_f: torch.Tensor,
                                      x_m: torch.Tensor, m_m: torch.Tensor,
                                      x_c: torch.Tensor, m_c: torch.Tensor,
                                      pooling: str = 'sum', collect_stats: bool = True):
    """Return (corrected F, detached scalar diagnostics), using observations only.

For mean observations, the 2x2/4x4 means are converted to sums. All observed
values must correspond to this exact aligned hierarchy. An observed M region
owns its missing F entries; its entire area is excluded from C's residual
region, so later corrections cannot undo an already satisfied M constraint.
"""
    if pooling not in ('sum', 'mean') or raw.ndim != 5 or weights.shape != raw.shape or x_f.shape != raw.shape:
        raise ValueError('Invalid residual-region input shape or pooling mode')
    dtype = torch.float64 if raw.dtype == torch.float64 else torch.float32
    z, w = raw.to(dtype), weights.to(dtype)
    values, masks = [], []
    for x, mask, factor in ((x_f, m_f, 1), (x_m, m_m, 2), (x_c, m_c, 4)):
        expected = (*raw.shape[:3], raw.shape[-2] // factor, raw.shape[-1] // factor)
        if tuple(x.shape) != expected or mask.ndim != 5 or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('Invalid observation shape or non-binary mask')
        try:
            visible = mask.bool().expand_as(x)
        except RuntimeError as exc:
            raise ValueError('Observation mask cannot broadcast across channels') from exc
        value = torch.where(visible, x.to(dtype), 0.)
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError('Hard consistency requires finite nonnegative observations')
        values.append(value)
        masks.append(visible)
    vf, vm, vc = values
    mf, mm, mc = masks
    vmass = vm * (4 if pooling == 'mean' else 1)
    cmass = vc * (16 if pooling == 'mean' else 1)

    fine_m = _blocks(~mf, 2)
    eligible_m = fine_m & mm.unsqueeze(-1)
    known_m = _blocks(vf, 2).sum(-1)
    budget_m = _validated_budget(vmass - known_m, vmass, known_m,
                                 fine_m.sum(-1), mm, 'M')
    projected_m = weighted_simplex_projection(_blocks(z, 2), _blocks(w, 2), eligible_m, budget_m)
    assign_m = _unblocks(eligible_m, 2)
    corrected = torch.where(assign_m, _unblocks(projected_m, 2), z)

    mid_area = _expand_cells(mm, 2)
    residual_region = (~mf) & (~mid_area)
    fine_c = _blocks(residual_region, 4)
    eligible_c = fine_c & mc.unsqueeze(-1)
    # Use observed M totals once. F observations inside those cells are already
    # represented by that total and must not be subtracted a second time.
    deducted_c = _blocks(vmass, 2).sum(-1) + _blocks(torch.where(mid_area, 0., vf), 4).sum(-1)
    budget_c = _validated_budget(cmass - deducted_c, cmass, deducted_c,
                                 fine_c.sum(-1), mc, 'C')
    projected_c = weighted_simplex_projection(_blocks(z, 4), _blocks(w, 4), eligible_c, budget_c)
    assign_c = _unblocks(eligible_c, 4)
    corrected = torch.where(assign_c, _unblocks(projected_c, 4), corrected)
    corrected = torch.where(mf, vf, corrected)
    constrained = assign_m | assign_c
    count_missing = (~mf).sum().clamp_min(1)
    if not collect_stats:
        return corrected, {}
    with torch.no_grad():
        stats = {
            'group_count_M': (mm & (fine_m.sum(-1) > 0)).sum().to(dtype),
            'group_count_C': (mc & (fine_c.sum(-1) > 0)).sum().to(dtype),
            'constrained_fraction': constrained.sum().to(dtype) / count_missing,
            'weight_mean': w.mean(), 'weight_min': w.min(), 'weight_max': w.max(),
            'correction_abs': torch.where(~mf, (corrected - z).abs(), 0.).sum() / count_missing,
            'negative_raw_fraction': ((z < 0) & constrained).sum().to(dtype) / constrained.sum().clamp_min(1),
        }
        completed_raw = torch.where(mf, vf, z)
        for name, value, mask, factor in (('M', vm, mm, 2), ('C', vc, mc, 4)):
            multiplier = factor * factor if pooling == 'sum' else 1
            for label, prediction in (('before', completed_raw), ('after', corrected)):
                aggregate = _blocks(prediction, factor).mean(-1) * multiplier
                error = torch.where(mask, (aggregate - value).abs(), 0.)
                stats[f'{label}_observed_{pooling}_residual_{name}'] = error.sum() / mask.sum().clamp_min(1)
        stats = {k: v.detach() for k, v in stats.items()}
    return corrected, stats


class ObservedConsistencyCoE(ObservedMultiScaleCoE):
    """B2 plus a small observation-only, weighted residual-region projection."""

    with_history = False

    @classmethod
    def from_config(cls, cfg):
        coe = cfg['model']['coe']
        spec = coe.get('observation_consistency', {})
        if spec.get('enabled') is not True or spec.get('mode') != 'residual_projection':
            raise ValueError('ObservedConsistencyCoE requires residual_projection')
        if set(spec) - {'enabled', 'mode', 'weight_bound', 'weight_mode'}:
            raise ValueError('Unknown observation-consistency configuration')
        weight_mode = spec.get('weight_mode', 'learned')
        if weight_mode not in ('learned', 'uniform'):
            raise ValueError('Expected learned or uniform consistency weights')
        bound = float(spec.get('weight_bound', 2.0))
        if not math.isfinite(bound) or not 0 < bound <= 8:
            raise ValueError('weight_bound must lie in (0, 8]')
        if cfg['data'].get('input_scales') != 'FMC' or coe.get('initial_observation_scales', 'FMC') != 'FMC':
            raise ValueError('I05 requires real F/M/C observations and fused FMC H0')
        if bool(coe.get('observed_history', {}).get('enabled')) != cls.with_history:
            raise ValueError('History configuration and consistency model must agree')
        if coe.get('scale_delta_exploration', {}).get('enabled') and not getattr(cls, 'with_free_scales', False):
            raise ValueError('Consistency excludes free-scale extensions')
        if cfg['data'].get('return_multiscale_targets', False):
            raise ValueError('Observed projection excludes full coarse target labels')
        model = super().from_config(cfg)
        if model.core_spec['communication'] not in ('conditional', 'none'):
            raise ValueError('Expected controlled difference or sequential state only')
        model.consistency_weight_bound = bound
        model.consistency_weight_mode = weight_mode
        mix = cfg.get('loss', {}).get('learnable_f_mix', {})
        if mix.get('enabled', False):
            initial = float(mix.get('initial_raw_weight', 0.3))
            if not math.isfinite(initial) or not 0 < initial < 1:
                raise ValueError('initial_raw_weight must be finite in (0,1)')
            # A single global training scalar; constant construction consumes no RNG.
            model.x_loss_mix_logit = nn.Parameter(torch.tensor(math.log(initial/(1-initial)), dtype=torch.float32))
        if weight_mode == 'uniform':
            return model
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(cfg['seed']) + 105501)
            model.x_consistency = nn.Sequential(
                nn.Conv3d(model.dim + 4 * model.c_in, max(8, model.dim // 2), 1),
                nn.SiLU(),
                nn.Conv3d(max(8, model.dim // 2), model.c_in, 1),
            )
            nn.init.zeros_(model.x_consistency[-1].weight)
            nn.init.zeros_(model.x_consistency[-1].bias)
        return model

    def forward(self, x_f, m_f, x_m, m_m, x_c, m_c, r_m=None, r_c=None, **kwargs):
        out = super().forward(x_f, m_f, x_m, m_m, x_c, m_c, r_m=r_m, r_c=r_c, **kwargs)
        # Projection arithmetic and its small head stay FP32 under training AMP.
        # Missing input slots are never read by the normalizer or projection.
        with torch.autocast(device_type=x_f.device.type, enabled=False):
            raw = out['x_hat_main'].float()
            hidden = out['h_st_aux'].float()
            mf = m_f.bool().expand_as(x_f)
            mm = m_m.bool().expand_as(x_m)
            mc = m_c.bool().expand_as(x_c)
            if self.consistency_weight_mode == 'uniform':
                weights = torch.ones_like(raw)
            else:
                squares, counts = [], []
                for x, mask, factor in ((x_f, mf, 1), (x_m, mm, 2), (x_c, mc, 4)):
                    density = torch.where(mask, x.float(), 0.) / (factor * factor if self.observation_pooling == 'sum' else 1)
                    squares.append(density.square().sum((2, 3, 4), keepdim=True))
                    counts.append(mask.sum((2, 3, 4), keepdim=True))
                scale = (sum(squares) / sum(counts).clamp_min(1)).sqrt().clamp_min(1).detach()
                features = torch.cat((hidden, raw / scale, mf.float(),
                                      _expand_cells(mm.float(), 2), _expand_cells(mc.float(), 4)), 1)
                weights = (self.consistency_weight_bound * self.x_consistency(features).tanh()).exp()
            corrected, stats = project_observed_residual_regions(
                raw, weights, x_f, m_f, x_m, m_m, x_c, m_c, pooling=self.observation_pooling, collect_stats=collect_diagnostics(self))
        out['x_hat_raw_consistency'] = out['x_hat_main']
        out['x_hat_main'] = corrected
        if hasattr(self, 'x_loss_mix_logit'):
            out['f_loss_raw_weight'] = self.x_loss_mix_logit.float().sigmoid()
        info = out['coe']
        info['x_consistency_diagnostics'] = stats
        # The fourth entry is the actual model output after output-level
        # correction. Keep raw expert outputs explicitly available for audits.
        for key in ('predictions', 'completions', 'candidate_predictions', 'candidate_completions'):
            if key in info:
                info['x_consistency_raw_' + key] = list(info[key])
                info[key] = [*info[key][:-1], corrected]
        for key in ('changes', 'candidate_changes'):
            if key in info:
                previous = info['predictions'][-2]
                info[key] = [*info[key][:-1], torch.where(mf, 0., (corrected - previous).abs())]
        for key, value in stats.items():
            info['diagnostics']['v25_consistency_' + key] = value
        for name in (('M', 'C') if stats else ()):
            key = 'observed_' + self.observation_pooling + '_residual_' + name
            info['diagnostics']['v25_' + key] = stats['after_' + key]
        return out


class ObservedHistoryConsistencyCoE(ObservedConsistencyCoE, ObservedHistoryCoE):
    """I08: I03 causal history inside rounds, I05 projection after round four.

    Cooperative super() builds/executes the unchanged history backbone first,
    then attaches the independently initialized consistency head and projection.
    """

    with_history = True


from .scale_delta_exploration_coe import ScaleDeltaExplorationCoE


class FreeScaleConsistencyCoE(ObservedConsistencyCoE, ScaleDeltaExplorationCoE):
    """Free scales and observed projection, without a prediction-history pool."""

    with_free_scales = True


class FreeScaleHistoryConsistencyCoE(ObservedConsistencyCoE, ObservedHistoryCoE,
                                     ScaleDeltaExplorationCoE):
    """I09/I10/I11: I08 with native single-scale ST execution.

    Cooperative MRO applies controlled delta then causal history before scale
    dispatch; output projection runs once after all four unprojected rounds.
    """
    with_history = True
    with_free_scales = True

    @classmethod
    def from_config(cls, cfg):
        coe = cfg['model']['coe']
        communication = coe.get('core_validation', {}).get('communication')
        expected = {'enabled': True, 'path': 'free', 'top_k_per_step': [2]*4,
                    'delta_steps': [] if communication == 'none' else [2,3,4], 'same_scale_only': False,
                    'scale_gradient': 'selected_branch_st',
                    'top1_gradient': 'selected_probability_st'}
        exploration = dict(coe.get('scale_delta_exploration', {}))
        last_fine = exploration.pop('last_round_fine', False)
        first_three_unique = exploration.pop('first_three_unique', False)
        if (exploration != expected or not isinstance(last_fine, bool)
                or not isinstance(first_three_unique, bool)
                or (first_three_unique and not last_fine)):
            raise ValueError('Requires I02 free-scale protocol; first_three_unique also requires last_round_fine')
        if coe.get('observation_consistency') != {'enabled': True,
                'mode': 'residual_projection', 'weight_bound': 2.0}:
            raise ValueError('I09 requires the original I05 learned projection')
        return super().from_config(cfg)
