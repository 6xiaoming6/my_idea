"""Observation-conditioned local fits, not calibrated uncertainty or a theorem.

Weighted least squares/normalized convolution are classical. The experimental
addition is a target-dependent, mode-wise restriction of the learned regional
prior before aligned readout. The final neural imputer remains unconstrained.
Only raw visible values enter the sufficient statistics (never hidden features).
"""
from __future__ import annotations

import math

import torch
from torch import nn


def spacetime_basis(t, h, w, device):
    """Fixed [1, x, y, time], normalized on the FULL reference grid, not mask.

    Singleton axes are zero. No trainable rescaling can inflate the Gram matrix.
    Time refers to the offline input window, not to future forecasting context.
    """
    def axis(n):
        x = torch.linspace(-1, 1, n, device=device) if n > 1 else torch.zeros(1, device=device)
        return x / x.square().mean().sqrt().clamp_min(1e-6)
    tt, yy, xx = torch.meshgrid(axis(t), axis(h), axis(w), indexing='ij')
    return torch.stack((torch.ones_like(xx), xx, yy, tt), -1).reshape(t*h*w, 4)


def observation_system(assignment, mask, values, basis):
    """G, b, coverage for each [batch, expert, region]. Always FP32.

    Full-region mass normalizes weights, so arbitrary expert gate factors do
    not change ridge strength. Missing entries are removed BEFORE arithmetic.
    Assignment is [B,E,T,N,K]; values [B,C,T,H,W]. Memory is O(BETNK),
    with small 4x4 solves; no N-by-N covariance or query-by-region-by-4x4 tensor.
    """
    with torch.autocast(device_type=values.device.type, enabled=False):
        b, e, t, n, k = assignment.shape
        a = assignment.float().reshape(b, e, t*n, k).transpose(-1, -2)
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
        m = mask.reshape(b, 1, 1, t*n).bool()
        weights = a * m
        z = torch.where(mask.bool(), values.float(), torch.zeros_like(values, dtype=torch.float32))
        z = z.permute(0, 2, 3, 4, 1).reshape(b, 1, t*n, -1)
        phi = basis.float()
        # Small loops avoid einsum selecting an enormous intermediate tensor.
        gram = torch.stack([torch.matmul(weights * phi[:, r], phi) for r in range(4)], -2)
        rhs = torch.stack([torch.matmul(weights * phi[:, r], z) for r in range(4)], -2)
        coverage = weights.sum(-1)
        return gram, rhs, coverage


def ridge_decomposition(gram, rhs, ridge):
    """c_obs=(G+lambda I)^-1 b; Q=lambda(G+lambda I)^-1.

    Q is a soft unconstrained-mode operator, not an exact null-space projector.
    The local regularized solution is c_obs + Q c_prior.
    """
    with torch.autocast(device_type=gram.device.type, enabled=False):
        eye = torch.eye(4, device=gram.device, dtype=torch.float32).expand_as(gram)
        g = (gram.float() + gram.float().transpose(-1, -2)) * .5
        solution = torch.linalg.solve(g + ridge*eye, torch.cat((rhs.float(), ridge*eye), -1))
        return solution[..., :rhs.shape[-1]], solution[..., rhs.shape[-1]:]


def aligned_evaluate(assignment, coefficients, basis):
    """Restore each expert's OWN coefficients before mixing expert outputs."""
    b, e, t, n, k = assignment.shape
    a = assignment.float().reshape(b, e, t*n, k)
    value = sum(torch.matmul(a, coefficients[..., r, :]) * basis[:, r:r+1] for r in range(4))
    return value.reshape(b, e, t, n, -1)


class RecoverabilityReadout(nn.Module):
    MODES = {'scalar', 'matrix', 'constrained', 'fit_only'}

    def __init__(self, dim, channels, options):
        super().__init__()
        self.mode = options.get('mode', 'off')
        self.ridge = options.get('ridge', .05)
        self.strength = options.get('strength', .1)
        if self.mode not in self.MODES:
            raise ValueError('recoverability mode must be scalar/matrix/constrained/fit_only')
        for name, value in [('ridge', self.ridge), ('strength', self.strength)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'recoverability {name} must be finite and positive')
        if set(options) - {'mode', 'ridge', 'strength'}:
            raise ValueError('Unknown recoverability options; basis is fixed normalized affine space-time')
        self.channels = channels
        # All enabled controls have identical parameter shapes/initialization.
        # fit_only retains this head for matching, but its output is multiplied
        # by zero (reported explicitly; these parameters have no task effect).
        self.prior = nn.Linear(dim, 4*channels)
        nn.init.normal_(self.prior.weight, std=.01)
        nn.init.zeros_(self.prior.bias)
        self.inject = nn.Linear(channels+1, dim)

    def forward(self, aggregation, regions, mask, values):
        b, c, t, h, w = values.shape
        a = aggregation['assignment']
        with torch.autocast(device_type=values.device.type, enabled=False):
            phi = spacetime_basis(t, h, w, values.device)
            g, rhs, coverage = observation_system(a, mask, values, phi)
            if self.mode == 'scalar':
                # Density-only control: constant observed mean, isotropic prior
                # attenuation. No directional moment enters its prediction.
                q_scalar = self.ridge / (coverage + self.ridge)
                obs = torch.zeros_like(rhs)
                obs[..., 0, :] = rhs[..., 0, :] / (coverage[..., None] + self.ridge)
                q = q_scalar[..., None, None] * torch.eye(4, device=values.device)
            else:
                obs, q = ridge_decomposition(g, rhs, self.ridge)
            # Nodes have common learned slot identities through time; prediction
            # of local coefficients uses contextual regions, not hidden labels.
            prior = self.prior(regions.float().mean(2)).reshape(*coverage.shape, 4, c)
            if self.mode in {'scalar', 'constrained'}:
                prior = torch.matmul(q, prior)
            elif self.mode == 'fit_only':
                prior = prior * 0.
            reconstruction = aligned_evaluate(a, obs+prior, phi)
            # Query-dependent weakness: phi(q)^T Q phi(q) / ||phi(q)||^2.
            # Detached diagnostic/conditioning prevents direct gaming of this
            # scalar, while the fit remains differentiable w.r.t. memberships.
            with torch.no_grad():
                weakness = aligned_evaluate(a.detach(), q.detach(), phi)
                weakness = (weakness * phi.reshape(1, 1, t, h*w, 4)).sum(-1, keepdim=True)
                weakness = (weakness / phi.square().sum(-1).reshape(1, 1, t, h*w, 1)).clamp(0, 1)
                cov = torch.matmul(a.detach().reshape(b, a.shape[1], t*h*w, -1), coverage.detach()[..., None])
                cov = cov.reshape(b, a.shape[1], t, h*w, 1)
            delta = self.strength * self.inject(torch.cat((reconstruction, weakness), -1))
        return delta, {'prediction': reconstruction, 'weakness': weakness, 'coverage': cov}
