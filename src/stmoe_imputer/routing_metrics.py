"""Exact, target-free diagnostics for sample-level Top-K expert routing."""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from .metrics import MaskedMetricAccumulator


class BackendDiagnosticAccumulator:
    """Opt-in, detached diagnostics; never enter the objective/router inputs.

    Density is observed fraction in a border-corrected 3x3 spatial window
    (including its missing center), same time step. Bins [0,1/3), [1/3,2/3),
    [2/3,1] are predeclared. Errors/wins use original-unit missing CHANNEL
    elements. Pair correlations are of absolute errors, not gate weights.
    """
    def __init__(self):
        self.bins = {}
        self.pairs = {}
        self.alpha = torch.zeros(3, dtype=torch.float64)

    @torch.no_grad()
    def update(self, outputs, batch):
        import torch.nn.functional as F
        mask = batch['m_f'].detach().float()
        target = batch['x_f_gt'].detach().float()
        valid = (1-mask).bool().expand_as(target)
        preds = [outputs['scale_predictions'][s].detach().float() for s in ('fine','mid','coarse')]
        errs = torch.stack([(p-target).abs() for p in preds])
        final = (outputs['x_hat_main'].detach().float()-target).abs()
        # Exclude padded cells, not treat outside-grid space as missing data.
        def avg(x):
            return F.avg_pool3d(F.pad(x, (1,1,1,1)), (1,3,3), stride=1)
        density = (avg(mask)/avg(torch.ones_like(mask))).expand_as(target)
        best = errs.amin(0)
        ties = (errs == best.unsqueeze(0)).float()
        wins = ties/ties.sum(0, keepdim=True)
        chosen = outputs['completion_gates'].detach().argmax(1, keepdim=True).expand_as(target)
        hit = (chosen.unsqueeze(0) == torch.arange(3, device=target.device).reshape(3,1,1,1,1,1)).float()*wins
        dominant = (outputs['completion_gates'].detach().amax(1,keepdim=True) > .9).expand_as(target)
        for label, region in [('all', valid), ('low', valid & (density < 1/3)),
                              ('medium', valid & (density >= 1/3) & (density < 2/3)),
                              ('high', valid & (density >= 2/3))]:
            # Only small sufficient-statistic vectors cross to CPU. Empty bins
            # report count=0, omit undefined scores (not artificial zero errors).
            values = [region.sum(), final[region].sum(), final[region].square().sum(),
                      best[region].sum(), dominant[region].sum(), hit.sum(0)[region].sum()]
            values += [e[region].sum() for e in errs]
            values += [w[region].sum() for w in wins]
            v = torch.stack(values).double().cpu()
            self.bins[label] = self.bins.get(label, torch.zeros_like(v))+v
        for i,j in ((0,1),(0,2),(1,2)):
            a,b = errs[i][valid],errs[j][valid]
            v = torch.stack((a.new_tensor(a.numel()),a.sum(),b.sum(),a.square().sum(),b.square().sum(),(a*b).sum(),(preds[i]-preds[j]).abs()[valid].sum())).double().cpu()
            self.pairs[(i,j)] = self.pairs.get((i,j),torch.zeros_like(v))+v
        n = valid.sum()
        a = outputs['completion_alpha'].detach()
        self.alpha += torch.stack((n, a*n, a.square()*n)).double().cpu()

    def compute(self):
        out = {}
        labels = ('fine','mid','coarse')
        for label,v in self.bins.items():
            prefix = 'backend_diag_'+label+'_'
            n = float(v[0]); out[prefix+'count'] = n
            if not n:
                continue
            out.update({prefix+'mae': float(v[1]/n), prefix+'rmse': math.sqrt(float(v[2]/n)),
                        prefix+'oracle_expert_mae': float(v[3]/n),
                        prefix+'dominant_fraction': float(v[4]/n), prefix+'top_weight_expert_hit': float(v[5]/n)})
            for i,s in enumerate(labels):
                out[prefix+s+'_mae'] = float(v[6+i]/n)
                out[prefix+s+'_win_fraction'] = float(v[9+i]/n)
        for (i,j),v in self.pairs.items():
            n = float(v[0])
            if not n:
                continue
            prefix = f'backend_diag_{labels[i]}_{labels[j]}_'
            va,vb = max(float(v[3]-v[1]*v[1]/n),0),max(float(v[4]-v[2]*v[2]/n),0)
            out[prefix+'prediction_gap'] = float(v[6]/n)
            out[prefix+'corr_defined'] = float(va > 0 and vb > 0)
            if va > 0 and vb > 0:
                out[prefix+'abs_error_corr'] = max(-1.,min(1.,float(v[5]-v[1]*v[2]/n)/math.sqrt(va*vb)))
        if self.alpha[0] > 0:
            mean = float(self.alpha[1]/self.alpha[0])
            out['backend_diag_alpha_mean'] = mean
            out['backend_diag_alpha_std'] = math.sqrt(max(0.,float(self.alpha[2]/self.alpha[0])-mean**2))
        return out


class DualMoEMetricAccumulator:
    """Exact spatial gate moments: observed sources vs missing destinations.

    Gate means are weighted by voxel count, not batch count. Per-expert errors
    use the same original-unit missing elements as the final prediction.
    """
    def __init__(self):
        self.moments = {}
        self.experts = {s: MaskedMetricAccumulator() for s in ("fine", "mid", "coarse")}
        self.support = {}
        self.regions = {}
        self.correction = torch.zeros(4, dtype=torch.float64)
        self.topk = {}
        self.active = False
        self.backend = None

    def _add(self, name, gate, mask):
        p, m = gate.detach().double(), mask.detach().double()
        dims = (0, 2, 3, 4)
        entropy = -(p*p.clamp_min(1e-12).log()).sum(1, keepdim=True)
        values = torch.cat((m.sum().reshape(1), (p*m).sum(dims),
                            (p.square()*m).sum(dims), (entropy*m).sum().reshape(1))).cpu()
        self.moments[name] = self.moments.get(name, torch.zeros_like(values))+values

    @torch.no_grad()
    def update(self, outputs, batch):
        if outputs.get("architecture") != "dual_moe":
            return
        self.active = True
        if outputs.get('backend_diagnostics'):
            if self.backend is None:
                self.backend = BackendDiagnosticAccumulator()
            self.backend.update(outputs, batch)
        domain = outputs.get('aggregation_gate_domain', 'observed')
        front_mask = 1-batch['m_f'] if domain == 'missing' else batch['m_f']
        for scale, gate in outputs["aggregation_gates"].items():
            self._add(f"aggregation_{scale}_{domain}", gate, front_mask)
            support = outputs["aggregation_support"][scale].detach().double()
            mass = outputs["aggregation_mass"][scale]
            count = mass.numel()
            values = torch.cat((support.sum((0, 1, 2, 3)).cpu(),
                                torch.tensor([float((mass == 0).sum()), count], dtype=torch.float64)))
            self.support[scale] = self.support.get(scale, torch.zeros_like(values))+values
            assignment = outputs['region_assignments'][scale]
            if assignment is None:
                # Hard regular-bin membership: conditional entropy is exactly
                # zero. Infer the observed marginal from compact bin counts.
                count = mass.sum(-1)
                marginal = mass/count[..., None].clamp_min(1)
                entropy = -(marginal*marginal.clamp_min(1e-12).log()).sum(-1)
                valid = count > 0
                values = torch.stack((count.sum()*0, count.sum(),
                                      (entropy.exp()*valid).sum(), valid.float().sum())).double().cpu()
            else:
                a = assignment.detach().float()
                b, e, t, n, k = a.shape
                m = batch['m_f'].reshape(b, 1, t, n).float()
                count = m.sum(-1)
                valid = (count > 0).expand(b, e, t)
                conditional = -(a*a.clamp_min(1e-12).log()).sum(-1)
                marginal = (a*m[..., None]).sum(-2)/count[..., None].clamp_min(1)
                entropy = -(marginal*marginal.clamp_min(1e-12).log()).sum(-1)
                values = torch.stack(((conditional*m).double().sum(), m.double().sum()*e,
                                      (entropy.exp()*valid).double().sum(), valid.double().sum())).cpu()
            self.regions[scale] = self.regions.get(scale, torch.zeros_like(values))+values
        self._add("completion_missing", outputs["completion_gates"], 1-batch["m_f"])
        for scale, pred in outputs["scale_predictions"].items():
            self.experts[scale].update(pred, batch["x_f_gt"], batch["m_f"])
        for name, routing in outputs.get('routing_details', {}).items():
            valid = (batch['m_f'] if name.startswith('aggregation_') else 1-batch['m_f']).double()
            if routing.get('valid_domain') == 'missing':
                valid = (1-batch['m_f']).double()
            probs, selected = routing['probabilities'].detach().double(), routing['selected'].detach().double()
            values = torch.cat((valid.sum().reshape(1), (probs*valid).sum((0,2,3,4)),
                                (selected*valid).sum((0,2,3,4)))).cpu()
            self.topk[name] = self.topk.get(name, torch.zeros_like(values))+values
        if 'normalized_correction' in outputs:
            correction = outputs['normalized_correction'].detach().float()
            missing = (1-batch['m_f']).expand_as(correction)
            residuals = [v.detach().float() for v in outputs['normalized_residuals'].values()]
            bound = outputs['residual_bound']
            self.correction += torch.stack(((correction.abs()*missing).sum(), missing.sum(),
                                            sum(((r.abs() > .95*bound)*missing).sum() for r in residuals),
                                            missing.sum()*len(residuals))).double().cpu()

    def compute(self):
        if not self.active:
            return {}
        result = {}
        if self.backend is not None:
            result.update(self.backend.compute())
        for name, values in self.moments.items():
            count = float(values[0])
            result[f"{name}_count"] = count
            # No observations/missing points means undefined routing, not NaN.
            # Omit moments and expose count=0 rather than imply uniform routing.
            if not count:
                continue
            experts = (values.numel()-2)//2
            mean = values[1:1+experts]/count
            std = (values[1+experts:1+2*experts]/count-mean.square()).clamp_min(0).sqrt()
            labels = ("fine", "mid", "coarse") if name == "completion_missing" else tuple(f"e{i}" for i in range(experts))
            for i, label in enumerate(labels):
                result[f"{name}_{label}_mean"] = float(mean[i])
                result[f"{name}_{label}_std"] = float(std[i])
            result[f"{name}_entropy"] = float(values[-1]/count)
        for scale, values in self.support.items():
            for index, label in enumerate(("mass_support", "effective_support", "heterogeneity", "empty_fraction")):
                result[f"aggregation_{scale}_{label}"] = float(values[index]/values[4])
        for scale, metrics in self.experts.items():
            for key, value in metrics.compute().items():
                result[f"{key}_expert_{scale}"] = value
        for scale, values in self.regions.items():
            if values[1] > 0:
                result[f"aggregation_{scale}_assignment_entropy"] = float(values[0]/values[1])
                result[f"aggregation_{scale}_effective_regions"] = float(values[2]/values[3])
        if self.correction[1] > 0:
            result['normalized_correction_abs_mean'] = float(self.correction[0]/self.correction[1])
            result['residual_saturation_fraction'] = float(self.correction[2]/self.correction[3])
        for name, values in self.topk.items():
            count = float(values[0]); experts = (len(values)-1)//2
            result[f'topk_{name}_token_count'] = count
            if not count:
                continue
            importance = values[1:1+experts]/count
            selected = values[1+experts:]
            activation = selected/count
            load = selected/selected.sum().clamp_min(1)
            for e in range(experts):
                result[f'topk_{name}_e{e}_probability'] = float(importance[e])
                result[f'topk_{name}_e{e}_activation'] = float(activation[e])
                result[f'topk_{name}_e{e}_load'] = float(load[e])
            result[f'topk_{name}_selected_per_token'] = float(activation.sum())
            result[f'topk_{name}_load_cv2'] = float(experts*(load-1/experts).square().sum())
        return result


def active_routing_scales(scale_mode: str) -> tuple[str, ...]:
    mapping = {
        "fine": ("fine",),
        "fine_mid": ("fine", "mid"),
        "fine_mid_coarse": ("fine", "mid", "coarse"),
    }
    try:
        return mapping[scale_mode]
    except KeyError as error:
        raise ValueError(f"Unknown scale_mode: {scale_mode}") from error


class RoutingMetricAccumulator:
    """Accumulate routing statistics over every sample instead of batch means."""

    def __init__(
        self,
        scale_names: tuple[str, ...],
        dead_threshold: float = 0.01,
        always_threshold: float = 0.99,
        eps: float = 1e-12,
    ) -> None:
        if not scale_names:
            raise ValueError("scale_names must contain at least one active scale")
        self.scale_names = tuple(scale_names)
        self.dead_threshold = float(dead_threshold)
        self.always_threshold = float(always_threshold)
        self.eps = float(eps)
        self._gate_sum: dict[str, torch.Tensor] = {}
        self._load_sum: dict[str, torch.Tensor] = {}
        self._sample_count: dict[str, int] = defaultdict(int)
        self._margin_sum: dict[str, float] = defaultdict(float)
        self._margin_count: dict[str, int] = defaultdict(int)

    def _add(
        self,
        name: str,
        gate: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> None:
        if gate.ndim != 2 or selected_mask.shape != gate.shape:
            raise ValueError(
                "Expected gate and selected_mask shaped [batch, experts], got "
                f"{tuple(gate.shape)} and {tuple(selected_mask.shape)}"
            )
        gate_cpu = gate.detach().to(device="cpu", dtype=torch.float64)
        selected_cpu = selected_mask.detach().to(device="cpu", dtype=torch.float64)
        if name not in self._gate_sum:
            self._gate_sum[name] = torch.zeros(gate_cpu.shape[1], dtype=torch.float64)
            self._load_sum[name] = torch.zeros(gate_cpu.shape[1], dtype=torch.float64)
        if self._gate_sum[name].numel() != gate_cpu.shape[1]:
            raise ValueError(f"Expert count changed while accumulating {name}")

        self._gate_sum[name] += gate_cpu.sum(dim=0)
        self._load_sum[name] += selected_cpu.sum(dim=0)
        self._sample_count[name] += int(gate_cpu.shape[0])

        selected_per_sample = selected_cpu.sum(dim=1)
        if selected_per_sample.numel() == 0:
            return
        top_k = int(round(float(selected_per_sample[0])))
        if (
            0 < top_k < gate_cpu.shape[1]
            and torch.allclose(
                selected_per_sample,
                torch.full_like(selected_per_sample, float(top_k)),
            )
        ):
            sorted_gate = gate_cpu.sort(dim=1, descending=True).values
            margins = sorted_gate[:, top_k - 1] - sorted_gate[:, top_k]
            self._margin_sum[name] += float(margins.sum())
            self._margin_count[name] += int(margins.numel())

    def update(
        self,
        gates: dict[str, torch.Tensor],
        selected_masks: dict[str, torch.Tensor] | None,
    ) -> None:
        if selected_masks is None:
            return
        active: list[tuple[torch.Tensor, torch.Tensor]] = []
        for scale in self.scale_names:
            gate = gates.get(scale)
            selected = selected_masks.get(scale)
            if not torch.is_tensor(gate) or not torch.is_tensor(selected):
                continue
            self._add(scale, gate, selected)
            active.append((gate, selected))
        if active:
            self._add(
                "all",
                torch.cat([item[0] for item in active], dim=0),
                torch.cat([item[1] for item in active], dim=0),
            )

    def _compute_group(self, name: str) -> dict[str, float]:
        count = self._sample_count[name]
        if count <= 0:
            return {}
        importance = self._gate_sum[name] / count
        hard_load = self._load_sum[name] / count
        load_distribution = hard_load / hard_load.sum().clamp_min(self.eps)
        mean_load = hard_load.mean()
        load_cv = hard_load.std(unbiased=False) / mean_load.clamp_min(self.eps)
        entropy = -(
            load_distribution
            * load_distribution.clamp_min(self.eps).log()
        ).sum()
        if load_distribution.numel() > 1:
            entropy = entropy / math.log(load_distribution.numel())
        soft_hard_gap = (importance - load_distribution).abs().sum()
        prefix = f"routing_{name}"
        result = {
            f"{prefix}_hard_load_cv": float(load_cv),
            f"{prefix}_selection_entropy": float(entropy),
            f"{prefix}_dead_expert_rate": float(
                (hard_load < self.dead_threshold).to(torch.float64).mean()
            ),
            f"{prefix}_always_selected_rate": float(
                (hard_load > self.always_threshold).to(torch.float64).mean()
            ),
            f"{prefix}_soft_hard_l1_gap": float(soft_hard_gap),
        }
        if self._margin_count[name] > 0:
            result[f"{prefix}_topk_boundary_margin"] = (
                self._margin_sum[name] / self._margin_count[name]
            )
        for index, value in enumerate(importance):
            result[f"{prefix}_soft_importance_{index}"] = float(value)
        for index, value in enumerate(hard_load):
            result[f"{prefix}_hard_load_{index}"] = float(value)
        return result

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in (*self.scale_names, "all"):
            if self._sample_count[name] > 0:
                result.update(self._compute_group(name))
        active_names = [
            name for name in self.scale_names if self._sample_count[name] > 0
        ]
        for metric in (
            "hard_load_cv",
            "selection_entropy",
            "dead_expert_rate",
            "always_selected_rate",
            "soft_hard_l1_gap",
            "topk_boundary_margin",
        ):
            values = [
                result[f"routing_{name}_{metric}"]
                for name in active_names
                if f"routing_{name}_{metric}" in result
            ]
            if values:
                result[f"routing_scales_mean_{metric}"] = sum(values) / len(values)
                result[f"routing_scales_max_{metric}"] = max(values)
        return result
