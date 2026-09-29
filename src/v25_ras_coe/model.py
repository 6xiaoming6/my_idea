"""V25 selective repair backbone with an unchanged v24 disabled path.

The v24 routing/expert core is inherited. Its forward loop is mirrored here
because latent acceptance must happen before the next round reads the state.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from stmoe_imputer.models.coe_router import PairRouter, observed_pattern_features
from stmoe_imputer.models.temporal_spatial_coe import (
    SUPPORT_FEATURE_NAMES, TemporalSpatialCoE, compute_observation_support,
)
from .repair_gate import RepairAcceptanceGate


class FeedbackRouter(nn.Module):
    """Add rejection statistics to C2 or native routing, preserving zero-init logits."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if not isinstance(base, (PairRouter, nn.Sequential)):
            raise TypeError("Repair feedback requires a legacy pair or native router")
        self.base = base
        layers = base.layers if isinstance(base, PairRouter) else base
        self.feedback_projection = nn.Linear(2, layers[1].out_features, bias=False)
        nn.init.zeros_(self.feedback_projection.weight)

    def forward(self, features: torch.Tensor):
        old_features, repair = features[:, :-2], features[:, -2:]
        layers = self.base.layers if isinstance(self.base, PairRouter) else self.base
        hidden = layers[1](layers[0](old_features))
        hidden = layers[2](hidden + self.feedback_projection(repair))
        logits = layers[3](hidden)
        if isinstance(self.base, PairRouter):
            return logits, (self.base.pair_head(hidden) if self.base.pair_head is not None else None)
        return logits


class RASCoE(TemporalSpatialCoE):
    """C2 expert chain with opt-in, position-wise selective latent repair."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.repair_acceptance = "none"
        self.repair_feedback_to_router = False
        self.repair_accept_aux_head_only = True
        self.repair_acceptance_gate = None

    @classmethod
    def from_config(cls, cfg: dict) -> "RASCoE":
        model = super().from_config(cfg)
        coe = cfg["model"].get("coe", {})
        mode = coe.get("repair_acceptance", "none")
        feedback = coe.get("repair_feedback_to_router", False)
        aux_only = coe.get("repair_accept_aux_head_only", True)
        if mode not in {"none", "latent_point"}:
            raise ValueError("repair_acceptance must be none or latent_point")
        if type(feedback) is not bool or type(aux_only) is not bool:
            raise ValueError("repair feedback and aux-head-only flags must be boolean")
        if mode == "latent_point" and (
            model.acceptance != "none" or model.router_state != "dynamic" or
            model.expert_state != "dynamic" or not model.completion_feedback or
            model.routing_mode != "hard" or model.top_k != 2 or
            not ((model.pair_mode == "partner_residual" and model.partner_fusion == "corrected") or
                 (model.pair_mode == "native" and model.partner_fusion == "individual")) or
            model.state_update_mode != "residual"
        ):
            raise ValueError("latent_point requires dynamic C2 or native hard Top-2 with completion feedback and no legacy acceptance")
        if feedback and (mode != "latent_point" or model.router_features != "legacy"):
            raise ValueError("repair_feedback_to_router requires latent_point and legacy router features")
        if mode == "latent_point":
            model.repair_acceptance_gate = RepairAcceptanceGate(
                model.dim, model.c_in, coe.get("repair_accept_init_prob", 0.99)
            )
        model.repair_acceptance = mode
        model.repair_feedback_to_router = feedback
        model.repair_accept_aux_head_only = aux_only
        if feedback:
            model.routers = nn.ModuleList(FeedbackRouter(router) for router in model.routers)
        return model

    def _router_features(self, hidden, values, support_summary, change, missing,
                         pattern_summary=None, repair_signal=None):
        original = super()._router_features(
            hidden, values, support_summary, change, missing, pattern_summary,
        )
        if not self.repair_feedback_to_router:
            return original
        if repair_signal is None:
            repair_signal = hidden.new_zeros((hidden.shape[0], 1, *hidden.shape[2:]))
        point_missing = missing.bool().any(dim=1, keepdim=True).to(repair_signal.dtype)
        repair_global = repair_signal.float().mean(dim=(2, 3, 4))
        repair_missing = self._missing_pool(repair_signal.float(), point_missing)
        return torch.cat((original, repair_global.to(original.dtype),
                          repair_missing.to(original.dtype)), dim=1)

    def forward(self, x_f: torch.Tensor, m_f: torch.Tensor, **kwargs: object) -> dict:
        if self.repair_acceptance == "none":
            return super().forward(x_f, m_f, **kwargs)
        if x_f.ndim != 5 or any(size < 1 for size in x_f.shape):
            raise ValueError("x_f must have nonempty shape [B, C, T, H, W]")
        if not x_f.is_floating_point():
            raise ValueError("x_f must have a floating-point dtype")
        if x_f.shape[1] != self.c_in:
            raise ValueError(f"Expected c_in={self.c_in}, got {x_f.shape[1]}")
        if (
            m_f.ndim != 5
            or m_f.shape[0] != x_f.shape[0]
            or m_f.shape[2:] != x_f.shape[2:]
            or m_f.shape[1] not in (1, self.c_in)
        ):
            raise ValueError("m_f must match x_f with either 1 or c_in mask channels")
        if m_f.device != x_f.device:
            raise ValueError("x_f and m_f must be on the same device")
        if not bool(torch.all((m_f == 0) | (m_f == 1))):
            raise ValueError("m_f must contain only finite binary values")
        forced_partner_step = kwargs.get("forced_partner_step")
        forced_partner_ids = kwargs.get("forced_partner_ids")
        forced_pair_step = kwargs.get("forced_pair_step")
        forced_pair_indices = kwargs.get("forced_pair_indices")
        if self.pair_mode not in {"partner", "partner_residual"} and forced_partner_step is not None:
            raise ValueError("forced_partner_step requires partner routing")
        if forced_partner_step is not None and (
            type(forced_partner_step) is not int or not 0 <= forced_partner_step < self.num_steps
            or not torch.is_tensor(forced_partner_ids)
            or forced_partner_ids.shape != (x_f.shape[0],)
        ):
            raise ValueError("forced partner needs a valid step and one partner per sample")
        if forced_pair_step is not None and (
            type(forced_pair_step) is not int or not 0 <= forced_pair_step < self.num_steps
            or not torch.is_tensor(forced_pair_indices)
            or forced_pair_indices.shape != (x_f.shape[0], 2)
        ):
            raise ValueError("forced pair needs a valid step and two experts per sample")
        observed = m_f.bool().expand_as(x_f)
        if not bool(torch.all(torch.isfinite(x_f) | ~observed)):
            raise ValueError("Input observed values must be finite; mark missing values with m_f=0")
        original_mask = observed.to(x_f.dtype)
        x_input = torch.where(observed, x_f, torch.zeros_like(x_f))
        support = compute_observation_support(
            original_mask, self.temporal_kernel, self.spatial_kernel
        ).to(x_f.dtype)
        missing = 1 - original_mask
        support_missing = missing.repeat(1, len(SUPPORT_FEATURE_NAMES), 1, 1, 1)
        support_summary = torch.cat(
            [
                support.mean(dim=(2, 3, 4)),
                support.std(dim=(2, 3, 4), unbiased=False),
                support.amax(dim=(2, 3, 4)),
                self._missing_pool(support, support_missing),
            ],
            dim=1,
        )
        position = self._position(x_input)
        hidden = self.encoder(torch.cat([x_input, original_mask, support, position], dim=1))
        prediction = self.decoder(hidden)
        completion = torch.where(observed, x_input, prediction)
        change = torch.zeros_like(prediction)
        previous_rejection = x_input.new_zeros((x_input.shape[0], 1, *x_input.shape[2:]))
        repair_logits_history, repair_aux_logits_history = [], []
        repair_weight_history, repair_rejection_history, repair_commit_history = [], [], []
        initial_hidden = hidden
        initial_prediction, initial_completion = prediction, completion
        pattern_summary = (
            observed_pattern_features(x_input, observed, self.router_value_std)
            if self.router_features == "grouped" else None
        )
        initial_router_features = self._router_features(
            hidden, completion, support_summary, change, missing, pattern_summary
        )
        hard_fraction, uniform_mix, sampling_temperature = self.routing_schedule()

        predictions, completions, changes = [], [], []
        candidate_predictions, candidate_completions, candidate_changes = [], [], []
        acceptance_history = []
        logits_history, probability_history, weight_history, path_history = [], [], [], []
        selected_history, pair_id_history, pair_logit_history, pair_prob_history = [], [], [], []
        importance_history, partner_score_history, primary_history, partner_history = [], [], [], []
        partner_rank_score_history, partner_fusion_weight_history = [], []
        partner_correction_abs_history = []
        fusion_second_history, fusion_shift_history = [], []
        previous_choice_history = []
        previous_candidate_prediction = initial_prediction
        for step in range(self.num_steps):
            forced_expert = self.fixed_expert_steps[step]
            forced_index = self.expert_names.index(forced_expert) if forced_expert is not None else None
            previous_choice = None
            if self.previous_expert_context:
                previous_choice = (F.one_hot(path_history[-1].detach(), self.num_experts).float()
                                   if step else hidden.new_zeros((hidden.shape[0], self.num_experts)))
                previous_choice_history.append(previous_choice)
            pair_bias = None
            if self.use_routed and self.routing_mode != "fixed" and forced_index is None:
                router_features = (
                    initial_router_features
                    if self.router_state == "initial"
                    else self._router_features(
                        hidden,
                        completion if self.completion_feedback else initial_completion,
                        support_summary,
                        change if self.completion_feedback else torch.zeros_like(change),
                        missing, pattern_summary, previous_rejection,
                    )
                )
                router_features = self._add_router_input_noise(router_features, step)
                if self.global_route_weights:
                    logits = self.global_route_logits[step].to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0).expand(hidden.shape[0], -1)
                else:
                    with torch.autocast(device_type=hidden.device.type, enabled=False) if self.router_fp32 else nullcontext():
                        features = router_features.float() if self.router_fp32 else router_features
                        scored = (self.routers[step](features, previous_choice)
                                  if self.previous_expert_context and step else self.routers[step](features))
                        if self.pair_mode == "native":
                            logits = scored
                        else:
                            logits, pair_bias = scored
            elif forced_index is not None:
                logits = hidden.new_full((hidden.shape[0], self.num_experts), -20.0)
                logits[:, forced_index] = 20.0
            else:
                logits = hidden.new_zeros((hidden.shape[0], self.num_experts))
            # The partner route observes the first expert's proposal before
            # choosing a companion; all variants reuse this same U and shared update.
            expert_hidden = initial_hidden if self.expert_state == "initial" else hidden
            expert_completion = (initial_completion if self.expert_state == "initial" or
                                 not self.completion_feedback else completion)
            unified = self.state_norm(self.state_projection(torch.cat(
                [expert_hidden, expert_completion, original_mask, support, position], dim=1
            )))
            shared_update = (self.shared_scale_logits[step].sigmoid() * self.shared_expert(unified)
                             if self.use_shared else torch.zeros_like(hidden))
            primary_update = None
            partner_scores = None
            # Gumbel hard argmax is invariant to tau: its clean categorical
            # probabilities are softmax(logits). Tau controls the surrogate
            # gradient; only the soft-mixture variant tempers its actual weights.
            probability_logits = (
                logits / self.temperature
                if self.routing_mode in {"soft", "parallel"} else logits / sampling_temperature
            )
            probabilities = F.softmax(probability_logits.float(), dim=-1)
            if not self.use_routed:
                weights = torch.zeros_like(probabilities)
                paths = torch.full((hidden.shape[0],), -1, device=hidden.device, dtype=torch.long)
            elif forced_index is not None:
                paths = torch.full((hidden.shape[0],), forced_index, device=hidden.device, dtype=torch.long)
                weights = F.one_hot(paths, self.num_experts).to(hidden.dtype)
            elif self.routing_mode == "fixed":
                expert_index = self.expert_names.index(self.fixed_path[step])
                paths = torch.full((hidden.shape[0],), expert_index, device=hidden.device, dtype=torch.long)
                weights = F.one_hot(paths, self.num_experts).to(hidden.dtype)
            elif self.routing_mode in {"soft", "parallel"}:
                weights = probabilities
                paths = weights.argmax(dim=-1)
            elif self.pair_mode in {"partner", "partner_residual"}:
                native_top2 = logits.float().topk(2, dim=-1).indices
                primary = (native_top2[:, 0] if self.pair_mode == "partner_residual"
                           else logits.argmax(dim=-1))
                primary_update = self._dispatch(unified, primary, step)
                proposal_hidden = (hidden + shared_update +
                                   self.routed_scale_logits[step].sigmoid() * primary_update)
                proposal = self.decoder(proposal_hidden)
                proposal_change = torch.where(observed, torch.zeros_like(proposal),
                                              proposal - prediction)
                point_missing = missing.mean(dim=1, keepdim=True)
                partner_features = torch.cat((
                    router_features[:, :-2] if self.repair_feedback_to_router else router_features,
                    primary_update.mean(dim=(2, 3, 4)),
                    self._missing_pool(primary_update, point_missing),
                    proposal_change.mean(dim=(2, 3, 4)),
                    self._missing_pool(proposal_change, missing),
                ), dim=1)
                correction = self.partner_scorer(partner_features, primary)
                if self.pair_mode == "partner_residual":
                    primary_mask = F.one_hot(primary, self.num_experts).bool()
                    partner_correction_abs_history.append(
                        correction.detach().masked_fill(primary_mask, 0.).abs().sum(-1).div(self.num_experts - 1).mean()
                    )
                    # Zero-initialized correction preserves native Top-2,
                    # including torch.topk's tie rule.
                    partner_scores = logits.float() + correction.float()
                    candidate_scores = partner_scores.masked_fill(primary_mask, -torch.inf)
                    best = candidate_scores.max(dim=-1).values
                    native_second = native_top2[:, 1]
                    native_score = candidate_scores.gather(1, native_second[:, None]).squeeze(1)
                    partner = torch.where(native_score == best, native_second,
                                          candidate_scores.argmax(dim=-1))
                else:
                    partner_scores = correction
                    partner = partner_scores.argmax(dim=-1)
                if self.partner_aux_head_only:
                    rank_correction = self.partner_scorer(partner_features.detach(), primary)
                    partner_rank_score_history.append(logits.detach().float() + rank_correction.float())
                if forced_partner_step == step:
                    partner = forced_partner_ids.to(device=primary.device, dtype=torch.long)
                    if bool(((partner == primary) | (partner < 0) |
                             (partner >= self.num_experts)).any()):
                        raise ValueError("forced partner must differ from the primary expert")
                selected = torch.stack((primary, partner), dim=-1)
                if forced_pair_step == step:
                    selected = forced_pair_indices.to(device=primary.device, dtype=torch.long)
                if bool(((selected < 0) | (selected >= self.num_experts)).any()) or bool(
                    (selected[:, 0] == selected[:, 1]).any()
                ):
                    raise ValueError("forced pair must contain two distinct valid experts")
                fusion_logits = logits.float().gather(1, selected)
                if self.partner_fusion == "corrected" and forced_pair_step != step:
                    fusion_logits = torch.stack((
                        fusion_logits[:, 0], partner_scores.gather(1, selected[:, 1:]).squeeze(1)
                    ), dim=-1)
                within = F.softmax(fusion_logits / sampling_temperature, dim=-1)
                weights = torch.zeros_like(probabilities).scatter(1, selected, within)
                if self.partner_fusion == "corrected" and forced_pair_step != step:
                    partner_fusion_weight_history.append(within[:, 1].detach().float().mean())
                paths = weights.argmax(dim=-1)
                ordered = selected.sort(dim=-1).values
                first, second = ordered.unbind(dim=-1)
                pair_ids = first * (2 * self.num_experts - first - 1) // 2 + (second - first - 1)
                left, right = self.pair_indices.unbind(dim=-1)
                pair_logits = torch.where(
                    primary[:, None] == left[None], partner_scores[:, right],
                    torch.where(primary[:, None] == right[None], partner_scores[:, left],
                                partner_scores.new_full((len(primary), len(left)), -1e4)),
                )
                pair_probs = F.softmax(pair_logits.float() / sampling_temperature, dim=-1)
                partner_score_history.append(partner_scores)
                primary_history.append(primary)
                partner_history.append(partner)
            elif self.pair_mode != "native":
                dense_fraction = (1 - hard_fraction if step < self.pair_dense_warmup_steps else 0.)
                weights, pair_ids, pair_logits, pair_probs = self._pair_route(
                    logits, pair_bias, sampling_temperature, dense_fraction, uniform_mix
                )
                paths = weights.argmax(dim=-1)
            elif forced_pair_step == step:
                selected = forced_pair_indices.to(device=logits.device, dtype=torch.long)
                if bool(((selected < 0) | (selected >= self.num_experts)).any()) or bool(
                    (selected[:, 0] == selected[:, 1]).any()
                ):
                    raise ValueError("forced pair must contain two distinct valid experts")
                within = F.softmax(logits.float().gather(1, selected) /
                                   sampling_temperature, dim=-1)
                weights = torch.zeros_like(probabilities).scatter(1, selected, within)
                paths = weights.argmax(dim=-1)
            elif self.training:
                if self.top_k > 1:
                    # Top-K keeps only the selected experts and renormalizes
                    # their clean probabilities. This is a differentiable
                    # weighted fusion inside a discrete selected set.
                    sample_logits = (logits.float() if self.router_fp32 else logits) / sampling_temperature
                    selected = sample_logits.topk(self.top_k, dim=-1).indices
                    selection_mask = F.one_hot(selected, self.num_experts).any(dim=-2).to(probabilities.dtype)
                    weights = probabilities * selection_mask
                    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                else:
                    with torch.autocast(device_type=hidden.device.type, enabled=False) if self.router_fp32 else nullcontext():
                        sample_logits = (logits.float() if self.router_fp32 else logits) / sampling_temperature
                        if hard_fraction < 1:
                            soft = uniform_mix / self.num_experts + (1 - uniform_mix) * probabilities
                            if hard_fraction == 0:
                                weights = soft
                            else:
                                hard = F.gumbel_softmax(sample_logits, tau=self.temperature, hard=True, dim=-1)
                                weights = (1 - hard_fraction) * soft + hard_fraction * hard
                        else:
                            weights = F.gumbel_softmax(sample_logits, tau=self.temperature, hard=True, dim=-1)
                paths = weights.argmax(dim=-1)
            else:
                if self.top_k > 1:
                    selected = logits.topk(self.top_k, dim=-1).indices
                    selection_mask = F.one_hot(selected, self.num_experts).any(dim=-2).to(probabilities.dtype)
                    weights = probabilities * selection_mask
                    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    paths = weights.argmax(dim=-1)
                else:
                    paths = logits.argmax(dim=-1)
                    weights = F.one_hot(paths, self.num_experts).to(hidden.dtype)

            if self.use_routed and self.routing_mode == "hard" and self.top_k == 2:
                if self.pair_mode == "native":
                    selected = weights.topk(2, dim=-1).indices.sort(dim=-1).values
                    first, second = selected.unbind(dim=-1)
                    pair_ids = first * (2 * self.num_experts - first - 1) // 2 + (second - first - 1)
                    individual = logits.float()
                    pair_logits = (individual[:, self.pair_indices[:, 0]] +
                                   individual[:, self.pair_indices[:, 1]])
                    pair_probs = F.softmax(pair_logits / sampling_temperature, dim=-1)
                else:
                    selected = self.pair_indices[pair_ids]
                selected_history.append(selected)
                pair_id_history.append(pair_ids)
                pair_logit_history.append(pair_logits)
                pair_prob_history.append(pair_probs)
                corrected_scores = (partner_scores if self.pair_mode == "partner_residual" and
                                    self.partner_fusion == "corrected" else None)
                importance_history.append(self._pair_importance(
                    logits, pair_probs, sampling_temperature, corrected_scores,
                    primary if corrected_scores is not None else None,
                ))
            else:
                importance_history.append(probabilities)

            precomputed_routed_update = None
            if self.fusion_mode != "original":
                active_ids = (forced_pair_indices.to(device=logits.device, dtype=torch.long)
                              if forced_pair_step == step else logits.float().topk(2, dim=-1).indices)
                precomputed_routed_update, second_mean, shift_mean = self._conditional_pair_update(
                    unified, active_ids, logits, router_features, missing, original_mask,
                    support, hidden, shared_update, completion, x_input,
                    step, sampling_temperature,
                )
                fusion_second_history.append(second_mean)
                fusion_shift_history.append(shift_mean)
            update = shared_update
            if self.use_routed:
                if precomputed_routed_update is not None:
                    routed_update = precomputed_routed_update
                elif self.routing_mode in {"soft", "parallel"}:
                    routed_update = torch.zeros_like(unified)
                    for expert_index, expert in enumerate(self.routed_experts(step)):
                        routed_update = routed_update + (
                            weights[:, expert_index, None, None, None, None] * expert(unified)
                        )
                elif self.routing_mode == "hard":
                    if (self.pair_mode in {"partner", "partner_residual"} and primary_update is not None and
                            forced_pair_step != step):
                        companion_update = self._dispatch(unified, partner, step)
                        primary_weight = weights.gather(1, primary[:, None]).flatten()
                        companion_weight = weights.gather(1, partner[:, None]).flatten()
                        routed_update = (
                            primary_update * primary_weight[:, None, None, None, None] +
                            companion_update * companion_weight[:, None, None, None, None]
                        )
                    else:
                        routed_update = self._dispatch_weighted(unified, weights, step)
                else:
                    routed_update = self._dispatch(unified, paths, step)
                update = update + self.routed_scale_logits[step].sigmoid() * routed_update
            hidden_before = hidden
            old_completion = completion
            previous_change = change
            candidate_hidden = hidden_before + update
            candidate_prediction = self.decoder(candidate_hidden)
            candidate_completion = torch.where(observed, x_input, candidate_prediction)
            candidate_change = torch.where(observed, torch.zeros_like(candidate_prediction),
                                           (candidate_prediction - previous_candidate_prediction).abs())
            gate_features = self.repair_acceptance_gate.features(
                hidden_before, update, old_completion, candidate_completion,
                previous_change, original_mask, support,
            )
            acceptance_logits = self.repair_acceptance_gate(gate_features)
            acceptance_aux_logits = (
                self.repair_acceptance_gate(gate_features.detach())
                if self.repair_accept_aux_head_only else acceptance_logits
            )
            acceptance_weight = torch.sigmoid(acceptance_logits)
            point_missing = missing.bool().any(dim=1, keepdim=True)
            commit_weight = torch.where(point_missing, acceptance_weight,
                                        torch.ones_like(acceptance_weight))
            hidden = hidden_before + commit_weight * update
            accepted_prediction = self.decoder(hidden)
            completion = torch.where(observed, x_input, accepted_prediction)
            prediction = accepted_prediction
            change = torch.where(observed, torch.zeros_like(prediction),
                                 (completion - old_completion).abs())
            previous_rejection = torch.where(point_missing, 1.0 - acceptance_weight,
                                             torch.zeros_like(acceptance_weight))
            repair_logits_history.append(acceptance_logits)
            repair_aux_logits_history.append(acceptance_aux_logits)
            repair_weight_history.append(acceptance_weight)
            repair_rejection_history.append(previous_rejection)
            repair_commit_history.append(commit_weight)
            previous_candidate_prediction = candidate_prediction
            predictions.append(prediction)
            completions.append(completion)
            changes.append(change)
            candidate_predictions.append(candidate_prediction)
            candidate_completions.append(candidate_completion)
            candidate_changes.append(candidate_change)
            acceptance_history.append(acceptance_weight)
            logits_history.append(logits)
            probability_history.append(probabilities)
            weight_history.append(weights)
            path_history.append(paths)

        route_probabilities = torch.stack(probability_history, dim=1)
        route_importance = torch.stack(importance_history, dim=1)
        route_weights = torch.stack(weight_history, dim=1)
        effective_mode = ("soft" if self.routing_mode == "hard" and hard_fraction < 1
                          and (self.top_k == 1 or self.pair_dense_warmup_steps > 0)
                          else self.routing_mode)
        diagnostics = {
            "hard_fraction": hidden.new_tensor(hard_fraction),
            "sampling_temperature": hidden.new_tensor(sampling_temperature),
            "update_abs_mean": changes[-1].detach().mean(),
            "missing_fraction": missing.detach().mean(),
            "shared_residual_scale": self.shared_scale_logits.detach().sigmoid().mean(),
            "routed_residual_scale": self.routed_scale_logits.detach().sigmoid().mean(),
            "acceptance_mean": torch.stack(acceptance_history).detach().float().mean(),
            "pair_dense_warmup_fraction": hidden.new_tensor(
                1 - hard_fraction if self.pair_dense_warmup_steps else 0.
            ),
        }
        point_missing_float = missing.bool().any(dim=1, keepdim=True).float()
        point_count = point_missing_float.sum().clamp_min(1.0)
        diagnostics["repair_accept_mean"] = torch.stack([
            (weight.detach().float() * point_missing_float).sum() / point_count
            for weight in repair_weight_history
        ]).mean()
        diagnostics["repair_reject_mean"] = 1.0 - diagnostics["repair_accept_mean"]
        for step, step_change in enumerate(changes, start=1):
            weight = repair_weight_history[step - 1].detach().float()
            rejection = repair_rejection_history[step - 1].detach().float()
            proposal = (candidate_completions[step - 1] -
                        (initial_completion if step == 1 else completions[step - 2])).detach().float()
            diagnostics[f"step{step}_repair_accept_mean"] = (weight * point_missing_float).sum() / point_count
            diagnostics[f"step{step}_repair_accept_std"] = (((weight - diagnostics[f"step{step}_repair_accept_mean"]).square() * point_missing_float).sum() / point_count).sqrt()
            diagnostics[f"step{step}_repair_reject_mean"] = (rejection * point_missing_float).sum() / point_count
            diagnostics[f"step{step}_repair_proposal_abs_mean"] = (proposal.abs() * missing.float()).sum() / missing.float().sum().clamp_min(1.)
            diagnostics[f"step{step}_repair_commit_below_05_rate"] = ((weight < 0.5).float() * point_missing_float).sum() / point_count
            if self.previous_expert_context and step > 1:
                diagnostics[f'step{step}_previous_expert_embedding_norm'] = self.routers[step - 1].previous_expert_embedding.detach().float().norm()
            step_logits = logits_history[step - 1].detach().float()
            diagnostics[f"step{step}_logit_abs_max"] = step_logits.abs().max()
            diagnostics[f"step{step}_logit_range"] = (step_logits.max(-1).values - step_logits.min(-1).values).mean()
            if self.num_experts > 1:
                top2 = step_logits.topk(2, dim=-1).values
                diagnostics[f"step{step}_top12_margin"] = (top2[:, 0] - top2[:, 1]).mean()
            diagnostics[f"step{step}_update_abs_mean"] = step_change.detach().float().mean()
            if self.fusion_mode != "original":
                diagnostics[f"step{step}_fusion_second_weight_mean"] = fusion_second_history[step - 1]
                diagnostics[f"step{step}_fusion_shift_abs_mean"] = fusion_shift_history[step - 1]
            if step <= self.pair_dense_warmup_steps:
                diagnostics[f"step{step}_dense_route_fraction"] = hidden.new_tensor(1 - hard_fraction)
            if self.partner_fusion == "corrected" and len(partner_fusion_weight_history) >= step:
                diagnostics[f"step{step}_partner_fusion_weight_mean"] = (
                    partner_fusion_weight_history[step - 1]
                )
            if self.pair_mode == "partner_residual" and len(partner_correction_abs_history) >= step:
                diagnostics[f"step{step}_partner_correction_abs_mean"] = (
                    partner_correction_abs_history[step - 1].detach().float()
                )
                ranked = step_logits.topk(3, dim=-1).values
                diagnostics[f"step{step}_native_partner_margin"] = (ranked[:, 1] - ranked[:, 2]).mean()
            if self.routing_mode == "hard" and self.top_k == 2:
                native_pair = step_logits.topk(2, dim=-1).indices.sort(dim=-1).values
                chosen_pair = selected_history[step - 1]
                diagnostics[f"step{step}_pair_vs_top2_disagreement_rate"] = (
                    (chosen_pair != native_pair).any(dim=-1).float().mean()
                )
        if self.use_routed and self.routing_mode != "fixed":
            diagnostic_probabilities = route_probabilities.detach().float()
            diagnostics["route_entropy"] = -(
                diagnostic_probabilities * diagnostic_probabilities.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            candidate_marginal = route_importance.detach().float()
            diagnostics["candidate_importance_entropy"] = -(
                candidate_marginal * candidate_marginal.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            diagnostics["candidate_importance_vs_selected_l1"] = (
                candidate_marginal - route_weights.detach().float()
            ).abs().sum(dim=-1).mean()
        return {
            "x_hat_main": prediction,
            "h_st_aux": hidden,
            "diagnostics": {"coe": diagnostics},
            "coe": {
                "predictions": predictions,
                "completions": completions,
                "candidate_predictions": candidate_predictions,
                "candidate_completions": candidate_completions,
                "candidate_changes": candidate_changes,
                "acceptance_weights": torch.stack(acceptance_history, dim=1),
                "repair_acceptance_logits": torch.stack(repair_logits_history, dim=1),
                "repair_acceptance_aux_logits": torch.stack(repair_aux_logits_history, dim=1),
                "repair_acceptance_weights": torch.stack(repair_weight_history, dim=1),
                "repair_rejection_maps": torch.stack(repair_rejection_history, dim=1),
                "repair_commit_weights": torch.stack(repair_commit_history, dim=1),
                "selected_experts": (torch.stack(selected_history, dim=1) if selected_history else None),
                "pair_ids": (torch.stack(pair_id_history, dim=1) if pair_id_history else None),
                "pair_logits": (torch.stack(pair_logit_history, dim=1) if pair_logit_history else None),
                "pair_probs": (torch.stack(pair_prob_history, dim=1) if pair_prob_history else None),
                "initial_prediction": initial_prediction,
                "initial_completion": initial_completion,
                "changes": changes,
                "route_logits": torch.stack(logits_history, dim=1),
                "route_probs": route_probabilities,
                "route_importance": route_importance,
                "route_weights": route_weights,
                "partner_scores": (torch.stack(partner_score_history, dim=1)
                                   if partner_score_history else None),
                "partner_rank_scores": (torch.stack(partner_rank_score_history, dim=1)
                                        if partner_rank_score_history else None),
                "primary_ids": (torch.stack(primary_history, dim=1)
                                if primary_history else None),
                "partner_ids": (torch.stack(partner_history, dim=1)
                                if partner_history else None),
                "paths": torch.stack(path_history, dim=1),
                "paths_are_discrete": self.use_routed and effective_mode not in {"soft", "parallel"},
                "routing_mode": effective_mode,
                "configured_routing_mode": self.routing_mode,
                "router_state": self.router_state,
                "expert_state": self.expert_state,
                "pair_mode": self.pair_mode,
                "partner_fusion": self.partner_fusion,
                "partner_aux_head_only": self.partner_aux_head_only,
                "fusion_mode": self.fusion_mode,
                "pair_dense_warmup_steps": self.pair_dense_warmup_steps,
                "acceptance": self.acceptance,
                "expert_sharing": self.expert_sharing,
                "completion_feedback": self.completion_feedback,
                "previous_expert_context": self.previous_expert_context,
                "previous_expert_choices": (torch.stack(previous_choice_history, dim=1)
                                            if previous_choice_history else None),
                "use_shared": self.use_shared,
                "use_routed": self.use_routed,
                "expert_names": self.expert_names,
                "num_experts": self.num_experts,
                "num_steps": self.num_steps,
                "top_k": self.top_k,
                "support": support,
                "support_feature_names": SUPPORT_FEATURE_NAMES,
                "observation_mask": original_mask,
                "diagnostics": diagnostics,
            },
        }
