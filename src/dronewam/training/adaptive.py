"""Minimal staged optimization for adaptive rollout."""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from dronewam.config import ExperimentConfig
from dronewam.data import build_preference_dataset, load_records
from dronewam.geometry import build_timeline
from dronewam.metrics import (normalized_trajectory_cost,
                              trajectory_metric_tensors)
from dronewam.models.adaptive import (ContinuationPolicy,
                                      GoalConditionedContinuationPolicy,
                                      IterativeDepthPlanner,
                                      PlannerStateHorizonHead,
                                      dpo_pairwise_loss,
                                      dpo_reward_probability,
                                      pool_policy_tokens)
from dronewam.models.wam import DroneWAM
from dronewam.preferences import (PreferenceDepthDataset,
                                  PreferencePairDataset,
                                  PreferenceWindowDataset,
                                  preference_group_split_indices,
                                  preference_split_indices,
                                  rank_rollout_depths,
                                  representative_group_indices)
from dronewam.training.distributed import DistributedContext
from dronewam.training.losses import _discounted_latent_l1

logger = logging.getLogger(__name__)

POLICY_OBJECTIVE = "best_vs_rest_plan_reward_v2"


@dataclass(frozen=True)
class MultiLengthTrainingOutput:
    """Differentiable all-depth world and planning outputs for one batch."""

    latent_loss: torch.Tensor
    trajectory_loss: torch.Tensor
    actions_by_depth: torch.Tensor
    context_by_step: torch.Tensor
    future_by_step: torch.Tensor
    observed_kinematics: torch.Tensor
    goal_tokens: torch.Tensor
    relative_ate: torch.Tensor
    relative_rpe: torch.Tensor
    planner_states: torch.Tensor | None


def _gate_aligned_task_cost(
    costs_by_depth: torch.Tensor,
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Train only the candidate selected by the deployable DPO reward."""
    if costs_by_depth.ndim != 2 or costs_by_depth.shape[1] < 2:
        raise ValueError("costs_by_depth must be [B,H>=2]")
    expected_logits_shape = (*costs_by_depth.shape, 2)
    if (
        policy_logits.shape != expected_logits_shape
        or reference_logits.shape != expected_logits_shape
    ):
        raise ValueError(f"policy and reference logits must be {expected_logits_shape}")
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (costs_by_depth, policy_logits, reference_logits)
    ):
        raise ValueError("Gate-aligned task tensors must be finite")
    with torch.no_grad():
        rewards = dpo_reward_probability(
            policy_logits.flatten(0, 1),
            reference_logits.flatten(0, 1),
            beta,
        ).reshape_as(costs_by_depth)
        selected_indices = rewards.argmax(dim=1)
    rows = torch.arange(costs_by_depth.shape[0], device=costs_by_depth.device)
    return costs_by_depth[rows, selected_indices].mean()


def _reward_selection_metrics(
    reward_probabilities: torch.Tensor,
    normalized_costs: torch.Tensor,
    relative_ate: torch.Tensor,
    relative_rpe: torch.Tensor,
    best_depths: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Measure the deployable full-horizon reward argmax on train calibration data."""
    if reward_probabilities.ndim != 2 or reward_probabilities.shape[1] < 2:
        raise ValueError("reward_probabilities must be [N,H>=2]")
    if (
        normalized_costs.shape != reward_probabilities.shape
        or relative_ate.shape != reward_probabilities.shape
        or relative_rpe.shape != reward_probabilities.shape
        or best_depths.shape != (reward_probabilities.shape[0],)
    ):
        raise ValueError("Reward-selection tensors must align")
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (
            reward_probabilities,
            normalized_costs,
            relative_ate,
            relative_rpe,
        )
    ):
        raise ValueError("Reward-selection tensors must be finite")
    if torch.any(reward_probabilities < 0.0) or torch.any(reward_probabilities > 1.0):
        raise ValueError("reward_probabilities must lie in [0,1]")
    if torch.any(best_depths < 1) or torch.any(
        best_depths > reward_probabilities.shape[1]
    ):
        raise ValueError("best_depths are outside the rollout horizon")
    rows = torch.arange(reward_probabilities.shape[0])
    selected_indices = reward_probabilities.argmax(dim=1)
    return (
        float(normalized_costs[rows, selected_indices].mean().item()),
        float(relative_ate[rows, selected_indices].mean().item()),
        float(relative_rpe[rows, selected_indices].mean().item()),
        float((selected_indices + 1 == best_depths).float().mean().item()),
    )


def _torch_load(path: str | Path) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    checkpoint = torch.load(path, **kwargs)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} must contain a dictionary")
    return checkpoint


def _atomic_save(state: Dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def load_adaptive_checkpoint(
    model: DroneWAM,
    path: str | Path,
    *,
    require_planner: bool,
    require_policy: bool,
    require_predictor: bool = False,
    load_planner: bool = True,
) -> int:
    if require_planner and not load_planner:
        raise ValueError("require_planner=true is incompatible with load_planner=false")
    checkpoint = _torch_load(path)
    required = []
    if require_predictor and "predictor" not in checkpoint:
        required.append("predictor")
    if require_planner and "adaptive_planner" not in checkpoint:
        required.append("adaptive_planner")
    if require_policy and "continuation_policy" not in checkpoint:
        required.append("continuation_policy")
    if require_policy and "reference_policy" not in checkpoint:
        required.append("reference_policy")
    if require_policy and "policy_token_count" not in checkpoint:
        required.append("policy_token_count")
    if require_policy and checkpoint.get("policy_objective") != POLICY_OBJECTIVE:
        required.append(f"policy_objective={POLICY_OBJECTIVE}")
    if required:
        raise KeyError(f"Adaptive checkpoint {path} is missing {required}")
    if (
        require_policy
        and model.continuation_policy is not None
        and checkpoint["policy_token_count"] != model.continuation_policy.token_count
    ):
        raise ValueError(
            "Adaptive checkpoint policy_token_count does not match the configured continuation policy"
        )
    for name in ("predictor", "adaptive_planner", "continuation_policy"):
        if name == "adaptive_planner" and not load_planner:
            continue
        if name == "continuation_policy" and not require_policy:
            continue
        state = checkpoint.get(name)
        if state is None:
            continue
        module = getattr(model, name)
        if module is None:
            raise ValueError(f"Checkpoint has {name}, but model does not")
        module.load_state_dict(state, strict=True)
    if require_policy:
        if model.continuation_reference is None:
            raise ValueError("Checkpoint has reference_policy, but model does not")
        model.continuation_reference.load_state_dict(
            checkpoint["reference_policy"], strict=True
        )
        model.continuation_reference.eval().requires_grad_(False)
    return int(checkpoint.get("epoch", -1)) + 1


def load_reference_policy(policy: ContinuationPolicy, path: str | Path) -> None:
    checkpoint = _torch_load(path)
    if "reference_policy" not in checkpoint:
        raise KeyError(f"Adaptive checkpoint {path} has no reference_policy")
    if checkpoint.get("policy_token_count") != policy.token_count:
        raise ValueError(
            "Adaptive checkpoint policy_token_count does not match the reference policy"
        )
    if checkpoint.get("policy_objective") != POLICY_OBJECTIVE:
        raise ValueError(
            f"Adaptive checkpoint must use policy_objective={POLICY_OBJECTIVE}"
        )
    policy.load_state_dict(checkpoint["reference_policy"], strict=True)
    policy.eval().requires_grad_(False)


def save_adaptive_checkpoint(
    model: DroneWAM,
    path: str | Path,
    *,
    stage: str,
    epoch: int,
    reference_policy: ContinuationPolicy | None = None,
) -> None:
    if stage not in ("planner", "multilength", "dpo", "joint"):
        raise ValueError(f"Unsupported adaptive checkpoint stage={stage!r}")
    if model.adaptive_planner is None:
        raise RuntimeError("Adaptive checkpoint requires an adaptive planner")
    state: Dict[str, Any] = {
        "format_version": 3,
        "stage": stage,
        "epoch": int(epoch),
        "adaptive_planner": model.adaptive_planner.state_dict(),
    }
    if stage == "multilength":
        state["predictor"] = model.predictor.state_dict()
    if stage in ("dpo", "joint"):
        if model.continuation_policy is None:
            raise RuntimeError(
                f"Adaptive checkpoint stage={stage!r} requires a continuation policy"
            )
        state.update(
            {
                "policy_token_count": model.continuation_policy.token_count,
                "policy_objective": POLICY_OBJECTIVE,
                "continuation_policy": model.continuation_policy.state_dict(),
            }
        )
    if reference_policy is not None:
        state["reference_policy"] = reference_policy.state_dict()
    _atomic_save(state, path)


def save_joint_horizon_checkpoint(
    model: DroneWAM,
    horizon_policy: torch.nn.Module,
    path: str | Path,
    *,
    epoch: int,
) -> None:
    """Save only the three modules optimized by joint horizon training."""
    if model.adaptive_planner is None:
        raise RuntimeError("Joint horizon checkpoint requires an adaptive planner")
    hidden_dim = getattr(horizon_policy, "hidden_dim", None)
    if not isinstance(hidden_dim, int):
        raise TypeError("Joint horizon policy must expose integer hidden_dim")
    plan_conditioned = isinstance(horizon_policy, PlannerStateHorizonHead)
    _atomic_save(
        {
            "format_version": 1,
            "stage": "joint_horizon",
            "epoch": int(epoch),
            "predictor": model.predictor.state_dict(),
            "adaptive_planner": model.adaptive_planner.state_dict(),
            "horizon_policy": horizon_policy.state_dict(),
            "policy_type": type(horizon_policy).__name__,
            "policy_hidden_dim": hidden_dim,
            "policy_token_count": int(getattr(horizon_policy, "token_count", 0)),
            "policy_horizon": int(getattr(horizon_policy, "horizon", 0)),
            "policy_expansion": int(getattr(horizon_policy, "expansion", 1)),
            "policy_dropout": float(getattr(horizon_policy, "dropout", 0.0)),
            "policy_action_dim": (
                int(getattr(horizon_policy, "action_dim", 0)) if plan_conditioned else 0
            ),
            "policy_plan_horizon": (
                int(getattr(horizon_policy, "plan_horizon", 0))
                if plan_conditioned
                else 0
            ),
        },
        path,
    )


def _unwrap_objective(objective: torch.nn.Module) -> torch.nn.Module:
    return (
        objective.module
        if isinstance(objective, DistributedDataParallel)
        else objective
    )


def _set_adaptive_trainability(model: DroneWAM, stage: str) -> None:
    model.requires_grad_(False)
    if stage == "planner":
        assert model.adaptive_planner is not None
        model.adaptive_planner.requires_grad_(True)
    elif stage == "joint":
        assert (
            model.adaptive_planner is not None and model.continuation_policy is not None
        )
        model.adaptive_planner.requires_grad_(True)
        model.continuation_policy.requires_grad_(True)
    else:
        raise ValueError(f"Unsupported adaptive model stage={stage!r}")


def _set_multilength_trainability(model: DroneWAM) -> None:
    model.requires_grad_(False)
    if model.adaptive_planner is None:
        raise RuntimeError("Multi-length WAM training requires an adaptive planner")
    model.set_predictor_trainable(True)
    model.adaptive_planner.requires_grad_(True)


def _set_joint_horizon_trainability(
    model: DroneWAM,
    horizon_policy: torch.nn.Module,
) -> None:
    _set_multilength_trainability(model)
    horizon_policy.requires_grad_(True)


def _set_frozen_wam_eval(model: DroneWAM) -> None:
    """Keep frozen WAM modules deterministic after a parent module enters train mode."""
    model.encoder.eval()
    model.target_encoder.eval()
    model.predictor.eval()
    model.action_head.eval()
    if model.continuation_reference is not None:
        model.continuation_reference.eval()
    if model.token_ae is not None:
        model.token_ae.eval()


def _set_multilength_backbone_eval(model: DroneWAM) -> None:
    """Keep frozen visual and action modules deterministic during joint prefix training."""
    model.encoder.eval()
    model.target_encoder.eval()
    model.action_head.eval()
    if model.token_ae is not None:
        model.token_ae.eval()
    if model.continuation_policy is not None:
        model.continuation_policy.eval()
    if model.continuation_reference is not None:
        model.continuation_reference.eval()


class PlannerObjective(torch.nn.Module):
    """Apply the canonical normalized trajectory cost at every rollout depth."""

    def __init__(
        self,
        model: DroneWAM,
        config: ExperimentConfig,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "oracle", "final", "final_plus_shallow"):
            raise ValueError(f"Unsupported planner reduction={reduction!r}")
        self.model = model
        self.ate_target = config.adaptive_training.ate_target
        self.rpe_target = config.adaptive_training.rpe_target
        self.shallow_weight = config.adaptive_training.planner_shallow_weight
        self.metric_power = config.adaptive_training.planner_metric_power
        self.reduction = reduction

    def forward(
        self, frames: torch.Tensor, states: torch.Tensor, goal_frames: torch.Tensor
    ) -> torch.Tensor:
        observed_frames = frames[:, :, : self.model.num_observed_frames]
        observed_states = states[:, : self.model.num_observed_frames]
        with torch.no_grad():
            world = self.model.rollout_world(observed_frames, observed_states)
            goal_tokens = self.model.encode_goal(goal_frames)
        timeline_states, _ = build_timeline(states, self.model.frame_stride)
        horizon = self.model.num_steps - self.model.observed_steps
        costs = []
        for depth in range(1, horizon + 1):
            depths = torch.full(
                (frames.shape[0],), depth, dtype=torch.long, device=frames.device
            )
            future_prefix = world.future_latents[
                :, : depth * self.model.tokens_per_step
            ]
            actions = self.model.plan_actions(
                world.observed_context,
                future_prefix,
                world.observed_kinematics,
                depths,
                goal_tokens=goal_tokens,
            )
            costs.append(
                normalized_trajectory_cost(
                    actions,
                    timeline_states,
                    self.model.observed_steps,
                    ate_target=self.ate_target,
                    rpe_target=self.rpe_target,
                    power=self.metric_power,
                )
            )
        costs_by_depth = torch.stack(costs, dim=1)
        if self.reduction == "oracle":
            return costs_by_depth.min(dim=1).values.mean()
        if self.reduction == "final":
            return costs_by_depth[:, -1].mean()
        if self.reduction == "final_plus_shallow":
            return (
                costs_by_depth[:, -1].mean()
                + self.shallow_weight * costs_by_depth[:, :-1].mean()
            )
        return costs_by_depth.mean()


class MultiLengthWAMObjective(torch.nn.Module):
    """Train world prediction and full-horizon planning from every causal prefix depth."""

    def __init__(self, model: DroneWAM, config: ExperimentConfig) -> None:
        super().__init__()
        if model.adaptive_planner is None:
            raise RuntimeError(
                "Multi-length WAM objective requires an adaptive planner"
            )
        self.model = model
        self.latent_weight = config.adaptive_training.world_latent_weight
        self.trajectory_weight = config.adaptive_training.world_trajectory_weight
        self.ar_discount = config.loss.ar_discount
        self.ate_target = config.adaptive_training.ate_target
        self.rpe_target = config.adaptive_training.rpe_target
        self.metric_power = config.adaptive_training.planner_metric_power

    def _plan_all_depths(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        observed_kinematics: torch.Tensor,
        goal_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Plan every prefix depth in one batched planner call.

        Returns
        -------
        torch.Tensor
            Full-horizon actions shaped ``[B, C, H, A]``, where the
            second dimension indexes every available rollout candidate.
        """
        actions, _ = self._plan_all_depths_with_state(
            context, future, observed_kinematics, goal_tokens
        )
        return actions

    def _plan_all_depths_with_state(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        observed_kinematics: torch.Tensor,
        goal_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.model.adaptive_planner is None or self.model.action_head is None:
            raise RuntimeError(
                "Multi-length WAM planning requires an adaptive planner and action head"
            )
        context_by_step = self.model._tokens_by_step(context)
        future_by_step = self.model._tokens_by_step(future)
        batch_size, horizon = future_by_step.shape[:2]
        expected_horizon = self.model.num_steps - self.model.observed_steps
        if horizon != expected_horizon:
            raise ValueError(f"Expected {expected_horizon} future steps, got {horizon}")
        if isinstance(self.model.adaptive_planner, IterativeDepthPlanner):
            return self.model.adaptive_planner.rollout_all_depths(
                context_by_step,
                future_by_step,
                observed_kinematics,
                self.model.action_head,
                goal_tokens,
                include_zero_depth=True,
            )

        expanded_context = (
            context_by_step[:, None]
            .expand(-1, horizon, -1, -1, -1)
            .reshape(
                batch_size * horizon,
                *context_by_step.shape[1:],
            )
        )
        expanded_future = (
            future_by_step[:, None]
            .expand(-1, horizon, -1, -1, -1)
            .reshape(
                batch_size * horizon,
                *future_by_step.shape[1:],
            )
        )
        expanded_kinematics = (
            observed_kinematics[:, None]
            .expand(-1, horizon, -1)
            .reshape(
                batch_size * horizon,
                observed_kinematics.shape[-1],
            )
        )
        expanded_goal = (
            goal_tokens[:, None]
            .expand(-1, horizon, -1, -1)
            .reshape(
                batch_size * horizon,
                *goal_tokens.shape[1:],
            )
        )
        depths = torch.arange(1, horizon + 1, device=future.device).repeat(batch_size)
        actions = self.model.adaptive_planner(
            expanded_context,
            expanded_future,
            expanded_kinematics,
            depths,
            self.model.action_head,
            expanded_goal,
        )
        return actions.reshape(batch_size, horizon, horizon, actions.shape[-1]), None

    def forward(
        self,
        frames: torch.Tensor,
        states: torch.Tensor,
        goal_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward_all_depths(frames, states, goal_frames)
        total = (
            self.latent_weight * output.latent_loss
            + self.trajectory_weight * output.trajectory_loss
        )
        return total, output.latent_loss, output.trajectory_loss

    def forward_all_depths(
        self,
        frames: torch.Tensor,
        states: torch.Tensor,
        goal_frames: torch.Tensor,
    ) -> MultiLengthTrainingOutput:
        """Return the shared differentiable rollout used by all three joint losses."""
        observed_frames = frames[:, :, : self.model.num_observed_frames]
        observed_states = states[:, : self.model.num_observed_frames]
        with torch.no_grad():
            context, _, _, observed_kinematics = self.model._prepare_observed_rollout(
                observed_frames,
                observed_states,
            )
            target = self.model._encode(
                self.model.target_encoder, frames, self.model.num_raw_frames
            )
            target_future = target[
                :, self.model.observed_steps * self.model.tokens_per_step :
            ]
            goal_tokens = self.model.encode_goal(goal_frames)
            timeline_states, timeline_actions = build_timeline(
                states, self.model.frame_stride
            )
        future = self.model._autoregressive_rollout(
            context, timeline_actions, timeline_states
        )
        latent_loss = _discounted_latent_l1(
            future,
            target_future,
            self.model.tokens_per_step,
            self.ar_discount,
        )
        actions_by_depth, planner_states = self._plan_all_depths_with_state(
            context,
            future,
            observed_kinematics,
            goal_tokens,
        )
        batch_size, candidate_count = actions_by_depth.shape[:2]
        expanded_states = (
            timeline_states[:, None]
            .expand(-1, candidate_count, -1, -1)
            .reshape(
                batch_size * candidate_count,
                *timeline_states.shape[1:],
            )
        )
        metrics = trajectory_metric_tensors(
            actions_by_depth.flatten(0, 1),
            expanded_states,
            self.model.observed_steps,
        )
        relative_ate = metrics.relative_ate.reshape(batch_size, candidate_count)
        relative_rpe = metrics.relative_rpe.reshape(batch_size, candidate_count)
        trajectory_cost = (relative_ate / self.ate_target).pow(self.metric_power) + (
            relative_rpe / self.rpe_target
        ).pow(self.metric_power)
        return MultiLengthTrainingOutput(
            latent_loss=latent_loss,
            trajectory_loss=trajectory_cost.mean(),
            actions_by_depth=actions_by_depth,
            context_by_step=self.model._tokens_by_step(context),
            future_by_step=self.model._tokens_by_step(future),
            observed_kinematics=observed_kinematics,
            goal_tokens=goal_tokens,
            relative_ate=relative_ate,
            relative_rpe=relative_rpe,
            planner_states=planner_states,
        )


def _relative_rank_targets(
    relative_ate: torch.Tensor, relative_rpe: torch.Tensor
) -> torch.Tensor:
    """Return earliest best depth from tie-aware within-sample metric ranks."""
    return _relative_rank_scores(relative_ate, relative_rpe).argmin(dim=1) + 1


def _relative_rank_scores(
    relative_ate: torch.Tensor, relative_rpe: torch.Tensor
) -> torch.Tensor:
    """Return the mean within-sample rel.ATE/rel.RPE rank for every depth."""
    if (
        relative_ate.ndim != 2
        or relative_ate.shape != relative_rpe.shape
        or relative_ate.shape[1] < 2
    ):
        raise ValueError("relative_ate and relative_rpe must align as [B,H>=2]")
    horizon = relative_ate.shape[1]
    ate_rank = (
        (relative_ate.unsqueeze(2) > relative_ate.unsqueeze(1)).sum(dim=2).float()
    )
    rpe_rank = (
        (relative_rpe.unsqueeze(2) > relative_rpe.unsqueeze(1)).sum(dim=2).float()
    )
    return 0.5 * (ate_rank + rpe_rank) / (horizon - 1)


def _causal_horizon_loss(
    logits: torch.Tensor,
    target_depths: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    class_priors: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train every deployable prefix up to each sample's optimal stopping depth."""
    if logits.ndim != 3 or logits.shape[1] != logits.shape[2]:
        raise ValueError("logits must be [B,H,H]")
    batch_size, horizon, _ = logits.shape
    if target_depths.shape != (batch_size,) or target_depths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("target_depths must be integer [B]")
    if class_weights is not None and (
        class_weights.shape != (horizon,)
        or torch.any(class_weights <= 0.0)
        or not torch.isfinite(class_weights).all()
    ):
        raise ValueError("class_weights must be finite and positive [H]")
    if class_priors is not None and (
        class_priors.shape != (horizon,)
        or torch.any(class_priors <= 0.0)
        or not torch.isfinite(class_priors).all()
    ):
        raise ValueError("class_priors must be finite and positive [H]")
    prefix_indices = (
        torch.arange(horizon, device=logits.device).unsqueeze(0).expand(batch_size, -1)
    )
    valid = prefix_indices < target_depths.unsqueeze(1)
    targets = (target_depths - 1).unsqueeze(1).expand(-1, horizon)
    safe_targets = torch.where(valid, targets, prefix_indices)
    adjusted_logits = logits.float()
    if class_priors is not None:
        adjusted_logits = adjusted_logits + class_priors.log().view(1, 1, horizon)
    losses = torch.nn.functional.cross_entropy(
        adjusted_logits.flatten(0, 1),
        safe_targets.flatten(),
        weight=class_weights,
        reduction="none",
    ).reshape(batch_size, horizon)
    return ((losses * valid).sum(dim=1) / target_depths.float()).mean()


def _causal_pairwise_rank_loss(
    logits: torch.Tensor,
    rank_scores: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank all still-reachable candidates at every causal rollout prefix."""
    if logits.ndim != 3 or logits.shape[1] != logits.shape[2]:
        raise ValueError("logits must be [B,H,H]")
    if rank_scores.shape != logits.shape[:2] or not torch.isfinite(rank_scores).all():
        raise ValueError("rank_scores must be finite [B,H]")
    candidate_count = logits.shape[1]
    if class_weights is not None and (
        class_weights.shape != (candidate_count,)
        or torch.any(class_weights <= 0.0)
        or not torch.isfinite(class_weights).all()
    ):
        raise ValueError("class_weights must be finite and positive [H]")
    predicted_differences = logits.unsqueeze(3) - logits.unsqueeze(2)
    target_utilities = -rank_scores.float()
    target_differences = target_utilities.unsqueeze(2) - target_utilities.unsqueeze(1)
    target_signs = target_differences.sign().unsqueeze(1)
    prefixes = torch.arange(candidate_count, device=logits.device).view(1, -1, 1, 1)
    left = torch.arange(candidate_count, device=logits.device).view(1, 1, -1, 1)
    right = torch.arange(candidate_count, device=logits.device).view(1, 1, 1, -1)
    valid = (left >= prefixes) & (right > left) & (target_signs != 0)
    safe_differences = torch.where(valid, predicted_differences.float(), 0.0)
    losses = torch.nn.functional.softplus(-target_signs * safe_differences)
    counts = valid.sum(dim=(2, 3)).clamp_min(1)
    sample_losses = ((losses * valid).sum(dim=(2, 3)) / counts).mean(dim=1)
    if class_weights is None:
        return sample_losses.mean()
    target_indices = rank_scores.argmin(dim=1)
    sample_weights = class_weights[target_indices]
    return (sample_losses * sample_weights).sum() / sample_weights.sum()


def _causal_rank_regression_loss(
    utilities: torch.Tensor,
    rank_scores: torch.Tensor,
) -> torch.Tensor:
    """Regress each reachable quality curve after removing its arbitrary offset."""
    if utilities.ndim != 3 or utilities.shape[1] != utilities.shape[2]:
        raise ValueError("utilities must be [B,H,H]")
    if (
        rank_scores.shape != utilities.shape[:2]
        or not torch.isfinite(rank_scores).all()
    ):
        raise ValueError("rank_scores must be finite [B,H]")
    candidate_count = utilities.shape[1]
    prefixes = torch.arange(candidate_count, device=utilities.device).view(1, -1, 1)
    candidates = torch.arange(candidate_count, device=utilities.device).view(1, 1, -1)
    valid = candidates >= prefixes
    counts = valid.sum(dim=2).clamp_min(1)
    predicted = torch.where(valid, utilities.float(), 0.0)
    targets = -rank_scores.float().unsqueeze(1).expand_as(predicted)
    targets = torch.where(valid, targets, 0.0)
    predicted = predicted - predicted.sum(dim=2, keepdim=True) / counts.unsqueeze(2)
    targets = targets - targets.sum(dim=2, keepdim=True) / counts.unsqueeze(2)
    losses = torch.nn.functional.smooth_l1_loss(predicted, targets, reduction="none")
    return ((losses * valid).sum(dim=2) / counts).mean()


class JointHorizonObjective(torch.nn.Module):
    """Jointly make world rollouts accurate and their optimal depth causally readable."""

    def __init__(
        self,
        model: DroneWAM,
        horizon_policy: GoalConditionedContinuationPolicy | PlannerStateHorizonHead,
        config: ExperimentConfig,
    ) -> None:
        super().__init__()
        self.world = MultiLengthWAMObjective(model, config)
        self.horizon_policy = horizon_policy
        self.horizon_weight = config.adaptive_training.world_horizon_weight
        self.classification_weight = (
            config.adaptive_training.world_horizon_classification_weight
        )
        self.rank_weight = config.adaptive_training.world_horizon_rank_weight
        self.regression_weight = (
            config.adaptive_training.world_horizon_regression_weight
        )
        configured_weights = config.adaptive_training.horizon_class_weights
        if configured_weights is None:
            self.register_buffer("horizon_class_weights", None)
        else:
            weights = torch.tensor(configured_weights, dtype=torch.float32)
            if weights.shape != (horizon_policy.horizon,):
                raise ValueError(
                    "horizon_class_weights must match the joint Gate candidate count"
                )
            self.register_buffer("horizon_class_weights", weights)
        configured_priors = config.adaptive_training.horizon_class_priors
        if configured_priors is None:
            self.register_buffer("horizon_class_priors", None)
        else:
            priors = torch.tensor(configured_priors, dtype=torch.float32)
            if priors.shape != (horizon_policy.horizon,):
                raise ValueError(
                    "horizon_class_priors must match the joint Gate candidate count"
                )
            self.register_buffer("horizon_class_priors", priors / priors.sum())

    def _horizon_logits(self, output: MultiLengthTrainingOutput) -> torch.Tensor:
        if isinstance(self.horizon_policy, PlannerStateHorizonHead):
            if output.planner_states is None:
                raise RuntimeError(
                    "Planner-state horizon policy requires an iterative planner"
                )
            planner_states = output.planner_states
            planned_actions = output.actions_by_depth
            if planner_states.shape[1] == self.horizon_policy.horizon + 1:
                planner_states = planner_states[:, 1:]
                planned_actions = planned_actions[:, 1:]
            return self.horizon_policy(planner_states, planned_actions)
        batch_size, horizon = output.actions_by_depth.shape[:2]
        policy_tokens = self.horizon_policy.token_count
        context_tokens = pool_policy_tokens(
            output.context_by_step[:, -1], policy_tokens
        )
        goal_tokens = pool_policy_tokens(output.goal_tokens, policy_tokens)
        future_spatial = pool_policy_tokens(output.future_by_step, policy_tokens)
        future_pooled = output.future_by_step.mean(dim=2)
        context_last = output.context_by_step[:, -1].mean(dim=1)
        goal_last = output.goal_tokens.mean(dim=1)

        def repeat(value: torch.Tensor) -> torch.Tensor:
            return (
                value[:, None]
                .expand(-1, horizon, *([-1] * (value.ndim - 1)))
                .flatten(0, 1)
            )

        depths = torch.arange(1, horizon + 1, device=context_last.device).repeat(
            batch_size
        )
        logits = self.horizon_policy(
            repeat(context_last),
            repeat(context_tokens),
            repeat(goal_last),
            repeat(goal_tokens),
            repeat(future_pooled),
            repeat(future_spatial),
            repeat(output.actions_by_depth),
            repeat(output.observed_kinematics),
            depths,
        ).reshape(batch_size, horizon, horizon)
        current = torch.arange(horizon, device=logits.device).view(1, horizon, 1)
        candidates = torch.arange(horizon, device=logits.device).view(1, 1, horizon)
        return logits.masked_fill(candidates < current, -torch.inf)

    def forward(
        self,
        frames: torch.Tensor,
        states: torch.Tensor,
        goal_frames: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        output = self.world.forward_all_depths(frames, states, goal_frames)
        logits = self._horizon_logits(output)
        policy_ate = output.relative_ate
        policy_rpe = output.relative_rpe
        if policy_ate.shape[1] == logits.shape[1] + 1:
            policy_ate = policy_ate[:, 1:]
            policy_rpe = policy_rpe[:, 1:]
        if policy_ate.shape[1] != logits.shape[1]:
            raise ValueError(
                "Joint horizon policy candidates do not align with trajectory metrics"
            )
        with torch.no_grad():
            rank_scores = _relative_rank_scores(policy_ate, policy_rpe)
            targets = rank_scores.argmin(dim=1) + 1
        classification_loss = _causal_horizon_loss(
            logits,
            targets,
            self.horizon_class_weights,
            self.horizon_class_priors,
        )
        rank_loss = _causal_pairwise_rank_loss(
            logits,
            rank_scores,
        )
        regression_loss = _causal_rank_regression_loss(logits, rank_scores)
        horizon_loss = (
            self.classification_weight * classification_loss
            + self.rank_weight * rank_loss
            + self.regression_weight * regression_loss
        )
        total = (
            self.world.latent_weight * output.latent_loss
            + self.world.trajectory_weight * output.trajectory_loss
            + self.horizon_weight * horizon_loss
        )
        predictions = logits.argmax(dim=2) + 1
        depths = torch.arange(1, logits.shape[1] + 1, device=logits.device).unsqueeze(0)
        selected = (predictions == depths).to(torch.int64).argmax(dim=1) + 1
        exact_hit = (selected == targets).float().mean()
        rows = torch.arange(selected.shape[0], device=selected.device)
        selected_indices = selected - 1
        selected_ate = policy_ate[rows, selected_indices].mean()
        selected_rpe = policy_rpe[rows, selected_indices].mean()
        return (
            total,
            output.latent_loss,
            output.trajectory_loss,
            horizon_loss,
            exact_hit,
            selected_ate,
            selected_rpe,
        )


class JointObjective(torch.nn.Module):
    """Align the planner with deployable reward selection while refreshing preferences."""

    def __init__(
        self,
        model: DroneWAM,
        reference_policy: ContinuationPolicy,
        config: ExperimentConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.reference_policy = reference_policy.requires_grad_(False)
        self.ate_target = config.adaptive_training.ate_target
        self.rpe_target = config.adaptive_training.rpe_target
        self.metric_power = config.adaptive_training.planner_metric_power
        self.dpo_beta = config.adaptive_training.dpo_beta
        self.dpo_weight = config.adaptive_training.dpo_weight
        self.compute_penalty = config.adaptive_training.compute_penalty
        self.preference_margin = config.adaptive_training.preference_margin

    def forward(
        self,
        frames: torch.Tensor,
        states: torch.Tensor,
        goal_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model.continuation_policy is None:
            raise RuntimeError("Joint objective requires continuation policy")
        world = self.model.rollout_world(
            frames[:, :, : self.model.num_observed_frames],
            states[:, : self.model.num_observed_frames],
        )
        with torch.no_grad():
            goal_tokens = self.model.encode_goal(goal_frames)
        timeline_states, _ = build_timeline(states, self.model.frame_stride)

        context_by_step = self.model._tokens_by_step(world.observed_context)
        future_by_step = self.model._tokens_by_step(world.future_latents)
        context_last_tokens = context_by_step[:, -1]
        context_last = context_last_tokens.mean(dim=1)
        context_tokens = pool_policy_tokens(
            context_last_tokens,
            self.model.continuation_policy.token_count,
        )
        future_pooled = future_by_step.mean(dim=2)
        future_spatial = pool_policy_tokens(
            future_by_step,
            self.model.continuation_policy.token_count,
        )
        policy_logits = []
        reference_logits = []
        costs = []
        horizon = future_pooled.shape[1]
        plan_history = []
        for depth in range(1, horizon + 1):
            depth_tensor = torch.full(
                (frames.shape[0],),
                depth,
                dtype=torch.long,
                device=frames.device,
            )
            current_plan = self.model.plan_actions(
                world.observed_context,
                world.future_latents[:, : depth * self.model.tokens_per_step],
                world.observed_kinematics,
                depth_tensor,
                goal_tokens=goal_tokens,
            )
            costs.append(
                normalized_trajectory_cost(
                    current_plan,
                    timeline_states,
                    self.model.observed_steps,
                    ate_target=self.ate_target,
                    rpe_target=self.rpe_target,
                    power=self.metric_power,
                )
            )
            plan_history.append(current_plan)
            plans = torch.stack(plan_history, dim=1).detach()
            policy_logits.append(
                self.model.continuation_policy(
                    context_last.detach(),
                    context_tokens.detach(),
                    future_pooled[:, :depth].detach(),
                    future_spatial[:, :depth].detach(),
                    plans,
                    world.observed_kinematics.detach(),
                    depth_tensor,
                )
            )
            with torch.no_grad():
                reference_logits.append(
                    self.reference_policy(
                        context_last.detach(),
                        context_tokens.detach(),
                        future_pooled[:, :depth].detach(),
                        future_spatial[:, :depth].detach(),
                        plans,
                        world.observed_kinematics.detach(),
                        depth_tensor,
                    )
                )
        policy_logits_tensor = torch.stack(policy_logits, dim=1)
        reference_logits_tensor = torch.stack(reference_logits, dim=1)
        costs_by_depth = torch.stack(costs, dim=1)
        task_cost = _gate_aligned_task_cost(
            costs_by_depth,
            policy_logits_tensor,
            reference_logits_tensor,
            self.dpo_beta,
        )
        best_indices = (
            rank_rollout_depths(
                costs_by_depth.detach(),
                compute_penalty=self.compute_penalty,
                preference_margin=self.preference_margin,
            )
            - 1
        )
        rows = torch.arange(frames.shape[0], device=frames.device)
        depth_indices = torch.arange(horizon, device=frames.device).expand(
            frames.shape[0], -1
        )
        rejected_indices = depth_indices[
            depth_indices != best_indices.unsqueeze(1)
        ].reshape(
            frames.shape[0],
            horizon - 1,
        )
        chosen_policy = (
            policy_logits_tensor[rows, best_indices]
            .unsqueeze(1)
            .expand(-1, horizon - 1, -1)
        )
        chosen_reference = (
            reference_logits_tensor[rows, best_indices]
            .unsqueeze(1)
            .expand(-1, horizon - 1, -1)
        )
        rejected_policy = policy_logits_tensor.gather(
            1,
            rejected_indices.unsqueeze(-1).expand(
                -1, -1, policy_logits_tensor.shape[-1]
            ),
        )
        rejected_reference = reference_logits_tensor.gather(
            1,
            rejected_indices.unsqueeze(-1).expand(
                -1, -1, reference_logits_tensor.shape[-1]
            ),
        )
        preference_loss = dpo_pairwise_loss(
            chosen_policy.flatten(0, 1),
            rejected_policy.flatten(0, 1),
            chosen_reference.flatten(0, 1),
            rejected_reference.flatten(0, 1),
            self.dpo_beta,
        )
        total = task_cost + self.dpo_weight * preference_loss
        return total, task_cost, preference_loss


def _distributed_mean(value: torch.Tensor) -> float:
    detached = value.detach().float()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(detached, op=dist.ReduceOp.SUM)
        detached /= dist.get_world_size()
    return float(detached.item())


def _set_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _autocast(config: ExperimentConfig, device: torch.device):
    enabled = device.type == "cuda" and config.optimization.dtype != "float32"
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.optimization.dtype]
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.no_grad()
def _evaluate_planner(
    objective: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    distributed: DistributedContext,
) -> float:
    objective.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=distributed.device)
    for batch in loader:
        frames = batch["frames"].to(distributed.device, non_blocking=True)
        states = batch["states"].to(distributed.device, non_blocking=True)
        goal_frames = batch["goal_frame"].to(distributed.device, non_blocking=True)
        with _autocast(config, distributed.device):
            loss = objective(frames, states, goal_frames)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite planner calibration loss")
        totals[0] += loss.double() * frames.shape[0]
        totals[1] += frames.shape[0]
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    if totals[1] <= 0:
        raise ValueError("Planner calibration split produced zero samples")
    return float((totals[0] / totals[1]).item())


@torch.no_grad()
def _evaluate_joint_selection(
    objective: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    distributed: DistributedContext,
) -> float:
    """Evaluate the actual reward-selected candidate on train calibration data."""
    objective.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=distributed.device)
    for batch in loader:
        frames = batch["frames"].to(distributed.device, non_blocking=True)
        states = batch["states"].to(distributed.device, non_blocking=True)
        goal_frames = batch["goal_frame"].to(distributed.device, non_blocking=True)
        with _autocast(config, distributed.device):
            _, selected_cost, _ = objective(frames, states, goal_frames)
        if not torch.isfinite(selected_cost):
            raise FloatingPointError("Non-finite joint calibration selection cost")
        totals[0] += selected_cost.double() * frames.shape[0]
        totals[1] += frames.shape[0]
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    if totals[1] <= 0:
        raise ValueError("Joint calibration split produced zero samples")
    return float((totals[0] / totals[1]).item())


def train_planner(
    model: DroneWAM,
    config: ExperimentConfig,
    distributed: DistributedContext,
    output_path: str | Path,
) -> None:
    _set_adaptive_trainability(model, "planner")
    model = model.to(distributed.device)
    planner_reduction = config.adaptive_training.planner_reduction
    objective: torch.nn.Module = PlannerObjective(
        model,
        config,
        reduction=planner_reduction,
    ).to(distributed.device)
    if distributed.world_size > 1:
        objective = DistributedDataParallel(
            objective,
            device_ids=[distributed.local_rank],
            broadcast_buffers=False,
        )
    core = _unwrap_objective(objective).model
    parameters = [
        parameter
        for parameter in core.adaptive_planner.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.adaptive_training.planner_learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    adaptive_train_indices, calibration_indices = preference_split_indices(
        len(load_records(config.data.train_csv)),
        calibration_fraction=config.adaptive_training.calibration_fraction,
        seed=config.optimization.seed,
    )
    train_dataset = build_preference_dataset(
        config.data,
        config.model.encoder.resolution,
        config.adaptive_training.preference_windows_per_record,
        adaptive_train_indices.tolist(),
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        drop_last=True,
    )
    calibration_dataset = build_preference_dataset(
        config.data,
        config.model.encoder.resolution,
        config.adaptive_training.preference_windows_per_record,
        calibration_indices.tolist(),
    )
    local_calibration_indices = range(
        distributed.rank, len(calibration_dataset), distributed.world_size
    )
    calibration_loader = DataLoader(
        Subset(calibration_dataset, local_calibration_indices),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        drop_last=False,
    )
    selection_objective = PlannerObjective(
        core, config, reduction=planner_reduction
    ).to(distributed.device)
    if distributed.rank == 0:
        logger.info(
            "preference split adaptive_train=%d planner_train_windows=%d calibration=%d "
            "planner_calibration_windows=%d",
            adaptive_train_indices.shape[0],
            len(train_dataset),
            calibration_indices.shape[0],
            len(calibration_dataset),
        )
    best_calibration_cost = _evaluate_planner(
        selection_objective,
        calibration_loader,
        config,
        distributed,
    )
    if distributed.rank == 0:
        logger.info(
            "adaptive planner initial calibration_%s_cost=%.6f",
            planner_reduction,
            best_calibration_cost,
        )
        save_adaptive_checkpoint(
            core,
            output_path,
            stage="planner",
            epoch=-1,
        )
    for epoch in range(config.adaptive_training.planner_epochs):
        objective.train()
        _set_frozen_wam_eval(core)
        _set_epoch(loader, epoch)
        total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            frames = batch["frames"].to(distributed.device, non_blocking=True)
            states = batch["states"].to(distributed.device, non_blocking=True)
            goal_frames = batch["goal_frame"].to(distributed.device, non_blocking=True)
            with _autocast(config, distributed.device):
                loss = objective(frames, states, goal_frames)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite planner loss at epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.optimization.grad_clip)
            optimizer.step()
            total += _distributed_mean(loss)
        calibration_cost = _evaluate_planner(
            selection_objective,
            calibration_loader,
            config,
            distributed,
        )
        if distributed.rank == 0:
            logger.info(
                "adaptive planner epoch=%d train_cost=%.6f calibration_%s_cost=%.6f",
                epoch + 1,
                total / len(loader),
                planner_reduction,
                calibration_cost,
            )
            if calibration_cost < best_calibration_cost:
                best_calibration_cost = calibration_cost
                save_adaptive_checkpoint(
                    core,
                    output_path,
                    stage="planner",
                    epoch=epoch,
                )


@torch.no_grad()
def _evaluate_multilength(
    objective: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    distributed: DistributedContext,
) -> tuple[float, float, float]:
    objective.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=distributed.device)
    for batch in loader:
        frames = batch["frames"].to(distributed.device, non_blocking=True)
        states = batch["states"].to(distributed.device, non_blocking=True)
        goal_frames = batch["goal_frame"].to(distributed.device, non_blocking=True)
        with _autocast(config, distributed.device):
            total, latent, trajectory = objective(frames, states, goal_frames)
        if not all(torch.isfinite(value) for value in (total, latent, trajectory)):
            raise FloatingPointError("Non-finite multi-length calibration loss")
        totals[:3] += (
            torch.stack([total, latent, trajectory]).double() * frames.shape[0]
        )
        totals[3] += frames.shape[0]
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    if totals[3] <= 0:
        raise ValueError("Multi-length calibration split produced zero samples")
    return tuple(float(value) for value in (totals[:3] / totals[3]).tolist())


def train_multilength_wam(
    model: DroneWAM,
    config: ExperimentConfig,
    distributed: DistributedContext,
    output_path: str | Path,
) -> None:
    _set_multilength_trainability(model)
    model = model.to(distributed.device)
    objective: torch.nn.Module = MultiLengthWAMObjective(model, config).to(
        distributed.device
    )
    if distributed.world_size > 1:
        objective = DistributedDataParallel(
            objective,
            device_ids=[distributed.local_rank],
            broadcast_buffers=False,
        )
    core = _unwrap_objective(objective).model
    predictor_parameters = [
        parameter
        for parameter in core.predictor.parameters()
        if parameter.requires_grad
    ]
    planner_parameters = [
        parameter
        for parameter in core.adaptive_planner.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": predictor_parameters, "lr": config.optimization.learning_rate},
            {
                "params": planner_parameters,
                "lr": config.adaptive_training.planner_learning_rate,
            },
        ],
        weight_decay=config.optimization.weight_decay,
    )
    adaptive_train_indices, calibration_indices = preference_split_indices(
        len(load_records(config.data.train_csv)),
        calibration_fraction=config.adaptive_training.calibration_fraction,
        seed=config.optimization.seed,
    )
    train_dataset = build_preference_dataset(
        config.data,
        config.model.encoder.resolution,
        config.adaptive_training.preference_windows_per_record,
        adaptive_train_indices.tolist(),
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        drop_last=True,
    )
    calibration_dataset = build_preference_dataset(
        config.data,
        config.model.encoder.resolution,
        config.adaptive_training.preference_windows_per_record,
        calibration_indices.tolist(),
    )
    local_calibration_indices = range(
        distributed.rank, len(calibration_dataset), distributed.world_size
    )
    calibration_loader = DataLoader(
        Subset(calibration_dataset, local_calibration_indices),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        drop_last=False,
    )
    selection_objective = MultiLengthWAMObjective(core, config).to(distributed.device)
    initial_calibration, initial_latent, initial_trajectory = _evaluate_multilength(
        selection_objective,
        calibration_loader,
        config,
        distributed,
    )
    best_calibration = float("inf")
    if distributed.rank == 0:
        logger.info(
            "multi-length split train_records=%d train_windows=%d calibration_records=%d "
            "calibration_windows=%d initial_total=%.6f initial_latent=%.6f initial_trajectory=%.6f",
            adaptive_train_indices.numel(),
            len(train_dataset),
            calibration_indices.numel(),
            len(calibration_dataset),
            initial_calibration,
            initial_latent,
            initial_trajectory,
        )
    for epoch in range(config.adaptive_training.multilength_epochs):
        objective.train()
        _set_multilength_backbone_eval(core)
        _set_epoch(loader, epoch)
        totals = torch.zeros(3, dtype=torch.float64, device=distributed.device)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            frames = batch["frames"].to(distributed.device, non_blocking=True)
            states = batch["states"].to(distributed.device, non_blocking=True)
            goal_frames = batch["goal_frame"].to(distributed.device, non_blocking=True)
            with _autocast(config, distributed.device):
                total, latent, trajectory = objective(frames, states, goal_frames)
            if not all(torch.isfinite(value) for value in (total, latent, trajectory)):
                raise FloatingPointError(
                    f"Non-finite multi-length loss at epoch={epoch}"
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                predictor_parameters + planner_parameters, config.optimization.grad_clip
            )
            optimizer.step()
            totals += torch.tensor(
                [
                    _distributed_mean(total),
                    _distributed_mean(latent),
                    _distributed_mean(trajectory),
                ],
                dtype=torch.float64,
                device=distributed.device,
            )
        calibration, cal_latent, cal_trajectory = _evaluate_multilength(
            selection_objective,
            calibration_loader,
            config,
            distributed,
        )
        if distributed.rank == 0:
            means = totals / len(loader)
            logger.info(
                "multi-length epoch=%d train_total=%.6f train_latent=%.6f train_trajectory=%.6f "
                "cal_total=%.6f cal_latent=%.6f cal_trajectory=%.6f",
                epoch + 1,
                *means.tolist(),
                calibration,
                cal_latent,
                cal_trajectory,
            )
            if calibration < best_calibration:
                best_calibration = calibration
                save_adaptive_checkpoint(
                    core, output_path, stage="multilength", epoch=epoch
                )


@torch.no_grad()
def _evaluate_joint_horizon(
    objective: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    distributed: DistributedContext,
) -> tuple[float, float, float, float, float, float, float]:
    objective.eval()
    totals = torch.zeros(8, dtype=torch.float64, device=distributed.device)
    for batch in loader:
        frames = batch["frames"].to(distributed.device, non_blocking=True)
        states = batch["states"].to(distributed.device, non_blocking=True)
        goal_frames = batch["goal_frame"].to(distributed.device, non_blocking=True)
        with _autocast(config, distributed.device):
            values = objective(frames, states, goal_frames)
        if not all(torch.isfinite(value) for value in values):
            raise FloatingPointError("Non-finite joint horizon calibration value")
        totals[:7] += torch.stack(values).double() * frames.shape[0]
        totals[7] += frames.shape[0]
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    if totals[7] <= 0:
        raise ValueError("Joint horizon calibration split produced zero samples")
    return tuple(float(value) for value in (totals[:7] / totals[7]).tolist())


def train_joint_horizon_wam(
    model: DroneWAM,
    config: ExperimentConfig,
    distributed: DistributedContext,
    output_path: str | Path,
) -> None:
    """Jointly optimize WAM accuracy and a deployable causal rollout horizon."""
    horizon = model.num_steps - model.observed_steps
    if isinstance(model.adaptive_planner, IterativeDepthPlanner):
        horizon_policy: torch.nn.Module = PlannerStateHorizonHead(
            hidden_dim=config.model.adaptive.planner_hidden_dim,
            horizon=horizon + 1,
            expansion=4,
            dropout=0.1,
            action_dim=config.model.predictor.action_dim,
            plan_horizon=horizon,
        )
    else:
        horizon_policy = GoalConditionedContinuationPolicy(
            embed_dim=config.model.encoder.embed_dim,
            hidden_dim=config.model.adaptive.policy_hidden_dim,
            horizon=horizon,
            action_dim=config.model.predictor.action_dim,
            kinematic_dim=7,
            token_count=config.model.adaptive.policy_token_count,
            output_dim=horizon,
        )
    _set_joint_horizon_trainability(model, horizon_policy)
    objective: torch.nn.Module = JointHorizonObjective(
        model, horizon_policy, config
    ).to(distributed.device)
    if distributed.world_size > 1:
        objective = DistributedDataParallel(
            objective,
            device_ids=[distributed.local_rank],
            broadcast_buffers=False,
        )
    core_objective = _unwrap_objective(objective)
    core = core_objective.world.model
    horizon_policy = core_objective.horizon_policy
    predictor_parameters = [
        parameter
        for parameter in core.predictor.parameters()
        if parameter.requires_grad
    ]
    planner_parameters = [
        parameter
        for parameter in core.adaptive_planner.parameters()
        if parameter.requires_grad
    ]
    policy_parameters = [
        parameter
        for parameter in horizon_policy.parameters()
        if parameter.requires_grad
    ]
    trainable_parameters = predictor_parameters + planner_parameters + policy_parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": predictor_parameters, "lr": config.optimization.learning_rate},
            {
                "params": planner_parameters,
                "lr": config.adaptive_training.planner_learning_rate,
            },
            {
                "params": policy_parameters,
                "lr": config.adaptive_training.policy_learning_rate,
            },
        ],
        weight_decay=config.optimization.weight_decay,
    )
    train_indices, calibration_indices = preference_split_indices(
        len(load_records(config.data.train_csv)),
        calibration_fraction=config.adaptive_training.calibration_fraction,
        seed=config.optimization.seed,
    )
    train_dataset = build_preference_dataset(
        config.data,
        config.model.encoder.resolution,
        config.adaptive_training.preference_windows_per_record,
        train_indices.tolist(),
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        drop_last=True,
    )
    calibration_dataset = build_preference_dataset(
        config.data,
        config.model.encoder.resolution,
        1,
        calibration_indices.tolist(),
    )
    local_calibration_indices = range(
        distributed.rank, len(calibration_dataset), distributed.world_size
    )
    calibration_loader = DataLoader(
        Subset(calibration_dataset, local_calibration_indices),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        drop_last=False,
    )
    initial = _evaluate_joint_horizon(
        objective, calibration_loader, config, distributed
    )
    if distributed.rank == 0:
        logger.info(
            "joint horizon split train_records=%d train_windows=%d calibration_records=%d "
            "initial_total=%.6f initial_latent=%.6f initial_trajectory=%.6f "
            "initial_horizon=%.6f initial_exact=%.4f initial_ate=%.6f initial_rpe=%.6f",
            train_indices.numel(),
            len(train_dataset),
            calibration_indices.numel(),
            *initial,
        )
    best_key: tuple[bool, float, float, float] | None = None
    for epoch in range(config.adaptive_training.joint_horizon_epochs):
        objective.train()
        _set_multilength_backbone_eval(core)
        _set_epoch(train_loader, epoch)
        totals = torch.zeros(8, dtype=torch.float64, device=distributed.device)
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            frames = batch["frames"].to(distributed.device, non_blocking=True)
            states = batch["states"].to(distributed.device, non_blocking=True)
            goal_frames = batch["goal_frame"].to(distributed.device, non_blocking=True)
            with _autocast(config, distributed.device):
                values = objective(frames, states, goal_frames)
            if not all(torch.isfinite(value) for value in values):
                raise FloatingPointError(
                    f"Non-finite joint horizon value at epoch={epoch}"
                )
            values[0].backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, config.optimization.grad_clip
            )
            optimizer.step()
            totals[:7] += (
                torch.stack([value.detach() for value in values]).double()
                * frames.shape[0]
            )
            totals[7] += frames.shape[0]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
        train_means = totals[:7] / totals[7]
        calibration = _evaluate_joint_horizon(
            objective, calibration_loader, config, distributed
        )
        cal_total, _, _, _, cal_exact, cal_ate, cal_rpe = calibration
        all_targets_met = (
            cal_exact > 0.85
            and cal_ate <= config.adaptive_training.ate_target
            and cal_rpe <= config.adaptive_training.rpe_target
        )
        key = (
            all_targets_met,
            -(
                cal_ate / config.adaptive_training.ate_target
                + cal_rpe / config.adaptive_training.rpe_target
            ),
            cal_exact,
            -cal_total,
        )
        if distributed.rank == 0:
            logger.info(
                "joint horizon epoch=%d train_total=%.6f train_latent=%.6f train_trajectory=%.6f "
                "train_horizon=%.6f train_exact=%.4f cal_total=%.6f cal_latent=%.6f "
                "cal_trajectory=%.6f cal_horizon=%.6f cal_exact=%.4f cal_ate=%.6f cal_rpe=%.6f",
                epoch + 1,
                *train_means[:5].tolist(),
                *calibration,
            )
            if best_key is None or key > best_key:
                best_key = key
                save_joint_horizon_checkpoint(
                    core, horizon_policy, output_path, epoch=epoch
                )


def train_dpo_policy(
    model: DroneWAM,
    artifact: Dict[str, Any],
    config: ExperimentConfig,
    device: torch.device,
    output_path: str | Path,
) -> None:
    if model.continuation_policy is None or model.continuation_reference is None:
        raise RuntimeError(
            "DPO training requires continuation policy and its fixed reference"
        )
    model.continuation_policy.to(device).train()
    model.continuation_policy.requires_grad_(True)
    reference = model.continuation_reference.to(device).eval().requires_grad_(False)
    adaptive_train_indices_all, calibration_indices_all = (
        preference_group_split_indices(
            artifact["record_indices"],
            calibration_fraction=config.adaptive_training.calibration_fraction,
            seed=config.optimization.seed,
        )
    )
    adaptive_train_indices = representative_group_indices(
        artifact["record_indices"],
        artifact["window_starts"],
        adaptive_train_indices_all,
    )
    calibration_indices = representative_group_indices(
        artifact["record_indices"],
        artifact["window_starts"],
        calibration_indices_all,
    )
    dataset = PreferencePairDataset(artifact, adaptive_train_indices)
    calibration_dataset = PreferencePairDataset(artifact, calibration_indices)
    calibration_depth_dataset = PreferenceDepthDataset(artifact, calibration_indices)
    loader = DataLoader(
        dataset,
        batch_size=max(config.data.batch_size * 32, 64),
        shuffle=True,
        num_workers=0,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=max(config.data.batch_size * 32, 64),
        shuffle=False,
        num_workers=0,
    )
    calibration_depth_loader = DataLoader(
        calibration_depth_dataset,
        batch_size=max(config.data.batch_size * 32, 64),
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.continuation_policy.parameters(),
        lr=config.adaptive_training.policy_learning_rate,
    )

    def batch_loss(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        context = batch["context_last"].to(device)
        context_tokens = batch["context_tokens"].to(device)
        observed_kinematics = batch["observed_kinematics"].to(device)

        def score_prefix(
            policy: ContinuationPolicy,
            prefix: str,
        ) -> torch.Tensor:
            return policy(
                context,
                context_tokens,
                batch[f"{prefix}_latent_history"].to(device),
                batch[f"{prefix}_spatial_history"].to(device),
                batch[f"{prefix}_plan_history"].to(device),
                observed_kinematics,
                batch[f"{prefix}_depth"].to(device),
            )

        chosen_policy_logits = score_prefix(
            model.continuation_policy,
            "chosen",
        )
        rejected_policy_logits = score_prefix(
            model.continuation_policy,
            "rejected",
        )
        with torch.no_grad():
            chosen_reference_logits = score_prefix(reference, "chosen")
            rejected_reference_logits = score_prefix(reference, "rejected")
        return dpo_pairwise_loss(
            chosen_policy_logits,
            rejected_policy_logits,
            chosen_reference_logits,
            rejected_reference_logits,
            config.adaptive_training.dpo_beta,
        )

    @torch.no_grad()
    def evaluate_calibration() -> tuple[float, float, float, float, float]:
        calibration_total = 0.0
        for batch in calibration_loader:
            loss = batch_loss(batch)
            calibration_total += float(loss.item()) * batch["context_last"].shape[0]
        probabilities = []
        for batch in calibration_depth_loader:
            context = batch["context_last"].to(device)
            context_tokens = batch["context_tokens"].to(device)
            latent_history = batch["latent_history"].to(device)
            spatial_history = batch["spatial_history"].to(device)
            plan_history = batch["plan_history"].to(device)
            observed_kinematics = batch["observed_kinematics"].to(device)
            depths = batch["depth"].to(device)
            logits = model.continuation_policy(
                context,
                context_tokens,
                latent_history,
                spatial_history,
                plan_history,
                observed_kinematics,
                depths,
            )
            reference_logits = reference(
                context,
                context_tokens,
                latent_history,
                spatial_history,
                plan_history,
                observed_kinematics,
                depths,
            )
            probabilities.append(
                dpo_reward_probability(
                    logits,
                    reference_logits,
                    config.adaptive_training.dpo_beta,
                ).cpu()
            )
        reward_probabilities = torch.cat(probabilities).reshape(
            calibration_indices.shape[0],
            -1,
        )
        selection = _reward_selection_metrics(
            reward_probabilities,
            artifact["normalized_costs"][calibration_indices],
            artifact["relative_ate"][calibration_indices],
            artifact["relative_rpe"][calibration_indices],
            artifact["best_depths"][calibration_indices],
        )
        return calibration_total / len(calibration_dataset), *selection

    model.continuation_policy.eval()
    (
        best_calibration_loss,
        best_selection_cost,
        selected_ate,
        selected_rpe,
        best_match,
    ) = evaluate_calibration()
    logger.info(
        "reward DPO initial calibration_loss=%.6f selection_cost=%.6f "
        "selected_ate=%.6f selected_rpe=%.6f best_match=%.6f",
        best_calibration_loss,
        best_selection_cost,
        selected_ate,
        selected_rpe,
        best_match,
    )
    save_adaptive_checkpoint(
        model,
        output_path,
        stage="dpo",
        epoch=-1,
        reference_policy=reference,
    )
    for epoch in range(config.adaptive_training.dpo_epochs):
        model.continuation_policy.train()
        train_total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite DPO loss at epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.continuation_policy.parameters(),
                config.optimization.grad_clip,
            )
            optimizer.step()
            train_total += float(loss.item()) * batch["context_last"].shape[0]

        model.continuation_policy.eval()
        train_loss = train_total / len(dataset)
        calibration_loss, selection_cost, selected_ate, selected_rpe, best_match = (
            evaluate_calibration()
        )
        logger.info(
            "reward DPO epoch=%d train_loss=%.6f calibration_loss=%.6f "
            "selection_cost=%.6f selected_ate=%.6f selected_rpe=%.6f best_match=%.6f",
            epoch + 1,
            train_loss,
            calibration_loss,
            selection_cost,
            selected_ate,
            selected_rpe,
            best_match,
        )
        if (selection_cost, calibration_loss) < (
            best_selection_cost,
            best_calibration_loss,
        ):
            best_selection_cost = selection_cost
            best_calibration_loss = calibration_loss
            save_adaptive_checkpoint(
                model,
                output_path,
                stage="dpo",
                epoch=epoch,
                reference_policy=reference,
            )


def train_joint(
    model: DroneWAM,
    artifact: Dict[str, Any],
    reference_policy: ContinuationPolicy,
    config: ExperimentConfig,
    distributed: DistributedContext,
    output_path: str | Path,
) -> None:
    _set_adaptive_trainability(model, "joint")
    adaptive_train_indices_all, calibration_indices_all = (
        preference_group_split_indices(
            artifact["record_indices"],
            calibration_fraction=config.adaptive_training.calibration_fraction,
            seed=config.optimization.seed,
        )
    )
    adaptive_train_indices = representative_group_indices(
        artifact["record_indices"],
        artifact["window_starts"],
        adaptive_train_indices_all,
    )
    calibration_record_indices = torch.unique(
        artifact["record_indices"][calibration_indices_all],
        sorted=True,
    )
    model = model.to(distributed.device)
    reference_policy = reference_policy.to(distributed.device)
    objective: torch.nn.Module = JointObjective(
        model,
        reference_policy,
        config,
    ).to(distributed.device)
    if distributed.world_size > 1:
        objective = DistributedDataParallel(
            objective,
            device_ids=[distributed.local_rank],
            broadcast_buffers=False,
        )
    core_objective = _unwrap_objective(objective)
    core = core_objective.model
    dataset = PreferenceWindowDataset(
        config.data,
        tuple(config.model.encoder.resolution),
        artifact,
        adaptive_train_indices,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        drop_last=True,
    )
    calibration_dataset = build_preference_dataset(
        config.data,
        config.model.encoder.resolution,
        1,
        calibration_record_indices.tolist(),
    )
    local_calibration_indices = range(
        distributed.rank, len(calibration_dataset), distributed.world_size
    )
    calibration_loader = DataLoader(
        Subset(calibration_dataset, local_calibration_indices),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        drop_last=False,
    )
    parameter_groups = [
        {
            "params": [
                parameter
                for parameter in core.adaptive_planner.parameters()
                if parameter.requires_grad
            ],
            "lr": config.adaptive_training.planner_learning_rate,
        },
        {
            "params": [
                parameter
                for parameter in core.continuation_policy.parameters()
                if parameter.requires_grad
            ],
            "lr": config.adaptive_training.policy_learning_rate,
        },
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=config.optimization.weight_decay
    )
    best_calibration_cost = _evaluate_joint_selection(
        objective,
        calibration_loader,
        config,
        distributed,
    )
    if distributed.rank == 0:
        logger.info(
            "adaptive joint initial calibration_selected_cost=%.6f train_windows=%d calibration_windows=%d",
            best_calibration_cost,
            len(dataset),
            len(calibration_dataset),
        )
        save_adaptive_checkpoint(
            core,
            output_path,
            stage="joint",
            epoch=-1,
            reference_policy=core_objective.reference_policy,
        )
    for epoch in range(config.adaptive_training.joint_epochs):
        objective.train()
        _set_frozen_wam_eval(core)
        core_objective.reference_policy.eval()
        sampler.set_epoch(epoch)
        totals = torch.zeros(3, device=distributed.device)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            frames = batch["frames"].to(distributed.device, non_blocking=True)
            states = batch["states"].to(distributed.device, non_blocking=True)
            goal_frames = batch["goal_frame"].to(distributed.device, non_blocking=True)
            with _autocast(config, distributed.device):
                total, task_cost, preference_loss = objective(
                    frames,
                    states,
                    goal_frames,
                )
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite joint loss at epoch={epoch}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in parameter_groups
                    for parameter in group["params"]
                ],
                config.optimization.grad_clip,
            )
            optimizer.step()
            totals += torch.stack(
                [total.detach(), task_cost.detach(), preference_loss.detach()]
            )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
            totals /= distributed.world_size
        totals /= len(loader)
        calibration_cost = _evaluate_joint_selection(
            objective,
            calibration_loader,
            config,
            distributed,
        )
        if distributed.rank == 0:
            logger.info(
                "adaptive joint epoch=%d total=%.6f selected_task=%.6f dpo=%.6f "
                "calibration_selected_cost=%.6f",
                epoch + 1,
                *totals.tolist(),
                calibration_cost,
            )
            if calibration_cost < best_calibration_cost:
                best_calibration_cost = calibration_cost
                save_adaptive_checkpoint(
                    core,
                    output_path,
                    stage="joint",
                    epoch=epoch,
                    reference_policy=core_objective.reference_policy,
                )
