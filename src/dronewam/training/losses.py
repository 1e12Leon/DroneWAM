"""Losses for latent world prediction and the initial action rollout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dronewam.config import LossConfig
from dronewam.geometry import integrate_ego_actions, to_observation_relative, wrap_angle
from dronewam.models.wam import WAMOutput


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    teacher_forced: torch.Tensor
    autoregressive: torch.Tensor
    action_regression: torch.Tensor
    rollout: torch.Tensor


def _discounted_latent_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tokens_per_step: int,
    discount: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Latent shapes differ: prediction={prediction.shape}, target={target.shape}")
    if prediction.shape[1] % tokens_per_step:
        raise ValueError("Latent token count must be divisible by tokens_per_step")
    num_steps = prediction.shape[1] // tokens_per_step
    losses = []
    weights = []
    for step in range(num_steps):
        start = step * tokens_per_step
        end = start + tokens_per_step
        losses.append(F.l1_loss(prediction[:, start:end].float(), target[:, start:end].float()))
        weights.append(discount**step)
    weight = prediction.new_tensor(weights, dtype=torch.float32)
    stacked = torch.stack(losses)
    return (stacked * weight).sum() / weight.sum()


def _weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dim_weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"Tensor shapes differ: prediction={prediction.shape}, target={target.shape}")
    weights = dim_weights / dim_weights.mean()
    loss = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none")
    return (loss * weights.view(*([1] * (loss.ndim - 1)), -1)).mean()


def _rollout_loss_from_actions(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    timeline_states: torch.Tensor,
    observed_steps: int,
    config: LossConfig,
    dim_weights: torch.Tensor,
) -> torch.Tensor:
    initial = timeline_states[:, observed_steps - 1]
    predicted = integrate_ego_actions(initial, predicted_actions)
    target = timeline_states[:, observed_steps - 1 :]
    if predicted.shape != target[..., :6].shape:
        raise ValueError(f"Trajectory shapes differ: prediction={predicted.shape}, target={target.shape}")
    predicted_relative = to_observation_relative(predicted, initial)
    target_relative = to_observation_relative(target, initial)
    predicted_delta = predicted_relative[:, 1:, :3] - predicted_relative[:, :-1, :3]
    target_delta = target_relative[:, 1:, :3] - target_relative[:, :-1, :3]
    trajectory_length = torch.linalg.vector_norm(target_delta, dim=-1).sum(dim=1)
    if torch.any(trajectory_length <= 0):
        indices = torch.nonzero(trajectory_length <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"GT trajectory length must be positive, bad sample indices={indices}")

    loss = predicted.new_tensor(0.0)
    if config.rollout_position_weight:
        loss = loss + config.rollout_position_weight * F.smooth_l1_loss(
            predicted_relative[:, 1:, :3],
            target_relative[:, 1:, :3],
        )
    if config.rollout_delta_weight:
        loss = loss + config.rollout_delta_weight * F.smooth_l1_loss(predicted_delta, target_delta)
    if config.rollout_action_delta_weight and predicted_actions.shape[1] > 1:
        loss = loss + config.rollout_action_delta_weight * _weighted_smooth_l1(
            predicted_actions[:, 1:] - predicted_actions[:, :-1],
            target_actions[:, 1:] - target_actions[:, :-1],
            dim_weights,
        )
    translation_error = torch.linalg.vector_norm(
        predicted_relative[..., :3] - target_relative[..., :3],
        dim=-1,
    )
    if config.rollout_relative_ate_weight:
        ate = torch.sqrt(translation_error.square().mean(dim=1).clamp_min(1.0e-12))
        loss = loss + config.rollout_relative_ate_weight * (ate / trajectory_length).mean()
    if config.rollout_relative_rpe_weight:
        rpe = torch.sqrt(
            torch.linalg.vector_norm(predicted_delta - target_delta, dim=-1).square().mean(dim=1).clamp_min(1.0e-12)
        )
        loss = loss + config.rollout_relative_rpe_weight * (rpe / trajectory_length).mean()
    if config.rollout_yaw_weight:
        yaw_error = wrap_angle(predicted_relative[:, 1:, 5] - target_relative[:, 1:, 5])
        loss = loss + config.rollout_yaw_weight * F.smooth_l1_loss(yaw_error, torch.zeros_like(yaw_error))
    return loss


def compute_action_plan_loss(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    timeline_states: torch.Tensor,
    observed_steps: int,
    config: LossConfig,
) -> torch.Tensor:
    """Return the established WAM action-regression plus rollout task loss."""
    if predicted_actions.shape != target_actions.shape:
        raise ValueError(f"Action shapes differ: prediction={predicted_actions.shape}, target={target_actions.shape}")
    dim_weights = predicted_actions.new_tensor(config.action_dim_weights, dtype=torch.float32)
    action_regression = _weighted_smooth_l1(
        predicted_actions,
        target_actions,
        dim_weights,
    )
    rollout = _rollout_loss_from_actions(
        predicted_actions,
        target_actions,
        timeline_states,
        observed_steps,
        config,
        dim_weights,
    )
    return config.action_weight * action_regression + rollout


def _rollout_loss(output: WAMOutput, config: LossConfig, dim_weights: torch.Tensor) -> torch.Tensor:
    if output.predicted_actions is None:
        return output.autoregressive_future.new_tensor(0.0)
    return _rollout_loss_from_actions(
        output.predicted_actions,
        output.target_actions,
        output.timeline_states,
        output.observed_steps,
        config,
        dim_weights,
    )


def compute_wam_loss(output: WAMOutput, config: LossConfig) -> LossBreakdown:
    if output.target_future is None or output.teacher_forced_future is None:
        raise ValueError("Training loss requires target and teacher-forced future tokens")
    teacher_forced = F.l1_loss(
        output.teacher_forced_future.float(),
        output.target_future.float(),
    )
    autoregressive = _discounted_latent_l1(
        output.autoregressive_future,
        output.target_future,
        output.tokens_per_step,
        config.ar_discount,
    )
    action_regression = output.autoregressive_future.new_tensor(0.0)
    dim_weights = output.autoregressive_future.new_tensor(config.action_dim_weights, dtype=torch.float32)
    if output.predicted_actions is not None:
        action_regression = _weighted_smooth_l1(
            output.predicted_actions,
            output.target_actions,
            dim_weights,
        )
    rollout = _rollout_loss(output, config, dim_weights)
    total = (
        config.teacher_forced_weight * teacher_forced
        + config.autoregressive_weight * autoregressive
        + config.action_weight * action_regression
        + rollout
    )
    return LossBreakdown(
        total=total,
        teacher_forced=teacher_forced,
        autoregressive=autoregressive,
        action_regression=action_regression,
        rollout=rollout,
    )
