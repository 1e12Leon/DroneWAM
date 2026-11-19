"""Offline metrics for the initial WAM action rollout."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dronewam.geometry import integrate_ego_actions, wrap_angle
from dronewam.models.wam import WAMOutput


@dataclass(frozen=True)
class BatchMetrics:
    count: int
    latent_l1: float
    action_rmse: float
    ate: float
    rpe: float
    relative_ate: float
    relative_rpe: float
    attitude_error: float


@dataclass(frozen=True)
class TrajectoryMetricTensors:
    """Per-sample differentiable trajectory metrics."""

    ate: torch.Tensor
    rpe: torch.Tensor
    relative_ate: torch.Tensor
    relative_rpe: torch.Tensor
    trajectory_length: torch.Tensor


def trajectory_metric_tensors(
    predicted_actions: torch.Tensor,
    timeline_states: torch.Tensor,
    observed_steps: int,
) -> TrajectoryMetricTensors:
    """Compute the canonical ATE/RPE metrics from a complete action plan."""
    if predicted_actions.ndim != 3 or predicted_actions.shape[-1] != 6:
        raise ValueError("predicted_actions must be [B,H,6]")
    initial = timeline_states[:, observed_steps - 1]
    predicted_trajectory = integrate_ego_actions(initial, predicted_actions)
    target_trajectory = timeline_states[:, observed_steps - 1 :]
    if predicted_trajectory.shape[:2] != target_trajectory.shape[:2]:
        raise ValueError(
            f"Trajectory shapes differ: prediction={predicted_trajectory.shape}, target={target_trajectory.shape}"
        )
    translation_error = torch.linalg.vector_norm(
        predicted_trajectory[..., :3] - target_trajectory[..., :3],
        dim=-1,
    )
    ate = torch.sqrt(translation_error.square().mean(dim=1).clamp_min(1.0e-12))
    predicted_delta = predicted_trajectory[:, 1:, :3] - predicted_trajectory[:, :-1, :3]
    target_delta = target_trajectory[:, 1:, :3] - target_trajectory[:, :-1, :3]
    rpe = torch.sqrt(
        torch.linalg.vector_norm(predicted_delta - target_delta, dim=-1).square().mean(dim=1).clamp_min(1.0e-12)
    )
    trajectory_length = torch.linalg.vector_norm(target_delta, dim=-1).sum(dim=1)
    if torch.any(trajectory_length <= 0):
        indices = torch.nonzero(trajectory_length <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"GT trajectory length must be positive, bad sample indices={indices}")
    return TrajectoryMetricTensors(
        ate=ate,
        rpe=rpe,
        relative_ate=ate / trajectory_length,
        relative_rpe=rpe / trajectory_length,
        trajectory_length=trajectory_length,
    )


def normalized_trajectory_cost(
    predicted_actions: torch.Tensor,
    timeline_states: torch.Tensor,
    observed_steps: int,
    *,
    ate_target: float,
    rpe_target: float,
    power: int = 1,
) -> torch.Tensor:
    """Single task objective aligned exactly with the two acceptance metrics."""
    if ate_target <= 0.0 or rpe_target <= 0.0:
        raise ValueError("Metric targets must be positive")
    if power not in (1, 2):
        raise ValueError("power must be 1 or 2")
    metrics = trajectory_metric_tensors(predicted_actions, timeline_states, observed_steps)
    normalized_ate = metrics.relative_ate / ate_target
    normalized_rpe = metrics.relative_rpe / rpe_target
    return normalized_ate.pow(power) + normalized_rpe.pow(power)


@torch.no_grad()
def compute_batch_metrics(output: WAMOutput) -> BatchMetrics:
    if output.target_future is None:
        raise ValueError("Offline metrics require target future tokens")
    if output.predicted_actions is None:
        raise ValueError("Offline action metrics require an action head")
    trajectory_metrics = trajectory_metric_tensors(
        output.predicted_actions,
        output.timeline_states,
        output.observed_steps,
    )
    initial = output.timeline_states[:, output.observed_steps - 1]
    predicted_trajectory = integrate_ego_actions(initial, output.predicted_actions)
    target_trajectory = output.timeline_states[:, output.observed_steps - 1 :]
    attitude = wrap_angle(predicted_trajectory[..., 3:6] - target_trajectory[..., 3:6]).abs().mean()
    return BatchMetrics(
        count=int(predicted_trajectory.shape[0]),
        latent_l1=float(
            torch.nn.functional.l1_loss(output.autoregressive_future.float(), output.target_future.float()).item()
        ),
        action_rmse=float(
            torch.sqrt((output.predicted_actions.float() - output.target_actions.float()).square().mean()).item()
        ),
        ate=float(trajectory_metrics.ate.mean().item()),
        rpe=float(trajectory_metrics.rpe.mean().item()),
        relative_ate=float(trajectory_metrics.relative_ate.mean().item()),
        relative_rpe=float(trajectory_metrics.relative_rpe.mean().item()),
        attitude_error=float(attitude.item()),
    )
