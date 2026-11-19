"""Causal simulator-in-the-loop evaluation for observation-only policies."""

from __future__ import annotations

import importlib
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

from dronewam.data import ImageTransform
from dronewam.geometry import wrap_angle


@dataclass(frozen=True)
class CausalObservation:
    """One RGB/state sample returned by the live environment."""

    frame: np.ndarray
    state: np.ndarray


@dataclass(frozen=True)
class ClosedLoopCommand:
    """One ego-frame action and the sampling contract for its execution."""

    action: np.ndarray
    control_duration_s: float
    observation_count: int


@dataclass(frozen=True)
class EnvironmentStep:
    """Fresh causal observations produced after executing a command."""

    observations: Sequence[CausalObservation]
    terminated: bool
    truncated: bool = False
    success: bool | None = None
    collision: bool | None = None


@dataclass(frozen=True)
class PlanResult:
    """A complete action plan produced from the current causal history."""

    actions: np.ndarray
    latency_ms: float
    rollout_depth: int | None = None
    plan_depth: int | None = None


class ClosedLoopEnvironment(Protocol):
    """Backend boundary implemented by AirSim or another interactive simulator."""

    def reset(self, episode_id: str) -> Sequence[CausalObservation]: ...

    def step(self, command: ClosedLoopCommand) -> EnvironmentStep: ...

    def reference_states(self) -> np.ndarray:
        """Return post-termination reference states aligned to executed control steps."""

    def close(self) -> None: ...


class ClosedLoopPolicy(Protocol):
    def plan(self, history: Sequence[CausalObservation]) -> PlanResult: ...


@dataclass(frozen=True)
class ClosedLoopEpisodeMetrics:
    episode_id: str
    control_steps: int
    independent_forwards: int
    ate: float
    rpe: float
    relative_ate: float
    relative_rpe: float
    attitude_error: float
    trajectory_length: float
    mean_latency_ms: float
    mean_rollout_depth: float | None
    mean_plan_depth: float | None
    success: bool | None
    collision: bool | None
    terminated: bool
    truncated: bool


def _validate_observation(observation: CausalObservation) -> None:
    if not isinstance(observation.frame, np.ndarray):
        raise TypeError("Environment RGB frames must be numpy arrays")
    if observation.frame.ndim != 3 or observation.frame.shape[-1] != 3:
        raise ValueError(f"Environment RGB frame must be [H,W,3], got {observation.frame.shape}")
    if observation.frame.dtype != np.uint8:
        raise ValueError(f"Environment RGB frame must use uint8, got {observation.frame.dtype}")
    state = np.asarray(observation.state)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("Environment state must be a finite seven-vector")


def closed_loop_trajectory_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compute canonical translational metrics on aligned executed/reference states."""
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.ndim != 2 or actual.shape[1] < 6 or actual.shape != reference.shape:
        raise ValueError(
            f"Actual/reference trajectories must align as [T,>=6], got {actual.shape} and {reference.shape}"
        )
    if actual.shape[0] < 2 or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Closed-loop trajectories require at least two finite states")
    translation_error = np.linalg.norm(actual[:, :3] - reference[:, :3], axis=1)
    ate = float(np.sqrt(np.mean(np.square(translation_error))))
    actual_delta = np.diff(actual[:, :3], axis=0)
    reference_delta = np.diff(reference[:, :3], axis=0)
    rpe = float(np.sqrt(np.mean(np.sum(np.square(actual_delta - reference_delta), axis=1))))
    trajectory_length = float(np.linalg.norm(reference_delta, axis=1).sum())
    if trajectory_length <= 0.0:
        raise ValueError("Reference trajectory length must be positive")
    angle_delta = torch.from_numpy(actual[:, 3:6] - reference[:, 3:6])
    attitude_error = float(wrap_angle(angle_delta).abs().mean().item())
    return {
        "ate": ate,
        "rpe": rpe,
        "relative_ate": ate / trajectory_length,
        "relative_rpe": rpe / trajectory_length,
        "attitude_error": attitude_error,
        "trajectory_length": trajectory_length,
    }


class ModelClosedLoopPolicy:
    """Convert raw causal history into one independent DroneWAM forward pass."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        transform: ImageTransform,
        *,
        stop_thresholds: torch.Tensor | None = None,
        min_rollout_depth: int = 1,
        dpo_beta: float = 0.1,
    ) -> None:
        self.model = model
        self.device = device
        self.transform = transform
        self.stop_thresholds = stop_thresholds
        self.min_rollout_depth = int(min_rollout_depth)
        self.dpo_beta = float(dpo_beta)

    @torch.inference_mode()
    def plan(self, history: Sequence[CausalObservation]) -> PlanResult:
        expected = int(self.model.num_observed_frames)
        if len(history) != expected:
            raise ValueError(f"Policy requires exactly {expected} causal observations, got {len(history)}")
        for observation in history:
            _validate_observation(observation)
        frames = self.transform(np.stack([observation.frame for observation in history])).unsqueeze(0)
        states = torch.as_tensor(
            np.stack([observation.state for observation in history]),
            dtype=torch.float32,
        ).unsqueeze(0)
        frames = frames.to(self.device, non_blocking=True)
        states = states.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            if self.stop_thresholds is None:
                output = self.model.rollout(frames, states)
            else:
                output = self.model.adaptive_rollout(
                    frames,
                    states,
                    min_rollout_depth=self.min_rollout_depth,
                    stop_thresholds=self.stop_thresholds,
                    dpo_beta=self.dpo_beta,
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if output.predicted_actions is None or output.predicted_actions.shape[0] != 1:
            raise RuntimeError("Closed-loop policy requires one complete batch-one action plan")
        actions = output.predicted_actions[0].float().cpu().numpy()
        rollout_depth = None
        plan_depth = None
        if self.stop_thresholds is not None:
            rollout_depth = int(output.rollout_depths[0].item())
            plan_depth = int(output.plan_depths[0].item())
        return PlanResult(actions, elapsed_ms, rollout_depth, plan_depth)


class ClosedLoopEvaluator:
    """Execute receding-horizon control without reading future or reference data online."""

    def __init__(
        self,
        environment: ClosedLoopEnvironment,
        policy: ClosedLoopPolicy,
        *,
        observed_frames: int,
        frame_stride: int,
        fps: int,
        action_horizon: int,
        max_control_steps: int,
    ) -> None:
        if observed_frames < 1 or frame_stride < 1 or fps < 1 or action_horizon < 1 or max_control_steps < 1:
            raise ValueError("Closed-loop dimensions, fps and step limit must be positive")
        self.environment = environment
        self.policy = policy
        self.observed_frames = int(observed_frames)
        self.frame_stride = int(frame_stride)
        self.control_duration_s = self.frame_stride / float(fps)
        self.action_horizon = int(action_horizon)
        self.max_control_steps = int(max_control_steps)

    def run_episode(self, episode_id: str) -> ClosedLoopEpisodeMetrics:
        initial = list(self.environment.reset(episode_id))
        if len(initial) != self.observed_frames:
            raise ValueError(
                f"Environment reset must return exactly {self.observed_frames} observations, got {len(initial)}"
            )
        for observation in initial:
            _validate_observation(observation)
        history = deque(initial, maxlen=self.observed_frames)
        actual_states = [np.asarray(initial[-1].state, dtype=np.float64)]
        latencies: list[float] = []
        rollout_depths: list[int] = []
        plan_depths: list[int] = []
        final_step: EnvironmentStep | None = None
        for _ in range(self.max_control_steps):
            plan = self.policy.plan(tuple(history))
            actions = np.asarray(plan.actions)
            if actions.shape != (self.action_horizon, 6):
                raise ValueError(
                    f"Policy action plan must be [{self.action_horizon},6], got {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise ValueError("Policy action plan contains non-finite values")
            command = ClosedLoopCommand(
                action=actions[0].astype(np.float32, copy=True),
                control_duration_s=self.control_duration_s,
                observation_count=self.frame_stride,
            )
            final_step = self.environment.step(command)
            observations = list(final_step.observations)
            if len(observations) != self.frame_stride:
                raise ValueError(
                    f"Environment step must return exactly {self.frame_stride} observations, got {len(observations)}"
                )
            for observation in observations:
                _validate_observation(observation)
                history.append(observation)
            actual_states.append(np.asarray(observations[-1].state, dtype=np.float64))
            latencies.append(float(plan.latency_ms))
            if plan.rollout_depth is not None:
                rollout_depths.append(plan.rollout_depth)
            if plan.plan_depth is not None:
                plan_depths.append(plan.plan_depth)
            if final_step.terminated or final_step.truncated:
                break
        if final_step is None:
            raise RuntimeError("Closed-loop episode executed no control steps")
        if not final_step.terminated and not final_step.truncated:
            raise RuntimeError(
                "Environment did not terminate or truncate before max_control_steps; "
                "reference trajectory was not read"
            )
        reference = np.asarray(self.environment.reference_states(), dtype=np.float64)
        actual = np.stack(actual_states)
        metrics = closed_loop_trajectory_metrics(actual, reference)
        return ClosedLoopEpisodeMetrics(
            episode_id=str(episode_id),
            control_steps=len(latencies),
            independent_forwards=len(latencies),
            mean_latency_ms=float(np.mean(latencies)),
            mean_rollout_depth=float(np.mean(rollout_depths)) if rollout_depths else None,
            mean_plan_depth=float(np.mean(plan_depths)) if plan_depths else None,
            success=final_step.success,
            collision=final_step.collision,
            terminated=bool(final_step.terminated),
            truncated=bool(final_step.truncated),
            **metrics,
        )


def summarize_closed_loop(metrics: Sequence[ClosedLoopEpisodeMetrics]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("Cannot summarize zero closed-loop episodes")
    numeric = ("ate", "rpe", "relative_ate", "relative_rpe", "attitude_error", "trajectory_length")
    result: dict[str, Any] = {
        "protocol": "simulator_in_the_loop_receding_horizon_v1",
        "episodes": len(metrics),
        "control_steps": sum(item.control_steps for item in metrics),
        "independent_observation_to_action_forwards": sum(item.independent_forwards for item in metrics),
        "metrics": {name: float(np.mean([getattr(item, name) for item in metrics])) for name in numeric},
        "mean_latency_ms": float(
            np.average([item.mean_latency_ms for item in metrics], weights=[item.control_steps for item in metrics])
        ),
        "episode_results": [asdict(item) for item in metrics],
    }
    for field in ("mean_rollout_depth", "mean_plan_depth"):
        values = [getattr(item, field) for item in metrics if getattr(item, field) is not None]
        result[field] = float(np.mean(values)) if values else None
    for field in ("success", "collision"):
        values = [getattr(item, field) for item in metrics if getattr(item, field) is not None]
        result[f"{field}_rate"] = float(np.mean(values)) if values else None
    return result


def load_environment_factory(specification: str) -> Callable[[Mapping[str, Any]], ClosedLoopEnvironment]:
    """Load a backend factory from ``module:function`` without simulator coupling."""
    if specification.count(":") != 1:
        raise ValueError("Environment factory must use module:function syntax")
    module_name, function_name = specification.split(":")
    if not module_name or not function_name:
        raise ValueError("Environment factory must use module:function syntax")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"Environment factory {specification!r} is not callable")
    return factory
