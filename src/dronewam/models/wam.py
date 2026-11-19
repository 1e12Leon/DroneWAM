"""The clean observation-to-world-to-action model."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from dronewam.backbones.ac_predictor import VisionTransformerPredictorAC
from dronewam.config import ExperimentConfig, TokenAEConfig
from dronewam.geometry import build_timeline
from dronewam.models.action_head import MeanPoolActionHead
from dronewam.models.adaptive import (
    ContinuationPolicy,
    IterativeDepthPlanner,
    PlannerStateStopHead,
    PlannerStateValueHead,
    VisualPlannerValueHead,
    VariableDepthPlanner,
    dpo_reward_probability,
    pool_policy_tokens,
)
from dronewam.models.encoder import build_encoder
from dronewam.models.token_ae import TokenAE


@dataclass(frozen=True)
class WAMOutput:
    """Model outputs aligned to the tubelet-level future timeline."""

    teacher_forced_future: Optional[torch.Tensor]
    autoregressive_future: torch.Tensor
    target_future: Optional[torch.Tensor]
    predicted_actions: Optional[torch.Tensor]
    target_actions: torch.Tensor
    timeline_states: torch.Tensor
    observed_steps: int
    tokens_per_step: int


@dataclass(frozen=True)
class RolloutOutput:
    """Deployable observation-only rollout."""

    future_latents: torch.Tensor
    predicted_actions: Optional[torch.Tensor]
    observed_steps: int
    tokens_per_step: int


@dataclass(frozen=True)
class AdaptiveRolloutOutput:
    """Variable-depth rollout followed by full-horizon planning."""

    future_latents: torch.Tensor
    predicted_actions: torch.Tensor
    rollout_depths: torch.Tensor
    plan_depths: torch.Tensor
    reward_probabilities: torch.Tensor
    observed_steps: int
    tokens_per_step: int


@dataclass(frozen=True)
class HorizonRolloutOutput:
    """Goal-conditioned rollout selected by the recursive Planner state."""

    future_latents: torch.Tensor
    predicted_actions: torch.Tensor
    rollout_depths: torch.Tensor
    stop_logits: torch.Tensor
    observed_steps: int
    tokens_per_step: int


@dataclass(frozen=True)
class ValueGateRolloutOutput:
    """Goal-conditioned rollout selected by causal reachable-depth values."""

    future_latents: torch.Tensor
    predicted_actions: torch.Tensor
    rollout_depths: torch.Tensor
    gate_values: torch.Tensor
    observed_steps: int
    tokens_per_step: int


@dataclass(frozen=True)
class WorldRolloutOutput:
    """Observation-only latent rollout before action planning."""

    observed_context: torch.Tensor
    future_latents: torch.Tensor
    observed_kinematics: torch.Tensor
    observed_steps: int
    tokens_per_step: int


def _torch_load(path: str | Path) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        kwargs["weights_only"] = True
    checkpoint = torch.load(path, **kwargs)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} must contain a dictionary")
    return checkpoint


def _load_token_ae(
    config: TokenAEConfig,
    embed_dim: int,
    raw_tokens_per_step: int,
    encoder_family: str,
    encoder_model_name: str,
) -> tuple[Optional[TokenAE], bool, Optional[tuple[int, int]]]:
    if not config.enabled:
        return None, False, None
    checkpoint = _torch_load(config.checkpoint)
    embedded = checkpoint.get("config", {})
    source_family = embedded.get("source_encoder_family")
    source_model = embedded.get("source_encoder_model")
    if encoder_family == "dinov2" and source_family is None:
        raise ValueError("DINOv2 requires a backbone-specific TokenAE checkpoint with source metadata")
    if source_family is not None and source_family != encoder_family:
        raise ValueError(
            f"TokenAE source_encoder_family={source_family!r} does not match Encoder family={encoder_family!r}"
        )
    if source_model is not None and source_model != encoder_model_name:
        raise ValueError(
            f"TokenAE source_encoder_model={source_model!r} does not match Encoder model={encoder_model_name!r}"
        )
    token_embed_dim = int(embedded.get("embed_dim", embed_dim))
    input_grid_size = tuple(embedded.get("input_grid_size", config.input_grid_size))
    latent_grid_size = tuple(embedded.get("latent_grid_size", config.latent_grid_size))
    num_latent_tokens = int(embedded.get("num_latent_tokens", config.num_latent_tokens))
    if token_embed_dim != embed_dim:
        raise ValueError(f"TokenAE embed_dim={token_embed_dim} does not match Encoder embed_dim={embed_dim}")
    if input_grid_size[0] * input_grid_size[1] != raw_tokens_per_step:
        raise ValueError(
            f"TokenAE input_grid_size={input_grid_size} does not match "
            f"Encoder tokens_per_step={raw_tokens_per_step}"
        )
    if latent_grid_size[0] * latent_grid_size[1] != num_latent_tokens:
        raise ValueError(
            f"TokenAE latent_grid_size={latent_grid_size} does not match " f"num_latent_tokens={num_latent_tokens}"
        )
    token_ae = TokenAE(
        embed_dim=token_embed_dim,
        tokens_per_frame=int(embedded.get("tokens_per_frame", raw_tokens_per_step)),
        num_latent_tokens=num_latent_tokens,
        num_heads=int(embedded.get("num_heads", config.num_heads)),
        encoder_depth=int(embedded.get("encoder_depth", config.encoder_depth)),
        decoder_depth=int(embedded.get("decoder_depth", config.decoder_depth)),
        mlp_ratio=float(embedded.get("mlp_ratio", config.mlp_ratio)),
        dropout=float(embedded.get("dropout", 0.0)),
        encoder_mode=str(embedded.get("encoder_mode", "parallel")),
        loss_type=str(embedded.get("loss_type", "smooth_l1")),
        cos_loss_weight=float(embedded.get("cos_loss_weight", 0.25)),
        latent_reg_weight=float(embedded.get("latent_reg_weight", 0.0)),
        pos_embed_type=str(embedded.get("pos_embed_type", "learnable")),
        input_grid_size=input_grid_size,
        latent_grid_size=latent_grid_size,
        temporal_depth=int(embedded.get("temporal_depth", config.temporal_depth)),
        temporal_num_heads=int(embedded.get("temporal_num_heads", config.temporal_num_heads)),
        temporal_mlp_ratio=float(embedded.get("temporal_mlp_ratio", config.mlp_ratio)),
        temporal_causal=bool(embedded.get("temporal_causal", config.temporal_causal)),
        temporal_mode=str(embedded.get("temporal_mode", config.temporal_mode)),
        temporal_pos_embed_type=str(embedded.get("temporal_pos_embed_type", "sincos")),
        temporal_loss_weight=float(embedded.get("temporal_loss_weight", 0.1)),
    )
    state = checkpoint["token_ae"] if "token_ae" in checkpoint else checkpoint
    token_ae.load_state_dict(state, strict=True)
    token_ae.eval()
    token_ae.requires_grad_(False)
    return token_ae, bool(embedded.get("normalize_reps", False)), latent_grid_size


class DroneWAM(torch.nn.Module):
    """Predict future visual latents and decode an initial 6-DoF action rollout."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        target_encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        action_head: Optional[torch.nn.Module],
        token_ae: Optional[torch.nn.Module],
        adaptive_planner: Optional[VariableDepthPlanner | IterativeDepthPlanner] = None,
        continuation_policy: Optional[ContinuationPolicy] = None,
        continuation_reference: Optional[ContinuationPolicy] = None,
        *,
        num_raw_frames: int,
        num_observed_frames: int,
        frame_stride: int,
        tokens_per_step: int,
        normalize_representations: bool = False,
    ) -> None:
        super().__init__()
        if num_raw_frames % frame_stride or num_observed_frames % frame_stride:
            raise ValueError("Raw and observed frame counts must be divisible by frame_stride")
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.action_head = action_head
        self.token_ae = token_ae
        self.adaptive_planner = adaptive_planner
        self.continuation_policy = continuation_policy
        self.continuation_reference = continuation_reference
        self.num_raw_frames = int(num_raw_frames)
        self.num_observed_frames = int(num_observed_frames)
        self.frame_stride = int(frame_stride)
        self.num_steps = self.num_raw_frames // self.frame_stride
        self.observed_steps = self.num_observed_frames // self.frame_stride
        self.tokens_per_step = int(tokens_per_step)
        self.normalize_representations = bool(normalize_representations)

    def _encode(self, encoder: torch.nn.Module, frames: torch.Tensor, raw_frames: int) -> torch.Tensor:
        tokens = encoder(frames, num_observed_frames=raw_frames)
        num_steps = raw_frames // self.frame_stride
        if self.token_ae is not None:
            token_ae_dtype = next(self.token_ae.parameters()).dtype
            tokens = self.token_ae.encode(tokens.to(dtype=token_ae_dtype), num_frames=num_steps)
        if self.normalize_representations:
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))
        return tokens

    def _masked_aux(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if actions.shape[1] != states.shape[1]:
            raise ValueError(f"Predictor actions/states must align, got {actions.shape[1]} and {states.shape[1]}")
        action_values = torch.zeros_like(actions)
        state_values = torch.zeros_like(states)
        known_actions = min(self.observed_steps - 1, actions.shape[1])
        known_states = min(self.observed_steps, states.shape[1])
        action_values[:, :known_actions] = actions[:, :known_actions]
        state_values[:, :known_states] = states[:, :known_states]
        action_mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        state_mask = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
        action_mask[:, :known_actions] = False
        state_mask[:, :known_states] = False
        return action_values, state_values, action_mask, state_mask

    def _predict(self, tokens: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        action_values, state_values, action_mask, state_mask = self._masked_aux(actions, states)
        predicted = self.predictor(
            tokens,
            action_values,
            state_values,
            action_mask=action_mask,
            state_mask=state_mask,
        )
        if self.normalize_representations:
            predicted = F.layer_norm(predicted, (predicted.shape[-1],))
        return predicted

    def _autoregressive_rollout(
        self,
        full_context: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        max_future_steps: Optional[int] = None,
    ) -> torch.Tensor:
        rolled = full_context[:, : self.observed_steps * self.tokens_per_step]
        future = []
        future_steps = self.num_steps - self.observed_steps
        requested_steps = future_steps if max_future_steps is None else int(max_future_steps)
        if not 0 <= requested_steps <= future_steps:
            raise ValueError(f"max_future_steps must be in [0, {future_steps}], got {requested_steps}")
        if requested_steps == 0:
            return rolled[:, :0]
        for step in range(self.observed_steps, self.observed_steps + requested_steps):
            prefix_actions = actions if step == self.num_steps - 1 else actions[:, :step]
            prefix_states = states[:, :step]
            next_tokens = self._predict(rolled, prefix_actions, prefix_states)[:, -self.tokens_per_step :]
            future.append(next_tokens)
            rolled = torch.cat([rolled, next_tokens], dim=1)
        return torch.cat(future, dim=1)

    def _tokens_by_step(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] % self.tokens_per_step:
            raise ValueError("tokens must be [B,T*P,D] aligned to tokens_per_step")
        return tokens.reshape(
            tokens.shape[0],
            tokens.shape[1] // self.tokens_per_step,
            self.tokens_per_step,
            tokens.shape[-1],
        )

    def plan_actions(
        self,
        observed_context: torch.Tensor,
        future_latents: torch.Tensor,
        observed_kinematics: torch.Tensor,
        rollout_depths: torch.Tensor,
        *,
        goal_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Plan the complete horizon while hiding latents beyond each depth."""
        if self.action_head is None:
            raise RuntimeError("Action planning requires an action head")
        context_btpd = self._tokens_by_step(observed_context)
        future_btpd = self._tokens_by_step(future_latents)
        horizon = self.num_steps - self.observed_steps
        if self.adaptive_planner is None:
            if future_btpd.shape[1] != horizon or torch.any(rollout_depths != horizon):
                raise RuntimeError("Variable-depth planning requires model.adaptive.enabled=true")
            return self.action_head(future_btpd)
        return self.adaptive_planner(
            context_btpd,
            future_btpd,
            observed_kinematics,
            rollout_depths,
            self.action_head,
            goal_tokens,
        )

    def _planner_kinematics(
        self,
        timeline_states: torch.Tensor,
        timeline_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return deployable ego motion: last 6-DoF action plus current speed."""
        if self.observed_steps < 2:
            raise ValueError("Planner kinematics require at least two observed timeline steps")
        last_action = timeline_actions[:, self.observed_steps - 2]
        current_speed = timeline_states[:, self.observed_steps - 1, 6:7]
        return torch.cat([last_action, current_speed], dim=-1)

    def _prepare_observed_rollout(
        self,
        observed_frames: torch.Tensor,
        observed_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observed_frames.ndim != 5 or observed_frames.shape[2] != self.num_observed_frames:
            raise ValueError(
                f"observed_frames must be [B,C,{self.num_observed_frames},H,W], " f"got {tuple(observed_frames.shape)}"
            )
        if (
            observed_states.ndim != 3
            or observed_states.shape[1] != self.num_observed_frames
            or observed_states.shape[-1] < 7
        ):
            raise ValueError(
                f"observed_states must be [B,{self.num_observed_frames},>=7], " f"got {tuple(observed_states.shape)}"
            )
        context = self._encode(self.encoder, observed_frames, self.num_observed_frames)
        observed_timeline_states, observed_timeline_actions = build_timeline(
            observed_states,
            self.frame_stride,
        )
        timeline_states = observed_states.new_zeros(
            observed_states.shape[0],
            self.num_steps,
            observed_states.shape[-1],
        )
        timeline_actions = observed_states.new_zeros(
            observed_states.shape[0],
            self.num_steps - 1,
            6,
        )
        timeline_states[:, : self.observed_steps] = observed_timeline_states
        timeline_actions[:, : self.observed_steps - 1] = observed_timeline_actions
        observed_kinematics = self._planner_kinematics(timeline_states, timeline_actions)
        return context, timeline_actions, timeline_states, observed_kinematics

    def rollout_world(
        self,
        observed_frames: torch.Tensor,
        observed_states: torch.Tensor,
        *,
        max_future_steps: Optional[int] = None,
    ) -> WorldRolloutOutput:
        """Roll out predicted world latents from observation-only inputs."""
        context, timeline_actions, timeline_states, observed_kinematics = self._prepare_observed_rollout(
            observed_frames,
            observed_states,
        )
        future = self._autoregressive_rollout(
            context,
            timeline_actions,
            timeline_states,
            max_future_steps=max_future_steps,
        )
        return WorldRolloutOutput(
            observed_context=context,
            future_latents=future,
            observed_kinematics=observed_kinematics,
            observed_steps=self.observed_steps,
            tokens_per_step=self.tokens_per_step,
        )

    def encode_goal(self, goal_frame: torch.Tensor) -> torch.Tensor:
        """Encode one deployable goal image as a single world-token step."""
        if goal_frame.ndim != 4:
            raise ValueError(f"goal_frame must be [B,C,H,W], got {tuple(goal_frame.shape)}")
        repeated = goal_frame.unsqueeze(2).expand(-1, -1, self.frame_stride, -1, -1).contiguous()
        tokens = self._encode(self.encoder, repeated, self.frame_stride)
        if tokens.shape[1] != self.tokens_per_step:
            raise RuntimeError("Goal encoder must produce exactly one token step")
        return tokens

    def forward(
        self,
        frames: torch.Tensor,
        raw_states: torch.Tensor,
        *,
        compute_targets: bool,
    ) -> WAMOutput:
        if frames.ndim != 5 or frames.shape[2] != self.num_raw_frames:
            raise ValueError(f"frames must be [B,C,{self.num_raw_frames},H,W], got {tuple(frames.shape)}")
        if raw_states.ndim != 3 or raw_states.shape[1] != self.num_raw_frames or raw_states.shape[-1] < 7:
            raise ValueError(f"raw_states must be [B,{self.num_raw_frames},>=7], got {tuple(raw_states.shape)}")
        timeline_states, timeline_actions = build_timeline(raw_states, self.frame_stride)

        context = self._encode(self.encoder, frames, self.num_raw_frames)
        if context.shape[1] != self.num_steps * self.tokens_per_step:
            raise ValueError("Encoder timeline does not match configured WAM timeline")

        teacher_forced = self._predict(
            context[:, : -self.tokens_per_step],
            timeline_actions,
            timeline_states[:, :-1],
        )
        teacher_forced_future = teacher_forced[:, (self.observed_steps - 1) * self.tokens_per_step :]
        autoregressive_future = self._autoregressive_rollout(
            context,
            timeline_actions,
            timeline_states,
        )

        target_future = None
        if compute_targets:
            with torch.no_grad():
                target = self._encode(self.target_encoder, frames, self.num_raw_frames)
            target_future = target[:, self.observed_steps * self.tokens_per_step :]

        future_steps = self.num_steps - self.observed_steps
        predicted_actions = None
        if self.action_head is not None:
            full_depths = torch.full(
                (frames.shape[0],),
                future_steps,
                dtype=torch.long,
                device=frames.device,
            )
            predicted_actions = self.plan_actions(
                context[:, : self.observed_steps * self.tokens_per_step],
                autoregressive_future,
                self._planner_kinematics(timeline_states, timeline_actions),
                full_depths,
            )
        target_actions = timeline_actions[:, self.observed_steps - 1 :]
        return WAMOutput(
            teacher_forced_future=teacher_forced_future,
            autoregressive_future=autoregressive_future,
            target_future=target_future,
            predicted_actions=predicted_actions,
            target_actions=target_actions,
            timeline_states=timeline_states,
            observed_steps=self.observed_steps,
            tokens_per_step=self.tokens_per_step,
        )

    def rollout(self, observed_frames: torch.Tensor, observed_states: torch.Tensor) -> RolloutOutput:
        """Predict the full future using only the observed prefix and masked metadata."""
        world = self.rollout_world(observed_frames, observed_states)
        context = world.observed_context
        future = world.future_latents
        future_steps = self.num_steps - self.observed_steps
        actions = None
        if self.action_head is not None:
            full_depths = torch.full(
                (future.shape[0],),
                future_steps,
                dtype=torch.long,
                device=future.device,
            )
            actions = self.plan_actions(context, future, world.observed_kinematics, full_depths)
        return RolloutOutput(
            future_latents=future,
            predicted_actions=actions,
            observed_steps=self.observed_steps,
            tokens_per_step=self.tokens_per_step,
        )

    def adaptive_rollout(
        self,
        observed_frames: torch.Tensor,
        observed_states: torch.Tensor,
        *,
        min_rollout_depth: int,
        stop_thresholds: torch.Tensor,
        dpo_beta: float,
    ) -> AdaptiveRolloutOutput:
        """Stop AR world rollout when the current plan reward reaches its threshold."""
        if self.adaptive_planner is None or self.continuation_policy is None or self.continuation_reference is None:
            raise RuntimeError("adaptive_rollout requires adaptive planner, policy and DPO reference")
        horizon = self.num_steps - self.observed_steps
        if not 1 <= min_rollout_depth <= horizon:
            raise ValueError(f"min_rollout_depth must be in [1, {horizon}]")
        if stop_thresholds.shape != (horizon - 1,):
            raise ValueError(f"stop_thresholds must contain {horizon - 1} values")
        if torch.any(stop_thresholds <= 0.0) or torch.any(stop_thresholds >= 1.0):
            raise ValueError("Every stop threshold must be in (0, 1)")

        context, timeline_actions, timeline_states, observed_kinematics = self._prepare_observed_rollout(
            observed_frames,
            observed_states,
        )

        batch_size = observed_frames.shape[0]
        rolled = context
        future_steps = []
        probabilities = []
        plan_history = []
        rollout_depths = torch.full(
            (batch_size,),
            horizon,
            dtype=torch.long,
            device=context.device,
        )
        plan_depths = torch.ones(batch_size, dtype=torch.long, device=context.device)
        best_rewards = torch.full(
            (batch_size,),
            -torch.inf,
            dtype=torch.float32,
            device=context.device,
        )
        active = torch.ones(batch_size, dtype=torch.bool, device=context.device)
        context_by_step = self._tokens_by_step(context)
        context_last_tokens = context_by_step[:, -1]
        context_last = context_last_tokens.mean(dim=1)
        context_tokens = pool_policy_tokens(
            context_last_tokens,
            self.continuation_policy.token_count,
        )
        for depth in range(1, horizon + 1):
            step = self.observed_steps + depth - 1
            prefix_actions = timeline_actions if step == self.num_steps - 1 else timeline_actions[:, :step]
            prefix_states = timeline_states[:, :step]
            next_tokens = self._predict(rolled, prefix_actions, prefix_states)[:, -self.tokens_per_step :]
            future_steps.append(next_tokens)
            rolled = torch.cat([rolled, next_tokens], dim=1)
            future = torch.cat(future_steps, dim=1)
            future_by_step = self._tokens_by_step(future)
            future_pooled = future_by_step.mean(dim=2)
            spatial_history = pool_policy_tokens(
                future_by_step,
                self.continuation_policy.token_count,
            )
            current_depths = torch.full((batch_size,), depth, dtype=torch.long, device=context.device)
            current_plan = self.plan_actions(
                context,
                future,
                observed_kinematics,
                current_depths,
            )
            plan_history.append(current_plan)
            plans = torch.stack(plan_history, dim=1)
            logits = self.continuation_policy(
                context_last,
                context_tokens,
                future_pooled,
                spatial_history,
                plans,
                observed_kinematics,
                current_depths,
            )
            reference_logits = self.continuation_reference(
                context_last,
                context_tokens,
                future_pooled,
                spatial_history,
                plans,
                observed_kinematics,
                current_depths,
            )
            reward_probability = dpo_reward_probability(
                logits,
                reference_logits,
                dpo_beta,
            )
            probabilities.append(torch.where(active, reward_probability, torch.nan))
            improved = active & (reward_probability > best_rewards)
            best_rewards = torch.where(improved, reward_probability, best_rewards)
            plan_depths[improved] = depth
            if depth == horizon:
                break
            if depth >= min_rollout_depth:
                stop_now = active & (reward_probability >= stop_thresholds[depth - 1])
                rollout_depths[stop_now] = depth
                active = active & ~stop_now
            if not torch.any(active):
                break

        future = torch.cat(future_steps, dim=1)
        actions = self.plan_actions(
            context,
            future,
            observed_kinematics,
            plan_depths,
        )
        return AdaptiveRolloutOutput(
            future_latents=self._tokens_by_step(future),
            predicted_actions=actions,
            rollout_depths=rollout_depths,
            plan_depths=plan_depths,
            reward_probabilities=torch.stack(probabilities, dim=1),
            observed_steps=self.observed_steps,
            tokens_per_step=self.tokens_per_step,
        )

    def horizon_rollout(
        self,
        observed_frames: torch.Tensor,
        observed_states: torch.Tensor,
        goal_frame: torch.Tensor,
        stop_head: PlannerStateStopHead,
        stop_thresholds: torch.Tensor,
    ) -> HorizonRolloutOutput:
        """Roll out until the causal Planner-state head selects the current depth."""
        if not isinstance(self.adaptive_planner, IterativeDepthPlanner) or self.action_head is None:
            raise RuntimeError("horizon_rollout requires IterativeDepthPlanner and Action Head")
        horizon = self.num_steps - self.observed_steps
        if stop_head.horizon != horizon or stop_head.hidden_dim != self.adaptive_planner.hidden_dim:
            raise ValueError("Stop Head dimensions do not match the recursive Planner")
        thresholds = torch.as_tensor(stop_thresholds, dtype=torch.float32, device=observed_frames.device)
        if thresholds.shape != (stop_head.candidate_count - 1,):
            raise ValueError(f"stop_thresholds must have shape [{stop_head.candidate_count - 1}]")
        if torch.any(thresholds < 0.0) or torch.any(thresholds > 1.0):
            raise ValueError("stop_thresholds must be in [0,1]")
        context, timeline_actions, timeline_states, observed_kinematics = self._prepare_observed_rollout(
            observed_frames,
            observed_states,
        )
        goal_tokens = self.encode_goal(goal_frame)
        batch_size = observed_frames.shape[0]
        rolled = context
        future_steps = []
        selected_actions = None
        rollout_depths = torch.full(
            (batch_size,),
            horizon,
            dtype=torch.long,
            device=context.device,
        )
        all_logits = context.new_full((batch_size, stop_head.candidate_count, 2), torch.nan)
        active = torch.ones(batch_size, dtype=torch.bool, device=context.device)
        previous_planner_state = None
        previous_plan = None
        context_steps = self._tokens_by_step(context)
        for depth in range(stop_head.min_rollout_depth, horizon + 1):
            if depth > 0:
                step = self.observed_steps + depth - 1
                prefix_actions = timeline_actions if step == self.num_steps - 1 else timeline_actions[:, :step]
                prefix_states = timeline_states[:, :step]
                next_tokens = self._predict(rolled, prefix_actions, prefix_states)[:, -self.tokens_per_step :]
                future_steps.append(next_tokens)
                rolled = torch.cat([rolled, next_tokens], dim=1)
            future = context[:, :0] if not future_steps else torch.cat(future_steps, dim=1)
            plans, planner_states = self.adaptive_planner.rollout_all_depths(
                context_steps,
                self._tokens_by_step(future),
                observed_kinematics,
                self.action_head,
                goal_tokens,
                include_zero_depth=stop_head.min_rollout_depth == 0,
            )
            current_plan = plans[:, -1]
            if selected_actions is None:
                selected_actions = torch.zeros_like(current_plan)
            current_planner_state = planner_states[:, -1]
            logits = stop_head.forward_step(
                current_planner_state,
                current_plan,
                depth,
                previous_planner_state,
                previous_plan,
            )
            all_logits[:, depth - stop_head.min_rollout_depth] = logits
            stop_probability = torch.softmax(logits.float(), dim=1)[:, stop_head.STOP]
            should_stop = depth == horizon
            if depth < horizon:
                should_stop = stop_probability >= thresholds[depth - stop_head.min_rollout_depth]
            stop = active & should_stop
            selected_actions[stop] = current_plan[stop]
            rollout_depths[stop] = depth
            active = active & ~stop
            previous_planner_state = current_planner_state
            previous_plan = current_plan
            if not torch.any(active):
                break
        if torch.any(active):
            raise RuntimeError("Horizon rollout did not terminate every sample")
        if selected_actions is None:
            raise RuntimeError("Horizon rollout produced no action plan")
        future = context[:, :0] if not future_steps else torch.cat(future_steps, dim=1)
        return HorizonRolloutOutput(
            future_latents=self._tokens_by_step(future),
            predicted_actions=selected_actions,
            rollout_depths=rollout_depths,
            stop_logits=all_logits,
            observed_steps=self.observed_steps,
            tokens_per_step=self.tokens_per_step,
        )

    def value_gate_rollout(
        self,
        observed_frames: torch.Tensor,
        observed_states: torch.Tensor,
        goal_frame: torch.Tensor,
        value_head: PlannerStateValueHead,
        candidate_bias: torch.Tensor,
    ) -> ValueGateRolloutOutput:
        """Roll out until the current depth has minimum predicted reachable value."""
        if not isinstance(self.adaptive_planner, IterativeDepthPlanner) or self.action_head is None:
            raise RuntimeError("value_gate_rollout requires IterativeDepthPlanner and Action Head")
        horizon = self.num_steps - self.observed_steps
        if value_head.horizon != horizon or value_head.hidden_dim != self.adaptive_planner.hidden_dim:
            raise ValueError("Value Gate dimensions do not match the recursive Planner")
        bias = torch.as_tensor(candidate_bias, dtype=torch.float32, device=observed_frames.device)
        if bias.shape != (value_head.candidate_count,) or not torch.isfinite(bias).all():
            raise ValueError(f"candidate_bias must be finite [{value_head.candidate_count}]")
        context, timeline_actions, timeline_states, observed_kinematics = self._prepare_observed_rollout(
            observed_frames,
            observed_states,
        )
        goal_tokens = self.encode_goal(goal_frame)
        batch_size = observed_frames.shape[0]
        rolled = context
        future_steps = []
        selected_actions = None
        rollout_depths = torch.full((batch_size,), horizon, dtype=torch.long, device=context.device)
        all_values = context.new_full(
            (batch_size, value_head.candidate_count, value_head.candidate_count),
            torch.nan,
        )
        active = torch.ones(batch_size, dtype=torch.bool, device=context.device)
        previous_planner_state = None
        previous_plan = None
        prefix_hidden = None
        context_steps = self._tokens_by_step(context)
        for depth in range(value_head.min_rollout_depth, horizon + 1):
            if depth > 0:
                step = self.observed_steps + depth - 1
                prefix_actions = timeline_actions if step == self.num_steps - 1 else timeline_actions[:, :step]
                prefix_states = timeline_states[:, :step]
                next_tokens = self._predict(rolled, prefix_actions, prefix_states)[:, -self.tokens_per_step :]
                future_steps.append(next_tokens)
                rolled = torch.cat([rolled, next_tokens], dim=1)
            future = context[:, :0] if not future_steps else torch.cat(future_steps, dim=1)
            plans, planner_states = self.adaptive_planner.rollout_all_depths(
                context_steps,
                self._tokens_by_step(future),
                observed_kinematics,
                self.action_head,
                goal_tokens,
                include_zero_depth=value_head.min_rollout_depth == 0,
            )
            current_plan = plans[:, -1]
            if selected_actions is None:
                selected_actions = torch.zeros_like(current_plan)
            current_planner_state = planner_states[:, -1]
            values, prefix_hidden = value_head.forward_step(
                current_planner_state,
                current_plan,
                depth,
                previous_planner_state,
                previous_plan,
                prefix_hidden,
            )
            depth_index = depth - value_head.min_rollout_depth
            all_values[:, depth_index] = values
            should_stop = depth == horizon
            if depth < horizon:
                should_stop = (values.float() + bias.unsqueeze(0)).argmin(dim=1) == depth_index
            stop = active & should_stop
            selected_actions[stop] = current_plan[stop]
            rollout_depths[stop] = depth
            active = active & ~stop
            previous_planner_state = current_planner_state
            previous_plan = current_plan
            if not torch.any(active):
                break
        if torch.any(active):
            raise RuntimeError("Value Gate rollout did not terminate every sample")
        if selected_actions is None:
            raise RuntimeError("Value Gate rollout produced no action plan")
        future = context[:, :0] if not future_steps else torch.cat(future_steps, dim=1)
        return ValueGateRolloutOutput(
            future_latents=self._tokens_by_step(future),
            predicted_actions=selected_actions,
            rollout_depths=rollout_depths,
            gate_values=all_values,
            observed_steps=self.observed_steps,
            tokens_per_step=self.tokens_per_step,
        )

    def visual_value_gate_rollout(
        self,
        observed_frames: torch.Tensor,
        observed_states: torch.Tensor,
        goal_frame: torch.Tensor,
        value_head: VisualPlannerValueHead,
        candidate_bias: torch.Tensor,
    ) -> ValueGateRolloutOutput:
        """Execute the rich visual Gate using only each online-available prefix."""
        if not isinstance(self.adaptive_planner, IterativeDepthPlanner) or self.action_head is None:
            raise RuntimeError("visual_value_gate_rollout requires IterativeDepthPlanner and Action Head")
        horizon = self.num_steps - self.observed_steps
        if value_head.horizon != horizon or value_head.planner_hidden_dim != self.adaptive_planner.hidden_dim:
            raise ValueError("Visual Value Gate dimensions do not match the recursive Planner")
        if value_head.observed_frames != observed_states.shape[1] or observed_states.shape[2] < value_head.state_dim:
            raise ValueError("Observed state history does not match the Visual Value Gate")
        bias = torch.as_tensor(candidate_bias, dtype=torch.float32, device=observed_frames.device)
        if bias.shape != (value_head.candidate_count,) or not torch.isfinite(bias).all():
            raise ValueError(f"candidate_bias must be finite [{value_head.candidate_count}]")
        context, timeline_actions, timeline_states, observed_kinematics = self._prepare_observed_rollout(
            observed_frames,
            observed_states,
        )
        goal_tokens = self.encode_goal(goal_frame)
        context_steps = self._tokens_by_step(context)
        pooled_context = pool_policy_tokens(context_steps[:, -1], value_head.token_count)
        pooled_goal = pool_policy_tokens(goal_tokens, value_head.token_count)
        history = observed_states[:, :, : value_head.state_dim]
        batch_size = observed_frames.shape[0]
        rolled = context
        future_steps = []
        selected_actions = None
        rollout_depths = torch.full((batch_size,), horizon, dtype=torch.long, device=context.device)
        all_values = context.new_full(
            (batch_size, value_head.candidate_count, value_head.candidate_count),
            torch.nan,
        )
        active = torch.ones(batch_size, dtype=torch.bool, device=context.device)
        previous_tokens = None
        previous_planner_state = None
        previous_plan = None
        prefix_hidden = None
        for depth in range(value_head.min_rollout_depth, horizon + 1):
            if depth > 0:
                step = self.observed_steps + depth - 1
                prefix_actions = timeline_actions if step == self.num_steps - 1 else timeline_actions[:, :step]
                prefix_states = timeline_states[:, :step]
                next_tokens = self._predict(rolled, prefix_actions, prefix_states)[:, -self.tokens_per_step :]
                future_steps.append(next_tokens)
                rolled = torch.cat([rolled, next_tokens], dim=1)
                current_tokens = pool_policy_tokens(next_tokens, value_head.token_count)
            else:
                current_tokens = pooled_context
            future = context[:, :0] if not future_steps else torch.cat(future_steps, dim=1)
            plans, planner_states = self.adaptive_planner.rollout_all_depths(
                context_steps,
                self._tokens_by_step(future),
                observed_kinematics,
                self.action_head,
                goal_tokens,
                include_zero_depth=value_head.min_rollout_depth == 0,
            )
            current_plan = plans[:, -1]
            if selected_actions is None:
                selected_actions = torch.zeros_like(current_plan)
            current_planner_state = planner_states[:, -1]
            values, prefix_hidden = value_head.forward_step(
                pooled_context,
                pooled_goal,
                current_tokens,
                history,
                observed_kinematics,
                current_planner_state,
                current_plan,
                depth,
                previous_tokens,
                previous_planner_state,
                previous_plan,
                prefix_hidden,
            )
            depth_index = depth - value_head.min_rollout_depth
            all_values[:, depth_index] = values
            should_stop = depth == horizon
            if depth < horizon:
                should_stop = (values.float() + bias.unsqueeze(0)).argmin(dim=1) == depth_index
            stop = active & should_stop
            selected_actions[stop] = current_plan[stop]
            rollout_depths[stop] = depth
            active = active & ~stop
            previous_tokens = current_tokens
            previous_planner_state = current_planner_state
            previous_plan = current_plan
            if not torch.any(active):
                break
        if torch.any(active):
            raise RuntimeError("Visual Value Gate rollout did not terminate every sample")
        if selected_actions is None:
            raise RuntimeError("Visual Value Gate rollout produced no action plan")
        future = context[:, :0] if not future_steps else torch.cat(future_steps, dim=1)
        return ValueGateRolloutOutput(
            future_latents=self._tokens_by_step(future),
            predicted_actions=selected_actions,
            rollout_depths=rollout_depths,
            gate_values=all_values,
            observed_steps=self.observed_steps,
            tokens_per_step=self.tokens_per_step,
        )

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1]")
        for student, teacher in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            teacher.mul_(momentum).add_(student, alpha=1.0 - momentum)

    def set_predictor_trainable(self, trainable: bool) -> None:
        """Set Predictor gradients while freezing structurally disabled inputs."""
        self.predictor.requires_grad_(trainable)
        extrinsics_encoder = getattr(self.predictor, "extrinsics_encoder", None)
        if extrinsics_encoder is not None and not bool(getattr(self.predictor, "use_extrinsics", False)):
            extrinsics_encoder.requires_grad_(False)


def build_model(config: ExperimentConfig) -> DroneWAM:
    """Construct the checkpoint-compatible base model without policy-side features."""
    encoder_cfg = config.model.encoder
    encoder = build_encoder(encoder_cfg, config.data)
    target_encoder = copy.deepcopy(encoder).eval()
    target_encoder.requires_grad_(False)

    raw_tokens_per_step = encoder.tokens_per_frame
    token_ae, normalize_representations, token_ae_grid = _load_token_ae(
        config.model.token_ae,
        embed_dim=encoder.embed_dim,
        raw_tokens_per_step=raw_tokens_per_step,
        encoder_family=encoder_cfg.family,
        encoder_model_name=encoder_cfg.model_name,
    )
    tokens_per_step = int(token_ae.num_latent_tokens) if token_ae is not None else int(raw_tokens_per_step)
    latent_grid = (
        token_ae_grid
        if token_ae is not None
        else (
            encoder_cfg.resolution[0] // encoder_cfg.patch_size,
            encoder_cfg.resolution[1] // encoder_cfg.patch_size,
        )
    )
    assert latent_grid is not None
    predictor_cfg = config.model.predictor
    with torch.random.fork_rng(devices=[]):
        # Keep trainable initialization identical across frozen representation ablations.
        torch.manual_seed(config.optimization.seed)
        predictor = VisionTransformerPredictorAC(
            img_size=(
                latent_grid[0] * encoder_cfg.patch_size,
                latent_grid[1] * encoder_cfg.patch_size,
            ),
            patch_size=encoder_cfg.patch_size,
            num_frames=512,
            tubelet_size=encoder_cfg.tubelet_size,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=predictor_cfg.hidden_dim,
            depth=predictor_cfg.depth,
            num_heads=predictor_cfg.num_heads,
            is_frame_causal=predictor_cfg.frame_causal,
            use_rope=predictor_cfg.use_rope,
            use_activation_checkpointing=predictor_cfg.use_activation_checkpointing,
            action_embed_dim=predictor_cfg.action_dim,
            state_embed_dim=predictor_cfg.state_dim,
            use_extrinsics=False,
        )
        action_head = None
        if config.model.action_head.enabled:
            action_head = MeanPoolActionHead(
                input_dim=encoder.embed_dim,
                hidden_dim=config.model.action_head.hidden_dim,
                action_dim=predictor_cfg.action_dim,
            )
    adaptive_planner = None
    continuation_policy = None
    continuation_reference = None
    if config.model.adaptive.enabled:
        adaptive_cfg = config.model.adaptive
        horizon = config.data.num_frames // encoder_cfg.tubelet_size
        horizon -= config.data.observed_frames // encoder_cfg.tubelet_size
        if adaptive_cfg.planner_mode == "iterative":
            adaptive_planner = IterativeDepthPlanner(
                embed_dim=encoder.embed_dim,
                hidden_dim=adaptive_cfg.planner_hidden_dim,
                action_dim=predictor_cfg.action_dim,
                kinematic_dim=predictor_cfg.action_dim + 1,
                horizon=horizon,
            )
        else:
            adaptive_planner = VariableDepthPlanner(
                embed_dim=encoder.embed_dim,
                hidden_dim=adaptive_cfg.planner_hidden_dim,
                action_dim=predictor_cfg.action_dim,
                kinematic_dim=predictor_cfg.action_dim + 1,
                horizon=horizon,
                num_layers=adaptive_cfg.planner_layers,
                num_heads=adaptive_cfg.planner_heads,
            )
        continuation_policy = ContinuationPolicy(
            embed_dim=encoder.embed_dim,
            hidden_dim=adaptive_cfg.policy_hidden_dim,
            horizon=horizon,
            action_dim=predictor_cfg.action_dim,
            kinematic_dim=predictor_cfg.action_dim + 1,
            token_count=adaptive_cfg.policy_token_count,
        )
        continuation_reference = copy.deepcopy(continuation_policy).eval()
        continuation_reference.requires_grad_(False)

    model = DroneWAM(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        action_head=action_head,
        token_ae=token_ae,
        adaptive_planner=adaptive_planner,
        continuation_policy=continuation_policy,
        continuation_reference=continuation_reference,
        num_raw_frames=config.data.num_frames,
        num_observed_frames=config.data.observed_frames,
        frame_stride=encoder_cfg.tubelet_size,
        tokens_per_step=tokens_per_step,
        normalize_representations=normalize_representations,
    )
    model.encoder.requires_grad_(config.runtime.train_encoder)
    model.set_predictor_trainable(config.runtime.train_predictor)
    if model.action_head is not None:
        model.action_head.requires_grad_(config.runtime.train_action_head)
    if model.adaptive_planner is not None:
        model.adaptive_planner.requires_grad_(config.runtime.train_adaptive_planner)
    if model.continuation_policy is not None:
        model.continuation_policy.requires_grad_(config.runtime.train_continuation_policy)
    if model.continuation_reference is not None:
        model.continuation_reference.requires_grad_(False)
    return model
