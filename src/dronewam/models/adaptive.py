"""Variable-depth planning and preference-trained continuation decisions."""

from __future__ import annotations

import torch

from dronewam.models.action_head import MeanPoolActionHead


def _validate_depths(
    depths: torch.Tensor, batch_size: int, available_steps: int, horizon: int
) -> None:
    if depths.shape != (batch_size,) or depths.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"depths must be integer [B], got shape={tuple(depths.shape)} dtype={depths.dtype}"
        )
    if (
        torch.any(depths < 1)
        or torch.any(depths > available_steps)
        or torch.any(depths > horizon)
    ):
        raise ValueError(
            f"depths must be in [1, min(available_steps={available_steps}, horizon={horizon})]"
        )


def pool_policy_tokens(tokens: torch.Tensor, output_tokens: int) -> torch.Tensor:
    """Deterministically pool raster-ordered world tokens without learned parameters."""
    if tokens.ndim < 3:
        raise ValueError("tokens must end in [P,D]")
    source_tokens = tokens.shape[-2]
    if output_tokens < 1 or source_tokens % output_tokens:
        raise ValueError(
            f"output_tokens={output_tokens} must evenly divide source tokens={source_tokens}"
        )
    group_size = source_tokens // output_tokens
    return tokens.reshape(
        *tokens.shape[:-2], output_tokens, group_size, tokens.shape[-1]
    ).mean(dim=-2)


class VariableDepthPlanner(torch.nn.Module):
    """Plan a full action horizon from any autoregressive latent prefix.

    With a goal image, unknown future slots interpolate from the terminal
    available world latent to the encoded goal latent.
    Every rollout depth may correct the complete provisional action-head plan.
    The zero-initialized residual preserves the legacy plan before training.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        action_dim: int,
        kinematic_dim: int,
        horizon: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if horizon < 2:
            raise ValueError("Adaptive planning requires horizon >= 2")
        if hidden_dim % num_heads:
            raise ValueError("planner_hidden_dim must be divisible by planner_heads")
        self.horizon = int(horizon)
        self.kinematic_dim = int(kinematic_dim)
        self.input_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, hidden_dim),
        )
        self.context_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, hidden_dim),
        )
        self.kinematic_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(kinematic_dim),
            torch.nn.Linear(kinematic_dim, hidden_dim),
        )
        self.temporal_embedding = torch.nn.Parameter(
            torch.zeros(1, horizon, hidden_dim)
        )
        self.visibility_embedding = torch.nn.Embedding(2, hidden_dim)
        self.depth_embedding = torch.nn.Embedding(horizon + 1, hidden_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_model = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.residual_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )
        torch.nn.init.zeros_(self.residual_head[-1].weight)
        torch.nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        context_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        observed_kinematics: torch.Tensor,
        depths: torch.Tensor,
        action_head: MeanPoolActionHead,
        goal_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return actions ``[B,H,A]`` using only each sample's available prefix."""
        if context_tokens.ndim != 4 or future_tokens.ndim != 4:
            raise ValueError("context_tokens and future_tokens must be [B,T,P,D]")
        if (
            context_tokens.shape[0] != future_tokens.shape[0]
            or context_tokens.shape[-1] != future_tokens.shape[-1]
        ):
            raise ValueError("Context and future latent batches must align")
        batch_size, available_steps, _, _ = future_tokens.shape
        _validate_depths(depths, batch_size, available_steps, self.horizon)
        if observed_kinematics.shape != (batch_size, self.kinematic_dim):
            raise ValueError(
                "observed_kinematics must be [B,K], got "
                f"{tuple(observed_kinematics.shape)}"
            )

        context_last = context_tokens[:, -1].mean(dim=1)
        future_pooled = future_tokens.mean(dim=2)
        terminal = future_pooled.gather(
            1,
            (depths - 1).view(batch_size, 1, 1).expand(-1, 1, future_pooled.shape[-1]),
        )
        if goal_tokens is None:
            boundary = terminal
            goal_offset = torch.zeros_like(context_last)
            unknown = terminal.expand(-1, self.horizon, -1)
        else:
            if (
                goal_tokens.ndim != 3
                or goal_tokens.shape[0] != batch_size
                or goal_tokens.shape[2] != future_tokens.shape[-1]
            ):
                raise ValueError(
                    "goal_tokens must be [B,P,D] aligned with world tokens"
                )
            boundary = goal_tokens.mean(dim=1, keepdim=True)
            goal_offset = boundary[:, 0] - context_last
            positions = torch.arange(self.horizon, device=future_tokens.device).view(
                1, -1, 1
            )
            remaining = (self.horizon - depths).clamp_min(1).view(batch_size, 1, 1)
            progress = (positions - depths.view(batch_size, 1, 1) + 1).clamp_min(
                0
            ) / remaining
            unknown = terminal + progress * (boundary - terminal)
        padded = unknown.clone()
        copy_steps = min(available_steps, self.horizon)
        padded[:, :copy_steps] = future_pooled[:, :copy_steps]
        positions = torch.arange(self.horizon, device=future_tokens.device).view(1, -1)
        known = positions < depths.view(-1, 1)
        filled = torch.where(known.unsqueeze(-1), padded, unknown)

        base_actions = action_head.decode_pooled(filled)
        hidden = self.input_projection(filled)
        hidden = hidden + self.context_projection(context_last).unsqueeze(1)
        hidden = hidden + self.context_projection(goal_offset).unsqueeze(1)
        hidden = hidden + self.kinematic_projection(observed_kinematics).unsqueeze(1)
        hidden = hidden + self.temporal_embedding
        hidden = hidden + self.visibility_embedding(known.long())
        hidden = hidden + self.depth_embedding(depths).unsqueeze(1)
        residual = self.residual_head(self.temporal_model(hidden))
        return base_actions + residual


class IterativeDepthPlanner(torch.nn.Module):
    """Refine one full-horizon plan as each new world latent becomes available."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        action_dim: int,
        kinematic_dim: int,
        horizon: int,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or horizon < 2:
            raise ValueError(
                "hidden_dim must be at least 8 and horizon must be at least 2"
            )
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.kinematic_dim = int(kinematic_dim)
        self.horizon = int(horizon)
        self.latent_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, hidden_dim),
        )
        self.relation_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim * 4),
            torch.nn.Linear(embed_dim * 4, hidden_dim),
            torch.nn.GELU(),
        )
        self.plan_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(horizon * action_dim),
            torch.nn.Linear(horizon * action_dim, hidden_dim),
        )
        self.kinematic_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(kinematic_dim),
            torch.nn.Linear(kinematic_dim, hidden_dim),
        )
        self.depth_embedding = torch.nn.Embedding(horizon + 1, hidden_dim)
        self.update = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.residual_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, horizon * action_dim),
        )
        torch.nn.init.zeros_(self.residual_head[-1].weight)
        torch.nn.init.zeros_(self.residual_head[-1].bias)

    def _base_plans(
        self,
        future_pooled: torch.Tensor,
        goal_pooled: torch.Tensor | None,
        action_head: MeanPoolActionHead,
    ) -> torch.Tensor:
        batch_size, available_steps, _ = future_pooled.shape
        positions = torch.arange(self.horizon, device=future_pooled.device).view(
            1, self.horizon, 1
        )
        plans = []
        for depth in range(1, available_steps + 1):
            terminal = future_pooled[:, depth - 1 : depth]
            if goal_pooled is None:
                unknown = terminal.expand(-1, self.horizon, -1)
            else:
                remaining = max(self.horizon - depth, 1)
                progress = (positions - depth + 1).clamp_min(0) / remaining
                unknown = terminal + progress * (goal_pooled - terminal)
            filled = unknown.clone()
            filled[:, :depth] = future_pooled[:, :depth]
            plans.append(action_head.decode_pooled(filled))
        return torch.stack(plans, dim=1)

    def _zero_depth_plan(
        self,
        context_last: torch.Tensor,
        goal_pooled: torch.Tensor | None,
        action_head: MeanPoolActionHead,
    ) -> torch.Tensor:
        """Plan from observations and the goal before any world rollout step."""
        target = context_last.unsqueeze(1) if goal_pooled is None else goal_pooled
        progress = (
            torch.arange(
                1,
                self.horizon + 1,
                device=context_last.device,
                dtype=context_last.dtype,
            )
            .view(1, self.horizon, 1)
            .div(self.horizon)
        )
        interpolated = context_last.unsqueeze(1) + progress * (
            target - context_last.unsqueeze(1)
        )
        return action_head.decode_pooled(interpolated)

    def rollout_all_depths(
        self,
        context_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        observed_kinematics: torch.Tensor,
        action_head: MeanPoolActionHead,
        goal_tokens: torch.Tensor | None = None,
        *,
        include_zero_depth: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nested plans and recurrent states for every available depth."""
        if context_tokens.ndim != 4 or future_tokens.ndim != 4:
            raise ValueError("context_tokens and future_tokens must be [B,T,P,D]")
        batch_size, available_steps, _, embed_dim = future_tokens.shape
        if (
            context_tokens.shape[0] != batch_size
            or context_tokens.shape[-1] != embed_dim
        ):
            raise ValueError("Context and future latent batches must align")
        if not 0 <= available_steps <= self.horizon:
            raise ValueError(f"available future steps must be in [0,{self.horizon}]")
        if available_steps == 0 and not include_zero_depth:
            raise ValueError(
                "zero available future steps require include_zero_depth=True"
            )
        if observed_kinematics.shape != (batch_size, self.kinematic_dim):
            raise ValueError(f"observed_kinematics must be [B,{self.kinematic_dim}]")
        if goal_tokens is not None and goal_tokens.shape != (
            batch_size,
            future_tokens.shape[2],
            embed_dim,
        ):
            raise ValueError("goal_tokens must be [B,P,D] aligned with future tokens")

        context_last = context_tokens[:, -1].mean(dim=1)
        future_pooled = future_tokens.mean(dim=2)
        goal_pooled = (
            None if goal_tokens is None else goal_tokens.mean(dim=1, keepdim=True)
        )
        relation_target = context_last if goal_pooled is None else goal_pooled[:, 0]
        initial_relation = self._relation(context_last, relation_target)
        hidden = torch.tanh(
            initial_relation + self.kinematic_projection(observed_kinematics)
        )
        plans = []
        states = []
        if include_zero_depth:
            zero_plan = self._zero_depth_plan(context_last, goal_pooled, action_head)
            zero_residual = self.residual_head(hidden).reshape(
                batch_size, self.horizon, self.action_dim
            )
            plans.append(zero_plan + zero_residual)
            states.append(hidden)
        if available_steps == 0:
            return torch.stack(plans, dim=1), torch.stack(states, dim=1)
        base_plans = self._base_plans(future_pooled, goal_pooled, action_head)
        for step in range(available_steps):
            current = future_pooled[:, step]
            update = (
                self.latent_projection(current)
                + self._relation(current, relation_target)
                + self.plan_projection(base_plans[:, step].flatten(1))
                + self.depth_embedding.weight[step + 1]
            )
            hidden = self.update(update, hidden)
            residual = self.residual_head(hidden).reshape(
                batch_size, self.horizon, self.action_dim
            )
            plans.append(base_plans[:, step] + residual)
            states.append(hidden)
        return torch.stack(plans, dim=1), torch.stack(states, dim=1)

    def forward(
        self,
        context_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        observed_kinematics: torch.Tensor,
        depths: torch.Tensor,
        action_head: MeanPoolActionHead,
        goal_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, available_steps = future_tokens.shape[:2]
        if depths.shape != (batch_size,) or depths.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                f"depths must be integer [B], got shape={tuple(depths.shape)} dtype={depths.dtype}"
            )
        if (
            torch.any(depths < 0)
            or torch.any(depths > available_steps)
            or torch.any(depths > self.horizon)
        ):
            raise ValueError(
                f"depths must be in [0, min(available_steps={available_steps}, horizon={self.horizon})]"
            )
        include_zero_depth = bool(torch.any(depths == 0))
        plans, _ = self.rollout_all_depths(
            context_tokens,
            future_tokens,
            observed_kinematics,
            action_head,
            goal_tokens,
            include_zero_depth=include_zero_depth,
        )
        rows = torch.arange(batch_size, device=depths.device)
        indices = depths if include_zero_depth else depths - 1
        return plans[rows, indices]

    def _relation(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.relation_projection(
            torch.cat(
                [source, target, target - source, (target - source).abs()], dim=-1
            )
        )


class PlannerStateHorizonHead(torch.nn.Module):
    """Predict the best remaining depth from recursive state and its action plan."""

    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        expansion: int = 4,
        dropout: float = 0.1,
        action_dim: int = 0,
        plan_horizon: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or horizon < 2 or expansion < 1:
            raise ValueError(
                "hidden_dim must be at least 8, horizon at least 2, and expansion positive"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if (
            (action_dim == 0) != (plan_horizon == 0)
            or action_dim < 0
            or plan_horizon < 0
        ):
            raise ValueError(
                "action_dim and plan_horizon must both be zero or both be positive"
            )
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.expansion = int(expansion)
        self.dropout = float(dropout)
        self.action_dim = int(action_dim)
        self.plan_horizon = int(plan_horizon)
        if self.action_dim:
            plan_dim = self.action_dim * self.plan_horizon
            self.plan_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(plan_dim),
                torch.nn.Linear(plan_dim, hidden_dim),
                torch.nn.GELU(),
            )
            self.fusion = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim * 2),
                torch.nn.Linear(hidden_dim * 2, hidden_dim),
                torch.nn.GELU(),
            )
        else:
            self.plan_projection = None
            self.fusion = None
        inner_dim = hidden_dim * expansion
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, inner_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(inner_dim, inner_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(inner_dim, horizon),
        )

    def _fuse(
        self,
        planner_states: torch.Tensor,
        planned_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.plan_projection is None or self.fusion is None:
            if planned_actions is not None:
                raise ValueError(
                    "planned_actions require a plan-conditioned horizon head"
                )
            return planner_states
        expected = (*planner_states.shape[:-1], self.plan_horizon, self.action_dim)
        if planned_actions is None or planned_actions.shape != expected:
            raise ValueError(f"planned_actions must be {expected}")
        plan_features = self.plan_projection(planned_actions.flatten(-2))
        return self.fusion(torch.cat((planner_states, plan_features), dim=-1))

    def forward(
        self,
        planner_states: torch.Tensor,
        planned_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if planner_states.ndim != 3 or planner_states.shape[1:] != (
            self.horizon,
            self.hidden_dim,
        ):
            raise ValueError(
                f"planner_states must be [B,{self.horizon},{self.hidden_dim}]"
            )
        logits = self.head(self._fuse(planner_states, planned_actions))
        current = torch.arange(self.horizon, device=logits.device).view(
            1, self.horizon, 1
        )
        candidates = torch.arange(self.horizon, device=logits.device).view(
            1, 1, self.horizon
        )
        return logits.masked_fill(candidates < current, -torch.inf)

    def forward_step(
        self,
        planner_state: torch.Tensor,
        current_depth: int,
        planned_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the best remaining depth from one causal recursive state."""
        if planner_state.ndim != 2 or planner_state.shape[1] != self.hidden_dim:
            raise ValueError(f"planner_state must be [B,{self.hidden_dim}]")
        if not 1 <= current_depth <= self.horizon:
            raise ValueError(f"current_depth must be in [1,{self.horizon}]")
        logits = self.head(self._fuse(planner_state, planned_actions))
        candidates = torch.arange(self.horizon, device=logits.device)
        return logits.masked_fill(candidates < current_depth - 1, -torch.inf)


class PlannerStateStopHead(torch.nn.Module):
    """Stop from causal Planner-state and action-plan convergence features."""

    CONTINUE = 0
    STOP = 1

    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        action_dim: int,
        min_rollout_depth: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or horizon < 2 or action_dim < 1:
            raise ValueError(
                "hidden_dim must be at least 8, horizon at least 2, and action_dim positive"
            )
        if min_rollout_depth not in (0, 1):
            raise ValueError("min_rollout_depth must be zero or one")
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.min_rollout_depth = int(min_rollout_depth)
        self.candidate_count = self.horizon - self.min_rollout_depth + 1
        plan_dim = horizon * action_dim
        self.state_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_delta_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.plan_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(plan_dim),
            torch.nn.Linear(plan_dim, hidden_dim),
        )
        self.plan_delta_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(plan_dim),
            torch.nn.Linear(plan_dim, hidden_dim),
        )
        self.change_magnitude_projection = torch.nn.Linear(2, hidden_dim)
        self.depth_embedding = torch.nn.Embedding(self.candidate_count, hidden_dim)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def _features(
        self,
        planner_states: torch.Tensor,
        planned_actions: torch.Tensor,
        previous_states: torch.Tensor,
        previous_actions: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        state_delta = planner_states - previous_states
        plan_delta = planned_actions - previous_actions
        magnitudes = torch.stack(
            (
                torch.log1p(state_delta.float().norm(dim=-1)),
                torch.log1p(plan_delta.float().flatten(-2).norm(dim=-1)),
            ),
            dim=-1,
        ).to(planner_states.dtype)
        return (
            self.state_projection(planner_states)
            + self.state_delta_projection(state_delta)
            + self.plan_projection(planned_actions.flatten(-2))
            + self.plan_delta_projection(plan_delta.flatten(-2))
            + self.change_magnitude_projection(magnitudes)
            + self.depth_embedding(depths)
        )

    def forward(
        self, planner_states: torch.Tensor, planned_actions: torch.Tensor
    ) -> torch.Tensor:
        if planner_states.ndim != 3 or planner_states.shape[1:] != (
            self.candidate_count,
            self.hidden_dim,
        ):
            raise ValueError(
                f"planner_states must be [B,{self.candidate_count},{self.hidden_dim}]"
            )
        if planned_actions.shape != (
            planner_states.shape[0],
            self.candidate_count,
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"planned_actions must be [B,{self.candidate_count},{self.horizon},{self.action_dim}]"
            )
        previous_states = torch.cat(
            (planner_states[:, :1], planner_states[:, :-1]), dim=1
        )
        previous_actions = torch.cat(
            (planned_actions[:, :1], planned_actions[:, :-1]), dim=1
        )
        depths = (
            torch.arange(self.candidate_count, device=planner_states.device)
            .view(1, -1)
            .expand(planner_states.shape[0], -1)
        )
        return self.head(
            self._features(
                planner_states,
                planned_actions,
                previous_states,
                previous_actions,
                depths,
            )
        )

    def forward_step(
        self,
        planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        current_depth: int,
        previous_state: torch.Tensor | None,
        previous_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        if planner_state.ndim != 2 or planner_state.shape[1] != self.hidden_dim:
            raise ValueError(f"planner_state must be [B,{self.hidden_dim}]")
        if planned_actions.shape != (
            planner_state.shape[0],
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"planned_actions must be [B,{self.horizon},{self.action_dim}]"
            )
        if not self.min_rollout_depth <= current_depth <= self.horizon:
            raise ValueError(
                f"current_depth must be in [{self.min_rollout_depth},{self.horizon}]"
            )
        if (previous_state is None) != (previous_actions is None):
            raise ValueError(
                "previous_state and previous_actions must either both be provided or both be None"
            )
        if previous_state is None:
            previous_state = planner_state
            previous_actions = planned_actions
        if (
            previous_state.shape != planner_state.shape
            or previous_actions.shape != planned_actions.shape
        ):
            raise ValueError(
                "Previous state and action shapes must match their current values"
            )
        depths = torch.full(
            (planner_state.shape[0],),
            current_depth - self.min_rollout_depth,
            dtype=torch.long,
            device=planner_state.device,
        )
        return self.head(
            self._features(
                planner_state,
                planned_actions,
                previous_state,
                previous_actions,
                depths,
            )
        )


class PlannerStateValueHead(torch.nn.Module):
    """Predict reachable rollout-depth values from one causal Planner prefix."""

    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        action_dim: int,
        min_rollout_depth: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim < 8 or horizon < 2 or action_dim < 1:
            raise ValueError(
                "hidden_dim must be at least 8, horizon at least 2, and action_dim positive"
            )
        if min_rollout_depth not in (0, 1):
            raise ValueError("min_rollout_depth must be zero or one")
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.min_rollout_depth = int(min_rollout_depth)
        self.candidate_count = self.horizon - self.min_rollout_depth + 1
        plan_dim = self.horizon * self.action_dim
        self.state_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_delta_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.plan_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(plan_dim),
            torch.nn.Linear(plan_dim, hidden_dim),
        )
        self.plan_delta_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(plan_dim),
            torch.nn.Linear(plan_dim, hidden_dim),
        )
        self.change_magnitude_projection = torch.nn.Linear(2, hidden_dim)
        self.depth_embedding = torch.nn.Embedding(self.candidate_count, hidden_dim)
        self.prefix_update = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.value_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, self.candidate_count),
        )

    def _features(
        self,
        planner_states: torch.Tensor,
        planned_actions: torch.Tensor,
        previous_states: torch.Tensor,
        previous_actions: torch.Tensor,
        depth_indices: torch.Tensor,
    ) -> torch.Tensor:
        state_delta = planner_states - previous_states
        plan_delta = planned_actions - previous_actions
        magnitudes = torch.stack(
            (
                torch.log1p(state_delta.float().norm(dim=-1)),
                torch.log1p(plan_delta.float().flatten(-2).norm(dim=-1)),
            ),
            dim=-1,
        ).to(planner_states.dtype)
        return (
            self.state_projection(planner_states)
            + self.state_delta_projection(state_delta)
            + self.plan_projection(planned_actions.flatten(-2))
            + self.plan_delta_projection(plan_delta.flatten(-2))
            + self.change_magnitude_projection(magnitudes)
            + self.depth_embedding(depth_indices)
        )

    def forward(
        self, planner_states: torch.Tensor, planned_actions: torch.Tensor
    ) -> torch.Tensor:
        """Return causal value predictions shaped ``[B,C,C]``."""
        if planner_states.ndim != 3 or planner_states.shape[1:] != (
            self.candidate_count,
            self.hidden_dim,
        ):
            raise ValueError(
                f"planner_states must be [B,{self.candidate_count},{self.hidden_dim}]"
            )
        expected_actions = (
            planner_states.shape[0],
            self.candidate_count,
            self.horizon,
            self.action_dim,
        )
        if planned_actions.shape != expected_actions:
            raise ValueError(f"planned_actions must be {expected_actions}")
        previous_states = torch.cat(
            (planner_states[:, :1], planner_states[:, :-1]), dim=1
        )
        previous_actions = torch.cat(
            (planned_actions[:, :1], planned_actions[:, :-1]), dim=1
        )
        depth_indices = (
            torch.arange(self.candidate_count, device=planner_states.device)
            .view(1, -1)
            .expand(planner_states.shape[0], -1)
        )
        features = self._features(
            planner_states,
            planned_actions,
            previous_states,
            previous_actions,
            depth_indices,
        )
        hidden = torch.zeros(
            planner_states.shape[0],
            self.hidden_dim,
            dtype=planner_states.dtype,
            device=planner_states.device,
        )
        predictions = []
        candidate_indices = torch.arange(
            self.candidate_count, device=planner_states.device
        )
        for current_index in range(self.candidate_count):
            hidden = self.prefix_update(features[:, current_index], hidden)
            values = self.value_head(hidden)
            predictions.append(
                values.masked_fill(candidate_indices < current_index, torch.inf)
            )
        return torch.stack(predictions, dim=1)

    def forward_step(
        self,
        planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        current_depth: int,
        previous_state: torch.Tensor | None,
        previous_actions: torch.Tensor | None,
        prefix_hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update one causal prefix and return reachable depth values."""
        if planner_state.ndim != 2 or planner_state.shape[1] != self.hidden_dim:
            raise ValueError(f"planner_state must be [B,{self.hidden_dim}]")
        if planned_actions.shape != (
            planner_state.shape[0],
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"planned_actions must be [B,{self.horizon},{self.action_dim}]"
            )
        if not self.min_rollout_depth <= current_depth <= self.horizon:
            raise ValueError(
                f"current_depth must be in [{self.min_rollout_depth},{self.horizon}]"
            )
        if (previous_state is None) != (previous_actions is None):
            raise ValueError(
                "previous_state and previous_actions must either both be provided or both be None"
            )
        if previous_state is None:
            previous_state = planner_state
            previous_actions = planned_actions
        if (
            previous_state.shape != planner_state.shape
            or previous_actions.shape != planned_actions.shape
        ):
            raise ValueError(
                "Previous state and action shapes must match their current values"
            )
        if prefix_hidden is None:
            prefix_hidden = torch.zeros_like(planner_state)
        if prefix_hidden.shape != planner_state.shape:
            raise ValueError("prefix_hidden must match planner_state")
        depth_index = current_depth - self.min_rollout_depth
        depth_indices = torch.full(
            (planner_state.shape[0],),
            depth_index,
            dtype=torch.long,
            device=planner_state.device,
        )
        features = self._features(
            planner_state,
            planned_actions,
            previous_state,
            previous_actions,
            depth_indices,
        )
        prefix_hidden = self.prefix_update(features, prefix_hidden)
        values = self.value_head(prefix_hidden)
        candidate_indices = torch.arange(
            self.candidate_count, device=planner_state.device
        )
        return (
            values.masked_fill(candidate_indices < depth_index, torch.inf),
            prefix_hidden,
        )


class VisualPlannerValueHead(torch.nn.Module):
    """Predict reachable rollout values from structured causal visual prefixes."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        if (
            min(
                embed_dim,
                hidden_dim,
                planner_hidden_dim,
                action_dim,
                state_dim,
                kinematic_dim,
                token_count,
            )
            < 1
        ):
            raise ValueError("Visual Value Gate dimensions must be positive")
        if (
            hidden_dim < 8
            or horizon < 2
            or observed_frames < 1
            or min_rollout_depth not in (0, 1)
        ):
            raise ValueError(
                "Visual Value Gate horizon, history, hidden dimension, or minimum depth is invalid"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if (
            transformer_layers < 1
            or transformer_heads < 1
            or hidden_dim % transformer_heads
        ):
            raise ValueError(
                "Transformer layers and heads must evenly divide the hidden dimension"
            )
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.planner_hidden_dim = int(planner_hidden_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.observed_frames = int(observed_frames)
        self.state_dim = int(state_dim)
        self.kinematic_dim = int(kinematic_dim)
        self.token_count = int(token_count)
        self.min_rollout_depth = int(min_rollout_depth)
        self.candidate_count = self.horizon - self.min_rollout_depth + 1
        plan_dim = self.horizon * self.action_dim
        self.visual_projection = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.visual_cls = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.visual_type_embedding = torch.nn.Embedding(4, hidden_dim)
        self.visual_slot_embedding = torch.nn.Embedding(token_count, hidden_dim)
        self.visual_depth_embedding = torch.nn.Embedding(
            self.candidate_count, hidden_dim
        )
        visual_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_encoder = torch.nn.TransformerEncoder(
            visual_layer, num_layers=transformer_layers
        )
        self.visual_norm = torch.nn.LayerNorm(hidden_dim)
        self.geometry_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(10),
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.planner_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(planner_hidden_dim),
            torch.nn.Linear(planner_hidden_dim, hidden_dim),
        )
        self.planner_delta_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(planner_hidden_dim),
            torch.nn.Linear(planner_hidden_dim, hidden_dim),
        )
        self.plan_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(plan_dim),
            torch.nn.Linear(plan_dim, hidden_dim),
        )
        self.plan_delta_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(plan_dim),
            torch.nn.Linear(plan_dim, hidden_dim),
        )
        self.history_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(observed_frames * state_dim),
            torch.nn.Linear(observed_frames * state_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.kinematic_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(kinematic_dim),
            torch.nn.Linear(kinematic_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.magnitude_projection = torch.nn.Linear(4, hidden_dim)
        self.depth_embedding = torch.nn.Embedding(self.candidate_count, hidden_dim)
        self.step_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 9),
            torch.nn.Linear(hidden_dim * 9, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.prefix_update = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.candidate_embedding = torch.nn.Embedding(self.candidate_count, hidden_dim)
        self.value_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 3),
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def _validate_static(
        self,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> None:
        batch_size = context_tokens.shape[0]
        token_shape = (batch_size, self.token_count, self.embed_dim)
        if context_tokens.shape != token_shape or goal_tokens.shape != token_shape:
            raise ValueError(f"context_tokens and goal_tokens must be {token_shape}")
        if observed_states.shape != (batch_size, self.observed_frames, self.state_dim):
            raise ValueError(
                f"observed_states must be [B,{self.observed_frames},{self.state_dim}]"
            )
        if observed_kinematics.shape != (batch_size, self.kinematic_dim):
            raise ValueError(f"observed_kinematics must be [B,{self.kinematic_dim}]")

    def _visual_features(
        self,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        previous_tokens: torch.Tensor,
        depth_indices: torch.Tensor,
    ) -> torch.Tensor:
        token_groups = (
            context_tokens,
            goal_tokens,
            current_tokens,
            current_tokens - previous_tokens,
        )
        depth = self.visual_depth_embedding(depth_indices).unsqueeze(1)
        slots = self.visual_slot_embedding.weight.unsqueeze(0)
        projected_groups = []
        for type_index, tokens in enumerate(token_groups):
            projected = self.visual_projection(tokens) + slots
            if type_index >= 2:
                projected = projected + depth
            projected_groups.append(
                projected + self.visual_type_embedding.weight[type_index]
            )
        cls = self.visual_cls.expand(context_tokens.shape[0], -1, -1) + depth
        visual_tokens = torch.cat((cls, *projected_groups), dim=1)
        return self.visual_norm(self.visual_encoder(visual_tokens)[:, 0])

    @staticmethod
    def _geometry_features(
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        previous_tokens: torch.Tensor,
    ) -> torch.Tensor:
        pairs = (
            (context_tokens, goal_tokens),
            (current_tokens, goal_tokens),
            (current_tokens, context_tokens),
            (current_tokens, previous_tokens),
        )
        rms_distances = []
        cosine_distances = []
        for source, target in pairs:
            delta = source.float() - target.float()
            rms_distances.append(torch.log1p(delta.square().mean(dim=(1, 2)).sqrt()))
            cosine = torch.nn.functional.cosine_similarity(
                source.float(), target.float(), dim=2
            ).mean(dim=1)
            cosine_distances.append(1.0 - cosine)
        initial_goal_distance = rms_distances[0].expm1().clamp_min(1.0e-6)
        current_goal_distance = rms_distances[1].expm1()
        step_distance = rms_distances[3].expm1()
        ratios = (
            torch.log1p(current_goal_distance / initial_goal_distance),
            torch.log1p(step_distance / initial_goal_distance),
        )
        return torch.stack((*rms_distances, *cosine_distances, *ratios), dim=1).to(
            context_tokens.dtype
        )

    def _static_features(
        self,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        history = self.history_projection(observed_states.flatten(1))
        kinematics = self.kinematic_projection(observed_kinematics)
        return torch.tanh(history + kinematics)

    def _step_features(
        self,
        current_tokens: torch.Tensor,
        previous_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        planner_state: torch.Tensor,
        previous_planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        previous_actions: torch.Tensor,
        observed_kinematics: torch.Tensor,
        depth_indices: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        planner_delta = planner_state - previous_planner_state
        plan_delta = planned_actions - previous_actions
        visual_features = self._visual_features(
            context_tokens,
            goal_tokens,
            current_tokens,
            previous_tokens,
            depth_indices,
        )
        geometry_features = self.geometry_projection(
            self._geometry_features(
                context_tokens, goal_tokens, current_tokens, previous_tokens
            )
        )
        magnitudes = torch.stack(
            (
                torch.log1p(
                    (current_tokens - previous_tokens).float().flatten(1).norm(dim=1)
                ),
                torch.log1p(planner_delta.float().norm(dim=1)),
                torch.log1p(plan_delta.float().flatten(1).norm(dim=1)),
                torch.log1p(planned_actions.float().flatten(1).norm(dim=1)),
            ),
            dim=1,
        ).to(planner_state.dtype)
        combined = torch.cat(
            (
                visual_features,
                geometry_features,
                self.planner_projection(planner_state),
                self.planner_delta_projection(planner_delta),
                self.plan_projection(planned_actions.flatten(1)),
                self.plan_delta_projection(plan_delta.flatten(1)),
                self.kinematic_projection(observed_kinematics),
                self.magnitude_projection(magnitudes),
                self.depth_embedding(depth_indices),
            ),
            dim=1,
        )
        return self.step_projection(combined), visual_features

    def _values(
        self, hidden: torch.Tensor, visual_features: torch.Tensor, depth_index: int
    ) -> torch.Tensor:
        candidates = self.candidate_embedding.weight.unsqueeze(0).expand(
            hidden.shape[0], -1, -1
        )
        expanded_hidden = hidden.unsqueeze(1).expand_as(candidates)
        expanded_visual = visual_features.unsqueeze(1).expand_as(candidates)
        values = self.value_head(
            torch.cat((expanded_hidden, expanded_visual, candidates), dim=2)
        ).squeeze(2)
        candidate_indices = torch.arange(self.candidate_count, device=hidden.device)
        return values.masked_fill(candidate_indices < depth_index, torch.inf)

    def forward(
        self,
        planner_states: torch.Tensor,
        planned_actions: torch.Tensor,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        """Return causal reachable values shaped ``[B,C,C]``."""
        self._validate_static(
            context_tokens, goal_tokens, observed_states, observed_kinematics
        )
        batch_size = context_tokens.shape[0]
        if planner_states.shape != (
            batch_size,
            self.candidate_count,
            self.planner_hidden_dim,
        ):
            raise ValueError("planner_states do not match the visual Value Gate")
        if planned_actions.shape != (
            batch_size,
            self.candidate_count,
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("planned_actions do not match the visual Value Gate")
        if future_tokens.shape != (
            batch_size,
            self.horizon,
            self.token_count,
            self.embed_dim,
        ):
            raise ValueError("future_tokens do not match the visual Value Gate")
        if self.min_rollout_depth != 0:
            raise NotImplementedError(
                "Visual Value Gate full-sequence training currently requires depth 0"
            )
        current_tokens = torch.cat((context_tokens.unsqueeze(1), future_tokens), dim=1)
        previous_tokens = torch.cat(
            (current_tokens[:, :1], current_tokens[:, :-1]), dim=1
        )
        previous_states = torch.cat(
            (planner_states[:, :1], planner_states[:, :-1]), dim=1
        )
        previous_actions = torch.cat(
            (planned_actions[:, :1], planned_actions[:, :-1]), dim=1
        )
        hidden = self._static_features(observed_states, observed_kinematics)
        predictions = []
        for depth_index in range(self.candidate_count):
            indices = torch.full(
                (batch_size,),
                depth_index,
                dtype=torch.long,
                device=context_tokens.device,
            )
            features, visual_features = self._step_features(
                current_tokens[:, depth_index],
                previous_tokens[:, depth_index],
                goal_tokens,
                planner_states[:, depth_index],
                previous_states[:, depth_index],
                planned_actions[:, depth_index],
                previous_actions[:, depth_index],
                observed_kinematics,
                indices,
                context_tokens,
            )
            hidden = self.prefix_update(features, hidden)
            predictions.append(self._values(hidden, visual_features, depth_index))
        return torch.stack(predictions, dim=1)

    def forward_step(
        self,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
        planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        current_depth: int,
        previous_tokens: torch.Tensor | None,
        previous_planner_state: torch.Tensor | None,
        previous_actions: torch.Tensor | None,
        prefix_hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update one online visual prefix and return values for reachable depths."""
        self._validate_static(
            context_tokens, goal_tokens, observed_states, observed_kinematics
        )
        batch_size = context_tokens.shape[0]
        if current_tokens.shape != context_tokens.shape:
            raise ValueError("current_tokens must match context_tokens")
        if planner_state.shape != (batch_size, self.planner_hidden_dim):
            raise ValueError("planner_state does not match the visual Value Gate")
        if planned_actions.shape != (batch_size, self.horizon, self.action_dim):
            raise ValueError("planned_actions do not match the visual Value Gate")
        if not self.min_rollout_depth <= current_depth <= self.horizon:
            raise ValueError("current_depth is outside the visual Value Gate range")
        previous_values = (previous_tokens, previous_planner_state, previous_actions)
        if any(value is None for value in previous_values) and not all(
            value is None for value in previous_values
        ):
            raise ValueError(
                "All previous visual Gate inputs must be jointly present or absent"
            )
        if previous_tokens is None:
            previous_tokens = current_tokens
            previous_planner_state = planner_state
            previous_actions = planned_actions
        if prefix_hidden is None:
            prefix_hidden = self._static_features(observed_states, observed_kinematics)
        depth_index = current_depth - self.min_rollout_depth
        indices = torch.full(
            (batch_size,), depth_index, dtype=torch.long, device=context_tokens.device
        )
        features, visual_features = self._step_features(
            current_tokens,
            previous_tokens,
            goal_tokens,
            planner_state,
            previous_planner_state,
            planned_actions,
            previous_actions,
            observed_kinematics,
            indices,
            context_tokens,
        )
        prefix_hidden = self.prefix_update(features, prefix_hidden)
        return self._values(prefix_hidden, visual_features, depth_index), prefix_hidden


class VisualPlannerStopHead(VisualPlannerValueHead):
    """Estimate the discrete stopping hazard from each causal visual prefix."""

    CONTINUE = 0
    STOP = 1

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__(
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            min_rollout_depth,
            dropout,
            transformer_layers,
            transformer_heads,
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim * 3),
            torch.nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.hidden_dim, 2),
        )

    def _values(
        self, hidden: torch.Tensor, visual_features: torch.Tensor, depth_index: int
    ) -> torch.Tensor:
        candidate = (
            self.candidate_embedding.weight[depth_index]
            .unsqueeze(0)
            .expand(hidden.shape[0], -1)
        )
        return self.value_head(torch.cat((hidden, visual_features, candidate), dim=1))


class CrossModalVisualPlannerStopHead(VisualPlannerStopHead):
    """Fuse visual, plan, and flight-state tokens before each causal stop decision."""

    VISUAL_MODALITIES = 4
    MODALITY_COUNT = 11

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__(
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            min_rollout_depth,
            dropout,
            transformer_layers=2,
            transformer_heads=transformer_heads,
        )
        if (
            transformer_layers < 1
            or transformer_heads < 1
            or hidden_dim % transformer_heads
        ):
            raise ValueError(
                "Cross-modal layers and heads must evenly divide the hidden dimension"
            )
        self.fusion_cls = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.fusion_modality_embedding = torch.nn.Embedding(
            self.MODALITY_COUNT, hidden_dim
        )
        self.action_time_embedding = torch.nn.Embedding(horizon, hidden_dim)
        self.state_time_embedding = torch.nn.Embedding(observed_frames, hidden_dim)
        self.action_token_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(action_dim),
            torch.nn.Linear(action_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.state_token_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(state_dim),
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.GELU(),
        )
        fusion_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = torch.nn.TransformerEncoder(
            fusion_layer, num_layers=transformer_layers
        )
        self.fusion_norm = torch.nn.LayerNorm(hidden_dim)

    def _step_features(
        self,
        current_tokens: torch.Tensor,
        previous_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        planner_state: torch.Tensor,
        previous_planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        previous_actions: torch.Tensor,
        observed_kinematics: torch.Tensor,
        depth_indices: torch.Tensor,
        context_tokens: torch.Tensor,
        observed_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observed_states is None:
            raise ValueError(
                "Cross-modal Gate requires observed_states at every causal step"
            )
        batch_size = context_tokens.shape[0]
        depth = self.depth_embedding(depth_indices).unsqueeze(1)
        slots = self.visual_slot_embedding.weight.unsqueeze(0)

        def modality(index: int) -> torch.Tensor:
            return self.fusion_modality_embedding.weight[index].reshape(1, 1, -1)

        visual_groups = (
            context_tokens,
            goal_tokens,
            current_tokens,
            current_tokens - previous_tokens,
        )
        fused_tokens = []
        for index, tokens in enumerate(visual_groups):
            projected = self.visual_projection(tokens) + slots + modality(index)
            if index >= 2:
                projected = projected + depth
            fused_tokens.append(projected)

        action_time = self.action_time_embedding.weight.unsqueeze(0)
        fused_tokens.extend(
            (
                self.action_token_projection(planned_actions)
                + action_time
                + modality(4)
                + depth,
                self.action_token_projection(planned_actions - previous_actions)
                + action_time
                + modality(5)
                + depth,
                self.planner_projection(planner_state).unsqueeze(1)
                + modality(6)
                + depth,
                self.planner_delta_projection(
                    planner_state - previous_planner_state
                ).unsqueeze(1)
                + modality(7)
                + depth,
                self.state_token_projection(observed_states)
                + self.state_time_embedding.weight.unsqueeze(0)
                + modality(8),
                self.kinematic_projection(observed_kinematics).unsqueeze(1)
                + modality(9),
                self.geometry_projection(
                    self._geometry_features(
                        context_tokens, goal_tokens, current_tokens, previous_tokens
                    )
                ).unsqueeze(1)
                + modality(10)
                + depth,
            )
        )
        cls = self.fusion_cls.expand(batch_size, -1, -1) + depth
        fused = self.fusion_norm(
            self.fusion_encoder(torch.cat((cls, *fused_tokens), dim=1))[:, 0]
        )
        return fused, fused

    def forward(
        self,
        planner_states: torch.Tensor,
        planned_actions: torch.Tensor,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        """Return stop logits after joint causal cross-modal fusion."""
        self._validate_static(
            context_tokens, goal_tokens, observed_states, observed_kinematics
        )
        batch_size = context_tokens.shape[0]
        if planner_states.shape != (
            batch_size,
            self.candidate_count,
            self.planner_hidden_dim,
        ):
            raise ValueError("planner_states do not match the cross-modal visual Gate")
        if planned_actions.shape != (
            batch_size,
            self.candidate_count,
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("planned_actions do not match the cross-modal visual Gate")
        if future_tokens.shape != (
            batch_size,
            self.horizon,
            self.token_count,
            self.embed_dim,
        ):
            raise ValueError("future_tokens do not match the cross-modal visual Gate")
        if self.min_rollout_depth != 0:
            raise NotImplementedError(
                "Cross-modal Gate full-sequence training currently requires depth 0"
            )
        current_tokens = torch.cat((context_tokens.unsqueeze(1), future_tokens), dim=1)
        previous_tokens = torch.cat(
            (current_tokens[:, :1], current_tokens[:, :-1]), dim=1
        )
        previous_states = torch.cat(
            (planner_states[:, :1], planner_states[:, :-1]), dim=1
        )
        previous_actions = torch.cat(
            (planned_actions[:, :1], planned_actions[:, :-1]), dim=1
        )
        hidden = self._static_features(observed_states, observed_kinematics)
        predictions = []
        for depth_index in range(self.candidate_count):
            indices = torch.full(
                (batch_size,),
                depth_index,
                dtype=torch.long,
                device=context_tokens.device,
            )
            features, visual_features = self._step_features(
                current_tokens[:, depth_index],
                previous_tokens[:, depth_index],
                goal_tokens,
                planner_states[:, depth_index],
                previous_states[:, depth_index],
                planned_actions[:, depth_index],
                previous_actions[:, depth_index],
                observed_kinematics,
                indices,
                context_tokens,
                observed_states,
            )
            hidden = self.prefix_update(features, hidden)
            predictions.append(self._values(hidden, visual_features, depth_index))
        return torch.stack(predictions, dim=1)

    def forward_step(
        self,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
        planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        current_depth: int,
        previous_tokens: torch.Tensor | None,
        previous_planner_state: torch.Tensor | None,
        previous_actions: torch.Tensor | None,
        prefix_hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the online cross-modal prefix and return current stop logits."""
        self._validate_static(
            context_tokens, goal_tokens, observed_states, observed_kinematics
        )
        batch_size = context_tokens.shape[0]
        if current_tokens.shape != context_tokens.shape:
            raise ValueError("current_tokens must match context_tokens")
        if planner_state.shape != (batch_size, self.planner_hidden_dim):
            raise ValueError("planner_state does not match the cross-modal visual Gate")
        if planned_actions.shape != (batch_size, self.horizon, self.action_dim):
            raise ValueError("planned_actions do not match the cross-modal visual Gate")
        if not self.min_rollout_depth <= current_depth <= self.horizon:
            raise ValueError(
                "current_depth is outside the cross-modal visual Gate range"
            )
        previous_values = (previous_tokens, previous_planner_state, previous_actions)
        if any(value is None for value in previous_values) and not all(
            value is None for value in previous_values
        ):
            raise ValueError(
                "All previous cross-modal Gate inputs must be jointly present or absent"
            )
        if previous_tokens is None:
            previous_tokens = current_tokens
            previous_planner_state = planner_state
            previous_actions = planned_actions
        if prefix_hidden is None:
            prefix_hidden = self._static_features(observed_states, observed_kinematics)
        depth_index = current_depth - self.min_rollout_depth
        indices = torch.full(
            (batch_size,), depth_index, dtype=torch.long, device=context_tokens.device
        )
        features, visual_features = self._step_features(
            current_tokens,
            previous_tokens,
            goal_tokens,
            planner_state,
            previous_planner_state,
            planned_actions,
            previous_actions,
            observed_kinematics,
            indices,
            context_tokens,
            observed_states,
        )
        prefix_hidden = self.prefix_update(features, prefix_hidden)
        return self._values(prefix_hidden, visual_features, depth_index), prefix_hidden


class CrossModalVisualPlannerDualGateHead(CrossModalVisualPlannerStopHead):
    """Decouple plan-cache updates from rollout-continuation decisions."""

    UPDATE = 0
    CONTINUE = 1

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__(
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            min_rollout_depth,
            dropout,
            transformer_layers,
            transformer_heads,
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.hidden_dim * 3),
            torch.nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.hidden_dim, 4),
        )

    def _values(
        self, hidden: torch.Tensor, visual_features: torch.Tensor, depth_index: int
    ) -> torch.Tensor:
        candidate = (
            self.candidate_embedding.weight[depth_index]
            .unsqueeze(0)
            .expand(hidden.shape[0], -1)
        )
        logits = self.value_head(torch.cat((hidden, visual_features, candidate), dim=1))
        return logits.reshape(hidden.shape[0], 2, 2)


class CrossModalVisualPlannerConvergenceDualGateHead(
    CrossModalVisualPlannerDualGateHead
):
    """Add explicit scale-invariant rollout-convergence evidence to the dual Gate."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__(
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            min_rollout_depth,
            dropout,
            transformer_layers,
            transformer_heads,
        )
        self.convergence_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(14),
            torch.nn.Linear(14, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.convergence_fusion = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

    def _step_features(
        self,
        current_tokens: torch.Tensor,
        previous_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        planner_state: torch.Tensor,
        previous_planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        previous_actions: torch.Tensor,
        observed_kinematics: torch.Tensor,
        depth_indices: torch.Tensor,
        context_tokens: torch.Tensor,
        observed_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused, _ = super()._step_features(
            current_tokens,
            previous_tokens,
            goal_tokens,
            planner_state,
            previous_planner_state,
            planned_actions,
            previous_actions,
            observed_kinematics,
            depth_indices,
            context_tokens,
            observed_states,
        )
        planner_delta = planner_state - previous_planner_state
        plan_delta = planned_actions - previous_actions
        magnitudes = torch.stack(
            (
                torch.log1p(
                    (current_tokens - previous_tokens).float().flatten(1).norm(dim=1)
                ),
                torch.log1p(planner_delta.float().norm(dim=1)),
                torch.log1p(plan_delta.float().flatten(1).norm(dim=1)),
                torch.log1p(planned_actions.float().flatten(1).norm(dim=1)),
            ),
            dim=1,
        )
        convergence = torch.cat(
            (
                self._geometry_features(
                    context_tokens, goal_tokens, current_tokens, previous_tokens
                ).float(),
                magnitudes,
            ),
            dim=1,
        ).to(fused.dtype)
        enhanced = self.convergence_fusion(
            torch.cat((fused, self.convergence_projection(convergence)), dim=1)
        )
        return enhanced, enhanced


class CrossModalVisualPlannerHorizonHead(CrossModalVisualPlannerStopHead):
    """Allocate a rollout horizon once from deployable depth-zero inputs."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__(
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            min_rollout_depth,
            dropout,
            transformer_layers,
            transformer_heads,
        )
        self.horizon_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, self.candidate_count),
        )

    def forward(
        self,
        planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for all rollout depths without reading future rollout tokens."""
        return self.horizon_head(
            self._horizon_condition(
                planner_state,
                planned_actions,
                context_tokens,
                goal_tokens,
                observed_states,
                observed_kinematics,
            )
        )

    def _horizon_condition(
        self,
        planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        """Build the shared causal condition used to compare rollout depths."""
        self._validate_static(
            context_tokens, goal_tokens, observed_states, observed_kinematics
        )
        batch_size = context_tokens.shape[0]
        if planner_state.shape != (batch_size, self.planner_hidden_dim):
            raise ValueError(
                "planner_state does not match the cross-modal horizon Gate"
            )
        if planned_actions.shape != (batch_size, self.horizon, self.action_dim):
            raise ValueError(
                "planned_actions do not match the cross-modal horizon Gate"
            )
        depth_indices = torch.zeros(
            batch_size, dtype=torch.long, device=context_tokens.device
        )
        features, fused = self._step_features(
            context_tokens,
            context_tokens,
            goal_tokens,
            planner_state,
            planner_state,
            planned_actions,
            planned_actions,
            observed_kinematics,
            depth_indices,
            context_tokens,
            observed_states,
        )
        hidden = self.prefix_update(
            features, self._static_features(observed_states, observed_kinematics)
        )
        return torch.cat((hidden, fused), dim=1)


class CandidateCostHorizonHead(torch.nn.Module):
    """Score rollout depths with shared condition features and explicit depth queries."""

    def __init__(
        self,
        hidden_dim: int,
        candidate_count: int,
        dropout: float,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__()
        if min(hidden_dim, candidate_count, transformer_layers, transformer_heads) < 1:
            raise ValueError("Candidate cost-head dimensions must be positive")
        if hidden_dim % transformer_heads:
            raise ValueError("Candidate cost-head heads must evenly divide hidden_dim")
        self.candidate_count = int(candidate_count)
        self.condition_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.depth_queries = torch.nn.Embedding(candidate_count, hidden_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.depth_encoder = torch.nn.TransformerEncoder(
            layer, num_layers=transformer_layers
        )
        self.utility_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError("Candidate cost-head condition must be [B,D]")
        shared = self.condition_projection(condition).unsqueeze(1)
        queries = self.depth_queries.weight.unsqueeze(0).expand(
            condition.shape[0], -1, -1
        )
        return self.utility_head(self.depth_encoder(shared + queries)).squeeze(-1)


class CandidateMarginHorizonHead(torch.nn.Module):
    """Predict optimality logits and margins for every cached depth query."""

    def __init__(
        self,
        hidden_dim: int,
        candidate_count: int,
        dropout: float,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__()
        if min(hidden_dim, candidate_count, transformer_layers, transformer_heads) < 1:
            raise ValueError("Candidate margin-head dimensions must be positive")
        if hidden_dim % transformer_heads:
            raise ValueError(
                "Candidate margin-head heads must evenly divide hidden_dim"
            )
        self.condition_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.depth_queries = torch.nn.Embedding(candidate_count, hidden_dim)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.depth_encoder = torch.nn.TransformerEncoder(
            layer, num_layers=transformer_layers
        )
        self.margin_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError("Candidate margin-head condition must be [B,D]")
        shared = self.condition_projection(condition).unsqueeze(1)
        queries = self.depth_queries.weight.unsqueeze(0).expand(
            condition.shape[0], -1, -1
        )
        return self.margin_head(self.depth_encoder(shared + queries))


class RelationalCandidateMarginHorizonHead(torch.nn.Module):
    """Compare cached plan features under the current causal prefix evidence."""

    def __init__(
        self,
        hidden_dim: int,
        candidate_count: int,
        dropout: float,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
        use_depth_queries: bool = True,
    ) -> None:
        super().__init__()
        if min(hidden_dim, candidate_count, transformer_layers, transformer_heads) < 1:
            raise ValueError("Relational margin-head dimensions must be positive")
        if hidden_dim % transformer_heads:
            raise ValueError(
                "Relational margin-head heads must evenly divide hidden_dim"
            )
        self.candidate_count = int(candidate_count)
        self.condition_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.candidate_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.interaction_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 4),
            torch.nn.Linear(hidden_dim * 4, hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.depth_queries = (
            torch.nn.Embedding(candidate_count, hidden_dim)
            if use_depth_queries
            else None
        )
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder = torch.nn.TransformerEncoder(
            layer, num_layers=transformer_layers
        )
        self.margin_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        condition: torch.Tensor,
        candidate_features: torch.Tensor,
        visible_count: int,
    ) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError("Relational margin-head condition must be [B,2D]")
        expected = (condition.shape[0], self.candidate_count, condition.shape[1] // 2)
        if candidate_features.shape != expected:
            raise ValueError(f"Relational candidate features must be {expected}")
        if not 1 <= visible_count <= self.candidate_count:
            raise ValueError("visible_count must be in [1,C]")
        shared = (
            self.condition_projection(condition)
            .unsqueeze(1)
            .expand(-1, self.candidate_count, -1)
        )
        candidates = self.candidate_projection(candidate_features)
        interactions = self.interaction_projection(
            torch.cat(
                (shared, candidates, candidates - shared, candidates * shared), dim=2
            )
        )
        tokens = interactions + candidates
        if self.depth_queries is not None:
            tokens = tokens + self.depth_queries.weight.unsqueeze(0)
        padding_mask = (
            torch.arange(self.candidate_count, device=condition.device).unsqueeze(0)
            >= visible_count
        )
        padding_mask = padding_mask.expand(condition.shape[0], -1)
        encoded = self.candidate_encoder(tokens, src_key_padding_mask=padding_mask)
        return self.margin_head(encoded)


class CrossModalVisualPlannerCostHorizonHead(CrossModalVisualPlannerHorizonHead):
    """Allocate rollout by comparing candidate-wise predicted utilities."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        candidate_layers: int = 2,
    ) -> None:
        super().__init__(
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            min_rollout_depth,
            dropout,
            transformer_layers,
            transformer_heads,
        )
        self.horizon_head = CandidateCostHorizonHead(
            hidden_dim,
            self.candidate_count,
            dropout,
            transformer_layers=candidate_layers,
            transformer_heads=transformer_heads,
        )


class CrossModalVisualPlannerSequentialHorizonHead(CrossModalVisualPlannerStopHead):
    """Predict the globally preferred horizon from every causal rollout prefix."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        candidate_layers: int = 2,
    ) -> None:
        super().__init__(
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            min_rollout_depth,
            dropout,
            transformer_layers,
            transformer_heads,
        )
        if min_rollout_depth != 0:
            raise NotImplementedError(
                "Sequential horizon Gate currently requires depth zero"
            )
        self.horizon_head = CandidateCostHorizonHead(
            hidden_dim,
            self.candidate_count,
            dropout,
            transformer_layers=candidate_layers,
            transformer_heads=transformer_heads,
        )

    def forward(
        self,
        planner_states: torch.Tensor,
        planned_actions: torch.Tensor,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        """Return global-depth utilities from every causal prefix as ``[B,C,C]``."""
        self._validate_static(
            context_tokens, goal_tokens, observed_states, observed_kinematics
        )
        batch_size = context_tokens.shape[0]
        if planner_states.shape != (
            batch_size,
            self.candidate_count,
            self.planner_hidden_dim,
        ):
            raise ValueError("planner_states do not match the sequential horizon Gate")
        if planned_actions.shape != (
            batch_size,
            self.candidate_count,
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("planned_actions do not match the sequential horizon Gate")
        if future_tokens.shape != (
            batch_size,
            self.horizon,
            self.token_count,
            self.embed_dim,
        ):
            raise ValueError("future_tokens do not match the sequential horizon Gate")

        current_tokens = torch.cat((context_tokens.unsqueeze(1), future_tokens), dim=1)
        previous_tokens = torch.cat(
            (current_tokens[:, :1], current_tokens[:, :-1]), dim=1
        )
        previous_states = torch.cat(
            (planner_states[:, :1], planner_states[:, :-1]), dim=1
        )
        previous_actions = torch.cat(
            (planned_actions[:, :1], planned_actions[:, :-1]), dim=1
        )
        hidden = self._static_features(observed_states, observed_kinematics)
        predictions = []
        for depth_index in range(self.candidate_count):
            indices = torch.full(
                (batch_size,),
                depth_index,
                dtype=torch.long,
                device=context_tokens.device,
            )
            features, fused = self._step_features(
                current_tokens[:, depth_index],
                previous_tokens[:, depth_index],
                goal_tokens,
                planner_states[:, depth_index],
                previous_states[:, depth_index],
                planned_actions[:, depth_index],
                previous_actions[:, depth_index],
                observed_kinematics,
                indices,
                context_tokens,
                observed_states,
            )
            hidden = self.prefix_update(features, hidden)
            predictions.append(self.horizon_head(torch.cat((hidden, fused), dim=1)))
        return torch.stack(predictions, dim=1)

    def forward_step(
        self,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
        planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        current_depth: int,
        previous_tokens: torch.Tensor | None,
        previous_planner_state: torch.Tensor | None,
        previous_actions: torch.Tensor | None,
        prefix_hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update one causal prefix and predict the preferred global depth."""
        self._validate_static(
            context_tokens, goal_tokens, observed_states, observed_kinematics
        )
        batch_size = context_tokens.shape[0]
        if current_tokens.shape != context_tokens.shape:
            raise ValueError("current_tokens must match context_tokens")
        if planner_state.shape != (batch_size, self.planner_hidden_dim):
            raise ValueError("planner_state does not match the sequential horizon Gate")
        if planned_actions.shape != (batch_size, self.horizon, self.action_dim):
            raise ValueError("planned_actions do not match the sequential horizon Gate")
        if not 0 <= current_depth <= self.horizon:
            raise ValueError(
                "current_depth is outside the sequential horizon Gate range"
            )
        previous_values = (previous_tokens, previous_planner_state, previous_actions)
        if any(value is None for value in previous_values) and not all(
            value is None for value in previous_values
        ):
            raise ValueError(
                "All previous sequential Gate inputs must be jointly present or absent"
            )
        if previous_tokens is None:
            previous_tokens = current_tokens
            previous_planner_state = planner_state
            previous_actions = planned_actions
        if prefix_hidden is None:
            prefix_hidden = self._static_features(observed_states, observed_kinematics)
        indices = torch.full(
            (batch_size,), current_depth, dtype=torch.long, device=context_tokens.device
        )
        features, fused = self._step_features(
            current_tokens,
            previous_tokens,
            goal_tokens,
            planner_state,
            previous_planner_state,
            planned_actions,
            previous_actions,
            observed_kinematics,
            indices,
            context_tokens,
            observed_states,
        )
        prefix_hidden = self.prefix_update(features, prefix_hidden)
        return (
            self.horizon_head(torch.cat((prefix_hidden, fused), dim=1)),
            prefix_hidden,
        )


class CrossModalVisualPlannerDiagonalMarginHead(
    CrossModalVisualPlannerSequentialHorizonHead
):
    """Estimate whether each causal rollout depth is globally cost-optimal."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__(
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            min_rollout_depth,
            dropout,
            transformer_layers,
            transformer_heads,
            candidate_layers=1,
        )
        self.horizon_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        predictions = super().forward(*args)
        if predictions.shape[-1] != 2:
            raise RuntimeError(
                "Diagonal margin head must return classification and regression values"
            )
        return predictions


class CrossModalVisualPlannerCachedMarginHead(
    CrossModalVisualPlannerSequentialHorizonHead
):
    """Re-score every cached plan from each newly observed causal rollout prefix."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        candidate_layers: int = 2,
    ) -> None:
        super().__init__(
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            min_rollout_depth,
            dropout,
            transformer_layers,
            transformer_heads,
            candidate_layers=1,
        )
        self.horizon_head = CandidateMarginHorizonHead(
            hidden_dim,
            self.candidate_count,
            dropout,
            transformer_layers=candidate_layers,
            transformer_heads=transformer_heads,
        )


class CrossModalVisualPlannerRelationalCachedMarginHead(
    CrossModalVisualPlannerSequentialHorizonHead
):
    """Score each cached plan from its own features and the latest causal prefix."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        candidate_layers: int = 3,
    ) -> None:
        super().__init__(
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            min_rollout_depth,
            dropout,
            transformer_layers,
            transformer_heads,
            candidate_layers=1,
        )
        self.horizon_head = RelationalCandidateMarginHorizonHead(
            hidden_dim,
            self.candidate_count,
            dropout,
            transformer_layers=candidate_layers,
            transformer_heads=transformer_heads,
        )

    def forward(
        self,
        planner_states: torch.Tensor,
        planned_actions: torch.Tensor,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        """Return cached-plan margins from every causal prefix as ``[B,C,C,2]``."""
        self._validate_static(
            context_tokens, goal_tokens, observed_states, observed_kinematics
        )
        batch_size = context_tokens.shape[0]
        if planner_states.shape != (
            batch_size,
            self.candidate_count,
            self.planner_hidden_dim,
        ):
            raise ValueError(
                "planner_states do not match the relational cached-margin Gate"
            )
        if planned_actions.shape != (
            batch_size,
            self.candidate_count,
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                "planned_actions do not match the relational cached-margin Gate"
            )
        if future_tokens.shape != (
            batch_size,
            self.horizon,
            self.token_count,
            self.embed_dim,
        ):
            raise ValueError(
                "future_tokens do not match the relational cached-margin Gate"
            )

        current_tokens = torch.cat((context_tokens.unsqueeze(1), future_tokens), dim=1)
        previous_tokens = torch.cat(
            (current_tokens[:, :1], current_tokens[:, :-1]), dim=1
        )
        previous_states = torch.cat(
            (planner_states[:, :1], planner_states[:, :-1]), dim=1
        )
        previous_actions = torch.cat(
            (planned_actions[:, :1], planned_actions[:, :-1]), dim=1
        )
        hidden = self._static_features(observed_states, observed_kinematics)
        cached_features = []
        predictions = []
        for depth_index in range(self.candidate_count):
            indices = torch.full(
                (batch_size,),
                depth_index,
                dtype=torch.long,
                device=context_tokens.device,
            )
            features, fused = self._step_features(
                current_tokens[:, depth_index],
                previous_tokens[:, depth_index],
                goal_tokens,
                planner_states[:, depth_index],
                previous_states[:, depth_index],
                planned_actions[:, depth_index],
                previous_actions[:, depth_index],
                observed_kinematics,
                indices,
                context_tokens,
                observed_states,
            )
            hidden = self.prefix_update(features, hidden)
            cached_features.append(fused)
            visible = torch.stack(cached_features, dim=1)
            padding = visible.new_zeros(
                batch_size, self.candidate_count - visible.shape[1], self.hidden_dim
            )
            candidates = torch.cat((visible, padding), dim=1)
            predictions.append(
                self.horizon_head(
                    torch.cat((hidden, fused), dim=1),
                    candidates,
                    visible_count=depth_index + 1,
                )
            )
        return torch.stack(predictions, dim=1)


class InvariantRelationalCachedMarginHead(torch.nn.Module):
    """Compare rollout candidates from scene-offset-invariant dynamic relations."""

    SCALAR_DIM = 7

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        candidate_layers: int = 3,
    ) -> None:
        super().__init__()
        dimensions = (
            embed_dim,
            hidden_dim,
            planner_hidden_dim,
            horizon,
            action_dim,
            observed_frames,
            state_dim,
            kinematic_dim,
            token_count,
            transformer_layers,
            transformer_heads,
            candidate_layers,
        )
        if min(dimensions) < 1 or horizon < 2:
            raise ValueError("Invariant relational Gate dimensions are invalid")
        if min_rollout_depth != 0:
            raise NotImplementedError(
                "Invariant relational Gate currently requires depth zero"
            )
        if hidden_dim % transformer_heads:
            raise ValueError(
                "Invariant relational Gate heads must evenly divide hidden_dim"
            )
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.planner_hidden_dim = int(planner_hidden_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.observed_frames = int(observed_frames)
        self.state_dim = int(state_dim)
        self.kinematic_dim = int(kinematic_dim)
        self.token_count = int(token_count)
        self.candidate_count = self.horizon + 1

        self.static_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(
                embed_dim * 2 + observed_frames * state_dim + kinematic_dim
            ),
            torch.nn.Linear(
                embed_dim * 2 + observed_frames * state_dim + kinematic_dim, hidden_dim
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.visual_relation_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim * 4),
            torch.nn.Linear(embed_dim * 4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.planner_relation_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(planner_hidden_dim * 2),
            torch.nn.Linear(planner_hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        flattened_action_dim = horizon * action_dim
        self.action_relation_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(flattened_action_dim * 3),
            torch.nn.Linear(flattened_action_dim * 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.scalar_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(self.SCALAR_DIM),
            torch.nn.Linear(self.SCALAR_DIM, hidden_dim),
            torch.nn.GELU(),
        )
        self.candidate_fusion = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 4),
            torch.nn.Linear(hidden_dim * 4, hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.prefix_model = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.horizon_head = RelationalCandidateMarginHorizonHead(
            hidden_dim,
            self.candidate_count,
            dropout,
            transformer_layers=candidate_layers,
            transformer_heads=transformer_heads,
            use_depth_queries=False,
        )

    def forward(
        self,
        planner_states: torch.Tensor,
        planned_actions: torch.Tensor,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        """Return invariant cached-plan scores from every causal prefix as ``[B,C,C,2]``."""
        batch_size = context_tokens.shape[0]
        token_shape = (batch_size, self.token_count, self.embed_dim)
        expected = {
            "planner_states": (
                batch_size,
                self.candidate_count,
                self.planner_hidden_dim,
            ),
            "planned_actions": (
                batch_size,
                self.candidate_count,
                self.horizon,
                self.action_dim,
            ),
            "context_tokens": token_shape,
            "goal_tokens": token_shape,
            "future_tokens": (
                batch_size,
                self.horizon,
                self.token_count,
                self.embed_dim,
            ),
            "observed_states": (batch_size, self.observed_frames, self.state_dim),
            "observed_kinematics": (batch_size, self.kinematic_dim),
        }
        values = {
            "planner_states": planner_states,
            "planned_actions": planned_actions,
            "context_tokens": context_tokens,
            "goal_tokens": goal_tokens,
            "future_tokens": future_tokens,
            "observed_states": observed_states,
            "observed_kinematics": observed_kinematics,
        }
        mismatched = [
            name for name, shape in expected.items() if values[name].shape != shape
        ]
        if mismatched:
            raise ValueError(f"Invariant relational Gate input mismatch: {mismatched}")
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise ValueError("Invariant relational Gate inputs must be finite")

        current = torch.cat((context_tokens.unsqueeze(1), future_tokens), dim=1)
        previous = torch.cat((current[:, :1], current[:, :-1]), dim=1)
        goal = goal_tokens.unsqueeze(1).expand_as(current)
        goal_delta = goal - current
        step_delta = current - previous
        visual_relations = torch.cat(
            (
                goal_delta.mean(dim=2),
                goal_delta.abs().mean(dim=2),
                step_delta.mean(dim=2),
                step_delta.abs().mean(dim=2),
            ),
            dim=2,
        )

        previous_planner = torch.cat(
            (planner_states[:, :1], planner_states[:, :-1]), dim=1
        )
        initial_planner = planner_states[:, :1].expand_as(planner_states)
        planner_relations = torch.cat(
            (planner_states - previous_planner, planner_states - initial_planner),
            dim=2,
        )
        previous_actions = torch.cat(
            (planned_actions[:, :1], planned_actions[:, :-1]), dim=1
        )
        initial_actions = planned_actions[:, :1].expand_as(planned_actions)
        action_relations = torch.cat(
            (
                planned_actions.flatten(2),
                (planned_actions - previous_actions).flatten(2),
                (planned_actions - initial_actions).flatten(2),
            ),
            dim=2,
        )
        depth = torch.linspace(
            0.0, 1.0, self.candidate_count, device=current.device, dtype=current.dtype
        )
        depth = depth.unsqueeze(0).expand(batch_size, -1)
        scalars = torch.stack(
            (
                _rms(goal_delta, dim=(2, 3)),
                _rms(step_delta, dim=(2, 3)),
                _rms(planned_actions - previous_actions, dim=(2, 3)),
                _rms(planned_actions, dim=(2, 3)),
                _rms(planner_states - previous_planner, dim=2),
                _rms(planner_states - initial_planner, dim=2),
                depth,
            ),
            dim=2,
        )
        candidate_features = self.candidate_fusion(
            torch.cat(
                (
                    self.visual_relation_projection(visual_relations),
                    self.planner_relation_projection(planner_relations),
                    self.action_relation_projection(action_relations),
                    self.scalar_projection(torch.log1p(scalars)),
                ),
                dim=2,
            )
        )

        context_goal = goal_tokens - context_tokens
        centered_states = observed_states - observed_states[:, -1:]
        static = self.static_projection(
            torch.cat(
                (
                    context_goal.mean(dim=1),
                    context_goal.abs().mean(dim=1),
                    centered_states.flatten(1),
                    observed_kinematics,
                ),
                dim=1,
            )
        )
        prefix_states, _ = self.prefix_model(candidate_features, static.unsqueeze(0))
        predictions = []
        for prefix in range(self.candidate_count):
            visible = candidate_features[:, : prefix + 1]
            padding = visible.new_zeros(
                batch_size, self.candidate_count - prefix - 1, self.hidden_dim
            )
            candidates = torch.cat((visible, padding), dim=1)
            condition = torch.cat(
                (prefix_states[:, prefix], candidate_features[:, prefix]), dim=1
            )
            predictions.append(
                self.horizon_head(condition, candidates, visible_count=prefix + 1)
            )
        return torch.stack(predictions, dim=1)


class InvariantVisualPlannerStopHead(torch.nn.Module):
    """Causal stopping Gate built from cross-scene invariant rollout relations."""

    CONTINUE = 0
    STOP = 1
    VISUAL_RELATION_DIM = 14
    PLANNER_RELATION_DIM = 3

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        planner_hidden_dim: int,
        horizon: int,
        action_dim: int,
        observed_frames: int,
        state_dim: int,
        kinematic_dim: int,
        token_count: int,
        min_rollout_depth: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if (
            min(
                embed_dim,
                hidden_dim,
                planner_hidden_dim,
                horizon,
                action_dim,
                observed_frames,
                state_dim,
            )
            < 1
        ):
            raise ValueError("Invariant visual Gate dimensions must be positive")
        if token_count < 1 or kinematic_dim < 1:
            raise ValueError(
                "Invariant visual Gate token and kinematic dimensions must be positive"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.planner_hidden_dim = int(planner_hidden_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.observed_frames = int(observed_frames)
        self.state_dim = int(state_dim)
        self.kinematic_dim = int(kinematic_dim)
        self.token_count = int(token_count)
        self.min_rollout_depth = int(min_rollout_depth)
        self.candidate_count = self.horizon - self.min_rollout_depth + 1
        if self.candidate_count < 2:
            raise ValueError(
                "Invariant visual Gate needs at least two candidate depths"
            )
        plan_dim = self.horizon * self.action_dim
        self.visual_relation_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(self.VISUAL_RELATION_DIM),
            torch.nn.Linear(self.VISUAL_RELATION_DIM, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.planner_relation_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(self.PLANNER_RELATION_DIM),
            torch.nn.Linear(self.PLANNER_RELATION_DIM, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.plan_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(plan_dim),
            torch.nn.Linear(plan_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.plan_delta_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(plan_dim),
            torch.nn.Linear(plan_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.history_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(observed_frames * state_dim),
            torch.nn.Linear(observed_frames * state_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.kinematic_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(kinematic_dim),
            torch.nn.Linear(kinematic_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.depth_embedding = torch.nn.Embedding(self.candidate_count, hidden_dim)
        self.step_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 6),
            torch.nn.Linear(hidden_dim * 6, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.prefix_update = torch.nn.GRUCell(hidden_dim, hidden_dim)
        self.stop_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 3),
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 2),
        )

    def _validate_static(
        self,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> None:
        batch_size = context_tokens.shape[0]
        token_shape = (batch_size, self.token_count, self.embed_dim)
        if context_tokens.shape != token_shape or goal_tokens.shape != token_shape:
            raise ValueError(f"context_tokens and goal_tokens must be {token_shape}")
        if observed_states.shape != (batch_size, self.observed_frames, self.state_dim):
            raise ValueError(
                f"observed_states must be [B,{self.observed_frames},{self.state_dim}]"
            )
        if observed_kinematics.shape != (batch_size, self.kinematic_dim):
            raise ValueError(f"observed_kinematics must be [B,{self.kinematic_dim}]")

    @staticmethod
    def _visual_relations(
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        previous_tokens: torch.Tensor,
    ) -> torch.Tensor:
        pairs = (
            (context_tokens, goal_tokens),
            (current_tokens, goal_tokens),
            (current_tokens, context_tokens),
            (current_tokens, previous_tokens),
        )
        rms_distances = []
        cosine_distances = []
        absolute_distances = []
        for source, target in pairs:
            delta = source.float() - target.float()
            rms_distances.append(torch.log1p(delta.square().mean(dim=(1, 2)).sqrt()))
            cosine = torch.nn.functional.cosine_similarity(
                source.float(), target.float(), dim=2
            ).mean(dim=1)
            cosine_distances.append(1.0 - cosine)
            absolute_distances.append(torch.log1p(delta.abs().mean(dim=(1, 2))))
        initial_goal = rms_distances[0].expm1().clamp_min(1.0e-6)
        current_goal = rms_distances[1].expm1()
        current_step = rms_distances[3].expm1()
        ratios = (
            torch.log1p(current_goal / initial_goal),
            torch.log1p(current_step / initial_goal),
        )
        return torch.stack(
            (*rms_distances, *cosine_distances, *absolute_distances, *ratios), dim=1
        ).to(context_tokens.dtype)

    @staticmethod
    def _planner_relations(
        planner_state: torch.Tensor, previous_planner_state: torch.Tensor
    ) -> torch.Tensor:
        current = planner_state.float()
        previous = previous_planner_state.float()
        delta = current - previous
        cosine = torch.nn.functional.cosine_similarity(current, previous, dim=1)
        return torch.stack(
            (
                torch.log1p(current.square().mean(dim=1).sqrt()),
                torch.log1p(delta.square().mean(dim=1).sqrt()),
                1.0 - cosine,
            ),
            dim=1,
        ).to(planner_state.dtype)

    def _initial_hidden(
        self,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        history = self.history_projection(observed_states.flatten(1))
        kinematics = self.kinematic_projection(observed_kinematics)
        return torch.tanh(history + kinematics)

    def _step(
        self,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        previous_tokens: torch.Tensor,
        planner_state: torch.Tensor,
        previous_planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        previous_actions: torch.Tensor,
        observed_kinematics: torch.Tensor,
        depth_index: int,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context_tokens.shape[0]
        depth_indices = torch.full(
            (batch_size,), depth_index, dtype=torch.long, device=context_tokens.device
        )
        visual = self.visual_relation_projection(
            self._visual_relations(
                context_tokens, goal_tokens, current_tokens, previous_tokens
            )
        )
        planner = self.planner_relation_projection(
            self._planner_relations(planner_state, previous_planner_state)
        )
        plan = self.plan_projection(planned_actions.flatten(1))
        plan_delta = self.plan_delta_projection(
            (planned_actions - previous_actions).flatten(1)
        )
        kinematics = self.kinematic_projection(observed_kinematics)
        depth = self.depth_embedding(depth_indices)
        features = self.step_projection(
            torch.cat((visual, planner, plan, plan_delta, kinematics, depth), dim=1)
        )
        hidden = self.prefix_update(features, hidden)
        return self.stop_head(torch.cat((hidden, visual, depth), dim=1)), hidden

    def forward(
        self,
        planner_states: torch.Tensor,
        planned_actions: torch.Tensor,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
    ) -> torch.Tensor:
        """Return STOP/CONTINUE logits for every causal rollout prefix."""
        self._validate_static(
            context_tokens, goal_tokens, observed_states, observed_kinematics
        )
        batch_size = context_tokens.shape[0]
        if planner_states.shape != (
            batch_size,
            self.candidate_count,
            self.planner_hidden_dim,
        ):
            raise ValueError("planner_states do not match the invariant visual Gate")
        if planned_actions.shape != (
            batch_size,
            self.candidate_count,
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("planned_actions do not match the invariant visual Gate")
        if future_tokens.shape != (
            batch_size,
            self.horizon,
            self.token_count,
            self.embed_dim,
        ):
            raise ValueError("future_tokens do not match the invariant visual Gate")
        if self.min_rollout_depth != 0:
            raise NotImplementedError(
                "Invariant visual Gate full-sequence training currently requires depth 0"
            )
        current_tokens = torch.cat((context_tokens.unsqueeze(1), future_tokens), dim=1)
        previous_tokens = torch.cat(
            (current_tokens[:, :1], current_tokens[:, :-1]), dim=1
        )
        previous_states = torch.cat(
            (planner_states[:, :1], planner_states[:, :-1]), dim=1
        )
        previous_actions = torch.cat(
            (planned_actions[:, :1], planned_actions[:, :-1]), dim=1
        )
        hidden = self._initial_hidden(observed_states, observed_kinematics)
        logits = []
        for depth_index in range(self.candidate_count):
            current_logits, hidden = self._step(
                context_tokens,
                goal_tokens,
                current_tokens[:, depth_index],
                previous_tokens[:, depth_index],
                planner_states[:, depth_index],
                previous_states[:, depth_index],
                planned_actions[:, depth_index],
                previous_actions[:, depth_index],
                observed_kinematics,
                depth_index,
                hidden,
            )
            logits.append(current_logits)
        return torch.stack(logits, dim=1)

    def forward_step(
        self,
        context_tokens: torch.Tensor,
        goal_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        observed_states: torch.Tensor,
        observed_kinematics: torch.Tensor,
        planner_state: torch.Tensor,
        planned_actions: torch.Tensor,
        current_depth: int,
        previous_tokens: torch.Tensor | None,
        previous_planner_state: torch.Tensor | None,
        previous_actions: torch.Tensor | None,
        prefix_hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update one deployable invariant rollout prefix."""
        self._validate_static(
            context_tokens, goal_tokens, observed_states, observed_kinematics
        )
        batch_size = context_tokens.shape[0]
        if current_tokens.shape != context_tokens.shape:
            raise ValueError("current_tokens must match context_tokens")
        if planner_state.shape != (batch_size, self.planner_hidden_dim):
            raise ValueError("planner_state does not match the invariant visual Gate")
        if planned_actions.shape != (batch_size, self.horizon, self.action_dim):
            raise ValueError("planned_actions do not match the invariant visual Gate")
        if not self.min_rollout_depth <= current_depth <= self.horizon:
            raise ValueError("current_depth is outside the invariant visual Gate range")
        previous_values = (previous_tokens, previous_planner_state, previous_actions)
        if any(value is None for value in previous_values) and not all(
            value is None for value in previous_values
        ):
            raise ValueError(
                "All previous invariant Gate inputs must be jointly present or absent"
            )
        if previous_tokens is None:
            previous_tokens = current_tokens
            previous_planner_state = planner_state
            previous_actions = planned_actions
        if prefix_hidden is None:
            prefix_hidden = self._initial_hidden(observed_states, observed_kinematics)
        return self._step(
            context_tokens,
            goal_tokens,
            current_tokens,
            previous_tokens,
            planner_state,
            previous_planner_state,
            planned_actions,
            previous_actions,
            observed_kinematics,
            current_depth - self.min_rollout_depth,
            prefix_hidden,
        )


class ContinuationPolicy(torch.nn.Module):
    """Score candidate-plan reward from one deployable rollout prefix."""

    LOW_REWARD = 0
    HIGH_REWARD = 1

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        horizon: int,
        action_dim: int,
        kinematic_dim: int,
        token_count: int,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.kinematic_dim = int(kinematic_dim)
        self.token_count = int(token_count)
        if self.token_count < 1:
            raise ValueError("token_count must be positive")
        self.feature_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, hidden_dim),
        )
        self.plan_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(horizon * action_dim),
            torch.nn.Linear(horizon * action_dim, hidden_dim),
        )
        self.kinematic_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(kinematic_dim),
            torch.nn.Linear(kinematic_dim, hidden_dim),
        )
        self.magnitude_projection = torch.nn.Linear(2, hidden_dim)
        self.spatial_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, hidden_dim),
        )
        self.spatial_attention = torch.nn.Linear(hidden_dim, 1)
        self.depth_embedding = torch.nn.Embedding(horizon + 1, hidden_dim)
        self.step_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 7),
            torch.nn.Linear(hidden_dim * 7, hidden_dim),
            torch.nn.GELU(),
        )
        self.prefix_model = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.policy = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 5),
            torch.nn.Linear(hidden_dim * 5, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        context_last: torch.Tensor,
        context_tokens: torch.Tensor,
        latent_history: torch.Tensor,
        spatial_history: torch.Tensor,
        plan_history: torch.Tensor,
        observed_kinematics: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate reward from rollout steps no later than ``depths``."""
        if context_last.ndim != 2 or latent_history.ndim != 3:
            raise ValueError(
                "context_last and latent_history must be [B,D] and [B,S,D]"
            )
        batch_size = context_last.shape[0]
        if (
            latent_history.shape[0] != batch_size
            or latent_history.shape[2] != context_last.shape[1]
        ):
            raise ValueError("latent_history must align with context_last")
        available_steps = latent_history.shape[1]
        _validate_depths(depths, batch_size, available_steps, self.horizon)
        if context_tokens.shape != (
            batch_size,
            self.token_count,
            context_last.shape[1],
        ):
            raise ValueError(
                "context_tokens must be [B,Q,D] aligned with the policy token count"
            )
        if spatial_history.shape != (
            batch_size,
            available_steps,
            self.token_count,
            context_last.shape[1],
        ):
            raise ValueError(
                "spatial_history must be [B,S,Q,D] aligned with latent_history"
            )
        expected_plan_shape = (
            batch_size,
            available_steps,
            self.horizon,
            self.action_dim,
        )
        if plan_history.shape != expected_plan_shape:
            raise ValueError(
                f"plan_history must be {expected_plan_shape}, got {tuple(plan_history.shape)}"
            )
        if observed_kinematics.shape != (batch_size, self.kinematic_dim):
            raise ValueError(
                f"observed_kinematics must be [B,{self.kinematic_dim}], got {tuple(observed_kinematics.shape)}"
            )
        previous_latents = torch.cat(
            [context_last.unsqueeze(1), latent_history[:, :-1]], dim=1
        )
        previous_plans = torch.cat(
            [torch.zeros_like(plan_history[:, :1]), plan_history[:, :-1]], dim=1
        )
        latent_delta = latent_history - previous_latents
        plan_delta = plan_history - previous_plans
        context_spatial = self._pool_spatial(context_tokens)
        spatial_features = self._pool_spatial(spatial_history)
        previous_spatial = torch.cat(
            [context_spatial.unsqueeze(1), spatial_features[:, :-1]], dim=1
        )
        spatial_delta = spatial_features - previous_spatial
        magnitudes = torch.stack(
            [
                latent_delta.square().mean(dim=2).sqrt(),
                plan_delta.square().mean(dim=(2, 3)).sqrt(),
            ],
            dim=2,
        )
        step_features = torch.cat(
            [
                self.feature_projection(latent_history),
                self.feature_projection(latent_delta),
                spatial_features,
                spatial_delta,
                self.plan_projection(plan_history.flatten(2)),
                self.plan_projection(plan_delta.flatten(2)),
                self.magnitude_projection(torch.log1p(magnitudes)),
            ],
            dim=-1,
        )
        valid_steps = torch.arange(available_steps, device=depths.device).unsqueeze(
            0
        ) < depths.unsqueeze(1)
        step_features = self.step_projection(step_features) * valid_steps.unsqueeze(2)
        context_features = self.feature_projection(context_last)
        kinematic_features = self.kinematic_projection(observed_kinematics)
        initial_state = torch.tanh(context_features + kinematic_features).unsqueeze(0)
        encoded_prefix, _ = self.prefix_model(step_features, initial_state)
        rows = torch.arange(batch_size, device=depths.device)
        prefix_state = encoded_prefix[rows, depths - 1]
        features = torch.cat(
            [
                prefix_state,
                context_features,
                context_spatial,
                kinematic_features,
                self.depth_embedding(depths),
            ],
            dim=-1,
        )
        return self.policy(features)

    def _pool_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.spatial_projection(tokens)
        weights = torch.softmax(self.spatial_attention(projected), dim=-2)
        return (projected * weights).sum(dim=-2)


class GoalConditionedContinuationPolicy(torch.nn.Module):
    """Rank rollout prefixes using only goal and online-available state."""

    CONTINUE = 0
    STOP = 1

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        horizon: int,
        action_dim: int,
        kinematic_dim: int,
        token_count: int,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        if horizon < 2 or token_count < 1:
            raise ValueError(
                "horizon must be at least 2 and token_count must be positive"
            )
        if output_dim < 2:
            raise ValueError("output_dim must be at least 2")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.kinematic_dim = int(kinematic_dim)
        self.token_count = int(token_count)
        self.output_dim = int(output_dim)
        self.feature_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, hidden_dim),
        )
        self.relation_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim * 4),
            torch.nn.Linear(embed_dim * 4, hidden_dim),
            torch.nn.GELU(),
        )
        self.spatial_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, hidden_dim),
        )
        self.spatial_attention = torch.nn.Linear(hidden_dim, 1)
        self.plan_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(horizon * action_dim),
            torch.nn.Linear(horizon * action_dim, hidden_dim),
        )
        self.kinematic_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(kinematic_dim),
            torch.nn.Linear(kinematic_dim, hidden_dim),
        )
        self.dynamics_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(6),
            torch.nn.Linear(6, hidden_dim),
            torch.nn.GELU(),
        )
        self.depth_embedding = torch.nn.Embedding(horizon + 1, hidden_dim)
        self.step_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 6),
            torch.nn.Linear(hidden_dim * 6, hidden_dim),
            torch.nn.GELU(),
        )
        self.prefix_model = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.policy = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 5),
            torch.nn.Linear(hidden_dim * 5, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        context_last: torch.Tensor,
        context_tokens: torch.Tensor,
        goal_last: torch.Tensor,
        goal_tokens: torch.Tensor,
        latent_history: torch.Tensor,
        spatial_history: torch.Tensor,
        plan_history: torch.Tensor,
        observed_kinematics: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """Return online decision logits for candidate prefixes."""
        if context_last.ndim != 2 or context_last.shape[1] != self.embed_dim:
            raise ValueError(f"context_last must be [B,{self.embed_dim}]")
        batch_size = context_last.shape[0]
        if goal_last.shape != context_last.shape:
            raise ValueError("goal_last must align with context_last")
        token_shape = (batch_size, self.token_count, self.embed_dim)
        if context_tokens.shape != token_shape or goal_tokens.shape != token_shape:
            raise ValueError(f"context_tokens and goal_tokens must be {token_shape}")
        if latent_history.ndim != 3 or latent_history.shape[0] != batch_size:
            raise ValueError("latent_history must be [B,S,D]")
        available_steps = latent_history.shape[1]
        if latent_history.shape[2] != self.embed_dim:
            raise ValueError(f"latent_history final dimension must be {self.embed_dim}")
        _validate_depths(depths, batch_size, available_steps, self.horizon)
        if spatial_history.shape != (
            batch_size,
            available_steps,
            self.token_count,
            self.embed_dim,
        ):
            raise ValueError("spatial_history must align as [B,S,Q,D]")
        if plan_history.shape != (
            batch_size,
            available_steps,
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("plan_history must align as [B,S,H,A]")
        if observed_kinematics.shape != (batch_size, self.kinematic_dim):
            raise ValueError(f"observed_kinematics must be [B,{self.kinematic_dim}]")

        previous_latents = torch.cat(
            [context_last.unsqueeze(1), latent_history[:, :-1]], dim=1
        )
        previous_plans = torch.cat(
            [torch.zeros_like(plan_history[:, :1]), plan_history[:, :-1]], dim=1
        )
        latent_delta = latent_history - previous_latents
        plan_delta = plan_history - previous_plans
        context_spatial = self._pool_spatial(context_tokens)
        goal_spatial = self._pool_spatial(goal_tokens)
        spatial_features = self._pool_spatial(spatial_history)
        previous_spatial = torch.cat(
            [context_spatial.unsqueeze(1), spatial_features[:, :-1]], dim=1
        )
        spatial_delta = spatial_features - previous_spatial
        goal_steps = goal_last.unsqueeze(1).expand_as(latent_history)
        goal_relations = self._relation(latent_history, goal_steps)
        goal_spatial_steps = goal_spatial.unsqueeze(1).expand_as(spatial_features)
        dynamics = torch.stack(
            [
                latent_delta.square().mean(dim=2).sqrt(),
                spatial_delta.square().mean(dim=2).sqrt(),
                plan_delta.square().mean(dim=(2, 3)).sqrt(),
                (goal_steps - latent_history).square().mean(dim=2).sqrt(),
                (goal_spatial_steps - spatial_features).square().mean(dim=2).sqrt(),
                plan_history.square().mean(dim=(2, 3)).sqrt(),
            ],
            dim=2,
        )
        step_features = torch.cat(
            [
                goal_relations,
                self.feature_projection(latent_delta),
                goal_spatial_steps - spatial_features,
                self.plan_projection(plan_history.flatten(2)),
                self.plan_projection(plan_delta.flatten(2)),
                self.dynamics_projection(torch.log1p(dynamics)),
            ],
            dim=-1,
        )
        valid_steps = torch.arange(available_steps, device=depths.device).unsqueeze(
            0
        ) < depths.unsqueeze(1)
        step_features = self.step_projection(step_features) * valid_steps.unsqueeze(2)
        initial_relation = self._relation(context_last, goal_last)
        initial_spatial_gap = goal_spatial - context_spatial
        kinematic_features = self.kinematic_projection(observed_kinematics)
        initial_state = torch.tanh(
            initial_relation + initial_spatial_gap + kinematic_features
        ).unsqueeze(0)
        encoded_prefix, _ = self.prefix_model(step_features, initial_state)
        rows = torch.arange(batch_size, device=depths.device)
        prefix_state = encoded_prefix[rows, depths - 1]
        policy_features = torch.cat(
            [
                prefix_state,
                initial_relation,
                initial_spatial_gap,
                kinematic_features,
                self.depth_embedding(depths),
            ],
            dim=-1,
        )
        return self.policy(policy_features)

    def _relation(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.relation_projection(
            torch.cat(
                [source, target, target - source, (target - source).abs()], dim=-1
            )
        )

    def _pool_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.spatial_projection(tokens)
        weights = torch.softmax(self.spatial_attention(projected), dim=-2)
        return (projected * weights).sum(dim=-2)


class GoalConvergenceHorizonPolicy(torch.nn.Module):
    """Predict a rollout horizon from scene-invariant online convergence signals."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        horizon: int,
        action_dim: int,
        kinematic_dim: int,
        token_count: int,
        include_plan_direction: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim < 1 or hidden_dim < 8 or horizon < 2 or action_dim < 1:
            raise ValueError(
                "model dimensions must be positive and horizon must be at least 2"
            )
        if kinematic_dim < 1 or token_count < 1:
            raise ValueError("kinematic_dim and token_count must be positive")
        self.embed_dim = int(embed_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.kinematic_dim = int(kinematic_dim)
        self.token_count = int(token_count)
        self.include_plan_direction = bool(include_plan_direction)
        self.signal_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(11),
            torch.nn.Linear(11, hidden_dim),
            torch.nn.GELU(),
        )
        if self.include_plan_direction:
            self.plan_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(horizon * action_dim),
                torch.nn.Linear(horizon * action_dim, hidden_dim),
            )
            self.plan_delta_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(horizon * action_dim),
                torch.nn.Linear(horizon * action_dim, hidden_dim),
            )
        self.kinematic_projection = torch.nn.Sequential(
            torch.nn.LayerNorm(kinematic_dim),
            torch.nn.Linear(kinematic_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.depth_embedding = torch.nn.Embedding(horizon + 1, hidden_dim)
        self.prefix_model = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.horizon_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 3),
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, horizon),
        )

    def forward(
        self,
        context_last: torch.Tensor,
        context_tokens: torch.Tensor,
        goal_last: torch.Tensor,
        goal_tokens: torch.Tensor,
        latent_history: torch.Tensor,
        spatial_history: torch.Tensor,
        plan_history: torch.Tensor,
        observed_kinematics: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:
        """Return horizon logits without exposing raw scene-identity features."""
        if context_last.ndim != 2 or context_last.shape[1] != self.embed_dim:
            raise ValueError(f"context_last must be [B,{self.embed_dim}]")
        batch_size = context_last.shape[0]
        if goal_last.shape != context_last.shape:
            raise ValueError("goal_last must align with context_last")
        token_shape = (batch_size, self.token_count, self.embed_dim)
        if context_tokens.shape != token_shape or goal_tokens.shape != token_shape:
            raise ValueError(f"context_tokens and goal_tokens must be {token_shape}")
        if latent_history.ndim != 3 or latent_history.shape[0] != batch_size:
            raise ValueError("latent_history must be [B,S,D]")
        available_steps = latent_history.shape[1]
        if latent_history.shape[2] != self.embed_dim:
            raise ValueError(f"latent_history final dimension must be {self.embed_dim}")
        _validate_depths(depths, batch_size, available_steps, self.horizon)
        if spatial_history.shape != (
            batch_size,
            available_steps,
            self.token_count,
            self.embed_dim,
        ):
            raise ValueError("spatial_history must align as [B,S,Q,D]")
        if plan_history.shape != (
            batch_size,
            available_steps,
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("plan_history must align as [B,S,H,A]")
        if observed_kinematics.shape != (batch_size, self.kinematic_dim):
            raise ValueError(f"observed_kinematics must be [B,{self.kinematic_dim}]")

        spatial_context = context_tokens.mean(dim=1)
        spatial_goal = goal_tokens.mean(dim=1)
        spatial_steps = spatial_history.mean(dim=2)
        previous_latents = torch.cat(
            [context_last.unsqueeze(1), latent_history[:, :-1]], dim=1
        )
        previous_spatial = torch.cat(
            [spatial_context.unsqueeze(1), spatial_steps[:, :-1]], dim=1
        )
        previous_plans = torch.cat(
            [torch.zeros_like(plan_history[:, :1]), plan_history[:, :-1]], dim=1
        )
        latent_step = _rms(latent_history - previous_latents, dim=2)
        spatial_step = _rms(spatial_steps - previous_spatial, dim=2)
        plan_step = _rms(plan_history - previous_plans, dim=(2, 3))
        latent_gap = _rms(latent_history - goal_last.unsqueeze(1), dim=2)
        spatial_gap = _rms(spatial_steps - spatial_goal.unsqueeze(1), dim=2)
        plan_scale = _rms(plan_history, dim=(2, 3))
        initial_latent_gap = _rms(context_last - goal_last, dim=1).unsqueeze(1)
        initial_spatial_gap = _rms(spatial_context - spatial_goal, dim=1).unsqueeze(1)
        previous_latent_gap = torch.cat([initial_latent_gap, latent_gap[:, :-1]], dim=1)
        previous_spatial_gap = torch.cat(
            [initial_spatial_gap, spatial_gap[:, :-1]], dim=1
        )
        epsilon = torch.finfo(latent_history.dtype).eps
        magnitude_signals = torch.stack(
            [latent_step, spatial_step, plan_step, latent_gap, spatial_gap, plan_scale],
            dim=2,
        ).log1p()
        ratio_signals = torch.stack(
            [
                latent_gap / initial_latent_gap.clamp_min(epsilon),
                spatial_gap / initial_spatial_gap.clamp_min(epsilon),
                plan_step / plan_scale.clamp_min(epsilon),
            ],
            dim=2,
        )
        progress_signals = torch.stack(
            [
                (previous_latent_gap - latent_gap)
                / initial_latent_gap.clamp_min(epsilon),
                (previous_spatial_gap - spatial_gap)
                / initial_spatial_gap.clamp_min(epsilon),
            ],
            dim=2,
        )
        signals = torch.cat([magnitude_signals, ratio_signals, progress_signals], dim=2)
        valid_steps = torch.arange(available_steps, device=depths.device).unsqueeze(
            0
        ) < depths.unsqueeze(1)
        step_features = self.signal_projection(signals)
        if self.include_plan_direction:
            step_features = (
                step_features
                + self.plan_projection(plan_history.flatten(2))
                + self.plan_delta_projection((plan_history - previous_plans).flatten(2))
            )
        step_features = step_features * valid_steps.unsqueeze(2)
        kinematic_features = self.kinematic_projection(observed_kinematics)
        encoded_prefix, _ = self.prefix_model(
            step_features, kinematic_features.unsqueeze(0)
        )
        rows = torch.arange(batch_size, device=depths.device)
        prefix_state = encoded_prefix[rows, depths - 1]
        return self.horizon_head(
            torch.cat(
                [prefix_state, kinematic_features, self.depth_embedding(depths)], dim=1
            )
        )


def _rms(values: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    return values.float().square().mean(dim=dim).sqrt().to(values.dtype)


def _dpo_reward_margin(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    if (
        policy_logits.shape != reference_logits.shape
        or policy_logits.ndim != 2
        or policy_logits.shape[-1] != 2
    ):
        raise ValueError(
            "policy_logits and reference_logits must have matching shape [N,2]"
        )
    policy_log_probs = torch.log_softmax(policy_logits.float(), dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits.float(), dim=-1)
    return (
        policy_log_probs[:, ContinuationPolicy.HIGH_REWARD]
        - policy_log_probs[:, ContinuationPolicy.LOW_REWARD]
        - reference_log_probs[:, ContinuationPolicy.HIGH_REWARD]
        + reference_log_probs[:, ContinuationPolicy.LOW_REWARD]
    )


def dpo_reward_probability(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Return the implicit DPO reward probability for one candidate plan."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    return torch.sigmoid(beta * _dpo_reward_margin(policy_logits, reference_logits))


def dpo_pairwise_loss(
    chosen_policy_logits: torch.Tensor,
    rejected_policy_logits: torch.Tensor,
    chosen_reference_logits: torch.Tensor,
    rejected_reference_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Directly prefer the lower-cost rollout candidate over the rejected one."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    chosen_reward = _dpo_reward_margin(
        chosen_policy_logits,
        chosen_reference_logits,
    )
    rejected_reward = _dpo_reward_margin(
        rejected_policy_logits,
        rejected_reference_logits,
    )
    return -torch.nn.functional.logsigmoid(
        beta * (chosen_reward - rejected_reward)
    ).mean()


def dpo_action_preference_loss(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    chosen_actions: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """DPO loss for a preferred STOP/CONTINUE action at one online state."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if (
        policy_logits.shape != reference_logits.shape
        or policy_logits.ndim != 2
        or policy_logits.shape[1] != 2
    ):
        raise ValueError("policy_logits and reference_logits must align as [N,2]")
    if chosen_actions.shape != (
        policy_logits.shape[0],
    ) or chosen_actions.dtype not in (torch.int32, torch.int64):
        raise ValueError("chosen_actions must be an integer [N] tensor")
    if torch.any(chosen_actions < 0) or torch.any(chosen_actions > 1):
        raise ValueError("chosen_actions must contain only 0 or 1")
    rejected_actions = 1 - chosen_actions
    policy_log_probs = torch.log_softmax(policy_logits.float(), dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits.float(), dim=-1)
    rows = torch.arange(policy_logits.shape[0], device=policy_logits.device)
    preference_margin = (
        policy_log_probs[rows, chosen_actions]
        - policy_log_probs[rows, rejected_actions]
        - reference_log_probs[rows, chosen_actions]
        + reference_log_probs[rows, rejected_actions]
    )
    return -torch.nn.functional.logsigmoid(beta * preference_margin).mean()


def dpo_stop_probability(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Return the implicit DPO probability of preferring STOP to CONTINUE."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if (
        policy_logits.shape != reference_logits.shape
        or policy_logits.ndim != 2
        or policy_logits.shape[1] != 2
    ):
        raise ValueError("policy_logits and reference_logits must align as [N,2]")
    policy_log_probs = torch.log_softmax(policy_logits.float(), dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits.float(), dim=-1)
    stop_margin = (
        policy_log_probs[:, GoalConditionedContinuationPolicy.STOP]
        - policy_log_probs[:, GoalConditionedContinuationPolicy.CONTINUE]
        - reference_log_probs[:, GoalConditionedContinuationPolicy.STOP]
        + reference_log_probs[:, GoalConditionedContinuationPolicy.CONTINUE]
    )
    return torch.sigmoid(beta * stop_margin)
