"""6-DoF state/action geometry shared by training and inference."""

from __future__ import annotations

import torch


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def states_to_ego_actions(states: torch.Tensor) -> torch.Tensor:
    """Convert global states ``[B,T,>=6]`` to ego-frame deltas ``[B,T-1,6]``."""
    if states.ndim != 3 or states.shape[-1] < 6 or states.shape[1] < 2:
        raise ValueError(f"states must be [B,T>=2,>=6], got {tuple(states.shape)}")
    current = states[:, :-1, :6]
    following = states[:, 1:, :6]
    global_delta = following[..., :3] - current[..., :3]
    yaw = current[..., 5]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    dx = cos_yaw * global_delta[..., 0] + sin_yaw * global_delta[..., 1]
    dy = -sin_yaw * global_delta[..., 0] + cos_yaw * global_delta[..., 1]
    angles = wrap_angle(following[..., 3:6] - current[..., 3:6])
    return torch.cat([dx.unsqueeze(-1), dy.unsqueeze(-1), global_delta[..., 2:3], angles], dim=-1)


def integrate_ego_actions(initial_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Differentiably integrate ``[dx,dy,dz,droll,dpitch,dyaw]`` actions."""
    if initial_state.ndim != 2 or initial_state.shape[-1] < 6:
        raise ValueError(f"initial_state must be [B,>=6], got {tuple(initial_state.shape)}")
    if actions.ndim != 3 or actions.shape[-1] != 6:
        raise ValueError(f"actions must be [B,T,6], got {tuple(actions.shape)}")
    current = initial_state[:, :6]
    trajectory = [current]
    for step in range(actions.shape[1]):
        action = actions[:, step]
        yaw = current[:, 5]
        next_x = current[:, 0] + torch.cos(yaw) * action[:, 0] - torch.sin(yaw) * action[:, 1]
        next_y = current[:, 1] + torch.sin(yaw) * action[:, 0] + torch.cos(yaw) * action[:, 1]
        current = torch.stack(
            [
                next_x,
                next_y,
                current[:, 2] + action[:, 2],
                wrap_angle(current[:, 3] + action[:, 3]),
                wrap_angle(current[:, 4] + action[:, 4]),
                wrap_angle(current[:, 5] + action[:, 5]),
            ],
            dim=-1,
        )
        trajectory.append(current)
    return torch.stack(trajectory, dim=1)


def to_observation_relative(trajectory: torch.Tensor, origin_state: torch.Tensor) -> torch.Tensor:
    """Transform a global 6-DoF trajectory into the observation-origin frame."""
    if trajectory.ndim != 3 or trajectory.shape[-1] < 6:
        raise ValueError(f"trajectory must be [B,T,>=6], got {tuple(trajectory.shape)}")
    if origin_state.ndim != 2 or origin_state.shape[0] != trajectory.shape[0] or origin_state.shape[-1] < 6:
        raise ValueError(f"origin_state must be [B,>=6], got {tuple(origin_state.shape)}")
    delta_x = trajectory[..., 0] - origin_state[:, None, 0]
    delta_y = trajectory[..., 1] - origin_state[:, None, 1]
    yaw = origin_state[:, None, 5]
    relative_x = torch.cos(yaw) * delta_x + torch.sin(yaw) * delta_y
    relative_y = -torch.sin(yaw) * delta_x + torch.cos(yaw) * delta_y
    relative_angles = wrap_angle(trajectory[..., 3:6] - origin_state[:, None, 3:6])
    return torch.cat(
        [
            relative_x.unsqueeze(-1),
            relative_y.unsqueeze(-1),
            (trajectory[..., 2] - origin_state[:, None, 2]).unsqueeze(-1),
            relative_angles,
        ],
        dim=-1,
    )


def build_timeline(states: torch.Tensor, frame_stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Align raw states/actions with tubelet-level predictor steps."""
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    indices = torch.arange(frame_stride - 1, states.shape[1], frame_stride, device=states.device)
    timeline_states = states.index_select(1, indices)
    return timeline_states, states_to_ego_actions(timeline_states)
