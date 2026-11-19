"""Training-only preference artifact for adaptive rollout decisions."""

from __future__ import annotations

import inspect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import Dataset

from dronewam.config import DataConfig
from dronewam.data import DroneDataset

PREFERENCE_FORMAT_VERSION = 7


@dataclass(frozen=True)
class ThresholdCalibration:
    """Train-only operating point for the sequential continuation policy."""

    stop_thresholds: tuple[float, ...]
    normalized_objective: float
    mean_rollout_depth: float
    relative_ate: float
    relative_rpe: float


def preference_split_indices(
    count: int,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic, disjoint adaptive-train and calibration indices."""
    if count < 2:
        raise ValueError("Preference splitting requires at least two records")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0,1)")
    calibration_count = min(count - 1, max(1, round(count * calibration_fraction)))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    calibration = permutation[:calibration_count].sort().values
    adaptive_train = permutation[calibration_count:].sort().values
    return adaptive_train, calibration


def preference_group_split_indices(
    record_indices: torch.Tensor,
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sample indices while keeping all windows from one trajectory together."""
    groups = torch.as_tensor(record_indices)
    if groups.ndim != 1 or groups.dtype not in (torch.int32, torch.int64):
        raise ValueError("record_indices must be a one-dimensional integer tensor")
    if groups.numel() < 2 or torch.any(groups < 0):
        raise ValueError("record_indices must contain at least two non-negative samples")
    unique_groups = torch.unique(groups, sorted=True)
    if unique_groups.numel() < 2:
        raise ValueError("Grouped preference splitting requires at least two trajectories")
    adaptive_groups, calibration_groups = preference_split_indices(
        unique_groups.shape[0],
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    adaptive_ids = unique_groups[adaptive_groups]
    calibration_ids = unique_groups[calibration_groups]
    sample_indices = torch.arange(groups.shape[0])
    adaptive = sample_indices[torch.isin(groups, adaptive_ids)]
    calibration = sample_indices[torch.isin(groups, calibration_ids)]
    return adaptive, calibration


def representative_group_indices(
    record_indices: torch.Tensor,
    window_starts: torch.Tensor,
    sample_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose one deterministic median-start window from every selected trajectory."""
    groups = torch.as_tensor(record_indices)
    starts = torch.as_tensor(window_starts)
    if groups.ndim != 1 or starts.shape != groups.shape:
        raise ValueError("record_indices and window_starts must align as one-dimensional tensors")
    if groups.dtype not in (torch.int32, torch.int64) or starts.dtype not in (torch.int32, torch.int64):
        raise ValueError("record_indices and window_starts must be integer tensors")
    selected = (
        torch.arange(groups.shape[0])
        if sample_indices is None
        else _validate_subset_indices(sample_indices, groups.shape[0])
    )
    representatives = []
    for group in torch.unique(groups[selected], sorted=True).tolist():
        candidates = selected[groups[selected] == group]
        ordered = candidates[starts[candidates].argsort(stable=True)]
        representatives.append(ordered[(ordered.shape[0] - 1) // 2])
    if not representatives:
        raise ValueError("Representative window selection produced zero samples")
    return torch.stack(representatives).sort().values


def load_threshold_calibration(path: str | Path) -> ThresholdCalibration:
    """Load a strict calibration artifact."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = set(ThresholdCalibration.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("Calibration JSON has invalid fields")
    stop_thresholds = raw["stop_thresholds"]
    if not isinstance(stop_thresholds, list) or not stop_thresholds:
        raise ValueError("Calibrated stop_thresholds must be a non-empty list")
    calibration = ThresholdCalibration(
        stop_thresholds=tuple(float(value) for value in stop_thresholds),
        normalized_objective=float(raw["normalized_objective"]),
        mean_rollout_depth=float(raw["mean_rollout_depth"]),
        relative_ate=float(raw["relative_ate"]),
        relative_rpe=float(raw["relative_rpe"]),
    )
    if any(not 0.0 < threshold < 1.0 for threshold in calibration.stop_thresholds):
        raise ValueError("Every calibrated stop threshold must be in (0,1)")
    if not all(
        math.isfinite(value)
        for value in (
            calibration.normalized_objective,
            calibration.mean_rollout_depth,
            calibration.relative_ate,
            calibration.relative_rpe,
        )
    ):
        raise ValueError("Calibration metrics must be finite")
    return calibration


def select_rollout_depths(
    reward_probabilities: torch.Tensor,
    *,
    stop_thresholds: Sequence[float] | torch.Tensor,
    min_rollout_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return online rollout depth and best-reward plan depth."""
    if reward_probabilities.ndim != 2:
        raise ValueError("reward_probabilities must be [N,H]")
    if not torch.isfinite(reward_probabilities).all():
        raise ValueError("reward_probabilities must be finite")
    if torch.any(reward_probabilities < 0.0) or torch.any(reward_probabilities > 1.0):
        raise ValueError("reward_probabilities must lie in [0,1]")
    horizon = reward_probabilities.shape[1]
    thresholds = torch.as_tensor(
        stop_thresholds,
        dtype=reward_probabilities.dtype,
        device=reward_probabilities.device,
    )
    if thresholds.shape != (horizon - 1,):
        raise ValueError(f"stop_thresholds must contain {horizon - 1} values")
    if torch.any(thresholds <= 0.0) or torch.any(thresholds >= 1.0):
        raise ValueError("Every stop threshold must be in (0,1)")
    if not 1 <= min_rollout_depth <= horizon:
        raise ValueError(f"min_rollout_depth must be in [1,{horizon}]")

    rollout_depths = torch.full(
        (reward_probabilities.shape[0],),
        horizon,
        dtype=torch.long,
        device=reward_probabilities.device,
    )
    plan_depths = torch.ones_like(rollout_depths)
    best_rewards = torch.full_like(reward_probabilities[:, 0], -torch.inf)
    active = torch.ones_like(rollout_depths, dtype=torch.bool)
    for depth in range(1, horizon + 1):
        reward = reward_probabilities[:, depth - 1]
        improved = active & (reward > best_rewards)
        best_rewards = torch.where(improved, reward, best_rewards)
        plan_depths[improved] = depth
        if depth == horizon or depth < min_rollout_depth:
            continue
        stop = active & (reward >= thresholds[depth - 1])
        rollout_depths[stop] = depth
        active &= ~stop
    return rollout_depths, plan_depths


def calibrate_stop_threshold(
    reward_probabilities: torch.Tensor,
    normalized_costs: torch.Tensor,
    relative_ate: torch.Tensor,
    relative_rpe: torch.Tensor,
    *,
    compute_penalty: float,
    min_rollout_depth: int,
) -> ThresholdCalibration:
    """Choose a policy threshold on training preferences without validation data."""
    if normalized_costs.ndim != 2 or normalized_costs.shape[1] < 2:
        raise ValueError("normalized_costs must be [N,H>=2]")
    if reward_probabilities.shape != normalized_costs.shape:
        raise ValueError("reward probabilities and per-depth costs must align")
    if relative_ate.shape != normalized_costs.shape or relative_rpe.shape != normalized_costs.shape:
        raise ValueError("relative_ate and relative_rpe must align with normalized_costs")
    if not torch.isfinite(normalized_costs).all():
        raise ValueError("normalized_costs must be finite")
    if compute_penalty < 0.0:
        raise ValueError("compute_penalty must be non-negative")

    horizon = normalized_costs.shape[1]
    row_indices = torch.arange(normalized_costs.shape[0], device=normalized_costs.device)

    def evaluate(thresholds: torch.Tensor) -> ThresholdCalibration:
        rollout_depths, plan_depths = select_rollout_depths(
            reward_probabilities,
            stop_thresholds=thresholds,
            min_rollout_depth=min_rollout_depth,
        )
        plan_indices = plan_depths - 1
        objective = (
            normalized_costs[row_indices, plan_indices]
            + compute_penalty * rollout_depths / horizon
        )
        return ThresholdCalibration(
            stop_thresholds=tuple(float(value) for value in thresholds.tolist()),
            normalized_objective=float(objective.mean().item()),
            mean_rollout_depth=float(rollout_depths.float().mean().item()),
            relative_ate=float(relative_ate[row_indices, plan_indices].mean().item()),
            relative_rpe=float(relative_rpe[row_indices, plan_indices].mean().item()),
        )

    def is_better(candidate: ThresholdCalibration, incumbent: ThresholdCalibration | None) -> bool:
        if incumbent is None:
            return True
        return (candidate.normalized_objective, candidate.mean_rollout_depth) < (
            incumbent.normalized_objective,
            incumbent.mean_rollout_depth,
        )

    # Descending search retains the larger threshold while a coordinate is
    # unreachable; once reachable, equal-cost solutions prefer less rollout.
    grid = torch.linspace(0.99, 0.01, 99, device=normalized_costs.device)
    best: ThresholdCalibration | None = None
    for initial in torch.linspace(0.1, 0.9, 9, device=normalized_costs.device):
        thresholds = torch.full(
            (horizon - 1,),
            float(initial.item()),
            device=normalized_costs.device,
        )
        for _ in range(3):
            changed = False
            for depth_index in reversed(range(horizon - 1)):
                local_best: ThresholdCalibration | None = None
                local_threshold = thresholds[depth_index].clone()
                for threshold in grid:
                    candidate_thresholds = thresholds.clone()
                    candidate_thresholds[depth_index] = threshold
                    candidate = evaluate(candidate_thresholds)
                    if is_better(candidate, local_best):
                        local_best = candidate
                        local_threshold = threshold
                if not torch.equal(local_threshold, thresholds[depth_index]):
                    thresholds[depth_index] = local_threshold
                    changed = True
            if not changed:
                break
        candidate = evaluate(thresholds)
        if is_better(candidate, best):
            best = candidate
    if best is None:
        raise RuntimeError("Threshold calibration produced no candidates")
    return best


def rank_rollout_depths(
    normalized_costs: torch.Tensor,
    *,
    compute_penalty: float,
    preference_margin: float,
) -> torch.Tensor:
    """Return the earliest rollout depth within a margin of the best candidate."""
    if normalized_costs.ndim != 2 or normalized_costs.shape[1] < 2:
        raise ValueError("normalized_costs must be [B,H>=2]")
    if not torch.isfinite(normalized_costs).all():
        raise ValueError("normalized_costs must be finite")
    if compute_penalty < 0.0 or preference_margin < 0.0:
        raise ValueError("compute_penalty and preference_margin must be non-negative")
    horizon = normalized_costs.shape[1]
    depth_fraction = torch.arange(
        1,
        horizon + 1,
        device=normalized_costs.device,
        dtype=normalized_costs.dtype,
    ) / horizon
    ranked_costs = normalized_costs + compute_penalty * depth_fraction.unsqueeze(0)
    optimum = ranked_costs.amin(dim=1, keepdim=True)
    acceptable = ranked_costs <= optimum + preference_margin
    return acceptable.to(torch.int64).argmax(dim=1) + 1


def validate_preference_artifact(artifact: Dict[str, Any]) -> None:
    required = {
        "format_version",
        "split",
        "record_indices",
        "window_starts",
        "context_last",
        "context_tokens",
        "future_pooled",
        "future_spatial",
        "planned_actions",
        "observed_kinematics",
        "relative_ate",
        "relative_rpe",
        "normalized_costs",
        "best_depths",
        "metadata",
    }
    missing = sorted(required - set(artifact))
    unknown = sorted(set(artifact) - required)
    if missing or unknown:
        raise ValueError(f"Invalid preference artifact keys: missing={missing}, unknown={unknown}")
    if artifact["format_version"] != PREFERENCE_FORMAT_VERSION:
        raise ValueError(f"Unsupported preference format_version={artifact['format_version']}")
    if artifact["split"] != "train":
        raise ValueError("Preference artifacts must be generated from split='train'")
    record_indices = artifact["record_indices"]
    window_starts = artifact["window_starts"]
    context_last = artifact["context_last"]
    context_tokens = artifact["context_tokens"]
    future_pooled = artifact["future_pooled"]
    future_spatial = artifact["future_spatial"]
    planned_actions = artifact["planned_actions"]
    observed_kinematics = artifact["observed_kinematics"]
    relative_ate = artifact["relative_ate"]
    relative_rpe = artifact["relative_rpe"]
    costs = artifact["normalized_costs"]
    best_depths = artifact["best_depths"]
    if record_indices.ndim != 1:
        raise ValueError("record_indices must be [N]")
    count = record_indices.shape[0]
    horizon = future_pooled.shape[1]
    if record_indices.dtype not in (torch.int32, torch.int64) or torch.any(record_indices < 0):
        raise ValueError("record_indices must contain non-negative integers")
    if (
        window_starts.shape != (count,)
        or context_last.ndim != 2
        or context_last.shape[0] != count
    ):
        raise ValueError("Preference artifact sample tensors must align")
    if window_starts.dtype not in (torch.int32, torch.int64) or torch.any(window_starts < 0):
        raise ValueError("window_starts must contain non-negative integers")
    sample_keys = torch.stack([record_indices.long(), window_starts.long()], dim=1)
    if torch.unique(sample_keys, dim=0).shape[0] != count:
        raise ValueError("Preference artifact contains duplicate trajectory windows")
    if (
        future_pooled.ndim != 3
        or context_tokens.ndim != 3
        or future_spatial.ndim != 4
        or context_tokens.shape[0] != count
        or future_spatial.shape[:2] != (count, horizon)
        or context_tokens.shape[1:] != future_spatial.shape[2:]
        or context_tokens.shape[-1] != context_last.shape[-1]
        or future_pooled.shape[-1] != context_last.shape[-1]
        or planned_actions.ndim != 4
        or planned_actions.shape[:3] != (count, horizon, horizon)
        or observed_kinematics.ndim != 2
        or observed_kinematics.shape[0] != count
        or costs.shape != (count, horizon)
        or relative_ate.shape != costs.shape
        or relative_rpe.shape != costs.shape
    ):
        raise ValueError(
            "context/future features, planned_actions, observed_kinematics and per-depth metrics must align"
        )
    if best_depths.shape != (count,):
        raise ValueError("best_depths shape does not match the sample count")
    if torch.any(best_depths < 1) or torch.any(best_depths > horizon):
        raise ValueError("best_depths are outside the rollout horizon")
    for name in (
        "context_last",
        "context_tokens",
        "future_pooled",
        "future_spatial",
        "planned_actions",
        "observed_kinematics",
        "relative_ate",
        "relative_rpe",
        "normalized_costs",
    ):
        if not torch.isfinite(artifact[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    metadata = artifact["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    if metadata.get("policy_token_count") != context_tokens.shape[1]:
        raise ValueError("metadata.policy_token_count must match cached spatial features")
    forbidden = {"gt_states", "gt_actions", "frames", "target_latents"}
    leaked = sorted(forbidden.intersection(metadata))
    if leaked:
        raise ValueError(f"Preference metadata contains forbidden target fields: {leaked}")


def save_preference_artifact(artifact: Dict[str, Any], path: str | Path) -> None:
    validate_preference_artifact(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, destination)


def load_preference_artifact(path: str | Path) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    artifact = torch.load(path, **kwargs)
    if not isinstance(artifact, dict):
        raise TypeError("Preference artifact must contain a dictionary")
    validate_preference_artifact(artifact)
    return artifact


def _validate_subset_indices(indices: torch.Tensor, count: int) -> torch.Tensor:
    indices = torch.as_tensor(indices)
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("record_indices must be a one-dimensional integer tensor")
    if indices.numel() == 0:
        raise ValueError("record_indices must not be empty")
    if torch.any(indices < 0) or torch.any(indices >= count):
        raise ValueError("record_indices are outside the preference artifact")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("record_indices must be unique")
    return indices.long().clone()


class _PreferenceFeatureDataset(Dataset):
    """Own cached prediction-only features shared by depth and pair datasets."""

    def __init__(
        self,
        artifact: Dict[str, Any],
        record_indices: torch.Tensor | None = None,
    ) -> None:
        validate_preference_artifact(artifact)
        self.context_last = artifact["context_last"].float()
        self.context_tokens = artifact["context_tokens"].float()
        self.future_pooled = artifact["future_pooled"].float()
        self.future_spatial = artifact["future_spatial"].float()
        self.planned_actions = artifact["planned_actions"].float()
        self.observed_kinematics = artifact["observed_kinematics"].float()
        compute_penalty = artifact["metadata"].get("compute_penalty")
        if not isinstance(compute_penalty, (int, float)) or compute_penalty < 0.0:
            raise ValueError("metadata.compute_penalty must be a non-negative number")
        preference_margin = artifact["metadata"].get("preference_margin")
        if not isinstance(preference_margin, (int, float)) or preference_margin < 0.0:
            raise ValueError("metadata.preference_margin must be a non-negative number")
        self.preference_margin = float(preference_margin)
        count, self.horizon, _ = self.future_pooled.shape
        self.record_indices = (
            torch.arange(count)
            if record_indices is None
            else _validate_subset_indices(record_indices, count)
        )
        depth_fraction = torch.arange(1, self.horizon + 1).float() / self.horizon
        self.ranked_costs = artifact["normalized_costs"].float() + float(compute_penalty) * depth_fraction

    def _common_features(self, sample_index: int) -> Dict[str, torch.Tensor]:
        return {
            "context_last": self.context_last[sample_index],
            "context_tokens": self.context_tokens[sample_index],
            "observed_kinematics": self.observed_kinematics[sample_index],
        }

    def _prefix_features(self, sample_index: int, depth: int) -> Dict[str, torch.Tensor]:
        latent_history = torch.zeros_like(self.future_pooled[sample_index])
        latent_history[:depth] = self.future_pooled[sample_index, :depth]
        spatial_history = torch.zeros_like(self.future_spatial[sample_index])
        spatial_history[:depth] = self.future_spatial[sample_index, :depth]
        plan_history = torch.zeros_like(self.planned_actions[sample_index])
        plan_history[:depth] = self.planned_actions[sample_index, :depth]
        return {
            "latent_history": latent_history,
            "spatial_history": spatial_history,
            "plan_history": plan_history,
            "depth": torch.tensor(depth, dtype=torch.long),
        }


class PreferenceDepthDataset(_PreferenceFeatureDataset):
    """Flatten every cached rollout depth for reward calibration and analysis."""

    def __len__(self) -> int:
        return self.record_indices.shape[0] * self.horizon

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        subset_index = index // self.horizon
        sample_index = int(self.record_indices[subset_index].item())
        depth = index % self.horizon + 1
        return {
            **self._common_features(sample_index),
            **self._prefix_features(sample_index, depth),
        }


class PreferencePairDataset(_PreferenceFeatureDataset):
    """Pair the preferred rollout depth with every non-preferred candidate."""

    def __init__(
        self,
        artifact: Dict[str, Any],
        record_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__(artifact, record_indices)
        self.best_depths = artifact["best_depths"].long()

    def __len__(self) -> int:
        return self.record_indices.shape[0] * (self.horizon - 1)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pair_count = self.horizon - 1
        subset_index = index // pair_count
        sample_index = int(self.record_indices[subset_index].item())
        chosen_depth = int(self.best_depths[sample_index].item())
        chosen_index = chosen_depth - 1
        rejected_index = index % pair_count
        if rejected_index >= chosen_index:
            rejected_index += 1
        rejected_depth = rejected_index + 1
        chosen = self._prefix_features(sample_index, chosen_depth)
        rejected = self._prefix_features(sample_index, rejected_depth)
        return {
            **self._common_features(sample_index),
            **{f"chosen_{name}": value for name, value in chosen.items()},
            **{f"rejected_{name}": value for name, value in rejected.items()},
        }


class PreferenceWindowDataset(Dataset):
    """Reload the exact train windows used to produce an artifact."""

    def __init__(
        self,
        config: DataConfig,
        resolution: tuple[int, int],
        artifact: Dict[str, Any],
        record_indices: torch.Tensor | None = None,
    ) -> None:
        validate_preference_artifact(artifact)
        self.base = DroneDataset(
            config,
            split="preference",
            resolution=resolution,
            sample_windows=list(
                zip(
                    artifact["record_indices"].tolist(),
                    artifact["window_starts"].tolist(),
                )
            ),
        )
        if len(self.base) != artifact["record_indices"].shape[0]:
            raise ValueError("Current training CSV no longer matches the preference artifact")
        self.record_indices = (
            torch.arange(len(self.base))
            if record_indices is None
            else _validate_subset_indices(record_indices, len(self.base))
        )
        self.source_record_indices = artifact["record_indices"].long()
        self.window_starts = artifact["window_starts"].long()
        self.best_depths = artifact["best_depths"].long()
        self.normalized_costs = artifact["normalized_costs"].float()

    def __len__(self) -> int:
        return self.record_indices.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_index = int(self.record_indices[index].item())
        sample = self.base[sample_index]
        if int(sample["record_index"]) != int(self.source_record_indices[sample_index]):
            raise RuntimeError("Preference window record index mismatch")
        if int(sample["window_start"]) != int(self.window_starts[sample_index]):
            raise RuntimeError("Preference window start mismatch")
        sample["best_depth"] = self.best_depths[sample_index]
        sample["normalized_costs"] = self.normalized_costs[sample_index]
        return sample
