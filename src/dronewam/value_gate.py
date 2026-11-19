"""Training and evaluation primitives for the causal rollout-value Gate."""

from __future__ import annotations

import math
from typing import Any

import torch


def causal_continuation_advantage_targets(rank_scores: torch.Tensor) -> torch.Tensor:
    """Return the signed benefit of continuing from every rollout prefix.

    Positive values mean that at least one strictly future depth has a lower
    relative-rank score. Zero or negative values mean that the current depth
    is already no worse than every future candidate and should stop.
    """
    if rank_scores.ndim != 2 or rank_scores.shape[1] < 2 or not torch.isfinite(rank_scores).all():
        raise ValueError("rank_scores must be finite [B,C>=2]")
    future_scores = rank_scores.float()[:, 1:]
    best_strict_future = torch.flip(
        torch.cummin(torch.flip(future_scores, dims=(1,)), dim=1).values,
        dims=(1,),
    )
    advantages = rank_scores.float()[:, :-1] - best_strict_future
    final_stop = -torch.ones_like(advantages[:, :1])
    return torch.cat((advantages, final_stop), dim=1)


def balanced_continuation_class_weights(rank_scores: torch.Tensor, *, power: float = 1.0) -> torch.Tensor:
    """Balance STOP/CONTINUE supervision independently at every depth."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("power must be in [0,1]")
    advantages = causal_continuation_advantage_targets(rank_scores)
    classes = torch.where(advantages > 0.0, 0, 1)
    counts = torch.nn.functional.one_hot(classes, num_classes=2).float().sum(dim=0)
    nonempty = counts > 0.0
    weights = torch.where(nonempty, counts.clamp_min(1.0).pow(-power), torch.zeros_like(counts))
    normalizer = (weights * counts).sum(dim=1, keepdim=True) / counts.sum(dim=1, keepdim=True)
    return weights / normalizer.clamp_min(1.0e-8)


def causal_continuation_advantage_loss(
    logits: torch.Tensor,
    rank_scores: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    classification_weight: float = 1.0,
    regression_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train every causal prefix to estimate the gain from more rollout."""
    if logits.ndim != 3 or logits.shape[2] != 2:
        raise ValueError("logits must be [B,C,2]")
    if rank_scores.shape != logits.shape[:2]:
        raise ValueError("rank_scores must match logits as [B,C]")
    if min(classification_weight, regression_weight) < 0.0:
        raise ValueError("Advantage loss weights must be non-negative")
    if classification_weight + regression_weight <= 0.0:
        raise ValueError("At least one advantage loss weight must be positive")

    advantages = causal_continuation_advantage_targets(rank_scores)
    classes = torch.where(advantages > 0.0, 0, 1)
    per_prefix_classification = torch.nn.functional.cross_entropy(
        logits.float().flatten(0, 1),
        classes.flatten(),
        reduction="none",
    ).reshape_as(advantages)
    predicted_advantages = logits.float()[:, :, 0] - logits.float()[:, :, 1]
    per_prefix_regression = torch.nn.functional.smooth_l1_loss(
        predicted_advantages,
        advantages,
        reduction="none",
    )
    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=torch.float32, device=logits.device)
        if weights.shape != (*logits.shape[1:2], 2) or torch.any(weights < 0.0):
            raise ValueError("class_weights must be non-negative [C,2]")
        if torch.any(weights.sum(dim=1) <= 0.0):
            raise ValueError("Every depth must have a positive class weight")
        selected_weights = weights.unsqueeze(0).expand(logits.shape[0], -1, -1).gather(
            2, classes.unsqueeze(2)
        ).squeeze(2)
        classification = (per_prefix_classification * selected_weights).mean()
        regression = (per_prefix_regression * selected_weights).mean()
    else:
        classification = per_prefix_classification.mean()
        regression = per_prefix_regression.mean()
    loss = classification_weight * classification + regression_weight * regression
    return loss, {
        "advantage_classification": classification,
        "advantage_regression": regression,
    }


def causal_observed_improvement_targets(rank_scores: torch.Tensor) -> torch.Tensor:
    """Return signed improvement over the best strictly earlier rollout."""
    if rank_scores.ndim != 2 or rank_scores.shape[1] < 2 or not torch.isfinite(rank_scores).all():
        raise ValueError("rank_scores must be finite [B,C>=2]")
    best_past = torch.cummin(rank_scores.float()[:, :-1], dim=1).values
    return best_past - rank_scores.float()[:, 1:]


def causal_cached_future_advantage_targets(
    rank_scores: torch.Tensor,
    *,
    compute_penalty: float = 0.0,
) -> torch.Tensor:
    """Return future quality gain after charging for additional rollout steps."""
    if rank_scores.ndim != 2 or rank_scores.shape[1] < 2 or not torch.isfinite(rank_scores).all():
        raise ValueError("rank_scores must be finite [B,C>=2]")
    if compute_penalty < 0.0:
        raise ValueError("compute_penalty must be non-negative")
    scores = rank_scores.float()
    best_observed = torch.cummin(scores, dim=1).values
    horizon = scores.shape[1] - 1
    future_values = []
    for depth in range(horizon):
        extra_steps = torch.arange(1, horizon - depth + 1, device=scores.device, dtype=scores.dtype)
        penalized = scores[:, depth + 1 :] + compute_penalty * extra_steps / horizon
        future_values.append(penalized.min(dim=1).values)
    best_strict_future = torch.stack(future_values, dim=1)
    advantages = best_observed[:, :-1] - best_strict_future
    final_stop = -torch.ones_like(advantages[:, :1])
    return torch.cat((advantages, final_stop), dim=1)


def balanced_cached_future_class_weights(
    rank_scores: torch.Tensor,
    *,
    power: float = 0.5,
    compute_penalty: float = 0.0,
) -> torch.Tensor:
    """Balance CONTINUE/STOP targets independently at every causal depth."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("power must be in [0,1]")
    advantages = causal_cached_future_advantage_targets(rank_scores, compute_penalty=compute_penalty)
    classes = torch.where(advantages > 0.0, 0, 1)
    counts = torch.nn.functional.one_hot(classes, num_classes=2).float().sum(dim=0)
    nonempty = counts > 0.0
    weights = torch.where(nonempty, counts.clamp_min(1.0).pow(-power), torch.zeros_like(counts))
    normalizer = (weights * counts).sum(dim=1, keepdim=True) / counts.sum(dim=1, keepdim=True)
    return weights / normalizer.clamp_min(1.0e-8)


def causal_dual_cost_control_loss(
    logits: torch.Tensor,
    normalized_cost: torch.Tensor,
    *,
    update_class_weights: torch.Tensor | None = None,
    continuation_class_weights: torch.Tensor | None = None,
    classification_weight: float = 1.0,
    regression_weight: float = 1.0,
    continuation_compute_penalty: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train independent cache-update and future-continuation decisions."""
    if logits.ndim != 4 or logits.shape[2:] != (2, 2) or normalized_cost.shape != logits.shape[:2]:
        raise ValueError("logits and normalized_cost must align as [B,C,2,2] and [B,C]")
    if torch.any(normalized_cost < 0.0) or not torch.isfinite(normalized_cost).all():
        raise ValueError("normalized_cost must be finite and non-negative")
    if min(classification_weight, regression_weight) < 0.0:
        raise ValueError("Dual-control loss weights must be non-negative")
    if classification_weight + regression_weight <= 0.0:
        raise ValueError("At least one dual-control loss weight must be positive")

    scores = torch.log1p(normalized_cost.float())
    update_targets = causal_observed_improvement_targets(scores)
    continuation_targets = causal_cached_future_advantage_targets(
        scores,
        compute_penalty=continuation_compute_penalty,
    )

    def decision_loss(
        decision_logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        classes = torch.where(targets > 0.0, 0, 1)
        per_step_classification = torch.nn.functional.cross_entropy(
            decision_logits.flatten(0, 1),
            classes.flatten(),
            reduction="none",
        ).reshape_as(targets)
        predicted_margins = decision_logits[:, :, 0] - decision_logits[:, :, 1]
        per_step_regression = torch.nn.functional.smooth_l1_loss(
            predicted_margins,
            targets,
            reduction="none",
        )
        if class_weights is None:
            return per_step_classification.mean(), per_step_regression.mean()
        weights = torch.as_tensor(class_weights, dtype=torch.float32, device=logits.device)
        if weights.shape != (*targets.shape[1:2], 2) or torch.any(weights < 0.0):
            raise ValueError("class_weights must be non-negative [steps,2]")
        if torch.any(weights.sum(dim=1) <= 0.0):
            raise ValueError("Every decision depth must have a positive class weight")
        selected_weights = weights.unsqueeze(0).expand(logits.shape[0], -1, -1).gather(
            2, classes.unsqueeze(2)
        ).squeeze(2)
        return (
            (per_step_classification * selected_weights).mean(),
            (per_step_regression * selected_weights).mean(),
        )

    update_classification, update_regression = decision_loss(
        logits.float()[:, 1:, 0],
        update_targets,
        update_class_weights,
    )
    continuation_classification, continuation_regression = decision_loss(
        logits.float()[:, :, 1],
        continuation_targets,
        continuation_class_weights,
    )
    classification = 0.5 * (update_classification + continuation_classification)
    regression = 0.5 * (update_regression + continuation_regression)
    loss = classification_weight * classification + regression_weight * regression
    return loss, {
        "update_classification": update_classification,
        "update_regression": update_regression,
        "continuation_classification": continuation_classification,
        "continuation_regression": continuation_regression,
    }


def balanced_observed_improvement_class_weights(
    rank_scores: torch.Tensor,
    *,
    power: float = 0.5,
) -> torch.Tensor:
    """Balance IMPROVED/STALE labels independently after every rollout step."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("power must be in [0,1]")
    margins = causal_observed_improvement_targets(rank_scores)
    classes = torch.where(margins > 0.0, 0, 1)
    counts = torch.nn.functional.one_hot(classes, num_classes=2).float().sum(dim=0)
    nonempty = counts > 0.0
    weights = torch.where(nonempty, counts.clamp_min(1.0).pow(-power), torch.zeros_like(counts))
    normalizer = (weights * counts).sum(dim=1, keepdim=True) / counts.sum(dim=1, keepdim=True)
    return weights / normalizer.clamp_min(1.0e-8)


def causal_observed_improvement_loss(
    logits: torch.Tensor,
    rank_scores: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    classification_weight: float = 1.0,
    regression_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Classify and regress observed improvement at every non-initial depth."""
    if logits.ndim != 3 or logits.shape[2] != 2 or rank_scores.shape != logits.shape[:2]:
        raise ValueError("logits and rank_scores must align as [B,C,2] and [B,C]")
    if min(classification_weight, regression_weight) < 0.0:
        raise ValueError("Observed-improvement loss weights must be non-negative")
    if classification_weight + regression_weight <= 0.0:
        raise ValueError("At least one observed-improvement loss weight must be positive")
    margins = causal_observed_improvement_targets(rank_scores)
    classes = torch.where(margins > 0.0, 0, 1)
    decision_logits = logits.float()[:, 1:]
    per_step_classification = torch.nn.functional.cross_entropy(
        decision_logits.flatten(0, 1),
        classes.flatten(),
        reduction="none",
    ).reshape_as(margins)
    predicted_margins = decision_logits[:, :, 0] - decision_logits[:, :, 1]
    per_step_regression = torch.nn.functional.smooth_l1_loss(
        predicted_margins,
        margins,
        reduction="none",
    )
    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=torch.float32, device=logits.device)
        if weights.shape != (*margins.shape[1:2], 2) or torch.any(weights < 0.0):
            raise ValueError("class_weights must be non-negative [C-1,2]")
        selected_weights = weights.unsqueeze(0).expand(logits.shape[0], -1, -1).gather(
            2, classes.unsqueeze(2)
        ).squeeze(2)
        classification = (per_step_classification * selected_weights).mean()
        regression = (per_step_regression * selected_weights).mean()
    else:
        classification = per_step_classification.mean()
        regression = per_step_regression.mean()
    loss = classification_weight * classification + regression_weight * regression
    return loss, {
        "improvement_classification": classification,
        "improvement_regression": regression,
    }


def causal_current_cost_loss(
    logits: torch.Tensor,
    normalized_cost: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress the realized trajectory cost at every causal rollout prefix."""
    if logits.ndim != 3 or logits.shape[2] != 2 or normalized_cost.shape != logits.shape[:2]:
        raise ValueError("logits and normalized_cost must align as [B,C,2] and [B,C]")
    if torch.any(normalized_cost < 0.0) or not torch.isfinite(normalized_cost).all():
        raise ValueError("normalized_cost must be finite and non-negative")
    predicted = logits.float()[:, :, 0] - logits.float()[:, :, 1]
    target = torch.log1p(normalized_cost.float())
    regression = torch.nn.functional.smooth_l1_loss(predicted, target)
    return regression, {"current_cost_regression": regression}


def select_patience_current_cost_depths(
    logits: torch.Tensor,
    *,
    patience: int = 3,
    min_depth: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache the lowest predicted-cost plan and stop after stale updates."""
    if logits.ndim != 3 or logits.shape[2] != 2 or logits.shape[1] < 2:
        raise ValueError("logits must be [B,C>=2,2]")
    candidate_count = logits.shape[1]
    if patience < 1 or patience >= candidate_count:
        raise ValueError("patience must be in [1,C-1]")
    values = logits.float()[:, :, 0] - logits.float()[:, :, 1]
    batch_size = logits.shape[0]
    best_depths = torch.full((batch_size,), min_depth, dtype=torch.long, device=logits.device)
    best_values = values[:, 0]
    stale = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    active = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
    compute_depths = torch.full_like(best_depths, min_depth + candidate_count - 1)
    for index in range(1, candidate_count):
        improved = active & (values[:, index] < best_values)
        best_values = torch.where(improved, values[:, index], best_values)
        best_depths = torch.where(improved, torch.full_like(best_depths, min_depth + index), best_depths)
        stale = torch.where(active, torch.where(improved, torch.zeros_like(stale), stale + 1), stale)
        stop = active & (stale >= patience)
        compute_depths[stop] = min_depth + index
        active = active & ~stop
    return best_depths, compute_depths


def select_patience_improvement_depths(
    logits: torch.Tensor,
    *,
    patience: int = 3,
    improvement_thresholds: torch.Tensor | None = None,
    min_depth: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply predicted observed improvements with patience and a cached best plan."""
    if logits.ndim != 3 or logits.shape[2] != 2 or logits.shape[1] < 2:
        raise ValueError("logits must be [B,C>=2,2]")
    candidate_count = logits.shape[1]
    if patience < 1 or patience >= candidate_count:
        raise ValueError("patience must be in [1,C-1]")
    if min_depth != 0:
        raise NotImplementedError("Observed-improvement selection currently requires min_depth=0")
    if improvement_thresholds is None:
        thresholds = torch.full((candidate_count - 1,), 0.5, device=logits.device)
    else:
        thresholds = torch.as_tensor(improvement_thresholds, dtype=torch.float32, device=logits.device)
        if thresholds.shape != (candidate_count - 1,) or torch.any((thresholds < 0.0) | (thresholds > 1.0)):
            raise ValueError("improvement_thresholds must be in [0,1] with shape [C-1]")
    probabilities = torch.softmax(logits.float()[:, 1:], dim=2)[:, :, 0]
    batch_size = logits.shape[0]
    best = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    selected = best.clone()
    stale = torch.zeros_like(best)
    active = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
    compute_depth = torch.full_like(best, candidate_count - 1)
    for depth in range(1, candidate_count):
        improved = active & (probabilities[:, depth - 1] >= thresholds[depth - 1])
        best = torch.where(improved, torch.full_like(best, depth), best)
        stale = torch.where(active, torch.where(improved, torch.zeros_like(stale), stale + 1), stale)
        stop = active & (stale >= patience)
        selected[stop] = best[stop]
        compute_depth[stop] = depth
        active = active & ~stop
    selected[active] = best[active]
    return selected, compute_depth


def select_hysteresis_improvement_depths(
    logits: torch.Tensor,
    *,
    patience: int = 2,
    stop_threshold: float = 0.3,
    update_threshold: float = 0.7,
    min_depth: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update or stop only on confident evidence, with full-depth fallback."""
    if logits.ndim != 3 or logits.shape[2] != 2 or logits.shape[1] < 2:
        raise ValueError("logits must be [B,C>=2,2]")
    candidate_count = logits.shape[1]
    if patience < 1 or patience >= candidate_count:
        raise ValueError("patience must be in [1,C-1]")
    if not 0.0 <= stop_threshold < update_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= stop < update <= 1")
    if min_depth != 0:
        raise NotImplementedError("Hysteresis improvement selection currently requires min_depth=0")
    probabilities = torch.softmax(logits.float()[:, 1:], dim=2)[:, :, 0]
    batch_size = logits.shape[0]
    best = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    selected = torch.full_like(best, candidate_count - 1)
    stale = torch.zeros_like(best)
    active = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
    compute_depth = torch.full_like(best, candidate_count - 1)
    for depth in range(1, candidate_count):
        current = probabilities[:, depth - 1]
        improved = active & (current >= update_threshold)
        stale_evidence = active & (current <= stop_threshold)
        best = torch.where(improved, torch.full_like(best, depth), best)
        stale = torch.where(
            active,
            torch.where(improved, torch.zeros_like(stale), torch.where(stale_evidence, stale + 1, stale)),
            stale,
        )
        stop = active & (stale >= patience)
        selected[stop] = best[stop]
        compute_depth[stop] = depth
        active = active & ~stop
    return selected + min_depth, compute_depth + min_depth


def select_dual_control_depths(
    logits: torch.Tensor,
    *,
    patience: int = 2,
    update_threshold: float = 0.7,
    continuation_threshold: float = 0.3,
    min_depth: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply independent cache updates and conservative early stopping online."""
    if logits.ndim != 4 or logits.shape[2:] != (2, 2) or logits.shape[1] < 2:
        raise ValueError("logits must be [B,C>=2,2,2]")
    candidate_count = logits.shape[1]
    if patience < 1 or patience >= candidate_count:
        raise ValueError("patience must be in [1,C-1]")
    if not 0.0 <= continuation_threshold <= 1.0 or not 0.0 <= update_threshold <= 1.0:
        raise ValueError("thresholds must be in [0,1]")
    if min_depth != 0:
        raise NotImplementedError("Dual-control selection currently requires min_depth=0")

    probabilities = torch.softmax(logits.float(), dim=3)[:, :, :, 0]
    batch_size = logits.shape[0]
    cached = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    selected = torch.full_like(cached, candidate_count - 1)
    stale = torch.zeros_like(cached)
    active = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
    compute_depth = torch.full_like(cached, candidate_count - 1)
    for depth in range(candidate_count - 1):
        if depth > 0:
            update = active & (probabilities[:, depth, 0] >= update_threshold)
            cached = torch.where(update, torch.full_like(cached, depth), cached)
        stop_evidence = active & (probabilities[:, depth, 1] <= continuation_threshold)
        stale = torch.where(active, torch.where(stop_evidence, stale + 1, torch.zeros_like(stale)), stale)
        stop = active & (stale >= patience)
        selected[stop] = cached[stop]
        compute_depth[stop] = depth
        active = active & ~stop
    return selected + min_depth, compute_depth + min_depth


def causal_survival_loss(
    logits: torch.Tensor,
    target_depths: torch.Tensor,
    *,
    min_depth: int = 0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train one STOP hazard at the oracle depth and CONTINUE before it."""
    if logits.ndim != 3 or logits.shape[2] != 2:
        raise ValueError("logits must be [B,C,2]")
    batch_size, candidate_count = logits.shape[:2]
    max_depth = min_depth + candidate_count - 1
    if target_depths.shape != (batch_size,) or target_depths.dtype not in (torch.int32, torch.int64):
        raise ValueError("target_depths must be integer [B]")
    if torch.any(target_depths < min_depth) or torch.any(target_depths > max_depth):
        raise ValueError(f"target_depths must be in [{min_depth},{max_depth}]")
    current = torch.arange(min_depth, max_depth + 1, device=logits.device).unsqueeze(0)
    reachable = current <= target_depths.unsqueeze(1)
    targets = (current == target_depths.unsqueeze(1)).long()
    per_prefix = torch.nn.functional.cross_entropy(
        logits.float().flatten(0, 1),
        targets.flatten(),
        reduction="none",
    ).reshape(batch_size, candidate_count)
    rows = torch.arange(batch_size, device=logits.device)
    stop_loss = per_prefix[rows, target_depths - min_depth]
    continue_mask = reachable & ~targets.bool()
    continue_count = continue_mask.sum(dim=1)
    continue_loss = (per_prefix * continue_mask).sum(dim=1) / continue_count.clamp_min(1)
    sample_loss = torch.where(continue_count > 0, 0.5 * (continue_loss + stop_loss), stop_loss)
    if class_weights is None:
        return sample_loss.mean()
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=logits.device)
    if weights.shape != (candidate_count,) or torch.any(weights <= 0.0):
        raise ValueError("class_weights must be positive [C]")
    sample_weights = weights[target_depths - min_depth]
    return (sample_loss * sample_weights).sum() / sample_weights.sum()


def causal_stop_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert prefix STOP logits to a normalized first-stop distribution."""
    if logits.ndim != 3 or logits.shape[2] != 2 or logits.shape[1] < 2:
        raise ValueError("logits must be [B,C>=2,2]")
    hazards = torch.softmax(logits.float(), dim=2)[:, :, 1]
    hazards = torch.cat((hazards[:, :-1], torch.ones_like(hazards[:, -1:])), dim=1)
    survival = torch.cumprod(
        torch.cat((torch.ones_like(hazards[:, :1]), 1.0 - hazards[:, :-1]), dim=1),
        dim=1,
    )
    probabilities = survival * hazards
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


def causal_expected_cost_loss(
    logits: torch.Tensor,
    trajectory_costs: torch.Tensor,
    *,
    compute_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Minimize expected trajectory cost plus normalized rollout compute."""
    if trajectory_costs.shape != logits.shape[:2] or not torch.isfinite(trajectory_costs).all():
        raise ValueError("trajectory_costs must be finite [B,C]")
    if compute_weight < 0.0:
        raise ValueError("compute_weight must be non-negative")
    probabilities = causal_stop_probabilities(logits)
    expected_cost = (probabilities * trajectory_costs.float()).sum(dim=1).mean()
    normalized_depths = torch.linspace(0.0, 1.0, logits.shape[1], device=logits.device)
    expected_compute = (probabilities * normalized_depths.unsqueeze(0)).sum(dim=1).mean()
    loss = expected_cost + compute_weight * expected_compute
    return loss, {"expected_cost": expected_cost, "expected_compute": expected_compute}


def _validate_values_and_ranks(values: torch.Tensor, rank_scores: torch.Tensor) -> tuple[int, int]:
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("values must be [B,C,C]")
    batch_size, candidate_count = values.shape[:2]
    if rank_scores.shape != (batch_size, candidate_count) or not torch.isfinite(rank_scores).all():
        raise ValueError("rank_scores must be finite [B,C]")
    reachable = torch.ones(candidate_count, candidate_count, dtype=torch.bool, device=values.device).triu()
    if not torch.isfinite(values[:, reachable]).all():
        raise ValueError("Every reachable value prediction must be finite")
    return batch_size, candidate_count


def reachable_value_gate_loss(
    values: torch.Tensor,
    rank_scores: torch.Tensor,
    *,
    regression_targets: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    regression_weight: float = 1.0,
    listwise_weight: float = 1.0,
    pairwise_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise every causal prefix with all of its reachable relative ranks."""
    _, candidate_count = _validate_values_and_ranks(values, rank_scores)
    if min(regression_weight, listwise_weight, pairwise_weight) < 0.0:
        raise ValueError("Loss weights must be non-negative")
    if regression_weight + listwise_weight + pairwise_weight <= 0.0:
        raise ValueError("At least one value-gate loss weight must be positive")
    if class_weights is not None:
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=values.device)
        if class_weights.shape != (candidate_count,) or torch.any(class_weights <= 0.0):
            raise ValueError("class_weights must be positive [C]")

    target = rank_scores.float().unsqueeze(1).expand(-1, candidate_count, -1)
    if regression_targets is None:
        regression_target = target
    else:
        if regression_targets.shape != rank_scores.shape or not torch.isfinite(regression_targets).all():
            raise ValueError("regression_targets must be finite [B,C]")
        regression_target = regression_targets.float().unsqueeze(1).expand_as(target)
    reachable = torch.ones(candidate_count, candidate_count, dtype=torch.bool, device=values.device).triu()
    mask = reachable.unsqueeze(0).expand(values.shape[0], -1, -1)
    predictions = values.float()
    regression = torch.nn.functional.smooth_l1_loss(predictions[mask], regression_target[mask])

    masked_target = target.masked_fill(~mask, torch.inf)
    minimum = masked_target.min(dim=2, keepdim=True).values
    tied_best = mask & torch.isclose(target, minimum, rtol=0.0, atol=1.0e-7)
    soft_targets = tied_best.float() / tied_best.sum(dim=2, keepdim=True)
    log_probabilities = torch.log_softmax(-predictions.masked_fill(~mask, torch.inf), dim=2)
    per_prefix_listwise = -(soft_targets * log_probabilities.masked_fill(~mask, 0.0)).sum(dim=2)
    if class_weights is not None:
        prefix_weights = (soft_targets * class_weights.view(1, 1, -1)).sum(dim=2)
        listwise = (per_prefix_listwise * prefix_weights).sum() / prefix_weights.sum()
    else:
        listwise = per_prefix_listwise.mean()

    target_differences = target.unsqueeze(3) - target.unsqueeze(2)
    prediction_differences = predictions.unsqueeze(3) - predictions.unsqueeze(2)
    pair_mask = mask.unsqueeze(3) & mask.unsqueeze(2) & (target_differences.abs() > 1.0e-7)
    pair_mask = pair_mask & torch.ones(candidate_count, candidate_count, dtype=torch.bool, device=values.device).triu(1)
    if torch.any(pair_mask):
        direction = target_differences.sign()
        pairwise = torch.nn.functional.softplus(-direction[pair_mask] * prediction_differences[pair_mask]).mean()
    else:
        pairwise = predictions.new_zeros(())
    total = regression_weight * regression + listwise_weight * listwise + pairwise_weight * pairwise
    return total, {"regression": regression, "listwise": listwise, "pairwise": pairwise}


def current_rank_value_loss(
    values: torch.Tensor,
    rank_scores: torch.Tensor,
    *,
    regression_targets: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    regression_weight: float = 1.0,
    listwise_weight: float = 1.0,
    pairwise_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress and rank the value of every already-observed rollout depth."""
    if values.ndim != 2 or rank_scores.shape != values.shape:
        raise ValueError("values and rank_scores must align as [B,C]")
    if not torch.isfinite(values).all() or not torch.isfinite(rank_scores).all():
        raise ValueError("values and rank_scores must be finite")
    if min(regression_weight, listwise_weight, pairwise_weight) < 0.0:
        raise ValueError("Current-value loss weights must be non-negative")
    if regression_weight + listwise_weight + pairwise_weight <= 0.0:
        raise ValueError("At least one current-value loss weight must be positive")
    candidate_count = values.shape[1]
    if class_weights is not None:
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=values.device)
        if class_weights.shape != (candidate_count,) or torch.any(class_weights <= 0.0):
            raise ValueError("class_weights must be positive [C]")

    predictions = values.float()
    targets = rank_scores.float()
    if regression_targets is None:
        regression_target = targets
    else:
        if regression_targets.shape != values.shape or not torch.isfinite(regression_targets).all():
            raise ValueError("regression_targets must be finite [B,C]")
        regression_target = regression_targets.float()
    regression = torch.nn.functional.smooth_l1_loss(predictions, regression_target)
    minimum = targets.min(dim=1, keepdim=True).values
    tied_best = torch.isclose(targets, minimum, rtol=0.0, atol=1.0e-7)
    soft_targets = tied_best.float() / tied_best.sum(dim=1, keepdim=True)
    per_sample_listwise = -(soft_targets * torch.log_softmax(-predictions, dim=1)).sum(dim=1)
    if class_weights is None:
        listwise = per_sample_listwise.mean()
    else:
        sample_weights = (soft_targets * class_weights.unsqueeze(0)).sum(dim=1)
        listwise = (per_sample_listwise * sample_weights).sum() / sample_weights.sum()

    target_differences = targets.unsqueeze(2) - targets.unsqueeze(1)
    prediction_differences = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    pair_mask = (target_differences.abs() > 1.0e-7) & torch.ones(
        candidate_count,
        candidate_count,
        dtype=torch.bool,
        device=values.device,
    ).triu(1)
    if torch.any(pair_mask):
        direction = target_differences.sign()
        pairwise = torch.nn.functional.softplus(
            -direction[pair_mask] * prediction_differences[pair_mask]
        ).mean()
    else:
        pairwise = predictions.new_zeros(())
    total = regression_weight * regression + listwise_weight * listwise + pairwise_weight * pairwise
    return total, {"regression": regression, "listwise": listwise, "pairwise": pairwise}


def soft_group_diagonal_loss(
    logits: torch.Tensor,
    normalized_cost: torch.Tensor,
    *,
    target_prior: torch.Tensor | None = None,
    assignment_temperature: float = 0.25,
    cost_temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make every softly predicted depth group prefer its diagonal cost."""
    if logits.ndim != 2 or normalized_cost.shape != logits.shape:
        raise ValueError("logits and normalized_cost must align as [B,C]")
    if not torch.isfinite(logits).all() or not torch.isfinite(normalized_cost).all():
        raise ValueError("logits and normalized_cost must be finite")
    if torch.any(normalized_cost < 0.0):
        raise ValueError("normalized_cost must be non-negative")
    if assignment_temperature <= 0.0 or cost_temperature <= 0.0:
        raise ValueError("Group temperatures must be positive")

    assignments = torch.softmax(logits.float() / assignment_temperature, dim=1)
    group_mass = assignments.sum(dim=0).clamp_min(1.0e-6)
    group_cost = assignments.transpose(0, 1) @ normalized_cost.float() / group_mass.unsqueeze(1)
    targets = torch.arange(logits.shape[1], device=logits.device)
    diagonal = torch.nn.functional.cross_entropy(-group_cost / cost_temperature, targets)

    predicted_prior = assignments.mean(dim=0).clamp_min(1.0e-8)
    if target_prior is None:
        prior = torch.full_like(predicted_prior, 1.0 / logits.shape[1])
    else:
        prior = torch.as_tensor(target_prior, dtype=torch.float32, device=logits.device)
        if prior.shape != (logits.shape[1],) or torch.any(prior <= 0.0) or not torch.isfinite(prior).all():
            raise ValueError("target_prior must be finite and positive [C]")
        prior = prior / prior.sum()
    balance = torch.sum(prior * (torch.log(prior) - torch.log(predicted_prior)))
    return diagonal, balance, group_cost


def select_sequential_horizon_depths(
    logits: torch.Tensor,
    candidate_bias: torch.Tensor | None = None,
    stop_thresholds: torch.Tensor | None = None,
    *,
    min_depth: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop when a causal prefix predicts that prefix as the globally best depth."""
    if logits.ndim != 3 or logits.shape[1] != logits.shape[2] or not torch.isfinite(logits).all():
        raise ValueError("Sequential horizon logits must be finite [B,C,C]")
    if min_depth != 0:
        raise NotImplementedError("Sequential horizon selection currently requires depth zero")
    candidate_count = logits.shape[1]
    if candidate_bias is None:
        bias = torch.zeros(candidate_count, dtype=logits.dtype, device=logits.device)
    else:
        bias = torch.as_tensor(candidate_bias, dtype=logits.dtype, device=logits.device)
        if bias.shape != (candidate_count,) or not torch.isfinite(bias).all():
            raise ValueError("candidate_bias must be finite [C]")
    biased_logits = logits + bias.view(1, 1, -1)
    predictions = biased_logits.argmax(dim=2)
    selected = predictions[:, -1].clone()
    compute_depth = torch.full_like(selected, candidate_count - 1)
    active = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    if stop_thresholds is None:
        confidence = None
    else:
        thresholds = torch.as_tensor(stop_thresholds, dtype=logits.dtype, device=logits.device)
        if thresholds.shape != (candidate_count - 1,) or not torch.all((thresholds > 0.0) & (thresholds <= 1.0)):
            raise ValueError("stop_thresholds must be in (0,1] with shape [C-1]")
        confidence = torch.softmax(biased_logits.float(), dim=2)
    for depth in range(candidate_count - 1):
        if confidence is None:
            stop = active & (predictions[:, depth] == depth)
        else:
            stop = active & (confidence[:, depth, depth] >= thresholds[depth])
        selected[stop] = depth
        compute_depth[stop] = depth
        active = active & ~stop
    return selected + min_depth, compute_depth + min_depth


def select_diagonal_margin_depths(
    logits: torch.Tensor,
    stop_thresholds: torch.Tensor,
    *,
    min_depth: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the first depth whose predicted diagonal-optimal probability is confident."""
    if logits.ndim != 3 or logits.shape[2] != 2 or not torch.isfinite(logits).all():
        raise ValueError("Diagonal margin logits must be finite [B,C,2]")
    if min_depth != 0:
        raise NotImplementedError("Diagonal margin selection currently requires depth zero")
    candidate_count = logits.shape[1]
    thresholds = torch.as_tensor(stop_thresholds, dtype=torch.float32, device=logits.device)
    if thresholds.shape != (candidate_count - 1,) or not torch.all((thresholds > 0.0) & (thresholds <= 1.0)):
        raise ValueError("stop_thresholds must be in (0,1] with shape [C-1]")
    probabilities = torch.sigmoid(logits[:, :, 0].float())
    selected = torch.full((logits.shape[0],), candidate_count - 1, dtype=torch.long, device=logits.device)
    active = torch.ones_like(selected, dtype=torch.bool)
    for depth in range(candidate_count - 1):
        stop = active & (probabilities[:, depth] >= thresholds[depth])
        selected[stop] = depth
        active = active & ~stop
    return selected + min_depth, selected + min_depth


def select_cached_margin_depths(
    logits: torch.Tensor,
    stop_thresholds: torch.Tensor,
    *,
    min_depth: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop from a causal prefix while allowing selection of any cached earlier plan."""
    if logits.ndim != 4 or logits.shape[1] != logits.shape[2] or logits.shape[3] != 2:
        raise ValueError("Cached margin logits must be [B,C,C,2]")
    if not torch.isfinite(logits).all():
        raise ValueError("Cached margin logits must be finite")
    if min_depth != 0:
        raise NotImplementedError("Cached margin selection currently requires depth zero")
    candidate_count = logits.shape[1]
    thresholds = torch.as_tensor(stop_thresholds, dtype=torch.float32, device=logits.device)
    if thresholds.shape != (candidate_count - 1,) or not torch.all((thresholds > 0.0) & (thresholds <= 1.0)):
        raise ValueError("stop_thresholds must be in (0,1] with shape [C-1]")
    probabilities = torch.sigmoid(logits[:, :, :, 0].float())
    selected = torch.full((logits.shape[0],), candidate_count - 1, dtype=torch.long, device=logits.device)
    compute_depth = torch.full_like(selected, candidate_count - 1)
    active = torch.ones_like(selected, dtype=torch.bool)
    for prefix in range(candidate_count - 1):
        cached_probabilities = probabilities[:, prefix, : prefix + 1]
        cached_thresholds = thresholds[: prefix + 1].unsqueeze(0)
        confidence = cached_probabilities - cached_thresholds
        best_confidence, best_depth = confidence.max(dim=1)
        stop = active & (best_confidence >= 0.0)
        selected[stop] = best_depth[stop]
        compute_depth[stop] = prefix
        active = active & ~stop
    if torch.any(active):
        selected[active] = probabilities[active, -1].argmax(dim=1)
    return selected + min_depth, compute_depth + min_depth


def select_relational_cached_margin_depths(
    logits: torch.Tensor,
    stop_thresholds: torch.Tensor,
    candidate_bias: torch.Tensor | None = None,
    *,
    min_depth: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop when the best cached plan beats the fixed-depth fallback by enough evidence."""
    if logits.ndim != 4 or logits.shape[1] != logits.shape[2] or logits.shape[3] != 2:
        raise ValueError("Relational cached margin logits must be [B,C,C,2]")
    if not torch.isfinite(logits).all():
        raise ValueError("Relational cached margin logits must be finite")
    if min_depth != 0:
        raise NotImplementedError("Relational cached margin selection currently requires depth zero")
    candidate_count = logits.shape[1]
    thresholds = torch.as_tensor(stop_thresholds, dtype=torch.float32, device=logits.device)
    if thresholds.shape != (candidate_count - 1,) or not torch.isfinite(thresholds).all():
        raise ValueError("stop_thresholds must be finite with shape [C-1]")
    if candidate_bias is None:
        bias = torch.zeros(candidate_count, dtype=torch.float32, device=logits.device)
    else:
        bias = torch.as_tensor(candidate_bias, dtype=torch.float32, device=logits.device)
        if bias.shape != (candidate_count,) or not torch.isfinite(bias).all():
            raise ValueError("candidate_bias must be finite [C]")
    scores = logits[:, :, :, 0].float() + bias.view(1, 1, -1)
    selected = torch.full((logits.shape[0],), candidate_count - 1, dtype=torch.long, device=logits.device)
    compute_depth = torch.full_like(selected, candidate_count - 1)
    active = torch.ones_like(selected, dtype=torch.bool)
    for prefix in range(candidate_count - 1):
        best_score, best_depth = scores[:, prefix, : prefix + 1].max(dim=1)
        evidence = best_score - scores[:, prefix, -1]
        stop = active & (evidence >= thresholds[prefix])
        selected[stop] = best_depth[stop]
        compute_depth[stop] = prefix
        active = active & ~stop
    selected[active] = scores[active, -1].argmax(dim=1)
    return selected + min_depth, compute_depth + min_depth


def select_relational_cached_metric_depths(
    predictions: torch.Tensor,
    stop_thresholds: torch.Tensor,
    *,
    min_depth: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select cached plans by predicted ATE+RPE cost relative to the fixed-depth fallback."""
    if predictions.ndim != 4 or predictions.shape[1] != predictions.shape[2] or predictions.shape[3] != 2:
        raise ValueError("Relational cached metric predictions must be [B,C,C,2]")
    if not torch.isfinite(predictions).all():
        raise ValueError("Relational cached metric predictions must be finite")
    if min_depth != 0:
        raise NotImplementedError("Relational cached metric selection currently requires depth zero")
    candidate_count = predictions.shape[1]
    thresholds = torch.as_tensor(stop_thresholds, dtype=torch.float32, device=predictions.device)
    if thresholds.shape != (candidate_count - 1,) or not torch.isfinite(thresholds).all():
        raise ValueError("stop_thresholds must be finite with shape [C-1]")
    costs = predictions.float().sum(dim=3)
    selected = torch.full((predictions.shape[0],), candidate_count - 1, dtype=torch.long, device=predictions.device)
    compute_depth = torch.full_like(selected, candidate_count - 1)
    active = torch.ones_like(selected, dtype=torch.bool)
    for prefix in range(candidate_count - 1):
        best_cost, best_depth = costs[:, prefix, : prefix + 1].min(dim=1)
        evidence = costs[:, prefix, -1] - best_cost
        stop = active & (evidence >= thresholds[prefix])
        selected[stop] = best_depth[stop]
        compute_depth[stop] = prefix
        active = active & ~stop
    selected[active] = costs[active, -1].argmin(dim=1)
    return selected + min_depth, compute_depth + min_depth


def select_patience_value_depths(
    values: torch.Tensor,
    *,
    patience: int = 3,
    improvement_margin: float = 0.0,
    min_depth: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop after repeated non-improvement and return the cached best depth."""
    if values.ndim != 2 or values.shape[1] < 2 or not torch.isfinite(values).all():
        raise ValueError("values must be finite [B,C>=2]")
    if patience < 1 or patience >= values.shape[1]:
        raise ValueError("patience must be in [1,C-1]")
    if improvement_margin < 0.0:
        raise ValueError("improvement_margin must be non-negative")
    if min_depth not in (0, 1):
        raise ValueError("min_depth must be zero or one")
    batch_size, candidate_count = values.shape
    rows = torch.arange(batch_size, device=values.device)
    best = torch.zeros(batch_size, dtype=torch.long, device=values.device)
    selected = best.clone()
    stale = torch.zeros_like(best)
    active = torch.ones(batch_size, dtype=torch.bool, device=values.device)
    compute_depth = torch.full_like(best, candidate_count - 1)
    for depth in range(1, candidate_count):
        improved = active & (values[:, depth] < values[rows, best] - improvement_margin)
        best = torch.where(improved, torch.full_like(best, depth), best)
        stale = torch.where(active, torch.where(improved, torch.zeros_like(stale), stale + 1), stale)
        stop = active & (stale >= patience)
        selected[stop] = best[stop]
        compute_depth[stop] = depth
        active = active & ~stop
    selected[active] = best[active]
    return selected + min_depth, compute_depth + min_depth


def select_value_gate_depths(
    values: torch.Tensor,
    candidate_bias: torch.Tensor | None = None,
    *,
    min_depth: int = 0,
) -> torch.Tensor:
    """Select the first prefix whose current plan has minimum predicted reachable value."""
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("values must be [B,C,C]")
    candidate_count = values.shape[1]
    if min_depth not in (0, 1):
        raise ValueError("min_depth must be zero or one")
    if candidate_bias is None:
        bias = values.new_zeros(candidate_count, dtype=torch.float32)
    else:
        bias = torch.as_tensor(candidate_bias, dtype=torch.float32, device=values.device)
        if bias.shape != (candidate_count,) or not torch.isfinite(bias).all():
            raise ValueError("candidate_bias must be finite [C]")
    adjusted = values.float() + bias.view(1, 1, -1)
    selected = torch.full((values.shape[0],), candidate_count - 1, dtype=torch.long, device=values.device)
    active = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    for prefix in range(candidate_count):
        stop = active & (adjusted[:, prefix].argmin(dim=1) == prefix)
        selected[stop] = prefix
        active = active & ~stop
    return selected + min_depth


def depth_group_summary(
    relative_ate: torch.Tensor,
    relative_rpe: torch.Tensor,
    selected_depths: torch.Tensor,
    *,
    ate_scale: float,
    rpe_scale: float,
    min_depth: int = 0,
) -> dict[str, Any]:
    """Build the offline 9x9 group matrix used as the Gate acceptance test."""
    if relative_ate.ndim != 2 or relative_rpe.shape != relative_ate.shape:
        raise ValueError("relative_ate and relative_rpe must align as [N,C]")
    count, candidate_count = relative_ate.shape
    if selected_depths.shape != (count,):
        raise ValueError("selected_depths must be [N]")
    if ate_scale <= 0.0 or rpe_scale <= 0.0:
        raise ValueError("Metric scales must be positive")
    indices = selected_depths.cpu().long() - min_depth
    if torch.any(indices < 0) or torch.any(indices >= candidate_count):
        raise ValueError("selected_depths are outside the candidate range")
    costs = relative_ate.float().cpu() / ate_scale + relative_rpe.float().cpu() / rpe_scale
    rows = []
    matrix: list[list[float] | None] = []
    for depth_index in range(candidate_count):
        members = indices == depth_index
        samples = int(members.sum())
        if samples == 0:
            matrix.append(None)
            rows.append(
                {
                    "group_depth": depth_index + min_depth,
                    "samples": 0,
                    "best_depth": None,
                    "diagonal_is_best": None,
                    "diagonal_regret": None,
                }
            )
            continue
        means = costs[members].mean(dim=0)
        best_index = int(means.argmin())
        regret = float(means[depth_index] - means[best_index])
        matrix.append([float(value) for value in means])
        rows.append(
            {
                "group_depth": depth_index + min_depth,
                "samples": samples,
                "best_depth": best_index + min_depth,
                "diagonal_is_best": math.isclose(regret, 0.0, rel_tol=0.0, abs_tol=1.0e-8),
                "diagonal_regret": regret,
            }
        )
    populated = [row for row in rows if row["samples"] > 0]
    diagonal = sum(bool(row["diagonal_is_best"]) for row in populated)
    return {
        "group_counts": [row["samples"] for row in rows],
        "rows": rows,
        "normalized_cost_matrix": matrix,
        "populated_groups": len(populated),
        "diagonal_optimal_groups": diagonal,
        "all_groups_populated": len(populated) == candidate_count,
        "all_groups_diagonal_optimal": len(populated) == candidate_count and diagonal == candidate_count,
        "mean_diagonal_regret": (
            sum(float(row["diagonal_regret"]) for row in populated) / len(populated) if populated else math.inf
        ),
    }


def inverse_sqrt_class_weights(target_depths: torch.Tensor, candidate_count: int, min_depth: int = 0) -> torch.Tensor:
    """Return bounded inverse-frequency weights for imbalanced oracle depths."""
    indices = torch.as_tensor(target_depths, dtype=torch.long) - min_depth
    if indices.ndim != 1 or torch.any(indices < 0) or torch.any(indices >= candidate_count):
        raise ValueError("target_depths are outside the candidate range")
    counts = torch.bincount(indices, minlength=candidate_count).float()
    if torch.any(counts == 0):
        raise ValueError("Every candidate depth needs at least one training target")
    weights = counts.rsqrt()
    weights = weights / weights.mean()
    return weights.clamp(max=4.0)
