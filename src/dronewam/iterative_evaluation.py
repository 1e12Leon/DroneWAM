"""Shared metrics for recursive rollout-depth selection."""

from __future__ import annotations

import torch


def validate_metric_scales(ate_scale: float, rpe_scale: float) -> None:
    if ate_scale <= 0.0 or rpe_scale <= 0.0:
        raise ValueError("Metric scales must be positive")


def selected_depth_metrics(
    relative_ate: torch.Tensor,
    relative_rpe: torch.Tensor,
    depths: torch.Tensor,
    *,
    ate_scale: float,
    rpe_scale: float,
    target_depths: torch.Tensor | None = None,
    rank_scores: torch.Tensor | None = None,
    min_depth: int = 1,
) -> dict[str, float | int]:
    """Summarize per-depth metrics selected by one deployable policy."""
    validate_metric_scales(ate_scale, rpe_scale)
    if relative_ate.ndim != 2 or relative_rpe.shape != relative_ate.shape:
        raise ValueError("relative_ate and relative_rpe must align as [N,H]")
    count, candidate_count = relative_ate.shape
    max_depth = min_depth + candidate_count - 1
    if depths.shape != (count,) or depths.dtype not in (torch.int32, torch.int64):
        raise ValueError("depths must be an integer [N] tensor")
    if torch.any(depths < min_depth) or torch.any(depths > max_depth):
        raise ValueError(f"depths must be in [{min_depth},{max_depth}]")
    rows = torch.arange(count)
    selected = depths.cpu() - min_depth
    ate = relative_ate.cpu()[rows, selected]
    rpe = relative_rpe.cpu()[rows, selected]
    result: dict[str, float | int] = {
        "samples": count,
        "relative_ate": float(ate.mean()),
        "relative_rpe": float(rpe.mean()),
        "normalized_cost": float((ate / ate_scale + rpe / rpe_scale).mean()),
        "mean_selected_depth": float(depths.float().mean()),
    }
    if target_depths is not None:
        if target_depths.shape != (count,):
            raise ValueError("target_depths must be [N]")
        result["exact_target_hit"] = float((depths.cpu() == target_depths.cpu()).float().mean())
    if rank_scores is not None:
        if rank_scores.shape != (count, candidate_count):
            raise ValueError("rank_scores must align with the metric candidates")
        ranks = rank_scores.cpu()
        best_ranks = ranks.min(dim=1).values
        selected_ranks = ranks[rows, selected]
        result["rank_hit"] = float((selected_ranks == best_ranks).float().mean())
        result["mean_rank_regret"] = float((selected_ranks - best_ranks).mean())
    return result


def best_fixed_depth(
    relative_ate: torch.Tensor,
    relative_rpe: torch.Tensor,
    *,
    ate_scale: float,
    rpe_scale: float,
    min_depth: int = 1,
) -> int:
    """Select one fixed depth using only a calibration split."""
    validate_metric_scales(ate_scale, rpe_scale)
    if relative_ate.ndim != 2 or relative_rpe.shape != relative_ate.shape:
        raise ValueError("relative_ate and relative_rpe must align as [N,H]")
    cost = relative_ate / ate_scale + relative_rpe / rpe_scale
    return int(cost.mean(dim=0).argmin()) + min_depth


def bootstrap_cost_delta(
    dynamic_ate: torch.Tensor,
    dynamic_rpe: torch.Tensor,
    fixed_ate: torch.Tensor,
    fixed_rpe: torch.Tensor,
    *,
    ate_scale: float,
    rpe_scale: float,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the paired normalized-cost difference, dynamic minus fixed."""
    validate_metric_scales(ate_scale, rpe_scale)
    tensors = (dynamic_ate, dynamic_rpe, fixed_ate, fixed_rpe)
    if any(value.ndim != 1 for value in tensors) or len({value.shape for value in tensors}) != 1:
        raise ValueError("Metric vectors must align as [N]")
    if dynamic_ate.numel() == 0:
        raise ValueError("Metric vectors must be non-empty")
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    delta = (dynamic_ate / ate_scale + dynamic_rpe / rpe_scale) - (fixed_ate / ate_scale + fixed_rpe / rpe_scale)
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(delta.numel(), (samples, delta.numel()), generator=generator)
    means = delta.cpu()[draws].mean(dim=1)
    quantiles = torch.quantile(means, torch.tensor([0.025, 0.975]))
    return {
        "dynamic_minus_fixed_mean": float(delta.mean()),
        "ci95_low": float(quantiles[0]),
        "ci95_high": float(quantiles[1]),
    }
