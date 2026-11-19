"""Fit global rollout-depth priors for a cached causal joint Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dronewam.cli.evaluate_joint_horizon import select_online_depths
from dronewam.iterative_evaluation import selected_depth_metrics
from dronewam.value_gate import depth_group_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ate-scale", type=float, default=0.04)
    parser.add_argument("--rpe-scale", type=float, default=0.017)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--min-group-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=239)
    return parser.parse_args()


def _summary(
    logits: torch.Tensor,
    relative_ate: torch.Tensor,
    relative_rpe: torch.Tensor,
    bias: torch.Tensor,
    *,
    ate_scale: float,
    rpe_scale: float,
    min_group_samples: int,
) -> dict:
    depths = select_online_depths(logits, bias)
    cost = relative_ate.float() / ate_scale + relative_rpe.float() / rpe_scale
    selection = selected_depth_metrics(
        relative_ate,
        relative_rpe,
        depths,
        ate_scale=ate_scale,
        rpe_scale=rpe_scale,
        target_depths=cost.argmin(dim=1),
        rank_scores=cost,
        min_depth=0,
    )
    groups = depth_group_summary(
        relative_ate,
        relative_rpe,
        depths,
        ate_scale=ate_scale,
        rpe_scale=rpe_scale,
        min_depth=0,
    )
    robust_rows = [
        row for row in groups["rows"] if int(row["samples"]) >= min_group_samples
    ]
    robust_regrets = [float(row["diagonal_regret"]) for row in robust_rows]
    groups["min_robust_group_samples"] = min_group_samples
    groups["robust_populated_groups"] = len(robust_rows)
    groups["robust_diagonal_optimal_groups"] = sum(
        bool(row["diagonal_is_best"]) for row in robust_rows
    )
    groups["robust_mean_diagonal_regret"] = (
        sum(robust_regrets) / len(robust_regrets) if robust_regrets else float("inf")
    )
    return {"selection": selection, "groups": groups}


def _selection_key(summary: dict, candidate_count: int) -> tuple:
    groups = summary["groups"]
    selection = summary["selection"]
    return (
        candidate_count - int(groups["populated_groups"]),
        candidate_count - int(groups["diagonal_optimal_groups"]),
        candidate_count - int(groups["robust_populated_groups"]),
        candidate_count - int(groups["robust_diagonal_optimal_groups"]),
        float(groups["robust_mean_diagonal_regret"]),
        float(selection["mean_rank_regret"]),
        float(selection["normalized_cost"]),
    )


@torch.no_grad()
def calibrate_candidate_bias(
    logits: torch.Tensor,
    relative_ate: torch.Tensor,
    relative_rpe: torch.Tensor,
    *,
    ate_scale: float,
    rpe_scale: float,
    sweeps: int,
    restarts: int,
    min_group_samples: int,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    if logits.ndim != 3 or logits.shape[1] != logits.shape[2]:
        raise ValueError("logits must be square [N,C,C]")
    if relative_ate.shape != logits.shape[:2] or relative_rpe.shape != logits.shape[:2]:
        raise ValueError("trajectory metrics must align with logits as [N,C]")
    if min(sweeps, restarts, min_group_samples) < 1 or min(ate_scale, rpe_scale) <= 0.0:
        raise ValueError("calibration dimensions and metric scales must be positive")
    candidate_count = logits.shape[1]
    grid = torch.tensor(
        (
            -1.5,
            -1.0,
            -0.75,
            -0.5,
            -0.35,
            -0.25,
            -0.15,
            -0.1,
            -0.05,
            0.0,
            0.05,
            0.1,
            0.15,
            0.25,
            0.35,
            0.5,
            0.75,
            1.0,
            1.5,
        )
    )
    generator = torch.Generator().manual_seed(seed)
    starts = [torch.zeros(candidate_count)]
    for _ in range(restarts - 1):
        start = torch.randn(candidate_count, generator=generator) * 0.35
        starts.append(start - start[-1])

    def evaluate(bias: torch.Tensor) -> tuple[tuple, dict]:
        summary = _summary(
            logits,
            relative_ate,
            relative_rpe,
            bias,
            ate_scale=ate_scale,
            rpe_scale=rpe_scale,
            min_group_samples=min_group_samples,
        )
        return _selection_key(summary, candidate_count), summary

    best: tuple | None = None
    for initial in starts:
        bias = initial.clone()
        for _ in range(sweeps):
            changed = False
            for depth in range(candidate_count - 1):
                local = None
                for value in grid:
                    candidate = bias.clone()
                    candidate[depth] = value
                    candidate = candidate - candidate[-1]
                    key, summary = evaluate(candidate)
                    result = (key, float(value), candidate, summary)
                    if local is None or result[0] < local[0]:
                        local = result
                assert local is not None
                changed |= not torch.equal(local[2], bias)
                bias = local[2]
            if not changed:
                break
        key, summary = evaluate(bias)
        result = (key, bias.clone(), summary)
        if best is None or result[0] < best[0]:
            best = result
    assert best is not None
    return best[1], best[2]


def main() -> None:
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    bias, summary = calibrate_candidate_bias(
        cache["logits"].float(),
        cache["relative_ate"].float(),
        cache["relative_rpe"].float(),
        ate_scale=args.ate_scale,
        rpe_scale=args.rpe_scale,
        sweeps=args.sweeps,
        restarts=args.restarts,
        min_group_samples=args.min_group_samples,
        seed=args.seed,
    )
    report = {
        "objective": "causal_joint_gate_candidate_bias_v1",
        "cache": str(Path(args.cache)),
        "candidate_bias": bias.tolist(),
        "calibration": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
