"""Evaluate a jointly trained zero-to-horizon Gate on held-out trajectories."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from dronewam.config import load_config
from dronewam.data import build_preference_dataset, load_records
from dronewam.geometry import build_timeline
from dronewam.iterative_evaluation import selected_depth_metrics
from dronewam.metrics import trajectory_metric_tensors
from dronewam.models.adaptive import IterativeDepthPlanner, PlannerStateHorizonHead
from dronewam.models.wam import build_model
from dronewam.preferences import preference_split_indices
from dronewam.training.checkpoint import load_checkpoint
from dronewam.training.distributed import initialize, shutdown
from dronewam.value_gate import depth_group_summary

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--windows-per-record", type=int, default=4)
    parser.add_argument("--calibration-subset", action="store_true")
    parser.add_argument("--candidate-bias")
    parser.add_argument("--cache-output")
    return parser.parse_args()


def select_online_depths(
    logits: torch.Tensor,
    candidate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stop at the first prefix that predicts itself, otherwise use the final depth."""
    if logits.ndim != 3 or logits.shape[1] != logits.shape[2]:
        raise ValueError("Joint horizon logits must be square [B,C,C]")
    candidate_count = logits.shape[1]
    if candidate_bias is None:
        bias = logits.new_zeros(candidate_count)
    else:
        bias = torch.as_tensor(candidate_bias, dtype=logits.dtype, device=logits.device)
        if bias.shape != (candidate_count,) or not torch.isfinite(bias).all():
            raise ValueError("candidate_bias must be finite [C]")
    decisions = (logits + bias.view(1, 1, -1)).argmax(dim=2)
    selected = torch.full(
        (logits.shape[0],),
        candidate_count - 1,
        dtype=torch.long,
        device=logits.device,
    )
    active = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    for depth in range(candidate_count - 1):
        stop = active & (decisions[:, depth] == depth)
        selected[stop] = depth
        active &= ~stop
    return selected


def _load_joint_checkpoint(
    model: torch.nn.Module, path: str | Path
) -> tuple[PlannerStateHorizonHead, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != "joint_horizon":
        raise ValueError("Expected a joint_horizon checkpoint")
    if checkpoint.get("policy_type") != "PlannerStateHorizonHead":
        raise ValueError(
            "Held-out iterative evaluation requires PlannerStateHorizonHead"
        )
    if not isinstance(model.adaptive_planner, IterativeDepthPlanner):
        raise TypeError("Joint horizon evaluation requires IterativeDepthPlanner")
    if (
        int(checkpoint["policy_action_dim"]) <= 0
        or int(checkpoint["policy_plan_horizon"]) <= 0
    ):
        raise ValueError(
            "Joint horizon checkpoint must contain a plan-conditioned Gate"
        )
    model.predictor.load_state_dict(checkpoint["predictor"], strict=True)
    model.adaptive_planner.load_state_dict(checkpoint["adaptive_planner"], strict=True)
    policy = PlannerStateHorizonHead(
        hidden_dim=int(checkpoint["policy_hidden_dim"]),
        horizon=int(checkpoint["policy_horizon"]),
        expansion=int(checkpoint["policy_expansion"]),
        dropout=float(checkpoint["policy_dropout"]),
        action_dim=int(checkpoint["policy_action_dim"]),
        plan_horizon=int(checkpoint["policy_plan_horizon"]),
    )
    policy.load_state_dict(checkpoint["horizon_policy"], strict=True)
    return policy, checkpoint


@torch.no_grad()
def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.windows_per_record < 1:
        raise ValueError("windows_per_record must be positive")
    config = load_config(args.config)
    if args.calibration_subset and args.split != "train":
        raise ValueError("--calibration-subset requires --split train")
    candidate_bias = None
    if args.candidate_bias:
        payload = json.loads(Path(args.candidate_bias).read_text(encoding="utf-8"))
        values = payload.get("candidate_bias") if isinstance(payload, dict) else payload
        candidate_bias = torch.tensor(values, dtype=torch.float32)
    distributed = initialize(config.optimization.seed)
    try:
        model = build_model(config)
        load_checkpoint(
            model,
            args.base_checkpoint,
            load_optimizer=False,
            allow_missing_adaptive=True,
        )
        policy, checkpoint = _load_joint_checkpoint(model, args.joint_checkpoint)
        if model.action_head is None:
            raise RuntimeError("Joint horizon evaluation requires an Action Head")
        model = model.to(distributed.device).eval().requires_grad_(False)
        policy = policy.to(distributed.device).eval().requires_grad_(False)
        candidate_count = model.num_steps - model.observed_steps + 1
        if policy.horizon != candidate_count:
            raise ValueError(
                f"Policy candidates={policy.horizon} do not match model candidates={candidate_count}"
            )

        record_indices = None
        if args.calibration_subset:
            _, calibration_indices = preference_split_indices(
                len(load_records(config.data.train_csv)),
                calibration_fraction=config.adaptive_training.calibration_fraction,
                seed=config.optimization.seed,
            )
            record_indices = calibration_indices.tolist()
        dataset = build_preference_dataset(
            config.data,
            config.model.encoder.resolution,
            args.windows_per_record,
            record_indices,
            source_split=args.split,
        )
        local_indices = list(
            range(distributed.rank, len(dataset), distributed.world_size)
        )
        loader = DataLoader(
            Subset(dataset, local_indices),
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True,
            persistent_workers=config.data.num_workers > 0,
        )
        selected_parts = []
        decision_parts = []
        logit_parts = []
        ate_parts = []
        rpe_parts = []
        offset = 0
        sample_indices = []
        for batch in loader:
            frames = batch["frames"].to(distributed.device, non_blocking=True)
            states = batch["states"].to(distributed.device, non_blocking=True)
            goals = batch["goal_frame"].to(distributed.device, non_blocking=True)
            with torch.autocast(
                device_type=distributed.device.type,
                dtype=torch.bfloat16,
                enabled=distributed.device.type == "cuda",
            ):
                observed_frames = frames[:, :, : model.num_observed_frames]
                observed_states = states[:, : model.num_observed_frames]
                world = model.rollout_world(observed_frames, observed_states)
                goal_tokens = model.encode_goal(goals)
                context_by_step = model._tokens_by_step(world.observed_context)
                future_by_step = model._tokens_by_step(world.future_latents)
                with torch.autocast(device_type=distributed.device.type, enabled=False):
                    actions, planner_states = model.adaptive_planner.rollout_all_depths(
                        context_by_step.float(),
                        future_by_step.float(),
                        world.observed_kinematics.float(),
                        model.action_head,
                        goal_tokens.float(),
                        include_zero_depth=True,
                    )
                    logits = policy(planner_states.float(), actions.float())
                selected = select_online_depths(logits, candidate_bias)
                bias = (
                    logits.new_zeros(candidate_count)
                    if candidate_bias is None
                    else candidate_bias.to(logits)
                )
                decisions = (logits + bias.view(1, 1, -1)).argmax(dim=2)
                timeline_states, _ = build_timeline(states, model.frame_stride)
                expanded_states = (
                    timeline_states[:, None]
                    .expand(-1, candidate_count, -1, -1)
                    .flatten(0, 1)
                )
                metrics = trajectory_metric_tensors(
                    actions.flatten(0, 1), expanded_states, model.observed_steps
                )
            count = frames.shape[0]
            selected_parts.append(selected.cpu())
            decision_parts.append(decisions.cpu())
            logit_parts.append(logits.cpu())
            ate_parts.append(metrics.relative_ate.reshape(count, candidate_count).cpu())
            rpe_parts.append(metrics.relative_rpe.reshape(count, candidate_count).cpu())
            sample_indices.extend(local_indices[offset : offset + count])
            offset += count
        local = {
            "indices": sample_indices,
            "selected": torch.cat(selected_parts),
            "decisions": torch.cat(decision_parts),
            "logits": torch.cat(logit_parts),
            "relative_ate": torch.cat(ate_parts),
            "relative_rpe": torch.cat(rpe_parts),
        }
        gathered = [None for _ in range(distributed.world_size)]
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if distributed.rank != 0:
            return
        rows = []
        for shard in gathered:
            assert shard is not None
            rows.extend(
                zip(
                    shard["indices"],
                    shard["selected"].tolist(),
                    shard["decisions"],
                    shard["logits"],
                    shard["relative_ate"],
                    shard["relative_rpe"],
                )
            )
        rows.sort(key=lambda row: row[0])
        selected = torch.tensor([row[1] for row in rows], dtype=torch.long)
        decisions = torch.stack([row[2] for row in rows])
        logits = torch.stack([row[3] for row in rows])
        relative_ate = torch.stack([row[4] for row in rows])
        relative_rpe = torch.stack([row[5] for row in rows])
        cost = relative_ate / config.adaptive_training.ate_target
        cost = cost + relative_rpe / config.adaptive_training.rpe_target
        selection = selected_depth_metrics(
            relative_ate,
            relative_rpe,
            selected,
            ate_scale=config.adaptive_training.ate_target,
            rpe_scale=config.adaptive_training.rpe_target,
            target_depths=cost.argmin(dim=1),
            rank_scores=cost,
            min_depth=0,
        )
        groups = depth_group_summary(
            relative_ate,
            relative_rpe,
            selected,
            ate_scale=config.adaptive_training.ate_target,
            rpe_scale=config.adaptive_training.rpe_target,
            min_depth=0,
        )
        upfront_selected = decisions[:, 0]
        upfront_selection = selected_depth_metrics(
            relative_ate,
            relative_rpe,
            upfront_selected,
            ate_scale=config.adaptive_training.ate_target,
            rpe_scale=config.adaptive_training.rpe_target,
            target_depths=cost.argmin(dim=1),
            rank_scores=cost,
            min_depth=0,
        )
        upfront_groups = depth_group_summary(
            relative_ate,
            relative_rpe,
            upfront_selected,
            ate_scale=config.adaptive_training.ate_target,
            rpe_scale=config.adaptive_training.rpe_target,
            min_depth=0,
        )
        prefix_decision_counts = [
            torch.bincount(decisions[:, prefix], minlength=candidate_count).tolist()
            for prefix in range(candidate_count)
        ]
        fixed = {}
        for depth in range(candidate_count):
            depths = torch.full_like(selected, depth)
            fixed[str(depth)] = selected_depth_metrics(
                relative_ate,
                relative_rpe,
                depths,
                ate_scale=config.adaptive_training.ate_target,
                rpe_scale=config.adaptive_training.rpe_target,
                min_depth=0,
            )
        report = {
            "objective": "joint_zero_to_horizon_gate_held_out_v1",
            "checkpoint": str(Path(args.joint_checkpoint)),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "split": args.split,
            "windows_per_record": args.windows_per_record,
            "samples": len(rows),
            "candidate_count": candidate_count,
            "candidate_bias": (
                torch.zeros(candidate_count).tolist()
                if candidate_bias is None
                else candidate_bias.tolist()
            ),
            "selection": selection,
            "groups": groups,
            "prefix_decision_counts": prefix_decision_counts,
            "upfront_selection": upfront_selection,
            "upfront_groups": upfront_groups,
            "fixed": fixed,
            "held_out_gate_accepted": bool(groups["all_groups_diagonal_optimal"]),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.cache_output:
            cache_output = Path(args.cache_output)
            cache_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format_version": 1,
                    "split": args.split,
                    "calibration_subset": bool(args.calibration_subset),
                    "logits": logits,
                    "relative_ate": relative_ate,
                    "relative_rpe": relative_rpe,
                },
                cache_output,
            )
        logger.info(
            "joint horizon held-out report=%s", json.dumps(report, sort_keys=True)
        )
    finally:
        shutdown()


if __name__ == "__main__":
    main()
