"""Run deployable observation-only latent/action rollout on a tensor sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dronewam.config import load_config
from dronewam.models.wam import build_model
from dronewam.preferences import load_threshold_calibration
from dronewam.training.adaptive import load_adaptive_checkpoint
from dronewam.training.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adaptive-checkpoint")
    parser.add_argument("--calibration")
    parser.add_argument("--input", required=True, help=".pt dict with normalized frames [C,T,H,W] and states [T,7]")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if bool(args.adaptive_checkpoint) != bool(args.calibration):
        raise ValueError("--adaptive-checkpoint and --calibration must be provided together")
    if args.adaptive_checkpoint and not config.model.adaptive.enabled:
        raise ValueError("Adaptive rollout requires model.adaptive.enabled=true")
    if config.model.adaptive.enabled and not args.adaptive_checkpoint:
        raise ValueError("Adaptive-enabled config requires --adaptive-checkpoint and --calibration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    load_checkpoint(
        model,
        args.checkpoint,
        load_optimizer=False,
        allow_missing_adaptive=bool(args.adaptive_checkpoint),
    )
    if args.adaptive_checkpoint:
        load_adaptive_checkpoint(
            model,
            args.adaptive_checkpoint,
            require_planner=True,
            require_policy=True,
        )
    sample = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(sample, dict) or set(sample) != {"frames", "states"}:
        raise ValueError("Input must contain exactly {'frames', 'states'}")
    frames = sample["frames"].unsqueeze(0).to(device)
    states = sample["states"].unsqueeze(0).to(device)
    with torch.no_grad():
        if args.adaptive_checkpoint:
            calibration = load_threshold_calibration(args.calibration)
            stop_thresholds = torch.tensor(
                calibration.stop_thresholds,
                device=device,
            )
            output = model.adaptive_rollout(
                frames,
                states,
                min_rollout_depth=config.model.adaptive.min_rollout_depth,
                stop_thresholds=stop_thresholds,
                dpo_beta=config.adaptive_training.dpo_beta,
            )
        else:
            output = model.rollout(frames, states)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "future_latents": output.future_latents.cpu(),
            "predicted_actions": None if output.predicted_actions is None else output.predicted_actions.cpu(),
            "rollout_depths": (
                output.rollout_depths.cpu() if args.adaptive_checkpoint else None
            ),
            "plan_depths": (
                output.plan_depths.cpu() if args.adaptive_checkpoint else None
            ),
            "reward_probabilities": (
                output.reward_probabilities.cpu() if args.adaptive_checkpoint else None
            ),
        },
        destination,
    )


if __name__ == "__main__":
    main()
