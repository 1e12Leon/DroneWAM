"""Train the iterative planner or the joint rollout-depth gate."""

from __future__ import annotations

import argparse
import logging

from dronewam.config import load_config
from dronewam.models.wam import build_model
from dronewam.training.adaptive import (
    load_adaptive_checkpoint,
    train_joint_horizon_wam,
    train_planner,
)
from dronewam.training.checkpoint import load_checkpoint
from dronewam.training.distributed import initialize, shutdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("planner", "joint-horizon"),
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--adaptive-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reset-planner", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    if not config.model.adaptive.enabled:
        raise ValueError("Adaptive training requires model.adaptive.enabled=true")
    distributed = initialize(config.optimization.seed)
    try:
        if args.reset_planner and args.stage != "joint-horizon":
            raise ValueError("--reset-planner is valid only for stage=joint-horizon")
        model = build_model(config)
        load_checkpoint(
            model,
            args.base_checkpoint,
            load_optimizer=False,
            allow_missing_adaptive=True,
        )
        if args.stage == "planner":
            if args.adaptive_checkpoint:
                load_adaptive_checkpoint(
                    model,
                    args.adaptive_checkpoint,
                    require_planner=True,
                    require_policy=False,
                )
            train_planner(model, config, distributed, args.output)
            return

        if not args.adaptive_checkpoint:
            raise ValueError("--adaptive-checkpoint is required for stage=joint-horizon")
        load_adaptive_checkpoint(
            model,
            args.adaptive_checkpoint,
            require_planner=not args.reset_planner,
            require_policy=False,
            require_predictor=args.reset_planner,
            load_planner=not args.reset_planner,
        )
        train_joint_horizon_wam(model, config, distributed, args.output)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
