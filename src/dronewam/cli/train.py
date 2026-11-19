"""Train the clean DroneWAM core."""

from __future__ import annotations

import argparse
import logging

from dronewam.config import load_config
from dronewam.data import build_loader
from dronewam.models.wam import build_model
from dronewam.training.distributed import initialize, shutdown
from dronewam.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    distributed = initialize(config.optimization.seed)
    try:
        loader = build_loader(
            config.data,
            config.model.encoder.resolution,
            split="train",
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        model = build_model(config)
        trainer = Trainer(model, config, distributed, steps_per_epoch=len(loader))
        trainer.fit(loader)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
