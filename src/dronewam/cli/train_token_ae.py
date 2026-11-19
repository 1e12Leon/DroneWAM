"""Train a backbone-specific TokenAE on frozen visual representations."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dronewam.config import load_token_ae_training_config
from dronewam.data import build_loader
from dronewam.models.encoder import build_encoder
from dronewam.training.distributed import initialize, shutdown
from dronewam.training.token_ae import TokenAETrainer, build_token_ae, save_token_ae_checkpoint

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_token_ae_training_config(args.config)
    distributed = initialize(config.optimization.seed)
    try:
        loader = build_loader(
            config.data,
            config.encoder.resolution,
            split="train",
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        encoder = build_encoder(config.encoder, config.data)
        token_ae = build_token_ae(config, encoder.embed_dim, encoder.tokens_per_frame)
        trainer = TokenAETrainer(encoder, token_ae, config, distributed, len(loader))
        output = Path(config.runtime.output_dir) / "token_ae_latest.pt"
        for epoch in range(config.optimization.epochs):
            metrics = trainer.train_epoch(loader, epoch)
            if distributed.rank == 0:
                logger.info("epoch=%d metrics=%s", epoch + 1, metrics)
                save_token_ae_checkpoint(
                    trainer.token_ae,
                    trainer.optimizer,
                    config,
                    output,
                    epoch,
                    metrics,
                    trainer.embed_dim,
                    trainer.tokens_per_step,
                )
    finally:
        shutdown()


if __name__ == "__main__":
    main()
