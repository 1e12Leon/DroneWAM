"""Offline validation for latent and initial-action prediction."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from dronewam.config import load_config
from dronewam.data import build_exact_eval_loader
from dronewam.metrics import BatchMetrics, compute_batch_metrics
from dronewam.models.wam import build_model
from dronewam.training.checkpoint import load_checkpoint
from dronewam.training.distributed import initialize, shutdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> BatchMetrics:
    model.eval()
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    for batch in loader:
        output = model(
            batch["frames"].to(device, non_blocking=True),
            batch["states"].to(device, non_blocking=True),
            compute_targets=True,
        )
        metrics = compute_batch_metrics(output)
        values = [
            metrics.count,
            metrics.latent_l1 * metrics.count,
            metrics.action_rmse * metrics.count,
            metrics.ate * metrics.count,
            metrics.rpe * metrics.count,
            metrics.relative_ate * metrics.count,
            metrics.relative_rpe * metrics.count,
            metrics.attitude_error * metrics.count,
        ]
        totals += torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    count = int(totals[0].item())
    if count == 0:
        raise ValueError("Validation loader produced zero samples")
    means = (totals[1:] / count).tolist()
    return BatchMetrics(count, *means)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    distributed = initialize(config.optimization.seed)
    try:
        loader = build_exact_eval_loader(
            config.data,
            config.model.encoder.resolution,
            distributed.rank,
            distributed.world_size,
        )
        model = build_model(config).to(distributed.device)
        load_checkpoint(model, args.checkpoint, load_optimizer=False)
        metrics = evaluate(model, loader, distributed.device)
        if distributed.rank == 0:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")
            logging.info("metrics=%s", asdict(metrics))
    finally:
        shutdown()


if __name__ == "__main__":
    main()
