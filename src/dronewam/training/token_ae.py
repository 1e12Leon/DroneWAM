"""Standalone TokenAE training on frozen encoder representations."""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from dronewam.config import TokenAETrainingConfig
from dronewam.models.token_ae import TokenAE
from dronewam.training.distributed import DistributedContext


def build_token_ae(config: TokenAETrainingConfig, embed_dim: int, tokens_per_step: int) -> TokenAE:
    """Build the exact Resampler architecture recorded in the training config."""
    token = config.token_ae
    return TokenAE(
        embed_dim=embed_dim,
        tokens_per_frame=tokens_per_step,
        num_latent_tokens=token.num_latent_tokens,
        num_heads=token.num_heads,
        encoder_depth=token.encoder_depth,
        decoder_depth=token.decoder_depth,
        mlp_ratio=token.mlp_ratio,
        dropout=token.dropout,
        encoder_mode=token.encoder_mode,
        loss_type=token.loss_type,
        cos_loss_weight=token.cos_loss_weight,
        latent_reg_weight=token.latent_reg_weight,
        pos_embed_type=token.pos_embed_type,
        input_grid_size=token.input_grid_size,
        latent_grid_size=token.latent_grid_size,
        temporal_depth=token.temporal_depth,
        temporal_num_heads=token.temporal_num_heads,
        temporal_mlp_ratio=token.temporal_mlp_ratio,
        temporal_causal=token.temporal_causal,
        temporal_mode=token.temporal_mode,
        temporal_pos_embed_type=token.temporal_pos_embed_type,
        temporal_loss_weight=token.temporal_loss_weight,
    )


def checkpoint_config(config: TokenAETrainingConfig, embed_dim: int, tokens_per_step: int) -> dict[str, Any]:
    """Serialize every field required by DroneWAM's strict TokenAE loader."""
    token = config.token_ae
    return {
        "embed_dim": int(embed_dim),
        "tokens_per_frame": int(tokens_per_step),
        "num_latent_tokens": token.num_latent_tokens,
        "num_heads": token.num_heads,
        "encoder_depth": token.encoder_depth,
        "decoder_depth": token.decoder_depth,
        "mlp_ratio": token.mlp_ratio,
        "dropout": token.dropout,
        "encoder_mode": token.encoder_mode,
        "loss_type": token.loss_type,
        "cos_loss_weight": token.cos_loss_weight,
        "latent_reg_weight": token.latent_reg_weight,
        "pos_embed_type": token.pos_embed_type,
        "input_grid_size": list(token.input_grid_size),
        "latent_grid_size": list(token.latent_grid_size),
        "temporal_depth": token.temporal_depth,
        "temporal_num_heads": token.temporal_num_heads,
        "temporal_mlp_ratio": token.temporal_mlp_ratio,
        "temporal_causal": token.temporal_causal,
        "temporal_mode": token.temporal_mode,
        "temporal_pos_embed_type": token.temporal_pos_embed_type,
        "temporal_loss_weight": token.temporal_loss_weight,
        "normalize_reps": token.normalize_reps,
        "source_encoder_family": config.encoder.family,
        "source_encoder_model": config.encoder.model_name,
        "source_encoder_checkpoint": config.encoder.checkpoint,
    }


def save_token_ae_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TokenAETrainingConfig,
    output_path: str | Path,
    epoch: int,
    metrics: dict[str, float],
    embed_dim: int,
    tokens_per_step: int,
) -> None:
    """Atomically save trainable TokenAE parameters without the frozen encoder."""
    core = model.module if hasattr(model, "module") else model
    if not isinstance(core, TokenAE):
        raise TypeError(f"Expected TokenAE, got {type(core).__name__}")
    state: dict[str, Any] = {
        "checkpoint_type": "token_ae_only",
        "upstream_encoder_included": False,
        "epoch": int(epoch),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "config": checkpoint_config(config, embed_dim, tokens_per_step),
        "token_ae": core.state_dict(),
    }
    if config.runtime.save_optimizer:
        state["optimizer"] = optimizer.state_dict()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


class TokenAETrainer:
    """Train only the Resampler while recomputing frozen source features online."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        token_ae: TokenAE,
        config: TokenAETrainingConfig,
        distributed: DistributedContext,
        steps_per_epoch: int,
    ) -> None:
        self.config = config
        self.distributed = distributed
        self.device = distributed.device
        self.encoder = encoder.to(self.device).eval().requires_grad_(False)
        self.token_ae = token_ae.to(self.device)
        self.embed_dim = int(getattr(encoder, "embed_dim"))
        self.tokens_per_step = int(getattr(encoder, "tokens_per_frame"))
        self.optimizer = torch.optim.AdamW(
            self.token_ae.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        self.accumulation_steps = config.optimization.accumulation_steps
        self.steps_per_epoch = math.ceil(steps_per_epoch / self.accumulation_steps)
        self.total_steps = config.optimization.epochs * self.steps_per_epoch
        self.warmup_steps = config.optimization.warmup_epochs * self.steps_per_epoch
        self.global_step = 0
        if distributed.world_size > 1:
            self.token_ae = DistributedDataParallel(
                self.token_ae,
                device_ids=[distributed.local_rank],
                broadcast_buffers=False,
            )

    def _learning_rate(self) -> float:
        optimization = self.config.optimization
        if self.global_step < self.warmup_steps:
            fraction = float(self.global_step + 1) / max(1, self.warmup_steps)
            return optimization.start_learning_rate + fraction * (
                optimization.learning_rate - optimization.start_learning_rate
            )
        progress = (self.global_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return optimization.final_learning_rate + (
            optimization.learning_rate - optimization.final_learning_rate
        ) * cosine

    def train_epoch(self, loader, epoch: int) -> dict[str, float]:
        self.encoder.eval()
        self.token_ae.train()
        loader.sampler.set_epoch(epoch)
        totals = {"loss": 0.0, "recon": 0.0, "cos": 0.0, "temporal": 0.0, "latent_reg": 0.0}
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            group_start = batch_index - batch_index % self.accumulation_steps
            group_size = min(self.accumulation_steps, len(loader) - group_start)
            should_step = (batch_index + 1) % self.accumulation_steps == 0 or batch_index + 1 == len(loader)
            frames = batch["frames"].to(self.device, non_blocking=True)
            learning_rate = self._learning_rate()
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=_dtype(self.config.optimization.dtype),
                enabled=self.device.type == "cuda" and self.config.optimization.dtype != "float32",
            ):
                target = self.encoder(frames, num_observed_frames=self.config.data.num_frames)
                if self.config.token_ae.normalize_reps:
                    target = torch.nn.functional.layer_norm(target, (target.shape[-1],))
            synchronization = (
                self.token_ae.no_sync()
                if isinstance(self.token_ae, DistributedDataParallel) and not should_step
                else nullcontext()
            )
            with synchronization:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=_dtype(self.config.optimization.dtype),
                    enabled=self.device.type == "cuda" and self.config.optimization.dtype != "float32",
                ):
                    output = self.token_ae(
                        target,
                        num_frames=self.config.data.num_frames // self.config.encoder.tubelet_size,
                    )
                    loss = output["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite TokenAE loss at epoch={epoch}, batch={batch_index}")
                (loss / group_size).backward()
            if should_step:
                torch.nn.utils.clip_grad_norm_(self.token_ae.parameters(), self.config.optimization.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            totals["loss"] += float(output["loss"].item())
            totals["recon"] += float(output["recon_loss"].item())
            totals["cos"] += float(output["cos_loss"].item())
            totals["temporal"] += float(output["temporal_loss"].item())
            totals["latent_reg"] += float(output["latent_reg"].item())
        return {key: value / len(loader) for key, value in totals.items()}


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]
