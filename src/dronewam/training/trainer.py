"""Focused trainer for the base DroneWAM objective."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from dronewam.config import ExperimentConfig
from dronewam.models.wam import DroneWAM
from dronewam.training.checkpoint import load_checkpoint, save_checkpoint
from dronewam.training.distributed import DistributedContext
from dronewam.training.losses import compute_wam_loss

logger = logging.getLogger(__name__)


def _dtype(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in mapping:
        raise ValueError(f"Unsupported optimization.dtype={name!r}")
    return mapping[name]


class Trainer:
    def __init__(
        self,
        model: DroneWAM,
        config: ExperimentConfig,
        distributed: DistributedContext,
        steps_per_epoch: int,
    ) -> None:
        self.config = config
        self.distributed = distributed
        self.device = distributed.device
        self.model = model.to(self.device)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("No trainable model parameters")
        action_parameters = (
            []
            if model.action_head is None
            else [parameter for parameter in model.action_head.parameters() if parameter.requires_grad]
        )
        action_parameter_ids = {id(parameter) for parameter in action_parameters}
        base_parameters = [parameter for parameter in trainable if id(parameter) not in action_parameter_ids]
        parameter_groups = []
        if base_parameters:
            parameter_groups.append({"params": base_parameters, "lr_scale": 1.0})
        if action_parameters:
            parameter_groups.append(
                {
                    "params": action_parameters,
                    "lr_scale": config.optimization.action_head_lr_multiplier,
                }
            )
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        self.steps_per_epoch = math.ceil(steps_per_epoch / config.optimization.accumulation_steps)
        self.total_steps = config.optimization.epochs * self.steps_per_epoch
        self.warmup_steps = config.optimization.warmup_epochs * self.steps_per_epoch
        self.global_step = 0
        self.start_epoch = 0
        if config.runtime.initial_checkpoint:
            load_checkpoint(
                self.model,
                config.runtime.initial_checkpoint,
                load_optimizer=False,
            )
        if config.runtime.resume_checkpoint:
            self.start_epoch = load_checkpoint(
                self.model,
                config.runtime.resume_checkpoint,
                self.optimizer,
                load_optimizer=config.runtime.resume_optimizer,
            )
            self.global_step = self.start_epoch * self.steps_per_epoch
        if distributed.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[distributed.local_rank],
                broadcast_buffers=False,
            )

    def _learning_rate(self) -> float:
        base = self.config.optimization.learning_rate
        start = self.config.optimization.start_learning_rate
        final = self.config.optimization.final_learning_rate
        if self.global_step < self.warmup_steps:
            fraction = float(self.global_step + 1) / max(1, self.warmup_steps)
            return start + fraction * (base - start)
        progress = (self.global_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return final + (base - final) * cosine

    def _teacher_momentum(self) -> float:
        start = self.config.runtime.ema_momentum_start
        end = self.config.runtime.ema_momentum_end
        progress = min(float(self.global_step) / max(1, self.total_steps - 1), 1.0)
        return start + progress * (end - start)

    def train_epoch(self, loader, epoch: int) -> dict[str, float]:
        self.model.train()
        core = self.model.module if hasattr(self.model, "module") else self.model
        core.target_encoder.eval()
        if core.token_ae is not None:
            core.token_ae.eval()
        loader.sampler.set_epoch(epoch)
        totals = {"loss": 0.0, "tf": 0.0, "ar": 0.0, "action": 0.0, "rollout": 0.0}
        accumulation_steps = self.config.optimization.accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            group_start = batch_index - batch_index % accumulation_steps
            group_size = min(accumulation_steps, len(loader) - group_start)
            should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            frames = batch["frames"].to(self.device, non_blocking=True)
            states = batch["states"].to(self.device, non_blocking=True)
            learning_rate = self._learning_rate()
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate * group["lr_scale"]
            amp_enabled = self.device.type == "cuda" and self.config.optimization.dtype != "float32"
            synchronization = (
                self.model.no_sync()
                if isinstance(self.model, DistributedDataParallel) and not should_step
                else nullcontext()
            )
            with synchronization:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=_dtype(self.config.optimization.dtype),
                    enabled=amp_enabled,
                ):
                    output = self.model(frames, states, compute_targets=True)
                    loss = compute_wam_loss(output, self.config.loss)
                if not torch.isfinite(loss.total):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={self.global_step}")
                (loss.total / group_size).backward()
            if should_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimization.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                core.update_teacher(self._teacher_momentum())
                self.global_step += 1
            totals["loss"] += float(loss.total.item())
            totals["tf"] += float(loss.teacher_forced.item())
            totals["ar"] += float(loss.autoregressive.item())
            totals["action"] += float(loss.action_regression.item())
            totals["rollout"] += float(loss.rollout.item())
        return {key: value / len(loader) for key, value in totals.items()}

    def fit(self, loader) -> None:
        output_dir = Path(self.config.runtime.output_dir)
        for epoch in range(self.start_epoch, self.config.optimization.epochs):
            metrics = self.train_epoch(loader, epoch)
            if self.distributed.rank == 0:
                logger.info("epoch=%d metrics=%s", epoch + 1, metrics)
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    output_dir / "latest.pt",
                    epoch,
                    include_optimizer=self.config.runtime.save_optimizer,
                )
                if (epoch + 1) % self.config.runtime.save_every_epochs == 0:
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        output_dir / f"epoch_{epoch + 1:03d}.pt",
                        epoch,
                        include_optimizer=self.config.runtime.save_optimizer,
                    )
