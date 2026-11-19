"""Strict configuration schema for the clean DroneWAM project."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class DataConfig:
    train_csv: str
    val_csv: str
    num_frames: int = 20
    observed_frames: int = 4
    fps: int = 7
    base_fps: int = 7
    batch_size: int = 2
    num_workers: int = 8
    crop_top_bottom: int = 28


@dataclass(frozen=True)
class EncoderConfig:
    family: str = "drive_jepa"
    model_name: str = "vit_large"
    resolution: Tuple[int, int] = (256, 512)
    patch_size: int = 16
    tubelet_size: int = 2
    embed_dim: int = 1024
    checkpoint: str = ""
    checkpoint_key: str = "target_encoder"
    use_grid_mask: bool = True
    use_causal_attention: bool = True
    use_rope: bool = True
    use_sdpa: bool = True
    use_activation_checkpointing: bool = True
    frame_batch_size: int = 20


@dataclass(frozen=True)
class TokenAEConfig:
    enabled: bool = True
    checkpoint: str = ""
    num_latent_tokens: int = 128
    num_heads: int = 16
    encoder_depth: int = 4
    decoder_depth: int = 2
    mlp_ratio: float = 4.0
    input_grid_size: Tuple[int, int] = (16, 32)
    latent_grid_size: Tuple[int, int] = (8, 16)
    temporal_depth: int = 2
    temporal_num_heads: int = 16
    temporal_mode: str = "block_causal"
    temporal_causal: bool = True
    dropout: float = 0.0
    encoder_mode: str = "parallel"
    loss_type: str = "smooth_l1"
    cos_loss_weight: float = 0.25
    latent_reg_weight: float = 0.0
    pos_embed_type: str = "learnable"
    temporal_mlp_ratio: float = 4.0
    temporal_pos_embed_type: str = "sincos"
    temporal_loss_weight: float = 0.1
    normalize_reps: bool = False


@dataclass(frozen=True)
class PredictorConfig:
    depth: int = 24
    hidden_dim: int = 1024
    num_heads: int = 16
    action_dim: int = 6
    state_dim: int = 7
    frame_causal: bool = True
    use_rope: bool = True
    use_activation_checkpointing: bool = True


@dataclass(frozen=True)
class ActionHeadConfig:
    enabled: bool = True
    hidden_dim: int = 192


@dataclass(frozen=True)
class AdaptiveConfig:
    enabled: bool = False
    planner_mode: str = "independent"
    planner_hidden_dim: int = 256
    planner_layers: int = 2
    planner_heads: int = 4
    policy_hidden_dim: int = 128
    policy_token_count: int = 32
    min_rollout_depth: int = 1


@dataclass(frozen=True)
class ModelConfig:
    encoder: EncoderConfig
    token_ae: TokenAEConfig
    predictor: PredictorConfig
    action_head: ActionHeadConfig
    adaptive: AdaptiveConfig


@dataclass(frozen=True)
class LossConfig:
    teacher_forced_weight: float = 1.0
    autoregressive_weight: float = 1.0
    action_weight: float = 0.35
    action_dim_weights: Tuple[float, ...] = (2.1, 2.1, 1.0, 0.2, 0.2, 0.8)
    ar_discount: float = 0.95
    rollout_position_weight: float = 0.07
    rollout_delta_weight: float = 2.5
    rollout_action_delta_weight: float = 1.0
    rollout_relative_ate_weight: float = 0.0
    rollout_relative_rpe_weight: float = 20.0
    rollout_yaw_weight: float = 0.06


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int = 30
    accumulation_steps: int = 1
    learning_rate: float = 5.0e-8
    start_learning_rate: float = 1.0e-8
    final_learning_rate: float = 1.0e-8
    weight_decay: float = 0.04
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    seed: int = 239
    action_head_lr_multiplier: float = 3.0


@dataclass(frozen=True)
class AdaptiveTrainingConfig:
    preference_path: str = ""
    preference_windows_per_record: int = 4
    planner_epochs: int = 5
    planner_reduction: str = "mean"
    planner_shallow_weight: float = 0.25
    planner_metric_power: int = 1
    multilength_epochs: int = 5
    world_latent_weight: float = 1.0
    world_trajectory_weight: float = 1.0
    world_horizon_weight: float = 1.0
    world_horizon_classification_weight: float = 1.0
    world_horizon_rank_weight: float = 1.0
    world_horizon_regression_weight: float = 0.0
    horizon_class_weights: Optional[List[float]] = None
    horizon_class_priors: Optional[List[float]] = None
    joint_horizon_epochs: int = 5
    dpo_epochs: int = 10
    joint_epochs: int = 5
    planner_learning_rate: float = 1.0e-4
    policy_learning_rate: float = 1.0e-4
    dpo_beta: float = 0.1
    dpo_weight: float = 1.0
    compute_penalty: float = 0.02
    preference_margin: float = 0.0
    calibration_fraction: float = 0.2
    ate_target: float = 0.04
    rpe_target: float = 0.013


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: str
    initial_checkpoint: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    resume_optimizer: bool = False
    save_optimizer: bool = False
    save_every_epochs: int = 5
    train_encoder: bool = False
    train_predictor: bool = True
    train_action_head: bool = True
    train_adaptive_planner: bool = False
    train_continuation_policy: bool = False
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 0.999


@dataclass(frozen=True)
class TokenAEOptimizationConfig:
    epochs: int = 80
    accumulation_steps: int = 1
    learning_rate: float = 1.0e-4
    start_learning_rate: float = 1.0e-6
    final_learning_rate: float = 0.0
    weight_decay: float = 0.04
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    seed: int = 239


@dataclass(frozen=True)
class TokenAETrainingRuntimeConfig:
    output_dir: str
    save_optimizer: bool = False


@dataclass(frozen=True)
class TokenAETrainingConfig:
    data: DataConfig
    encoder: EncoderConfig
    token_ae: TokenAEConfig
    optimization: TokenAEOptimizationConfig
    runtime: TokenAETrainingRuntimeConfig

    def validate(self) -> None:
        if self.encoder.family not in ("drive_jepa", "dinov2"):
            raise ValueError("encoder.family must be drive_jepa or dinov2")
        if not self.encoder.checkpoint:
            raise ValueError("encoder.checkpoint is required")
        if self.encoder.frame_batch_size < 1:
            raise ValueError("encoder.frame_batch_size must be positive")
        if self.data.num_frames % self.encoder.tubelet_size:
            raise ValueError(
                "data.num_frames must be divisible by encoder.tubelet_size"
            )
        raw_grid = (
            self.encoder.resolution[0] // self.encoder.patch_size,
            self.encoder.resolution[1] // self.encoder.patch_size,
        )
        if tuple(self.token_ae.input_grid_size) != raw_grid:
            raise ValueError(
                f"token_ae.input_grid_size={self.token_ae.input_grid_size} does not match encoder grid={raw_grid}"
            )
        if self.token_ae.num_latent_tokens != int(
            self.token_ae.latent_grid_size[0]
        ) * int(self.token_ae.latent_grid_size[1]):
            raise ValueError(
                "token_ae.latent_grid_size must match token_ae.num_latent_tokens"
            )
        if self.token_ae.checkpoint:
            raise ValueError(
                "TokenAE training config must not load a pretrained TokenAE checkpoint"
            )
        if self.token_ae.encoder_mode not in ("query", "parallel", "prumerge"):
            raise ValueError("Unsupported token_ae.encoder_mode")
        if self.token_ae.loss_type not in ("mse", "smooth_l1"):
            raise ValueError("token_ae.loss_type must be mse or smooth_l1")
        if (
            min(
                self.token_ae.cos_loss_weight,
                self.token_ae.latent_reg_weight,
                self.token_ae.temporal_loss_weight,
            )
            < 0.0
        ):
            raise ValueError("TokenAE loss weights must be non-negative")
        if self.optimization.epochs < 1 or self.optimization.accumulation_steps < 1:
            raise ValueError("TokenAE epochs and accumulation_steps must be positive")
        if self.optimization.learning_rate <= 0.0:
            raise ValueError("TokenAE learning_rate must be positive")
        if self.optimization.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("Unsupported TokenAE optimization.dtype")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    optimization: OptimizationConfig
    adaptive_training: AdaptiveTrainingConfig
    runtime: RuntimeConfig

    def validate(self) -> None:
        if self.runtime.initial_checkpoint and self.runtime.resume_checkpoint:
            raise ValueError(
                "runtime.initial_checkpoint and runtime.resume_checkpoint are mutually exclusive"
            )
        if self.model.encoder.family not in ("drive_jepa", "dinov2"):
            raise ValueError("model.encoder.family must be drive_jepa or dinov2")
        if self.model.encoder.frame_batch_size < 1:
            raise ValueError("model.encoder.frame_batch_size must be positive")
        if (
            self.data.observed_frames <= 0
            or self.data.observed_frames >= self.data.num_frames
        ):
            raise ValueError("data.observed_frames must be in [1, data.num_frames)")
        stride = self.model.encoder.tubelet_size
        if (
            self.data.num_frames % stride != 0
            or self.data.observed_frames % stride != 0
        ):
            raise ValueError(
                "num_frames and observed_frames must be divisible by encoder.tubelet_size"
            )
        if self.model.predictor.action_dim != 6:
            raise ValueError(
                "The clean DroneWAM data contract requires predictor.action_dim=6"
            )
        if self.model.predictor.state_dim != 7:
            raise ValueError(
                "The clean DroneWAM data contract requires predictor.state_dim=7"
            )
        if len(self.loss.action_dim_weights) != self.model.predictor.action_dim:
            raise ValueError("loss.action_dim_weights must match predictor.action_dim")
        if not 0.0 < self.loss.ar_discount <= 1.0:
            raise ValueError("loss.ar_discount must be in (0, 1]")
        rollout_weights = (
            self.loss.rollout_position_weight,
            self.loss.rollout_delta_weight,
            self.loss.rollout_action_delta_weight,
            self.loss.rollout_relative_ate_weight,
            self.loss.rollout_relative_rpe_weight,
            self.loss.rollout_yaw_weight,
        )
        if min(rollout_weights) < 0.0:
            raise ValueError("All rollout loss weights must be non-negative")
        if (
            self.optimization.start_learning_rate < 0.0
            or self.optimization.final_learning_rate < 0.0
        ):
            raise ValueError("Learning rates must be non-negative")
        if self.optimization.accumulation_steps < 1:
            raise ValueError("optimization.accumulation_steps must be positive")
        if self.optimization.action_head_lr_multiplier < 0.0:
            raise ValueError(
                "optimization.action_head_lr_multiplier must be non-negative"
            )
        future_steps = (self.data.num_frames - self.data.observed_frames) // stride
        if not 1 <= self.model.adaptive.min_rollout_depth <= future_steps:
            raise ValueError(
                "model.adaptive.min_rollout_depth must be within the future rollout horizon"
            )
        if self.model.adaptive.enabled and not self.model.action_head.enabled:
            raise ValueError("Adaptive rollout requires model.action_head.enabled=true")
        if self.model.adaptive.planner_mode not in ("independent", "iterative"):
            raise ValueError(
                "model.adaptive.planner_mode must be independent or iterative"
            )
        policy_source_tokens = (
            self.model.token_ae.num_latent_tokens
            if self.model.token_ae.enabled
            else (self.model.encoder.resolution[0] // self.model.encoder.patch_size)
            * (self.model.encoder.resolution[1] // self.model.encoder.patch_size)
        )
        if (
            self.model.adaptive.policy_token_count < 1
            or policy_source_tokens % self.model.adaptive.policy_token_count
        ):
            raise ValueError(
                "model.adaptive.policy_token_count must evenly divide world tokens per step"
            )
        if self.adaptive_training.dpo_beta <= 0.0:
            raise ValueError("adaptive_training.dpo_beta must be positive")
        if self.adaptive_training.preference_windows_per_record < 1:
            raise ValueError(
                "adaptive_training.preference_windows_per_record must be positive"
            )
        if self.adaptive_training.multilength_epochs < 1:
            raise ValueError("adaptive_training.multilength_epochs must be positive")
        if self.adaptive_training.world_latent_weight <= 0.0:
            raise ValueError("adaptive_training.world_latent_weight must be positive")
        if self.adaptive_training.world_trajectory_weight <= 0.0:
            raise ValueError(
                "adaptive_training.world_trajectory_weight must be positive"
            )
        if self.adaptive_training.world_horizon_weight <= 0.0:
            raise ValueError("adaptive_training.world_horizon_weight must be positive")
        horizon_loss_weights = (
            self.adaptive_training.world_horizon_classification_weight,
            self.adaptive_training.world_horizon_rank_weight,
            self.adaptive_training.world_horizon_regression_weight,
        )
        if (
            any(weight < 0.0 for weight in horizon_loss_weights)
            or sum(horizon_loss_weights) <= 0.0
        ):
            raise ValueError(
                "joint Gate loss weights must be non-negative with at least one positive value"
            )
        horizon_class_weights = self.adaptive_training.horizon_class_weights
        if horizon_class_weights is not None and (
            len(horizon_class_weights) < 2
            or any(weight <= 0.0 for weight in horizon_class_weights)
        ):
            raise ValueError(
                "adaptive_training.horizon_class_weights must contain at least two positive values"
            )
        horizon_class_priors = self.adaptive_training.horizon_class_priors
        if horizon_class_priors is not None and (
            len(horizon_class_priors) < 2
            or any(prior <= 0.0 for prior in horizon_class_priors)
        ):
            raise ValueError(
                "adaptive_training.horizon_class_priors must contain at least two positive values"
            )
        if self.adaptive_training.joint_horizon_epochs < 1:
            raise ValueError("adaptive_training.joint_horizon_epochs must be positive")
        if self.adaptive_training.planner_learning_rate <= 0.0:
            raise ValueError("adaptive_training.planner_learning_rate must be positive")
        if self.adaptive_training.planner_reduction not in (
            "mean",
            "final",
            "final_plus_shallow",
        ):
            raise ValueError(
                "adaptive_training.planner_reduction must be mean, final or final_plus_shallow"
            )
        if self.adaptive_training.planner_shallow_weight < 0.0:
            raise ValueError(
                "adaptive_training.planner_shallow_weight must be non-negative"
            )
        if self.adaptive_training.planner_metric_power not in (1, 2):
            raise ValueError("adaptive_training.planner_metric_power must be 1 or 2")
        if self.adaptive_training.policy_learning_rate <= 0.0:
            raise ValueError("adaptive_training.policy_learning_rate must be positive")
        if self.adaptive_training.dpo_weight < 0.0:
            raise ValueError("adaptive_training.dpo_weight must be non-negative")
        if self.adaptive_training.compute_penalty < 0.0:
            raise ValueError("adaptive_training.compute_penalty must be non-negative")
        if not 0.0 < self.adaptive_training.calibration_fraction < 1.0:
            raise ValueError("adaptive_training.calibration_fraction must be in (0,1)")
        if (
            self.adaptive_training.ate_target <= 0.0
            or self.adaptive_training.rpe_target <= 0.0
        ):
            raise ValueError("adaptive metric targets must be positive")
        if self.model.token_ae.enabled and not self.model.token_ae.checkpoint:
            raise ValueError(
                "model.token_ae.checkpoint is required when TokenAE is enabled"
            )
        if not self.model.encoder.checkpoint:
            raise ValueError("model.encoder.checkpoint is required")


def _strict_dataclass(cls: Type[T], raw: Dict[str, Any], path: str) -> T:
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must be a mapping")
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown fields in {path}: {unknown}")
    required = {
        field.name
        for field in fields(cls)
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Missing required fields in {path}: {missing}")
    return cls(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML without accepting legacy or silently ignored fields."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("Experiment config must be a mapping")
    expected = {"data", "model", "loss", "optimization", "adaptive_training", "runtime"}
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        raise ValueError(
            f"Invalid top-level config fields: unknown={unknown}, missing={missing}"
        )

    model_raw = raw["model"]
    if not isinstance(model_raw, dict):
        raise TypeError("model must be a mapping")
    expected_model = {"encoder", "token_ae", "predictor", "action_head", "adaptive"}
    if set(model_raw) != expected_model:
        raise ValueError(
            f"model fields must be exactly {sorted(expected_model)}, got {sorted(model_raw)}"
        )
    model = ModelConfig(
        encoder=_strict_dataclass(EncoderConfig, model_raw["encoder"], "model.encoder"),
        token_ae=_strict_dataclass(
            TokenAEConfig, model_raw["token_ae"], "model.token_ae"
        ),
        predictor=_strict_dataclass(
            PredictorConfig, model_raw["predictor"], "model.predictor"
        ),
        action_head=_strict_dataclass(
            ActionHeadConfig, model_raw["action_head"], "model.action_head"
        ),
        adaptive=_strict_dataclass(
            AdaptiveConfig, model_raw["adaptive"], "model.adaptive"
        ),
    )
    config = ExperimentConfig(
        data=_strict_dataclass(DataConfig, raw["data"], "data"),
        model=model,
        loss=_strict_dataclass(LossConfig, raw["loss"], "loss"),
        optimization=_strict_dataclass(
            OptimizationConfig, raw["optimization"], "optimization"
        ),
        adaptive_training=_strict_dataclass(
            AdaptiveTrainingConfig,
            raw["adaptive_training"],
            "adaptive_training",
        ),
        runtime=_strict_dataclass(RuntimeConfig, raw["runtime"], "runtime"),
    )
    config.validate()
    return config


def load_token_ae_training_config(path: str | Path) -> TokenAETrainingConfig:
    """Load the standalone frozen-encoder TokenAE training contract."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("TokenAE training config must be a mapping")
    expected = {"data", "encoder", "token_ae", "optimization", "runtime"}
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        raise ValueError(
            f"Invalid TokenAE config fields: unknown={unknown}, missing={missing}"
        )
    config = TokenAETrainingConfig(
        data=_strict_dataclass(DataConfig, raw["data"], "data"),
        encoder=_strict_dataclass(EncoderConfig, raw["encoder"], "encoder"),
        token_ae=_strict_dataclass(TokenAEConfig, raw["token_ae"], "token_ae"),
        optimization=_strict_dataclass(
            TokenAEOptimizationConfig, raw["optimization"], "optimization"
        ),
        runtime=_strict_dataclass(
            TokenAETrainingRuntimeConfig, raw["runtime"], "runtime"
        ),
    )
    config.validate()
    return config
