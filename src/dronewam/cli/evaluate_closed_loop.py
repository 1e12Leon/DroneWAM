"""Evaluate DroneWAM through a real interactive environment backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import torch

from dronewam.closed_loop import (
    ClosedLoopEvaluator,
    ModelClosedLoopPolicy,
    load_environment_factory,
    summarize_closed_loop,
)
from dronewam.config import load_config
from dronewam.data import ImageTransform
from dronewam.models.wam import build_model
from dronewam.preferences import load_threshold_calibration
from dronewam.training.adaptive import load_adaptive_checkpoint
from dronewam.training.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--environment-factory", required=True, help="Python module:function backend factory")
    parser.add_argument("--environment-config", help="Optional JSON object passed to the backend factory")
    parser.add_argument("--episodes", required=True, help="JSON list of environment episode IDs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-control-steps", type=int, default=256)
    parser.add_argument("--adaptive-checkpoint")
    parser.add_argument("--calibration")
    return parser.parse_args()


def _read_json(path: str | None, default):
    if path is None:
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if bool(args.adaptive_checkpoint) != bool(args.calibration):
        raise ValueError("--adaptive-checkpoint and --calibration must be provided together")
    episodes = _read_json(args.episodes, None)
    if not isinstance(episodes, list) or not episodes or not all(isinstance(item, str) for item in episodes):
        raise ValueError("--episodes must contain a non-empty JSON list of string IDs")
    environment_config = _read_json(args.environment_config, {})
    if not isinstance(environment_config, dict):
        raise ValueError("--environment-config must contain a JSON object")

    config = load_config(args.config)
    if args.adaptive_checkpoint and not config.model.adaptive.enabled:
        raise ValueError("Adaptive closed-loop evaluation requires model.adaptive.enabled=true")
    if config.model.adaptive.enabled and not args.adaptive_checkpoint:
        raise ValueError("Adaptive-enabled config requires --adaptive-checkpoint and --calibration")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    load_checkpoint(
        model,
        args.checkpoint,
        load_optimizer=False,
        allow_missing_adaptive=bool(args.adaptive_checkpoint),
    )
    stop_thresholds = None
    if args.adaptive_checkpoint:
        load_adaptive_checkpoint(model, args.adaptive_checkpoint, require_planner=True, require_policy=True)
        calibration = load_threshold_calibration(args.calibration)
        stop_thresholds = torch.tensor(calibration.stop_thresholds, device=device)
    model = model.to(device).eval()
    policy = ModelClosedLoopPolicy(
        model,
        device,
        ImageTransform(config.model.encoder.resolution, config.data.crop_top_bottom),
        stop_thresholds=stop_thresholds,
        min_rollout_depth=config.model.adaptive.min_rollout_depth,
        dpo_beta=config.adaptive_training.dpo_beta,
    )
    environment = load_environment_factory(args.environment_factory)(environment_config)
    evaluator = ClosedLoopEvaluator(
        environment,
        policy,
        observed_frames=model.num_observed_frames,
        frame_stride=model.frame_stride,
        fps=config.data.fps,
        action_horizon=model.num_steps - model.observed_steps,
        max_control_steps=args.max_control_steps,
    )
    results = []
    try:
        for index, episode_id in enumerate(episodes, start=1):
            result = evaluator.run_episode(episode_id)
            results.append(result)
            logger.info(
                "episode=%s progress=%d/%d steps=%d rel_ate=%.6f rel_rpe=%.6f",
                episode_id,
                index,
                len(episodes),
                result.control_steps,
                result.relative_ate,
                result.relative_rpe,
            )
    finally:
        environment.close()
    report = summarize_closed_loop(results)
    report["environment_factory"] = args.environment_factory
    report["mode"] = "adaptive" if args.adaptive_checkpoint else "fixed_depth_full_horizon"
    report["control_duration_s"] = model.frame_stride / float(config.data.fps)
    report["observations_per_control_step"] = model.frame_stride
    report["provenance"] = {
        "git_commit": _git_commit(),
        "config": str(Path(args.config)),
        "config_sha256": _sha256(args.config),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "adaptive_checkpoint": (
            None if args.adaptive_checkpoint is None else str(Path(args.adaptive_checkpoint))
        ),
        "adaptive_checkpoint_sha256": (
            None if args.adaptive_checkpoint is None else _sha256(args.adaptive_checkpoint)
        ),
        "calibration": None if args.calibration is None else str(Path(args.calibration)),
        "calibration_sha256": None if args.calibration is None else _sha256(args.calibration),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "dtype": "bfloat16" if device.type == "cuda" else "float32",
        "batch_size": 1,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("closed-loop report written to %s", destination)


if __name__ == "__main__":
    main()
