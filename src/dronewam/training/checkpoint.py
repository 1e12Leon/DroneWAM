"""Atomic, trainable-only DroneWAM checkpoint I/O."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from dronewam.models.wam import DroneWAM


def _core(model: torch.nn.Module) -> DroneWAM:
    candidate = model.module if hasattr(model, "module") else model
    if not isinstance(candidate, DroneWAM):
        raise TypeError(f"Expected DroneWAM, got {type(candidate).__name__}")
    return candidate


def _torch_load(path: str | Path) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    checkpoint = torch.load(path, **kwargs)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} must contain a dictionary")
    return checkpoint


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    load_optimizer: bool,
    allow_missing_adaptive: bool = False,
) -> int:
    checkpoint = _torch_load(path)
    module = _core(model)
    aliases = {
        "encoder": ("encoder",),
        "target_encoder": ("target_encoder",),
        "predictor": ("predictor",),
        "action_head": ("action_head", "lewm_action_head"),
        "adaptive_planner": ("adaptive_planner",),
        "continuation_policy": ("continuation_policy",),
    }
    required = []
    if any(parameter.requires_grad for parameter in module.encoder.parameters()):
        required.append("encoder")
    if any(parameter.requires_grad for parameter in module.predictor.parameters()):
        required.append("predictor")
    if module.action_head is not None and any(parameter.requires_grad for parameter in module.action_head.parameters()):
        required.append("action_head")
    if (
        module.adaptive_planner is not None
        and any(parameter.requires_grad for parameter in module.adaptive_planner.parameters())
        and not allow_missing_adaptive
    ):
        required.append("adaptive_planner")
    if (
        module.continuation_policy is not None
        and any(parameter.requires_grad for parameter in module.continuation_policy.parameters())
        and not allow_missing_adaptive
    ):
        required.append("continuation_policy")
    missing = [name for name in required if not any(alias in checkpoint for alias in aliases[name])]
    if missing:
        raise KeyError(f"Checkpoint {path} is missing trainable modules: {missing}")

    for name in (
        "encoder",
        "target_encoder",
        "predictor",
        "action_head",
        "adaptive_planner",
        "continuation_policy",
    ):
        present = [alias for alias in aliases[name] if alias in checkpoint]
        if len(present) > 1:
            raise ValueError(f"Checkpoint contains multiple aliases for {name}: {present}")
        state = checkpoint.get(present[0]) if present else None
        component = getattr(module, name)
        if state is not None:
            if component is None:
                raise ValueError(f"Checkpoint contains {name}, but the model does not")
            component.load_state_dict(state, strict=True)
    if load_optimizer:
        if optimizer is None:
            raise ValueError("load_optimizer=True requires an optimizer")
        if "optimizer" not in checkpoint:
            raise KeyError(f"Checkpoint {path} has no optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", -1)) + 1


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    output_path: str | Path,
    epoch: int,
    *,
    include_optimizer: bool,
) -> None:
    module = _core(model)
    state: Dict[str, Any] = {
        "format_version": 1,
        "epoch": int(epoch),
        "saved_modules": [],
    }
    if include_optimizer:
        if optimizer is None:
            raise ValueError("include_optimizer=True requires an optimizer")
        state["optimizer"] = optimizer.state_dict()
    for name in ("encoder", "predictor", "action_head", "adaptive_planner", "continuation_policy"):
        component = getattr(module, name)
        if component is not None and any(parameter.requires_grad for parameter in component.parameters()):
            state[name] = component.state_dict()
            state["saved_modules"].append(name)
    if any(parameter.requires_grad for parameter in module.encoder.parameters()):
        state["target_encoder"] = module.target_encoder.state_dict()
        state["saved_modules"].append("target_encoder")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
