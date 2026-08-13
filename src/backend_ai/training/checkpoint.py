"""Minimal PyTorch checkpoint persistence for Phase 2.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from backend_ai.training.config import TrainingConfig

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """State restored from a trainer checkpoint."""

    epoch: int
    global_step: int
    metrics: dict[str, Any]
    config: dict[str, Any]


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: TrainingConfig,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save the minimum state required to resume training."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "training_config": config.to_dict(),
        "metrics": metrics or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> CheckpointState:
    """Restore model and optimizer state and return resume metadata."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError(f"Unable to load checkpoint: {checkpoint_path}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported or corrupt Fodci checkpoint format.")
    required = {"model_state_dict", "optimizer_state_dict", "epoch", "global_step"}
    if not required.issubset(payload):
        missing = ", ".join(sorted(required - payload.keys()))
        raise ValueError(f"Checkpoint is missing required fields: {missing}")
    try:
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint state is incompatible with the model or optimizer.") from exc
    return CheckpointState(
        epoch=_validated_nonnegative_int(payload["epoch"], "epoch"),
        global_step=_validated_nonnegative_int(payload["global_step"], "global_step"),
        metrics=dict(payload.get("metrics", {})),
        config=dict(payload.get("training_config", {})),
    )


def _validated_nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Checkpoint field '{name}' must be a non-negative integer.")
    return value
