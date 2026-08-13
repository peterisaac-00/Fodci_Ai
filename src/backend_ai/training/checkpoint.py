"""Backward-compatible facade over the Phase 2.7 CheckpointManager."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from backend_ai.checkpoint import CheckpointManager, LoadedCheckpoint
from backend_ai.training.config import TrainingConfig

CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Legacy-shaped resume state backed by the new metadata-aware manager."""

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
    """Save through CheckpointManager for callers of the old helper API."""

    manager = CheckpointManager(Path(path).parent)
    return manager.save(
        model,
        optimizer,
        config,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics,
        path=path,
    )


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> CheckpointState:
    """Load through CheckpointManager and return the legacy-shaped state."""

    manager = CheckpointManager(Path(path).parent)
    loaded: LoadedCheckpoint = manager.load(
        path,
        model,
        optimizer,
        device=device,
    )
    return CheckpointState(
        epoch=loaded.epoch,
        global_step=loaded.global_step,
        metrics=loaded.metrics,
        config=loaded.config,
    )
