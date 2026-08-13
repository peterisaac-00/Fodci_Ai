"""Lightweight training and validation metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Metrics recorded for one completed training epoch."""

    epoch: int
    train_loss: float
    validation_loss: float | None
    training_steps: int
    validation_steps: int
    learning_rate: float
    train_perplexity: float
    validation_perplexity: float | None
    training_tokens: int
    validation_tokens: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def perplexity(loss: float | None) -> float | None:
    """Return exp(loss), guarding overflow and absent validation metrics."""

    if loss is None:
        return None
    try:
        return math.exp(loss) if loss < math.log(float("inf")) else float("inf")
    except OverflowError:
        return float("inf")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Complete in-memory summary returned by ``FodciTrainer.train``."""

    history: tuple[EpochMetrics, ...]
    global_step: int
    last_checkpoint: str | None
    elapsed_seconds: float = 0.0

    @property
    def final_metrics(self) -> EpochMetrics | None:
        return self.history[-1] if self.history else None
