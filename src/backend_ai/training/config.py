"""Configuration and device selection for the Phase 2.5 trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Small, explicit, CPU-friendly training configuration."""

    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float | None = 1.0
    device: str = "cpu"
    seed: int = 0
    log_interval: int = 0
    validation_interval: int = 1
    checkpoint_interval: int = 1
    output_dir: Path | str = Path("artifacts/checkpoints")

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive or None.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive or None.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        for name, value in (
            ("log_interval", self.log_interval),
            ("validation_interval", self.validation_interval),
            ("checkpoint_interval", self.checkpoint_interval),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")
        normalized_device = self.device.strip().lower()
        if normalized_device not in {"auto", "cpu", "cuda"} and not normalized_device.startswith("cuda:"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'.")
        object.__setattr__(self, "device", normalized_device)
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def resolve_device(self) -> torch.device:
        """Resolve the configured device without silently falling back from explicit CUDA."""

        if self.device == "cpu":
            return torch.device("cpu")
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{self.device}' is unavailable; CUDA is not available."
            )
        return torch.device(self.device)

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-safe representation of this configuration."""

        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values
