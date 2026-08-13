"""Configuration and result schemas for Phase 2.8 evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Deterministic, CPU-first evaluation settings."""

    batch_size: int = 2
    device: str = "cpu"
    seed: int = 2026
    model_version: str = "fodci-tiny-v1"
    tokenizer_version: int = 1
    dataset_path: Path | str = "data/fodci_tiny_v1/validation"
    checkpoint_dir: Path | str = "artifacts/checkpoints"
    dataset_split: str = "validation"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if self.device.strip().lower() not in {"auto", "cpu", "cuda"} and not self.device.strip().lower().startswith("cuda:"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'.")
        if not self.model_version:
            raise ValueError("model_version must be non-empty.")
        if self.tokenizer_version < 0:
            raise ValueError("tokenizer_version cannot be negative.")
        if not self.dataset_split:
            raise ValueError("dataset_split must be non-empty.")
        object.__setattr__(self, "device", self.device.strip().lower())
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["dataset_path"] = str(self.dataset_path)
        values["checkpoint_dir"] = str(self.checkpoint_dir)
        return values
