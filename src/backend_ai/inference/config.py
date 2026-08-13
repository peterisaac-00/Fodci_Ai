"""Configuration for local Fodci inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from backend_ai.tokenizer import TOKENIZER_VERSION


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Conservative local decoding configuration."""

    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int | None = None
    do_sample: bool = False
    stop_on_eos: bool = True
    device: str = "cpu"
    seed: int = 2026
    model_version: str = "fodci-tiny-v1"
    tokenizer_version: int = TOKENIZER_VERSION
    checkpoint_path: Path | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool):
            raise ValueError("max_new_tokens must be an integer.")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool):
            raise ValueError("temperature must be a finite positive number.")
        if not math.isfinite(float(self.temperature)) or float(self.temperature) <= 0:
            raise ValueError("temperature must be a finite positive number.")
        if self.top_k is not None:
            if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k <= 0:
                raise ValueError("top_k must be a positive integer or None.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not self.model_version:
            raise ValueError("model_version must not be empty.")
        if self.tokenizer_version < 0:
            raise ValueError("tokenizer_version must be non-negative.")
        if self.device not in {"cpu", "auto"} and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cpu', 'auto', or a CUDA device string.")

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        if isinstance(values["checkpoint_path"], Path):
            values["checkpoint_path"] = str(values["checkpoint_path"])
        return values
