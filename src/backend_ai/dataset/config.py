"""Configuration for the local Phase 2.4 dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONTEXT_LENGTH = 256


DEFAULT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".sql",
        ".json",
        ".yaml",
        ".yml",
        ".html",
        ".css",
        ".sh",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Small deterministic configuration for local document processing."""

    input_dir: Path | str
    supported_extensions: frozenset[str] = DEFAULT_EXTENSIONS
    max_file_size_mb: float = 2.0
    normalize_line_endings: bool = False
    context_length: int = DEFAULT_CONTEXT_LENGTH
    use_eos_document_boundaries: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dir", Path(self.input_dir))
        normalized_extensions = frozenset(
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in self.supported_extensions
        )
        object.__setattr__(self, "supported_extensions", normalized_extensions)
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive.")
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2.")
