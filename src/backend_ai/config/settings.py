"""Small, environment-backed settings boundary for Phase 0."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_VALID_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration shared by the foundation without agent-specific concerns."""

    project_root: Path
    log_level: str


def load_settings(
    environment: Mapping[str, str] | None = None,
    *,
    cwd: Path | None = None,
) -> Settings:
    """Load validated foundation settings from an environment-like mapping.

    `environment` is injectable to keep tests deterministic. When it is omitted,
    the process environment is used. A relative project root is resolved against
    the supplied working directory (or the current working directory).
    """

    source = os.environ if environment is None else environment
    base_directory = (cwd or Path.cwd()).resolve()
    project_root_value = source.get("PROJECT_ROOT", ".")
    project_root = Path(project_root_value).expanduser()
    if not project_root.is_absolute():
        project_root = base_directory / project_root

    return Settings(
        project_root=project_root.resolve(),
        log_level=_normalise_log_level(source.get("LOG_LEVEL", "INFO")),
    )


def _normalise_log_level(value: str) -> str:
    level = value.strip().upper()
    if level not in _VALID_LOG_LEVELS:
        supported = ", ".join(sorted(_VALID_LOG_LEVELS))
        raise ValueError(f"Unsupported LOG_LEVEL {value!r}. Use one of: {supported}.")
    return level
