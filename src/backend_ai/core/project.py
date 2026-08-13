"""Project-root context and validation for Phase 1.7."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend_ai.config import Settings


class InvalidProjectRootError(ValueError):
    """Raised when an explicitly resolved project root is unusable."""

    def __init__(self, root: Path) -> None:
        self.root = root
        super().__init__(
            f"Invalid project root: {root}\n"
            "Path does not exist or is not a directory."
        )


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """Minimal validated context for the project being operated on."""

    root: Path


def resolve_project_context(settings: Settings) -> ProjectContext:
    """Resolve and validate only the configured project-root directory.

    This function deliberately does not list, scan, inspect, or modify anything
    inside the root. The path is normalized before the existence and directory
    checks so downstream layers receive an absolute path.
    """

    root = settings.project_root.expanduser().resolve(strict=False)
    if not root.exists() or not root.is_dir():
        raise InvalidProjectRootError(root)
    return ProjectContext(root=root)
