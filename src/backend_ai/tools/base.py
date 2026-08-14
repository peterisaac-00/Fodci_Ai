"""Shared primitives for safe Agent tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ToolErrorCode(str, Enum):
    """Stable machine-readable error codes exposed by tools."""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    PATH_NOT_FOUND = "PATH_NOT_FOUND"
    NOT_DIRECTORY = "NOT_DIRECTORY"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    FILESYSTEM_ERROR = "FILESYSTEM_ERROR"
    DISCOVERY_LIMIT = "DISCOVERY_LIMIT"
    NOT_A_FILE = "NOT_A_FILE"
    PATH_OUTSIDE_ROOT = "PATH_OUTSIDE_ROOT"
    INVALID_UTF8 = "INVALID_UTF8"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"


class ToolError(RuntimeError):
    """Structured, user-safe failure from a tool boundary."""

    def __init__(
        self,
        code: ToolErrorCode,
        message: str,
        *,
        path: Path | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.path = path
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation suitable for an Agent."""

        return {
            "code": self.code.value,
            "message": self.message,
            "path": str(self.path) if self.path is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ToolMetadata:
    """Descriptive metadata shared by concrete tools."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


__all__ = ["ToolError", "ToolErrorCode", "ToolMetadata"]
