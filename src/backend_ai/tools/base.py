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
    FILE_EXISTS = "FILE_EXISTS"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    MATCH_NOT_FOUND = "MATCH_NOT_FOUND"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    CONCURRENT_MODIFICATION = "CONCURRENT_MODIFICATION"
    BACKUP_FAILED = "BACKUP_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    RECOVERY_FAILED = "RECOVERY_FAILED"
    RECOVERY_UNAVAILABLE = "RECOVERY_UNAVAILABLE"
    ATOMIC_PUBLISH_FAILED = "ATOMIC_PUBLISH_FAILED"
    TEMPORARY_FILE_CLEANUP_FAILED = "TEMPORARY_FILE_CLEANUP_FAILED"
    COMMAND_INVALID = "COMMAND_INVALID"
    COMMAND_FAILED = "COMMAND_FAILED"
    COMMAND_TIMEOUT = "COMMAND_TIMEOUT"
    EXECUTABLE_NOT_FOUND = "EXECUTABLE_NOT_FOUND"
    WORKING_DIRECTORY_INVALID = "WORKING_DIRECTORY_INVALID"
    OUTPUT_LIMIT = "OUTPUT_LIMIT"
    COMMAND_DENIED = "COMMAND_DENIED"
    COMMAND_NOT_ALLOWED = "COMMAND_NOT_ALLOWED"
    UNSAFE_ARGUMENT = "UNSAFE_ARGUMENT"
    UNSAFE_WORKING_DIRECTORY = "UNSAFE_WORKING_DIRECTORY"
    UNSAFE_EXECUTABLE = "UNSAFE_EXECUTABLE"
    SHELL_BYPASS_ATTEMPT = "SHELL_BYPASS_ATTEMPT"
    ENVIRONMENT_NOT_ALLOWED = "ENVIRONMENT_NOT_ALLOWED"
    GIT_MUTATION_DENIED = "GIT_MUTATION_DENIED"
    NETWORK_COMMAND_DENIED = "NETWORK_COMMAND_DENIED"
    PACKAGE_OPERATION_DENIED = "PACKAGE_OPERATION_DENIED"
    GIT_NOT_AVAILABLE = "GIT_NOT_AVAILABLE"
    GIT_COMMAND_FAILED = "GIT_COMMAND_FAILED"
    GIT_TIMEOUT = "GIT_TIMEOUT"
    INVALID_REGEX = "INVALID_REGEX"


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
