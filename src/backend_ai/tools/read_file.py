"""Safe, bounded file reading for the Agent tool layer."""

from __future__ import annotations

from dataclasses import dataclass
import ntpath
import os
from pathlib import Path, PureWindowsPath
import stat
from typing import Any, Mapping

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.filesystem import _validate_root

DEFAULT_MAX_READ_BYTES = 1_048_576


@dataclass(frozen=True, slots=True)
class ReadFileResult:
    """Exact decoded contents and cheap metadata for one project-relative file."""

    relative_path: str
    file_name: str
    content: str
    encoding: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable result without changing the content."""

        return {
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "content": self.content,
            "encoding": self.encoding,
            "size_bytes": self.size_bytes,
        }


class ReadFileTool:
    """First-class read-only Agent tool for bounded UTF-8 file reading."""

    name = "read_file"
    description = (
        "Read the exact UTF-8 contents of one regular file inside an explicit "
        "project root. Read-only, bounded by max_bytes, and rejects traversal "
        "and symbolic links."
    )
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root", "path"],
            "properties": {
                "project_root": {"type": "string", "description": "Explicit project directory."},
                "path": {"type": "string", "description": "File path relative to project_root."},
                "max_bytes": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_READ_BYTES},
            },
        },
    )

    def run(self, arguments: Mapping[str, Any]) -> ReadFileResult:
        """Validate a structured request and read one file."""

        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "read_file arguments must be a mapping.")
        missing = [name for name in ("project_root", "path") if name not in arguments]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ToolError(
                ToolErrorCode.INVALID_ARGUMENT,
                f"read_file requires: {names}.",
            )
        return read_file(
            arguments["project_root"],
            arguments["path"],
            max_bytes=arguments.get("max_bytes", DEFAULT_MAX_READ_BYTES),
        )


def read_file(
    project_root: Path | str,
    path: Path | str,
    *,
    max_bytes: int = DEFAULT_MAX_READ_BYTES,
) -> ReadFileResult:
    """Read one regular UTF-8 file without escaping or following symlinks."""

    root = _validate_root(project_root)
    _validate_max_bytes(max_bytes)
    relative_path, lexical_path = _resolve_requested_path(root, path)
    _reject_symlink_components(root, relative_path, lexical_path)

    try:
        resolved_path = lexical_path.resolve(strict=False)
        resolved_relative = resolved_path.relative_to(root)
    except ValueError as exc:
        raise ToolError(
            ToolErrorCode.PATH_OUTSIDE_ROOT,
            "Requested path resolves outside the project root.",
            path=lexical_path,
        ) from exc
    except PermissionError as exc:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            "Permission denied while resolving requested file.",
            path=lexical_path,
        ) from exc
    except OSError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to resolve requested file: {exc}",
            path=lexical_path,
        ) from exc

    try:
        metadata = resolved_path.stat()
    except FileNotFoundError as exc:
        raise ToolError(
            ToolErrorCode.PATH_NOT_FOUND,
            f"Requested file does not exist: {relative_path}",
            path=resolved_relative,
        ) from exc
    except PermissionError as exc:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Permission denied while inspecting: {relative_path}",
            path=resolved_relative,
        ) from exc
    except OSError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to inspect requested file: {relative_path}: {exc}",
            path=resolved_relative,
        ) from exc

    if not stat.S_ISREG(metadata.st_mode):
        raise ToolError(
            ToolErrorCode.NOT_A_FILE,
            f"Requested path is not a regular file: {relative_path}",
            path=resolved_relative,
        )
    if metadata.st_size > max_bytes:
        raise ToolError(
            ToolErrorCode.FILE_TOO_LARGE,
            f"File exceeds max_bytes: {relative_path} is {metadata.st_size} bytes; "
            f"maximum is {max_bytes}.",
            path=resolved_relative,
        )

    try:
        with resolved_path.open("rb") as stream:
            raw_content = stream.read(max_bytes + 1)
    except PermissionError as exc:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Permission denied while reading: {relative_path}",
            path=resolved_relative,
        ) from exc
    except OSError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to read requested file: {relative_path}: {exc}",
            path=resolved_relative,
        ) from exc

    if len(raw_content) > max_bytes:
        raise ToolError(
            ToolErrorCode.FILE_TOO_LARGE,
            f"File exceeded max_bytes while reading: {relative_path}; maximum is {max_bytes}.",
            path=resolved_relative,
        )
    try:
        content = raw_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(
            ToolErrorCode.INVALID_UTF8,
            f"File is not valid UTF-8: {relative_path}.",
            path=resolved_relative,
        ) from exc

    return ReadFileResult(
        relative_path=relative_path,
        file_name=resolved_path.name,
        content=content,
        encoding="utf-8",
        size_bytes=len(raw_content),
    )


def _validate_max_bytes(max_bytes: int) -> None:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            "max_bytes must be a non-negative integer.",
        )


def _resolve_requested_path(root: Path, requested_path: Path | str) -> tuple[str, Path]:
    if not isinstance(requested_path, (Path, str)):
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            "path must be a string or pathlib.Path.",
        )
    raw = str(requested_path)
    if not raw.strip():
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "path must not be empty.")
    normalized = raw.replace("\\", "/")
    if "\x00" in normalized:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "path must not contain a NUL byte.")
    if _looks_like_windows_absolute(normalized):
        candidate = Path(normalized)
        if not candidate.is_absolute():
            raise ToolError(
                ToolErrorCode.PATH_OUTSIDE_ROOT,
                "Absolute Windows and UNC paths cannot bypass the project root.",
                path=candidate,
            )
    else:
        candidate = Path(normalized)

    lexical_path = candidate if candidate.is_absolute() else root / candidate
    lexical_path = Path(os.path.normpath(str(lexical_path)))
    try:
        relative = lexical_path.relative_to(root)
    except ValueError as exc:
        raise ToolError(
            ToolErrorCode.PATH_OUTSIDE_ROOT,
            "Requested path is outside the project root.",
            path=lexical_path,
        ) from exc
    relative_text = relative.as_posix()
    if not relative_text or relative_text == ".":
        raise ToolError(
            ToolErrorCode.NOT_A_FILE,
            "Requested path refers to the project root, not a file.",
            path=relative,
        )
    return relative_text, lexical_path


def _looks_like_windows_absolute(path: str) -> bool:
    return ntpath.isabs(path) or PureWindowsPath(path).is_absolute()


def _reject_symlink_components(root: Path, relative_path: str, lexical_path: Path) -> None:
    try:
        relative_parts = Path(relative_path).parts
        current = root
        for part in relative_parts:
            current = current / part
            if current.is_symlink():
                raise ToolError(
                    ToolErrorCode.PATH_OUTSIDE_ROOT,
                    f"Symbolic links are not readable by read_file: {relative_path}",
                    path=Path(relative_path),
                )
    except ToolError:
        raise
    except PermissionError as exc:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Permission denied while checking requested path: {relative_path}",
            path=lexical_path,
        ) from exc
    except OSError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to check requested path: {relative_path}: {exc}",
            path=lexical_path,
        ) from exc


__all__ = [
    "DEFAULT_MAX_READ_BYTES",
    "ReadFileResult",
    "ReadFileTool",
    "read_file",
]
