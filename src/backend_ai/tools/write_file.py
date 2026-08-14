"""Safe, bounded creation of new UTF-8 files for the Agent tool layer."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.filesystem import _validate_root
from backend_ai.tools.read_file import _reject_symlink_components, _resolve_requested_path

DEFAULT_MAX_WRITE_BYTES = 1_048_576
DEFAULT_MAX_PARENT_DIRECTORIES = 32


@dataclass(frozen=True, slots=True)
class WriteFileResult:
    """Structured result for one newly created project-relative file."""

    relative_path: str
    file_name: str
    size_bytes: int
    encoding: str
    created: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible creation result."""

        return {
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "size_bytes": self.size_bytes,
            "encoding": self.encoding,
            "created": self.created,
        }


class WriteFileTool:
    """Agent tool for creating one new regular UTF-8 file without overwriting."""

    name = "write_file"
    description = (
        "Create exactly one new UTF-8 regular file inside an explicit project root. "
        "Missing parent directories may be created safely inside that root, while "
        "existing paths, traversal, and symbolic links are rejected. The operation "
        "is bounded and never overwrites."
    )
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root", "path", "content"],
            "properties": {
                "project_root": {"type": "string", "description": "Explicit project directory."},
                "path": {"type": "string", "description": "New file path relative to project_root."},
                "content": {"type": "string", "description": "UTF-8 text to write."},
                "max_bytes": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_WRITE_BYTES},
                "max_parent_directories": {
                    "type": "integer",
                    "minimum": 0,
                    "default": DEFAULT_MAX_PARENT_DIRECTORIES,
                },
            },
        },
    )

    def run(self, arguments: Mapping[str, Any]) -> WriteFileResult:
        """Validate a structured request and create one new file."""

        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "write_file arguments must be a mapping.")
        missing = [name for name in ("project_root", "path", "content") if name not in arguments]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, f"write_file requires: {names}.")
        return write_file(
            arguments["project_root"],
            arguments["path"],
            arguments["content"],
            max_bytes=arguments.get("max_bytes", DEFAULT_MAX_WRITE_BYTES),
            max_parent_directories=arguments.get(
                "max_parent_directories",
                DEFAULT_MAX_PARENT_DIRECTORIES,
            ),
        )


def write_file(
    project_root: Path | str,
    path: Path | str,
    content: str,
    *,
    max_bytes: int = DEFAULT_MAX_WRITE_BYTES,
    max_parent_directories: int = DEFAULT_MAX_PARENT_DIRECTORIES,
) -> WriteFileResult:
    """Create one new UTF-8 regular file inside an existing project root.

    Missing parent directories are created one component at a time with a
    bounded depth, only after root/path/symlink validation. Content is written
    and synchronized to a private temporary file, then atomically published
    through an exclusive hard-link operation. A concurrent or pre-existing
    target cannot be overwritten, and temporary artifacts are cleaned on
    handled failures.
    """

    root = _validate_root(project_root)
    _validate_max_bytes(max_bytes)
    _validate_max_parent_directories(max_parent_directories)
    encoded = _encode_content(content)
    if len(encoded) > max_bytes:
        raise ToolError(
            ToolErrorCode.FILE_TOO_LARGE,
            f"Content exceeds max_bytes: {len(encoded)} bytes; maximum is {max_bytes}.",
        )

    relative_path, lexical_path = _resolve_requested_path(root, path)
    _reject_symlink_components(root, relative_path, lexical_path)
    created_parent_directories = _ensure_parent_directories(
        root,
        lexical_path.parent,
        relative_path,
        max_parent_directories=max_parent_directories,
    )
    _reject_symlink_components(root, relative_path, lexical_path)

    try:
        lexical_path.stat()
    except FileNotFoundError:
        pass
    except PermissionError as exc:
        _cleanup_created_directories(created_parent_directories)
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Permission denied while inspecting target: {relative_path}",
            path=Path(relative_path),
        ) from exc
    except OSError as exc:
        _cleanup_created_directories(created_parent_directories)
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to inspect target: {relative_path}: {exc}",
            path=Path(relative_path),
        ) from exc
    else:
        _cleanup_created_directories(created_parent_directories)
        raise ToolError(
            ToolErrorCode.FILE_EXISTS,
            f"Target already exists: {relative_path}",
            path=Path(relative_path),
        )

    file_descriptor: int | None = None
    temporary_path: Path | None = None
    published = False
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{lexical_path.name}.fodci-",
            dir=str(lexical_path.parent),
        )
        temporary_path = Path(temporary_name)
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as stream:
            file_descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary_path), str(lexical_path))
        published = True
        temporary_path.unlink()
        temporary_path = None
    except FileExistsError as exc:
        _close_descriptor(file_descriptor)
        _remove_temporary_file(temporary_path)
        _cleanup_created_directories(created_parent_directories)
        raise ToolError(
            ToolErrorCode.FILE_EXISTS,
            f"Target already exists: {relative_path}",
            path=Path(relative_path),
        ) from exc
    except PermissionError as exc:
        _close_descriptor(file_descriptor)
        _remove_temporary_file(temporary_path)
        _remove_created_file(lexical_path, published)
        _cleanup_created_directories(created_parent_directories)
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Permission denied while creating: {relative_path}",
            path=Path(relative_path),
        ) from exc
    except OSError as exc:
        _close_descriptor(file_descriptor)
        _remove_temporary_file(temporary_path)
        _remove_created_file(lexical_path, published)
        _cleanup_created_directories(created_parent_directories)
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to create file: {relative_path}: {exc}",
            path=Path(relative_path),
        ) from exc

    return WriteFileResult(
        relative_path=relative_path,
        file_name=lexical_path.name,
        size_bytes=len(encoded),
        encoding="utf-8",
    )


def _encode_content(content: str) -> bytes:
    if not isinstance(content, str):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "content must be a string.")
    try:
        return content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ToolError(
            ToolErrorCode.INVALID_UTF8,
            "content cannot be encoded as UTF-8.",
        ) from exc


def _validate_max_bytes(max_bytes: int) -> None:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            "max_bytes must be a non-negative integer.",
        )


def _validate_max_parent_directories(max_parent_directories: int) -> None:
    if (
        not isinstance(max_parent_directories, int)
        or isinstance(max_parent_directories, bool)
        or max_parent_directories < 0
    ):
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            "max_parent_directories must be a non-negative integer.",
        )


def _ensure_parent_directories(
    root: Path,
    parent: Path,
    relative_path: str,
    *,
    max_parent_directories: int,
) -> tuple[Path, ...]:
    try:
        relative_parent = parent.relative_to(root)
    except ValueError as exc:
        raise ToolError(
            ToolErrorCode.PATH_OUTSIDE_ROOT,
            "Parent directory is outside the project root.",
            path=parent,
        ) from exc
    parts = () if str(relative_parent) == "." else relative_parent.parts
    if len(parts) > max_parent_directories:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            "Parent directory depth exceeds max_parent_directories.",
            path=Path(relative_path),
        )

    current = root
    created: list[Path] = []
    for part in parts:
        current = current / part
        if current.is_symlink():
            _cleanup_created_directories(tuple(created))
            raise ToolError(
                ToolErrorCode.PATH_OUTSIDE_ROOT,
                f"Symbolic links are not allowed in parent path: {relative_path}",
                path=Path(relative_path),
            )
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            if current.is_symlink():
                _cleanup_created_directories(tuple(created))
                raise ToolError(
                    ToolErrorCode.PATH_OUTSIDE_ROOT,
                    f"Symbolic links are not allowed in parent path: {relative_path}",
                    path=Path(relative_path),
                )
            try:
                metadata = current.stat()
            except PermissionError as exc:
                _cleanup_created_directories(tuple(created))
                raise ToolError(
                    ToolErrorCode.PERMISSION_DENIED,
                    f"Permission denied while inspecting parent: {relative_path}",
                    path=Path(relative_path),
                ) from exc
            except OSError as exc:
                _cleanup_created_directories(tuple(created))
                raise ToolError(
                    ToolErrorCode.FILESYSTEM_ERROR,
                    f"Unable to inspect parent: {relative_path}: {exc}",
                    path=Path(relative_path),
                ) from exc
            if not stat.S_ISDIR(metadata.st_mode):
                _cleanup_created_directories(tuple(created))
                raise ToolError(
                    ToolErrorCode.NOT_DIRECTORY,
                    f"Parent path is not a directory: {relative_path}",
                    path=Path(relative_path),
                )
        except PermissionError as exc:
            _cleanup_created_directories(tuple(created))
            raise ToolError(
                ToolErrorCode.PERMISSION_DENIED,
                f"Permission denied while creating parent: {relative_path}",
                path=Path(relative_path),
            ) from exc
        except OSError as exc:
            _cleanup_created_directories(tuple(created))
            raise ToolError(
                ToolErrorCode.FILESYSTEM_ERROR,
                f"Unable to create parent: {relative_path}: {exc}",
                path=Path(relative_path),
            ) from exc
        else:
            created.append(current)
    return tuple(created)


def _close_descriptor(file_descriptor: int | None) -> None:
    if file_descriptor is not None:
        try:
            os.close(file_descriptor)
        except OSError:
            pass


def _remove_created_file(path: Path, created: bool) -> None:
    if not created:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _remove_temporary_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _cleanup_created_directories(paths: tuple[Path, ...]) -> None:
    for path in reversed(paths):
        try:
            path.rmdir()
        except OSError:
            pass


__all__ = [
    "DEFAULT_MAX_PARENT_DIRECTORIES",
    "DEFAULT_MAX_WRITE_BYTES",
    "WriteFileResult",
    "WriteFileTool",
    "write_file",
]
