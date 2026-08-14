"""Safe deletion of existing regular files for the Agent tool layer."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.filesystem import _validate_root
from backend_ai.tools.read_file import _reject_symlink_components, _resolve_requested_path


@dataclass(frozen=True, slots=True)
class DeleteFileResult:
    """Structured result for deleting one project-relative regular file."""

    relative_path: str
    file_name: str
    size_bytes: int
    deleted: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible deletion result without file contents."""

        return {
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "size_bytes": self.size_bytes,
            "deleted": self.deleted,
        }


class DeleteFileTool:
    """Agent tool for deleting one existing regular file without recursion."""

    name = "delete_file"
    description = (
        "Delete exactly one existing regular file inside an explicit project root. "
        "Directories, symbolic links, FIFOs, devices, traversal, and paths outside "
        "the root are rejected. The tool never deletes parent directories, follows "
        "symlinks, creates paths, or invokes a shell."
    )
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root", "path"],
            "properties": {
                "project_root": {"type": "string", "description": "Explicit project directory."},
                "path": {"type": "string", "description": "Existing file path relative to project_root."},
            },
        },
    )

    def run(self, arguments: Mapping[str, Any]) -> DeleteFileResult:
        """Validate a structured request and delete one regular file."""

        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "delete_file arguments must be a mapping.")
        missing = [name for name in ("project_root", "path") if name not in arguments]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, f"delete_file requires: {names}.")
        return delete_file(arguments["project_root"], arguments["path"])


def delete_file(project_root: Path | str, path: Path | str) -> DeleteFileResult:
    """Delete one existing regular file inside an explicit project root.

    The operation does not read file contents, does not create or remove
    directories, and does not follow symlinks. The parent directory is opened
    as a directory where supported, and the target is revalidated with
    no-follow metadata immediately before an unlink relative to that directory.
    This substantially narrows TOCTOU risk but cannot claim an absolute
    race-free guarantee on every filesystem/platform.
    """

    root = _validate_root(project_root)
    relative_path, lexical_path = _resolve_requested_path(root, path)
    _reject_symlink_components(root, relative_path, lexical_path)
    initial = _lstat_regular_file(lexical_path, relative_path)
    parent_fd = _open_parent_directory(lexical_path.parent, relative_path)
    try:
        current = _stat_at_parent(parent_fd, lexical_path.name, relative_path)
        if _identity(current) != _identity(initial):
            raise ToolError(
                ToolErrorCode.CONCURRENT_MODIFICATION,
                f"Target changed before deletion: {relative_path}",
                path=Path(relative_path),
            )
        if not stat.S_ISREG(current.st_mode):
            raise ToolError(
                ToolErrorCode.NOT_A_FILE,
                f"Target is no longer a regular file: {relative_path}",
                path=Path(relative_path),
            )
        try:
            os.unlink(lexical_path.name, dir_fd=parent_fd)
        except FileNotFoundError as exc:
            raise ToolError(
                ToolErrorCode.FILE_NOT_FOUND,
                f"Target file does not exist: {relative_path}",
                path=Path(relative_path),
            ) from exc
        except PermissionError as exc:
            raise ToolError(
                ToolErrorCode.PERMISSION_DENIED,
                f"Permission denied while deleting: {relative_path}",
                path=Path(relative_path),
            ) from exc
        except OSError as exc:
            raise ToolError(
                ToolErrorCode.FILESYSTEM_ERROR,
                f"Unable to delete: {relative_path}: {exc}",
                path=Path(relative_path),
            ) from exc
    finally:
        _close_descriptor(parent_fd)

    return DeleteFileResult(
        relative_path=relative_path,
        file_name=lexical_path.name,
        size_bytes=initial.st_size,
        deleted=True,
    )


def _lstat_regular_file(path: Path, relative_path: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ToolError(
            ToolErrorCode.FILE_NOT_FOUND,
            f"Target file does not exist: {relative_path}",
            path=Path(relative_path),
        ) from exc
    except PermissionError as exc:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Permission denied while inspecting: {relative_path}",
            path=Path(relative_path),
        ) from exc
    except OSError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to inspect: {relative_path}: {exc}",
            path=Path(relative_path),
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ToolError(
            ToolErrorCode.NOT_A_FILE,
            f"Target is not a regular file: {relative_path}",
            path=Path(relative_path),
        )
    return metadata


def _open_parent_directory(parent: Path, relative_path: str) -> int | None:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(str(parent), flags)
    except FileNotFoundError as exc:
        raise ToolError(
            ToolErrorCode.FILE_NOT_FOUND,
            f"Parent directory does not exist: {relative_path}",
            path=Path(relative_path).parent,
        ) from exc
    except NotADirectoryError as exc:
        raise ToolError(
            ToolErrorCode.NOT_A_FILE,
            f"Parent path is not a directory: {relative_path}",
            path=Path(relative_path).parent,
        ) from exc
    except PermissionError as exc:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Permission denied while opening parent directory: {relative_path}",
            path=Path(relative_path).parent,
        ) from exc
    except OSError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to open parent directory: {relative_path}: {exc}",
            path=Path(relative_path).parent,
        ) from exc


def _stat_at_parent(parent_fd: int | None, file_name: str, relative_path: str) -> os.stat_result:
    try:
        if parent_fd is None:
            return Path(file_name).lstat()
        return os.stat(file_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ToolError(
            ToolErrorCode.FILE_NOT_FOUND,
            f"Target file does not exist: {relative_path}",
            path=Path(relative_path),
        ) from exc
    except PermissionError as exc:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Permission denied while inspecting: {relative_path}",
            path=Path(relative_path),
        ) from exc
    except OSError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to revalidate: {relative_path}: {exc}",
            path=Path(relative_path),
        ) from exc


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _close_descriptor(file_descriptor: int | None) -> None:
    if file_descriptor is not None:
        try:
            os.close(file_descriptor)
        except OSError:
            pass


__all__ = ["DeleteFileResult", "DeleteFileTool", "delete_file"]
