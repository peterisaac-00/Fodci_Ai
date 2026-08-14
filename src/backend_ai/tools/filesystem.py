"""Safe, deterministic filesystem discovery for the Agent tool layer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Any

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata

DEFAULT_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".coverage",
        "dist",
        "build",
        ".eggs",
    }
)
DEFAULT_MAX_FILES = 10_000
DEFAULT_MAX_DIRECTORIES = 10_000
DEFAULT_MAX_DEPTH = 32


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """Cheap metadata for one regular file, using a root-relative path."""

    relative_path: str
    name: str
    extension: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "name": self.name,
            "extension": self.extension,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class DiscoveredDirectory:
    """Metadata for one discovered directory, excluding the requested root."""

    relative_path: str
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class FileDiscoveryResult:
    """Structured, deterministic output from ``list_files``."""

    root: Path
    files: tuple[DiscoveredFile, ...]
    directories: tuple[DiscoveredDirectory, ...]
    total_files: int
    total_directories: int
    truncated: bool
    truncation_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result without reading file contents."""

        return {
            "root": str(self.root),
            "files": [item.to_dict() for item in self.files],
            "directories": [item.to_dict() for item in self.directories],
            "total_files": self.total_files,
            "total_directories": self.total_directories,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }


class ListFilesTool:
    """First-class read-only Agent tool for bounded project discovery."""

    name = "list_files"
    description = (
        "Recursively discover regular files and directories inside an explicit "
        "project root. Returns deterministic relative paths and cheap metadata. "
        "Read-only; symlinks and common generated directories are skipped."
    )
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root"],
            "properties": {
                "project_root": {"type": "string", "description": "Explicit project directory."},
                "max_files": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_FILES},
                "max_directories": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_DIRECTORIES},
                "max_depth": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_DEPTH},
                "include_hidden": {"type": "boolean", "default": True},
                "ignored_directories": {"type": "array", "items": {"type": "string"}},
            },
        },
    )

    def run(self, arguments: Mapping[str, Any]) -> FileDiscoveryResult:
        """Validate explicit arguments and run bounded discovery."""

        if not isinstance(arguments, Mapping):
            raise ToolError(
                ToolErrorCode.INVALID_ARGUMENT,
                "list_files arguments must be a mapping.",
            )
        if "project_root" not in arguments:
            raise ToolError(
                ToolErrorCode.INVALID_ARGUMENT,
                "list_files requires an explicit 'project_root'.",
            )
        return list_files(
            arguments["project_root"],
            max_files=arguments.get("max_files", DEFAULT_MAX_FILES),
            max_directories=arguments.get("max_directories", DEFAULT_MAX_DIRECTORIES),
            max_depth=arguments.get("max_depth", DEFAULT_MAX_DEPTH),
            include_hidden=arguments.get("include_hidden", True),
            ignored_directories=arguments.get("ignored_directories"),
        )


def list_files(
    project_root: Path | str,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_directories: int = DEFAULT_MAX_DIRECTORIES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    include_hidden: bool = True,
    ignored_directories: Iterable[str] | None = None,
) -> FileDiscoveryResult:
    """Discover a project tree without reading or modifying file contents.

    The root is explicit and normalized. Traversal is deterministic by
    normalized POSIX relative path. All symbolic links are skipped, including
    links to files, directories, and recursive links, so discovery never follows
    a link outside the requested root or enters a symlink loop.
    """

    root = _validate_root(project_root)
    ignored_values = _validate_options(
        max_files,
        max_directories,
        max_depth,
        include_hidden,
        ignored_directories,
    )
    ignored = set(DEFAULT_IGNORED_DIRECTORIES)
    ignored.update(ignored_values)

    files: list[DiscoveredFile] = []
    directories: list[DiscoveredDirectory] = []
    state = _DiscoveryState()

    def walk(directory: Path, current_depth: int) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(
                    iterator,
                    key=lambda entry: _relative_sort_key(root, Path(entry.path)),
                )
        except PermissionError as exc:
            raise ToolError(
                ToolErrorCode.PERMISSION_DENIED,
                f"Permission denied while listing directory: {directory}",
                path=directory,
            ) from exc
        except OSError as exc:
            raise ToolError(
                ToolErrorCode.FILESYSTEM_ERROR,
                f"Unable to list directory: {directory}: {exc}",
                path=directory,
            ) from exc

        for entry in entries:
            if state.truncated:
                return
            entry_path = Path(entry.path)
            if _should_skip(entry, include_hidden, ignored):
                continue
            entry_depth = current_depth + 1
            if entry_depth > max_depth:
                state.truncated = True
                state.truncation_reason = "max_depth"
                return
            try:
                metadata = entry.stat(follow_symlinks=False)
            except PermissionError as exc:
                raise ToolError(
                    ToolErrorCode.PERMISSION_DENIED,
                    f"Permission denied while inspecting: {entry_path}",
                    path=entry_path,
                ) from exc
            except OSError as exc:
                raise ToolError(
                    ToolErrorCode.FILESYSTEM_ERROR,
                    f"Unable to inspect filesystem entry: {entry_path}: {exc}",
                    path=entry_path,
                ) from exc

            relative_path = _relative_path(root, entry_path)
            if stat.S_ISDIR(metadata.st_mode):
                if len(directories) >= max_directories:
                    state.truncated = True
                    state.truncation_reason = "max_directories"
                    return
                directories.append(
                    DiscoveredDirectory(
                        relative_path=relative_path,
                        name=entry.name,
                    )
                )
                walk(entry_path, entry_depth)
            elif stat.S_ISREG(metadata.st_mode):
                if len(files) >= max_files:
                    state.truncated = True
                    state.truncation_reason = "max_files"
                    return
                files.append(
                    DiscoveredFile(
                        relative_path=relative_path,
                        name=entry.name,
                        extension=entry_path.suffix,
                        size=metadata.st_size,
                    )
                )
            # Sockets, devices, and other special entries are deliberately ignored.

    walk(root, 0)
    files.sort(key=lambda item: _path_sort_key(item.relative_path))
    directories.sort(key=lambda item: _path_sort_key(item.relative_path))
    return FileDiscoveryResult(
        root=root,
        files=tuple(files),
        directories=tuple(directories),
        total_files=len(files),
        total_directories=len(directories),
        truncated=state.truncated,
        truncation_reason=state.truncation_reason,
    )


@dataclass(slots=True)
class _DiscoveryState:
    truncated: bool = False
    truncation_reason: str | None = None


def _validate_root(project_root: Path | str) -> Path:
    if not isinstance(project_root, (Path, str)):
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            "project_root must be a path string or pathlib.Path.",
        )
    if isinstance(project_root, str) and not project_root.strip():
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "project_root must not be empty.")
    raw_root = Path(project_root).expanduser()
    try:
        root = raw_root.resolve(strict=False)
        exists = root.exists()
        is_directory = root.is_dir()
    except PermissionError as exc:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            "Permission denied while resolving project root.",
            path=raw_root,
        ) from exc
    except OSError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to resolve project root: {exc}",
            path=raw_root,
        ) from exc
    if not exists:
        raise ToolError(
            ToolErrorCode.PATH_NOT_FOUND,
            "Project root does not exist.",
            path=root,
        )
    if not is_directory:
        raise ToolError(
            ToolErrorCode.NOT_DIRECTORY,
            "Project root is not a directory.",
            path=root,
        )
    return root


def _validate_options(
    max_files: int,
    max_directories: int,
    max_depth: int,
    include_hidden: bool,
    ignored_directories: Iterable[str] | None,
) -> tuple[str, ...]:
    if not isinstance(max_files, int) or isinstance(max_files, bool) or max_files < 0:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "max_files must be a non-negative integer.")
    if not isinstance(max_directories, int) or isinstance(max_directories, bool) or max_directories < 0:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            "max_directories must be a non-negative integer.",
        )
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "max_depth must be a non-negative integer.")
    if not isinstance(include_hidden, bool):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "include_hidden must be a boolean.")
    if ignored_directories is None:
        return ()
    if isinstance(ignored_directories, (str, bytes)):
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            "ignored_directories must be an iterable of directory names, not a string.",
        )
    try:
        values = tuple(ignored_directories)
    except TypeError as exc:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            "ignored_directories must be an iterable of directory names.",
        ) from exc
    if any(not isinstance(value, str) or not value for value in values):
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            "ignored_directories must contain non-empty strings.",
        )
    return values


def _should_skip(entry: os.DirEntry[str], include_hidden: bool, ignored: set[str]) -> bool:
    if entry.is_symlink():
        return True
    if not include_hidden and entry.name.startswith("."):
        return True
    if entry.is_dir(follow_symlinks=False) and entry.name in ignored:
        return True
    if entry.is_file(follow_symlinks=False) and entry.name in ignored:
        return True
    return False


def _relative_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            "Discovery produced a path outside the requested project root.",
            path=path,
        ) from exc
    return relative.as_posix()


def _relative_sort_key(root: Path, path: Path) -> tuple[str, str]:
    relative = _relative_path(root, path)
    return _path_sort_key(relative)


def _path_sort_key(relative_path: str) -> tuple[str, str]:
    return (relative_path.casefold(), relative_path)


__all__ = [
    "DEFAULT_IGNORED_DIRECTORIES",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_DIRECTORIES",
    "DEFAULT_MAX_FILES",
    "DiscoveredDirectory",
    "DiscoveredFile",
    "FileDiscoveryResult",
    "ListFilesTool",
    "list_files",
]
