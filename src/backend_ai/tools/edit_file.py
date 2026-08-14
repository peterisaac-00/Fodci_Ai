"""Safe, bounded exact editing of existing UTF-8 files for the Agent tool layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.filesystem import _validate_root
from backend_ai.tools.read_file import _reject_symlink_components, _resolve_requested_path
from backend_ai.tools.write_file import DEFAULT_MAX_WRITE_BYTES

DEFAULT_MAX_FILE_BYTES = DEFAULT_MAX_WRITE_BYTES
DEFAULT_MAX_OLD_CONTENT_BYTES = DEFAULT_MAX_WRITE_BYTES
DEFAULT_MAX_NEW_CONTENT_BYTES = DEFAULT_MAX_WRITE_BYTES
DEFAULT_MAX_RESULT_BYTES = DEFAULT_MAX_WRITE_BYTES


@dataclass(frozen=True, slots=True)
class EditFileResult:
    """Structured result for one exact edit of an existing project file."""

    relative_path: str
    file_name: str
    original_size_bytes: int
    new_size_bytes: int
    bytes_changed: int
    match_count: int
    occurrence: int | None
    changed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible edit result without exposing file content."""

        return {
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "original_size_bytes": self.original_size_bytes,
            "new_size_bytes": self.new_size_bytes,
            "bytes_changed": self.bytes_changed,
            "match_count": self.match_count,
            "occurrence": self.occurrence,
            "changed": self.changed,
        }


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    """Private identity/content snapshot used for optimistic concurrency checks."""

    raw_content: bytes
    stat_fingerprint: tuple[int, int, int, int, int]
    digest: str


class EditFileTool:
    """Agent tool for one exact literal replacement in an existing UTF-8 file."""

    name = "edit_file"
    description = (
        "Replace exactly one literal old_content occurrence with new_content in an "
        "existing regular UTF-8 file inside an explicit project root. Missing files, "
        "ambiguous matches, traversal, symlinks, invalid UTF-8, and oversized edits "
        "are rejected; no fuzzy or regex matching is performed."
    )
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root", "path", "old_content", "new_content"],
            "properties": {
                "project_root": {"type": "string", "description": "Explicit project directory."},
                "path": {"type": "string", "description": "Existing file path relative to project_root."},
                "old_content": {"type": "string", "description": "Exact literal text to find once."},
                "new_content": {"type": "string", "description": "Replacement UTF-8 text."},
                "max_file_bytes": {
                    "type": "integer",
                    "minimum": 0,
                    "default": DEFAULT_MAX_FILE_BYTES,
                },
                "max_old_content_bytes": {
                    "type": "integer",
                    "minimum": 0,
                    "default": DEFAULT_MAX_OLD_CONTENT_BYTES,
                },
                "max_new_content_bytes": {
                    "type": "integer",
                    "minimum": 0,
                    "default": DEFAULT_MAX_NEW_CONTENT_BYTES,
                },
                "max_result_bytes": {
                    "type": "integer",
                    "minimum": 0,
                    "default": DEFAULT_MAX_RESULT_BYTES,
                },
            },
        },
    )

    def run(self, arguments: Mapping[str, Any]) -> EditFileResult:
        """Validate a structured request and perform one exact edit."""

        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "edit_file arguments must be a mapping.")
        missing = [
            name
            for name in ("project_root", "path", "old_content", "new_content")
            if name not in arguments
        ]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, f"edit_file requires: {names}.")
        return edit_file(
            arguments["project_root"],
            arguments["path"],
            arguments["old_content"],
            arguments["new_content"],
            max_file_bytes=arguments.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES),
            max_old_content_bytes=arguments.get(
                "max_old_content_bytes",
                DEFAULT_MAX_OLD_CONTENT_BYTES,
            ),
            max_new_content_bytes=arguments.get(
                "max_new_content_bytes",
                DEFAULT_MAX_NEW_CONTENT_BYTES,
            ),
            max_result_bytes=arguments.get("max_result_bytes", DEFAULT_MAX_RESULT_BYTES),
        )


def edit_file(
    project_root: Path | str,
    path: Path | str,
    old_content: str,
    new_content: str,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_old_content_bytes: int = DEFAULT_MAX_OLD_CONTENT_BYTES,
    max_new_content_bytes: int = DEFAULT_MAX_NEW_CONTENT_BYTES,
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
) -> EditFileResult:
    """Replace exactly one literal occurrence in an existing UTF-8 file.

    The target must already exist as a regular, non-symlink file. Matching is
    exact and case-sensitive over decoded text; no whitespace, line-ending,
    Unicode, regex, or fuzzy normalization occurs. The original is read and
    hashed before a bounded temporary replacement is atomically installed.
    """

    root = _validate_root(project_root)
    _validate_limits(
        max_file_bytes,
        max_old_content_bytes,
        max_new_content_bytes,
        max_result_bytes,
    )
    old_text, old_bytes = _encode_text(old_content, "old_content")
    new_text, new_bytes = _encode_text(new_content, "new_content")
    if not old_text:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "old_content must not be empty.")
    if len(old_bytes) > max_old_content_bytes:
        raise ToolError(
            ToolErrorCode.FILE_TOO_LARGE,
            f"old_content exceeds max_old_content_bytes: {len(old_bytes)} bytes; "
            f"maximum is {max_old_content_bytes}.",
        )
    if len(new_bytes) > max_new_content_bytes:
        raise ToolError(
            ToolErrorCode.FILE_TOO_LARGE,
            f"new_content exceeds max_new_content_bytes: {len(new_bytes)} bytes; "
            f"maximum is {max_new_content_bytes}.",
        )

    relative_path, lexical_path = _resolve_requested_path(root, path)
    _reject_symlink_components(root, relative_path, lexical_path)
    snapshot = _read_snapshot(lexical_path, relative_path, max_file_bytes)
    original_text = _decode_content(snapshot.raw_content, relative_path)
    match_count = original_text.count(old_text)
    if match_count == 0:
        raise ToolError(
            ToolErrorCode.MATCH_NOT_FOUND,
            f"Exact old_content was not found in: {relative_path}",
            path=Path(relative_path),
        )
    if match_count > 1:
        raise ToolError(
            ToolErrorCode.AMBIGUOUS_MATCH,
            f"Exact old_content matched {match_count} times in: {relative_path}; one match is required.",
            path=Path(relative_path),
        )

    replacement_text = original_text.replace(old_text, new_text, 1)
    replacement_bytes = replacement_text.encode("utf-8")
    if len(replacement_bytes) > max_result_bytes:
        raise ToolError(
            ToolErrorCode.FILE_TOO_LARGE,
            f"Result exceeds max_result_bytes: {len(replacement_bytes)} bytes; "
            f"maximum is {max_result_bytes}.",
            path=Path(relative_path),
        )

    original_size = len(snapshot.raw_content)
    new_size = len(replacement_bytes)
    if old_text == new_text:
        return EditFileResult(
            relative_path=relative_path,
            file_name=lexical_path.name,
            original_size_bytes=original_size,
            new_size_bytes=original_size,
            bytes_changed=0,
            match_count=match_count,
            occurrence=1,
            changed=False,
        )

    latest = _read_snapshot(lexical_path, relative_path, max_file_bytes)
    if latest != snapshot:
        raise ToolError(
            ToolErrorCode.CONCURRENT_MODIFICATION,
            f"Target changed while preparing edit: {relative_path}",
            path=Path(relative_path),
        )

    _atomic_replace(
        lexical_path,
        relative_path,
        replacement_bytes,
        root=root,
        mode=stat.S_IMODE(_stat_mode(lexical_path, relative_path)),
        expected=snapshot,
        max_file_bytes=max_file_bytes,
    )
    return EditFileResult(
        relative_path=relative_path,
        file_name=lexical_path.name,
        original_size_bytes=original_size,
        new_size_bytes=new_size,
        bytes_changed=new_size - original_size,
        match_count=match_count,
        occurrence=1,
        changed=True,
    )


def _read_snapshot(path: Path, relative_path: str, max_file_bytes: int) -> _FileSnapshot:
    metadata = _stat_regular_file(path, relative_path)
    if not os.access(path, os.R_OK):
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"File is not readable: {relative_path}",
            path=Path(relative_path),
        )
    if metadata.st_size > max_file_bytes:
        raise ToolError(
            ToolErrorCode.FILE_TOO_LARGE,
            f"File exceeds max_file_bytes: {relative_path} is {metadata.st_size} bytes; "
            f"maximum is {max_file_bytes}.",
            path=Path(relative_path),
        )
    try:
        with path.open("rb") as stream:
            raw_content = stream.read(max_file_bytes + 1)
    except PermissionError as exc:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Permission denied while reading: {relative_path}",
            path=Path(relative_path),
        ) from exc
    except OSError as exc:
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to read: {relative_path}: {exc}",
            path=Path(relative_path),
        ) from exc
    if len(raw_content) > max_file_bytes:
        raise ToolError(
            ToolErrorCode.FILE_TOO_LARGE,
            f"File exceeded max_file_bytes while reading: {relative_path}; "
            f"maximum is {max_file_bytes}.",
            path=Path(relative_path),
        )
    _decode_content(raw_content, relative_path)
    after_read = _stat_regular_file(path, relative_path)
    if _stat_fingerprint(after_read) != _stat_fingerprint(metadata):
        raise ToolError(
            ToolErrorCode.CONCURRENT_MODIFICATION,
            f"Target changed while reading: {relative_path}",
            path=Path(relative_path),
        )
    return _FileSnapshot(
        raw_content=raw_content,
        stat_fingerprint=_stat_fingerprint(after_read),
        digest=hashlib.sha256(raw_content).hexdigest(),
    )


def _atomic_replace(
    target: Path,
    relative_path: str,
    replacement: bytes,
    *,
    root: Path,
    mode: int,
    expected: _FileSnapshot,
    max_file_bytes: int,
) -> None:
    latest = _read_snapshot(target, relative_path, max_file_bytes)
    if latest != expected:
        raise ToolError(
            ToolErrorCode.CONCURRENT_MODIFICATION,
            f"Target changed before replacement: {relative_path}",
            path=Path(relative_path),
        )

    file_descriptor: int | None = None
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.fodci-edit-",
            dir=str(target.parent),
        )
        temporary_path = Path(temporary_name)
        os.fchmod(file_descriptor, mode)
        with os.fdopen(file_descriptor, "wb") as stream:
            file_descriptor = None
            stream.write(replacement)
            stream.flush()
            os.fsync(stream.fileno())
        _reject_symlink_components(root, relative_path, target)
        latest = _read_snapshot(target, relative_path, max_file_bytes)
        if latest != expected:
            raise ToolError(
                ToolErrorCode.CONCURRENT_MODIFICATION,
                f"Target changed before atomic replacement: {relative_path}",
                path=Path(relative_path),
            )
        os.replace(str(temporary_path), str(target))
        temporary_path = None
    except ToolError:
        _close_descriptor(file_descriptor)
        _remove_temporary_file(temporary_path)
        raise
    except PermissionError as exc:
        _close_descriptor(file_descriptor)
        _remove_temporary_file(temporary_path)
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Permission denied while replacing: {relative_path}",
            path=Path(relative_path),
        ) from exc
    except OSError as exc:
        _close_descriptor(file_descriptor)
        _remove_temporary_file(temporary_path)
        raise ToolError(
            ToolErrorCode.FILESYSTEM_ERROR,
            f"Unable to atomically replace: {relative_path}: {exc}",
            path=Path(relative_path),
        ) from exc


def _stat_regular_file(path: Path, relative_path: str) -> os.stat_result:
    try:
        metadata = path.stat()
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
    if metadata.st_mode & 0o222 == 0:
        raise ToolError(
            ToolErrorCode.PERMISSION_DENIED,
            f"Target is not writable: {relative_path}",
            path=Path(relative_path),
        )
    return metadata


def _stat_mode(path: Path, relative_path: str) -> int:
    return _stat_regular_file(path, relative_path).st_mode


def _stat_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _decode_content(raw_content: bytes, relative_path: str) -> str:
    try:
        return raw_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(
            ToolErrorCode.INVALID_UTF8,
            f"File is not valid UTF-8: {relative_path}.",
            path=Path(relative_path),
        ) from exc


def _encode_text(value: Any, name: str) -> tuple[str, bytes]:
    if not isinstance(value, str):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, f"{name} must be a string.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ToolError(
            ToolErrorCode.INVALID_UTF8,
            f"{name} cannot be encoded as UTF-8.",
        ) from exc
    return value, encoded


def _validate_limits(
    max_file_bytes: int,
    max_old_content_bytes: int,
    max_new_content_bytes: int,
    max_result_bytes: int,
) -> None:
    for name, value in (
        ("max_file_bytes", max_file_bytes),
        ("max_old_content_bytes", max_old_content_bytes),
        ("max_new_content_bytes", max_new_content_bytes),
        ("max_result_bytes", max_result_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, f"{name} must be a non-negative integer.")


def _close_descriptor(file_descriptor: int | None) -> None:
    if file_descriptor is not None:
        try:
            os.close(file_descriptor)
        except OSError:
            pass


def _remove_temporary_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_NEW_CONTENT_BYTES",
    "DEFAULT_MAX_OLD_CONTENT_BYTES",
    "DEFAULT_MAX_RESULT_BYTES",
    "EditFileResult",
    "EditFileTool",
    "edit_file",
]
