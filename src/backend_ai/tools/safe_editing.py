"""Reusable safety infrastructure for bounded file mutations.

This module wraps the existing write/edit/delete tools rather than replacing
or weakening them. It adds conservative policy checks, immutable snapshots,
internal deterministic diffs, optional controlled backups, and post-operation
verification without enabling AgentLoop mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Literal

from backend_ai.tools.base import ToolError, ToolErrorCode
from backend_ai.tools.delete_file import delete_file
from backend_ai.tools.edit_file import edit_file
from backend_ai.tools.filesystem import _validate_root
from backend_ai.tools.read_file import _reject_symlink_components, _resolve_requested_path
from backend_ai.tools.write_file import write_file

DEFAULT_SAFE_EDIT_MAX_FILE_SIZE = 1_048_576
DEFAULT_SAFE_EDIT_MAX_CONTENT_SIZE = 1_048_576
DEFAULT_SAFE_EDIT_MAX_DIFF_BYTES = 65_536
DEFAULT_SAFE_EDIT_MAX_DIFF_LINES = 2_000
DEFAULT_SAFE_EDIT_BACKUP_DIRECTORY = ".fodci/backups"
_HASH_CHUNK_SIZE = 65_536

OperationName = Literal["create", "edit", "delete"]


@dataclass(frozen=True, slots=True)
class SafeEditPolicy:
    """Conservative capabilities and resource limits for safe mutations.

    Existing tool guarantees cannot be disabled by this layer: explicit roots,
    symlink rejection, atomic writes, and concurrency detection are required.
    Mutation capabilities and optional backups/diffs/verification are opt-in.
    """

    require_project_root: bool = True
    allow_create: bool = False
    allow_edit: bool = False
    allow_delete: bool = False
    max_file_size: int = DEFAULT_SAFE_EDIT_MAX_FILE_SIZE
    max_content_size: int = DEFAULT_SAFE_EDIT_MAX_CONTENT_SIZE
    backup_enabled: bool = False
    retain_backup_on_success: bool = False
    diff_enabled: bool = True
    max_diff_bytes: int = DEFAULT_SAFE_EDIT_MAX_DIFF_BYTES
    max_diff_lines: int = DEFAULT_SAFE_EDIT_MAX_DIFF_LINES
    preserve_metadata: bool = True
    verify_after_write: bool = True
    reject_symlinks: bool = True
    atomic_write: bool = True
    detect_concurrent_modification: bool = True
    backup_directory: str = DEFAULT_SAFE_EDIT_BACKUP_DIRECTORY

    def __post_init__(self) -> None:
        for name in (
            "max_file_size",
            "max_content_size",
            "max_diff_bytes",
            "max_diff_lines",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        required_true = (
            "require_project_root",
            "preserve_metadata",
            "verify_after_write",
            "reject_symlinks",
            "atomic_write",
            "detect_concurrent_modification",
        )
        if any(getattr(self, name) is not True for name in required_true):
            raise ValueError("SafeEditPolicy cannot weaken required filesystem safety guarantees")
        if not isinstance(self.backup_enabled, bool) or not isinstance(self.diff_enabled, bool):
            raise ValueError("backup_enabled and diff_enabled must be booleans")
        if not isinstance(self.retain_backup_on_success, bool):
            raise ValueError("retain_backup_on_success must be a boolean")
        if not isinstance(self.backup_directory, str) or not self.backup_directory.strip():
            raise ValueError("backup_directory must be non-empty text")
        if "\\" in self.backup_directory or self.backup_directory.startswith("/"):
            raise ValueError("backup_directory must be a project-relative POSIX path")
        if any(part in ("", ".", "..") for part in Path(self.backup_directory).parts):
            raise ValueError("backup_directory must not contain traversal components")

    @classmethod
    def for_modification(
        cls,
        *,
        backup_enabled: bool = False,
        retain_backup_on_success: bool = False,
        diff_enabled: bool = True,
        **kwargs: Any,
    ) -> "SafeEditPolicy":
        """Build an explicit policy capable of all three mutation wrappers."""

        return cls(
            allow_create=True,
            allow_edit=True,
            allow_delete=True,
            backup_enabled=backup_enabled,
            retain_backup_on_success=retain_backup_on_success,
            diff_enabled=diff_enabled,
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Immutable metadata identity for one root-relative filesystem entry."""

    relative_path: str
    exists: bool
    size_bytes: int
    mtime_ns: int | None
    device: int | None
    inode: int | None
    content_hash: str | None
    file_type: str
    mode: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return metadata without exposing file contents."""

        return {
            "relative_path": self.relative_path,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
            "content_hash": self.content_hash,
            "file_type": self.file_type,
            "mode": self.mode,
        }

    def same_identity(self, other: "FileSnapshot") -> bool:
        """Compare the observable identity used for optimistic verification."""

        return (
            self.relative_path == other.relative_path
            and self.exists == other.exists
            and self.size_bytes == other.size_bytes
            and self.mtime_ns == other.mtime_ns
            and self.device == other.device
            and self.inode == other.inode
            and self.content_hash == other.content_hash
            and self.file_type == other.file_type
            and self.mode == other.mode
        )


@dataclass(frozen=True, slots=True)
class DiffResult:
    """Bounded deterministic internal unified diff, never a VCS diff."""

    operation: OperationName
    relative_path: str
    text: str
    line_count: int
    size_bytes: int
    truncated: bool
    available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "relative_path": self.relative_path,
            "text": self.text,
            "line_count": self.line_count,
            "size_bytes": self.size_bytes,
            "truncated": self.truncated,
            "available": self.available,
        }


@dataclass(frozen=True, slots=True)
class BackupResult:
    """Controlled backup metadata; backup content is never returned."""

    created: bool
    relative_path: str | None
    size_bytes: int
    retained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "retained": self.retained,
        }


@dataclass(frozen=True, slots=True)
class SafeEditResult:
    """Consistent result for one safe create/edit/delete operation."""

    operation: OperationName
    relative_path: str
    success: bool
    changed: bool
    created: bool
    deleted: bool
    old_size_bytes: int
    new_size_bytes: int
    old_hash: str | None
    new_hash: str | None
    diff: DiffResult | None
    backup: BackupResult | None
    verification_passed: bool
    concurrent_change_detected: bool = False
    verification: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "relative_path": self.relative_path,
            "success": self.success,
            "changed": self.changed,
            "created": self.created,
            "deleted": self.deleted,
            "old_size_bytes": self.old_size_bytes,
            "new_size_bytes": self.new_size_bytes,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "diff": self.diff.to_dict() if self.diff else None,
            "backup": self.backup.to_dict() if self.backup else None,
            "verification_passed": self.verification_passed,
            "concurrent_change_detected": self.concurrent_change_detected,
            "verification": self.verification.to_dict() if self.verification is not None else None,
        }


class SafeEditSession:
    """Policy-guarded wrapper around the existing file mutation tools."""

    def __init__(self, policy: SafeEditPolicy) -> None:
        if not isinstance(policy, SafeEditPolicy):
            raise TypeError("SafeEditSession requires SafeEditPolicy")
        self.policy = policy

    def snapshot(self, project_root: Path | str, path: Path | str) -> FileSnapshot:
        """Capture bounded metadata/hash identity without exposing content."""

        root, relative, target = _safe_target(project_root, path, self.policy)
        try:
            metadata = target.lstat()
        except FileNotFoundError:
            return FileSnapshot(relative, False, 0, None, None, None, None, "missing", None)
        except PermissionError as exc:
            raise ToolError(ToolErrorCode.PERMISSION_DENIED, "Permission denied while snapshotting target.", path=Path(relative)) from exc
        except OSError as exc:
            raise ToolError(ToolErrorCode.FILESYSTEM_ERROR, "Unable to snapshot target.", path=Path(relative)) from exc
        file_type = _file_type(metadata.st_mode)
        if stat.S_ISLNK(metadata.st_mode) and self.policy.reject_symlinks:
            raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "Symbolic links are not allowed.", path=Path(relative))
        if not stat.S_ISREG(metadata.st_mode):
            return FileSnapshot(relative, True, metadata.st_size, metadata.st_mtime_ns, metadata.st_dev, metadata.st_ino, None, file_type, stat.S_IMODE(metadata.st_mode))
        if metadata.st_size > self.policy.max_file_size:
            raise ToolError(ToolErrorCode.FILE_TOO_LARGE, "Target exceeds the safe snapshot size limit.", path=Path(relative))
        return FileSnapshot(
            relative,
            True,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_dev,
            metadata.st_ino,
            _hash_file(target, relative, self.policy.max_file_size),
            file_type,
            stat.S_IMODE(metadata.st_mode),
        )

    def diff_for_create(self, project_root: Path | str, path: Path | str, content: str) -> DiffResult | None:
        root, relative, _ = _safe_target(project_root, path, self.policy)
        _ = root
        content_bytes = _encode_bounded(content, self.policy.max_content_size)
        return self._diff("create", relative, "", _decode_diff_text(content_bytes))

    def diff_for_edit(self, project_root: Path | str, path: Path | str, old_content: str, new_content: str) -> DiffResult | None:
        _, relative, _ = _safe_target(project_root, path, self.policy)
        old_bytes = _encode_bounded(old_content, self.policy.max_content_size)
        new_bytes = _encode_bounded(new_content, self.policy.max_content_size)
        return self._diff("edit", relative, _decode_diff_text(old_bytes), _decode_diff_text(new_bytes))

    def diff_for_delete(self, project_root: Path | str, path: Path | str) -> DiffResult | None:
        _, relative, target = _safe_target(project_root, path, self.policy)
        try:
            raw = _read_bounded_bytes(target, relative, self.policy.max_file_size)
        except ToolError as exc:
            if exc.code is ToolErrorCode.INVALID_UTF8:
                return DiffResult("delete", relative, "", 0, 0, False, False)
            raise
        return self._diff("delete", relative, _decode_diff_text(raw), "")

    def create(self, project_root: Path | str, path: Path | str, content: str) -> SafeEditResult:
        self._require_capability("create")
        root, relative, _ = _safe_target(project_root, path, self.policy)
        content_bytes = _encode_bounded(content, self.policy.max_content_size)
        diff = self._diff("create", relative, "", _decode_diff_text(content_bytes))
        before = self.snapshot(root, relative)
        if before.exists:
            raise ToolError(ToolErrorCode.FILE_EXISTS, "Target already exists.", path=Path(relative))
        write_file(root, relative, content, max_bytes=self.policy.max_content_size)
        after = self.snapshot(root, relative)
        expected_hash = hashlib.sha256(content_bytes).hexdigest()
        self._verify(after.exists and after.content_hash == expected_hash, relative)
        from backend_ai.tools.modification_verification import ExpectedModification, verify_modification
        verification = verify_modification(root, [ExpectedModification.created(relative, expected_sha256=expected_hash, expected_size=len(content_bytes))], detect_unexpected=False, max_file_bytes=self.policy.max_file_size)
        self._verify(verification.success, relative)
        return SafeEditResult("create", relative, True, True, True, False, 0, len(content_bytes), None, expected_hash, diff, None, True, verification=verification)

    def edit(self, project_root: Path | str, path: Path | str, old_content: str, new_content: str) -> SafeEditResult:
        self._require_capability("edit")
        root, relative, target = _safe_target(project_root, path, self.policy)
        before = self.snapshot(root, relative)
        if not before.exists:
            raise ToolError(ToolErrorCode.FILE_NOT_FOUND, "Target file does not exist.", path=Path(relative))
        old_bytes = _encode_bounded(old_content, self.policy.max_content_size)
        new_bytes = _encode_bounded(new_content, self.policy.max_content_size)
        diff = self._diff("edit", relative, _decode_diff_text(old_bytes), _decode_diff_text(new_bytes))
        if old_content == new_content:
            result = edit_file(
                root,
                relative,
                old_content,
                new_content,
                max_file_bytes=self.policy.max_file_size,
                max_old_content_bytes=self.policy.max_content_size,
                max_new_content_bytes=self.policy.max_content_size,
                max_result_bytes=self.policy.max_content_size,
            )
            after = self.snapshot(root, relative)
            self._verify(after.exists and after.content_hash == before.content_hash, relative)
            from backend_ai.tools.modification_verification import ExpectedModification, verify_modification
            verification = verify_modification(root, [ExpectedModification.unchanged(relative, expected_sha256=before.content_hash, expected_size=before.size_bytes, before_snapshot=before)], detect_unexpected=False, max_file_bytes=self.policy.max_file_size)
            self._verify(verification.success, relative)
            return SafeEditResult("edit", relative, True, False, False, False, result.original_size_bytes, result.new_size_bytes, before.content_hash, after.content_hash, diff, None, True, verification=verification)
        backup = self._backup_if_enabled(root, relative, target, before)
        result = edit_file(
            root,
            relative,
            old_content,
            new_content,
            max_file_bytes=self.policy.max_file_size,
            max_old_content_bytes=self.policy.max_content_size,
            max_new_content_bytes=self.policy.max_content_size,
            max_result_bytes=self.policy.max_content_size,
        )
        after = self.snapshot(root, relative)
        self._verify(after.exists and after.content_hash is not None, relative)
        from backend_ai.tools.modification_verification import ExpectedModification, verify_modification
        verification = verify_modification(root, [ExpectedModification.modified(relative, expected_sha256=after.content_hash, expected_size=after.size_bytes, before_snapshot=before)], detect_unexpected=False, max_file_bytes=self.policy.max_file_size)
        self._verify(verification.success, relative)
        backup = self._finish_backup(root, backup)
        return SafeEditResult("edit", relative, True, result.changed, False, False, result.original_size_bytes, result.new_size_bytes, before.content_hash, after.content_hash, diff, backup, True, verification=verification)

    def delete(self, project_root: Path | str, path: Path | str) -> SafeEditResult:
        self._require_capability("delete")
        root, relative, target = _safe_target(project_root, path, self.policy)
        before = self.snapshot(root, relative)
        if not before.exists:
            raise ToolError(ToolErrorCode.FILE_NOT_FOUND, "Target file does not exist.", path=Path(relative))
        diff = self.diff_for_delete(root, relative) if self.policy.diff_enabled else None
        backup = self._backup_if_enabled(root, relative, target, before)
        result = delete_file(root, relative)
        after = self.snapshot(root, relative)
        self._verify(not after.exists, relative)
        from backend_ai.tools.modification_verification import ExpectedModification, verify_modification
        verification = verify_modification(root, [ExpectedModification.deleted(relative, before_snapshot=before)], detect_unexpected=False, max_file_bytes=self.policy.max_file_size)
        self._verify(verification.success, relative)
        backup = self._finish_backup(root, backup)
        return SafeEditResult("delete", relative, True, True, False, result.deleted, before.size_bytes, 0, before.content_hash, None, diff, backup, True, verification=verification)

    def _require_capability(self, operation: OperationName) -> None:
        if not getattr(self.policy, f"allow_{operation}"):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, f"Safe editing capability is disabled: {operation}.")

    def _verify(self, condition: bool, relative: str) -> None:
        if self.policy.verify_after_write and not condition:
            raise ToolError(ToolErrorCode.VERIFICATION_FAILED, f"Post-operation verification failed: {relative}", path=Path(relative))

    def _diff(self, operation: OperationName, relative: str, old_text: str, new_text: str) -> DiffResult | None:
        if not self.policy.diff_enabled:
            return None
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        lines = list(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{relative}", tofile=f"b/{relative}", lineterm=""))
        raw = "".join(lines)
        encoded = raw.encode("utf-8")
        truncated = len(lines) > self.policy.max_diff_lines or len(encoded) > self.policy.max_diff_bytes
        if truncated:
            marker = "\n[diff truncated]\n"
            marker_bytes = marker.encode("utf-8")
            if self.policy.max_diff_bytes < len(marker_bytes):
                encoded = marker_bytes[: self.policy.max_diff_bytes]
                raw = encoded.decode("ascii")
            else:
                kept: list[str] = []
                size = 0
                content_budget = self.policy.max_diff_bytes - len(marker_bytes)
                for line in lines[: self.policy.max_diff_lines]:
                    line_size = len(line.encode("utf-8"))
                    if size + line_size > content_budget:
                        break
                    kept.append(line)
                    size += line_size
                raw = "".join(kept) + marker
                encoded = raw.encode("utf-8")
        return DiffResult(operation, relative, raw, len(lines), len(encoded), truncated, True)

    def _backup_if_enabled(self, root: Path, relative: str, target: Path, before: FileSnapshot) -> BackupResult | None:
        if not self.policy.backup_enabled:
            return None
        raw = _read_bounded_bytes(target, relative, self.policy.max_file_size)
        current = self.snapshot(root, relative)
        if not before.same_identity(current):
            raise ToolError(ToolErrorCode.CONCURRENT_MODIFICATION, "Target changed before backup.", path=Path(relative))
        backup_relative = _backup_relative_path(self.policy.backup_directory, relative, before.content_hash or hashlib.sha256(raw).hexdigest())
        backup_path = _ensure_relative_path(root, backup_relative)
        _ensure_directory(backup_path.parent, root, backup_relative)
        try:
            _atomic_create_bytes(backup_path, raw)
        except FileExistsError:
            existing = _read_bounded_bytes(backup_path, Path(backup_relative).name, self.policy.max_file_size)
            if existing != raw:
                raise ToolError(ToolErrorCode.BACKUP_FAILED, "Backup path collision detected.", path=Path(backup_relative))
        except OSError as exc:
            raise ToolError(ToolErrorCode.BACKUP_FAILED, "Unable to create safe backup.", path=Path(backup_relative)) from exc
        return BackupResult(True, backup_relative, len(raw), self.policy.retain_backup_on_success)

    def _finish_backup(self, root: Path, backup: BackupResult | None) -> BackupResult | None:
        if backup is None or backup.retained:
            return backup
        backup_path = _ensure_relative_path(root, backup.relative_path or "")
        try:
            backup_path.unlink()
        except OSError as exc:
            raise ToolError(ToolErrorCode.BACKUP_FAILED, "Unable to clean temporary backup.", path=Path(backup.relative_path or "")) from exc
        return BackupResult(True, None, backup.size_bytes, False)


def _safe_target(project_root: Path | str, path: Path | str, policy: SafeEditPolicy) -> tuple[Path, str, Path]:
    if not policy.require_project_root:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "An explicit project_root is required.")
    root = _validate_root(project_root)
    relative, target = _resolve_requested_path(root, path)
    if policy.reject_symlinks:
        _reject_symlink_components(root, relative, target)
    return root, relative, target


def _file_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISBLK(mode):
        return "block_device"
    if stat.S_ISCHR(mode):
        return "character_device"
    return "special"


def _hash_file(path: Path, relative: str, max_bytes: int) -> str:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_SIZE):
                total += len(chunk)
                if total > max_bytes:
                    raise ToolError(ToolErrorCode.FILE_TOO_LARGE, "File exceeds the safe hash size limit.", path=Path(relative))
                digest.update(chunk)
    except PermissionError as exc:
        raise ToolError(ToolErrorCode.PERMISSION_DENIED, "Permission denied while hashing snapshot.", path=Path(relative)) from exc
    except OSError as exc:
        raise ToolError(ToolErrorCode.FILESYSTEM_ERROR, "Unable to hash snapshot.", path=Path(relative)) from exc
    return digest.hexdigest()


def _read_bounded_bytes(path: Path, relative: str, max_bytes: int) -> bytes:
    try:
        with path.open("rb") as stream:
            content = stream.read(max_bytes + 1)
    except FileNotFoundError as exc:
        raise ToolError(ToolErrorCode.FILE_NOT_FOUND, "File does not exist.", path=Path(relative)) from exc
    except PermissionError as exc:
        raise ToolError(ToolErrorCode.PERMISSION_DENIED, "Permission denied while reading safe-edit data.", path=Path(relative)) from exc
    except OSError as exc:
        raise ToolError(ToolErrorCode.FILESYSTEM_ERROR, "Unable to read safe-edit data.", path=Path(relative)) from exc
    if len(content) > max_bytes:
        raise ToolError(ToolErrorCode.FILE_TOO_LARGE, "Safe-edit content exceeds the configured limit.", path=Path(relative))
    return content


def _encode_bounded(content: str, max_bytes: int) -> bytes:
    if not isinstance(content, str):
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "Safe-edit content must be a string.")
    try:
        encoded = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ToolError(ToolErrorCode.INVALID_UTF8, "Safe-edit content is not valid UTF-8.") from exc
    if len(encoded) > max_bytes:
        raise ToolError(ToolErrorCode.FILE_TOO_LARGE, "Safe-edit content exceeds the configured limit.")
    return encoded


def _decode_diff_text(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def _backup_relative_path(directory: str, relative: str, content_hash: str) -> str:
    path_hash = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
    return f"{directory.rstrip('/')}/{path_hash}-{content_hash[:32]}.bak"


def _ensure_relative_path(root: Path, relative: str) -> Path:
    if not relative:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "Relative path must not be empty.")
    candidate = root / Path(relative)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "Safe-edit path is outside project root.", path=candidate) from exc
    return candidate


def _ensure_directory(path: Path, root: Path, relative: str) -> None:
    parts = path.relative_to(root).parts
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "Backup directory contains a symlink.", path=Path(relative))
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            if not current.is_dir():
                raise ToolError(ToolErrorCode.NOT_DIRECTORY, "Backup path component is not a directory.", path=Path(relative))
        except OSError as exc:
            raise ToolError(ToolErrorCode.BACKUP_FAILED, "Unable to create controlled backup directory.", path=Path(relative)) from exc


def _atomic_create_bytes(path: Path, content: bytes) -> None:
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
        temporary = Path(name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
        temporary.unlink()
        temporary = None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


__all__ = [
    "BackupResult",
    "DEFAULT_SAFE_EDIT_BACKUP_DIRECTORY",
    "DEFAULT_SAFE_EDIT_MAX_CONTENT_SIZE",
    "DEFAULT_SAFE_EDIT_MAX_DIFF_BYTES",
    "DEFAULT_SAFE_EDIT_MAX_DIFF_LINES",
    "DEFAULT_SAFE_EDIT_MAX_FILE_SIZE",
    "DiffResult",
    "FileSnapshot",
    "SafeEditPolicy",
    "SafeEditResult",
    "SafeEditSession",
]
