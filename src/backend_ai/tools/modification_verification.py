"""Read-only verification of explicit filesystem mutations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import stat
from typing import Any, Iterable, Literal, Mapping

from backend_ai.tools.base import ToolError, ToolErrorCode
from backend_ai.tools.filesystem import DEFAULT_MAX_DEPTH, DEFAULT_MAX_DIRECTORIES, DEFAULT_MAX_FILES, _validate_root, list_files
from backend_ai.tools.read_file import _resolve_requested_path
from backend_ai.tools.safe_editing import FileSnapshot

ExpectedState = Literal["created", "modified", "deleted", "unchanged"]
VerificationStatus = Literal[
    "VERIFIED",
    "MISSING",
    "UNEXPECTED_MODIFICATION",
    "UNEXPECTED_CREATION",
    "UNEXPECTED_DELETION",
    "TYPE_CHANGED",
    "CONTENT_MISMATCH",
    "HASH_MISMATCH",
    "VERIFICATION_ERROR",
    "VERIFICATION_UNAVAILABLE",
]

ACTUAL_MISSING = "missing"
ACTUAL_REGULAR = "present_regular_file"
ACTUAL_SYMLINK = "symlink"
ACTUAL_DIRECTORY = "directory"
ACTUAL_SPECIAL = "special_file"
ACTUAL_UNREADABLE = "unreadable"
ACTUAL_INVALID_UTF8 = "invalid_utf8"

DEFAULT_MAX_VERIFICATION_FILE_BYTES = 1_048_576
DEFAULT_MAX_VERIFICATION_FILES = DEFAULT_MAX_FILES


@dataclass(frozen=True, slots=True)
class ExpectedModification:
    """Explicit expected post-state; content is held privately and never serialized."""

    relative_path: str
    expected_state: ExpectedState
    expected_size: int | None = None
    expected_sha256: str | None = None
    expected_content: str | None = None
    before_snapshot: FileSnapshot | None = None

    def __post_init__(self) -> None:
        if not self.relative_path or not isinstance(self.relative_path, str):
            raise ValueError("relative_path must be non-empty text")
        if self.expected_state not in {"created", "modified", "deleted", "unchanged"}:
            raise ValueError("unsupported expected_state")
        if self.expected_size is not None and (not isinstance(self.expected_size, int) or self.expected_size < 0):
            raise ValueError("expected_size must be a non-negative integer")
        if self.expected_sha256 is not None and (
            not isinstance(self.expected_sha256, str)
            or len(self.expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 hex digest")
        if self.expected_content is not None and not isinstance(self.expected_content, str):
            raise ValueError("expected_content must be text")
        if self.expected_content is not None:
            encoded = self.expected_content.encode("utf-8")
            if self.expected_size is not None and self.expected_size != len(encoded):
                raise ValueError("expected_size does not match expected_content")
            object.__setattr__(self, "expected_size", len(encoded))
            if self.expected_sha256 is None:
                object.__setattr__(self, "expected_sha256", hashlib.sha256(encoded).hexdigest())

    @classmethod
    def created(cls, path: str, *, expected_sha256: str | None = None, expected_size: int | None = None, expected_content: str | None = None) -> "ExpectedModification":
        return cls(path, "created", expected_size, expected_sha256, expected_content)

    @classmethod
    def modified(cls, path: str, *, expected_sha256: str | None = None, expected_size: int | None = None, expected_content: str | None = None, before_snapshot: FileSnapshot | None = None) -> "ExpectedModification":
        return cls(path, "modified", expected_size, expected_sha256, expected_content, before_snapshot)

    @classmethod
    def deleted(cls, path: str, *, before_snapshot: FileSnapshot | None = None) -> "ExpectedModification":
        return cls(path, "deleted", before_snapshot=before_snapshot)

    @classmethod
    def unchanged(cls, path: str, *, expected_sha256: str | None = None, expected_size: int | None = None, expected_content: str | None = None, before_snapshot: FileSnapshot | None = None) -> "ExpectedModification":
        return cls(path, "unchanged", expected_size, expected_sha256, expected_content, before_snapshot)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "expected_state": self.expected_state,
            "expected_size": self.expected_size,
            "expected_sha256": self.expected_sha256,
            "has_expected_content": self.expected_content is not None,
            "before_snapshot": self.before_snapshot.to_dict() if self.before_snapshot else None,
        }


@dataclass(frozen=True, slots=True)
class ModificationVerificationItem:
    """Immutable per-target verification record without file contents."""

    relative_path: str
    expected_state: ExpectedState
    actual_state: str
    status: VerificationStatus
    expected_size: int | None
    actual_size: int | None
    expected_sha256: str | None
    actual_sha256: str | None
    file_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "expected_state": self.expected_state,
            "actual_state": self.actual_state,
            "status": self.status,
            "expected_size": self.expected_size,
            "actual_size": self.actual_size,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "file_type": self.file_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ModificationVerificationResult:
    """Immutable deterministic verification result."""

    success: bool
    project_root: Path
    operation: str
    verified_targets: tuple[ModificationVerificationItem, ...]
    unexpected_changes: tuple[ModificationVerificationItem, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    complete: bool
    truncated: bool
    truncation_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "project_root": str(self.project_root),
            "operation": self.operation,
            "verified_targets": [item.to_dict() for item in self.verified_targets],
            "unexpected_changes": [item.to_dict() for item in self.unexpected_changes],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "complete": self.complete,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }


class ModificationVerifier:
    """Strict read-only verifier for explicit expected filesystem states."""

    def __init__(
        self,
        *,
        max_file_bytes: int = DEFAULT_MAX_VERIFICATION_FILE_BYTES,
        max_files: int = DEFAULT_MAX_VERIFICATION_FILES,
        max_directories: int = DEFAULT_MAX_DIRECTORIES,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        for name, value in (("max_file_bytes", max_file_bytes), ("max_files", max_files), ("max_directories", max_directories), ("max_depth", max_depth)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.max_directories = max_directories
        self.max_depth = max_depth

    def verify(
        self,
        project_root: Path | str,
        expected_changes: Iterable[ExpectedModification],
        *,
        baseline: Mapping[str, FileSnapshot] | None = None,
        detect_unexpected: bool = True,
    ) -> ModificationVerificationResult:
        """Verify explicit targets and optionally compare a bounded project baseline."""

        root = _validate_root(project_root)
        expected = tuple(expected_changes)
        if not expected:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "expected_changes must not be empty.")
        normalized: list[ExpectedModification] = []
        for item in expected:
            if not isinstance(item, ExpectedModification):
                raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "expected_changes must contain ExpectedModification records.")
            relative, _ = _safe_verification_path(root, item.relative_path)
            normalized.append(ExpectedModification(relative, item.expected_state, item.expected_size, item.expected_sha256, item.expected_content, item.before_snapshot))
        normalized.sort(key=lambda item: (item.relative_path.casefold(), item.relative_path))
        target_paths = {item.relative_path for item in normalized}
        items = tuple(self._verify_item(root, item) for item in normalized)
        unexpected: tuple[ModificationVerificationItem, ...] = ()
        warnings: list[str] = []
        errors = [item.message for item in items if item.status in {"VERIFICATION_ERROR", "VERIFICATION_UNAVAILABLE"}]
        complete = True
        truncated = False
        reasons: list[str] = []
        if detect_unexpected and baseline is not None:
            unexpected, baseline_warnings, baseline_complete, baseline_truncated, baseline_reason = self._compare_baseline(root, baseline, target_paths)
            warnings.extend(baseline_warnings)
            complete = complete and baseline_complete
            truncated = truncated or baseline_truncated
            if baseline_reason:
                reasons.append(baseline_reason)
        elif detect_unexpected and baseline is None:
            warnings.append("No baseline supplied; verification covers explicit targets only.")
            complete = False
        success = complete and not unexpected and not errors and all(item.status == "VERIFIED" for item in items)
        return ModificationVerificationResult(
            success=success,
            project_root=root,
            operation="verify",
            verified_targets=items,
            unexpected_changes=unexpected,
            warnings=tuple(warnings),
            errors=tuple(errors),
            complete=complete,
            truncated=truncated,
            truncation_reason=";".join(dict.fromkeys(reasons)) if reasons else None,
        )

    def _verify_item(self, root: Path, expected: ExpectedModification) -> ModificationVerificationItem:
        actual = _inspect_entry(root, expected.relative_path, self.max_file_bytes)
        expected_hash = expected.expected_sha256
        expected_size = expected.expected_size
        if expected.expected_state == "deleted":
            if actual.actual_state == ACTUAL_MISSING:
                return _item(expected, actual, "VERIFIED", "Target is absent as expected.")
            if actual.actual_state in {ACTUAL_SYMLINK, ACTUAL_DIRECTORY, ACTUAL_SPECIAL}:
                return _item(expected, actual, "TYPE_CHANGED", "Deleted target path is occupied by an unexpected filesystem type.")
            return _item(expected, actual, "UNEXPECTED_CREATION", "Target still exists after expected deletion.")
        if actual.actual_state == ACTUAL_MISSING:
            status: VerificationStatus = "MISSING" if expected.expected_state == "created" else "UNEXPECTED_DELETION"
            return _item(expected, actual, status, "Expected target is missing.")
        if actual.actual_state in {ACTUAL_INVALID_UTF8, ACTUAL_UNREADABLE}:
            status: VerificationStatus = "VERIFICATION_ERROR" if actual.actual_state == ACTUAL_INVALID_UTF8 else "VERIFICATION_UNAVAILABLE"
            return _item(expected, actual, status, actual.error or "Target cannot be verified safely.")
        if actual.actual_state != ACTUAL_REGULAR:
            return _item(expected, actual, "TYPE_CHANGED", "Expected a regular file but found another filesystem type.")
        if actual.error:
            return _item(expected, actual, "VERIFICATION_UNAVAILABLE", actual.error)
        if expected.expected_content is not None and actual.decoded_content is None:
            return _item(expected, actual, "VERIFICATION_ERROR", "Actual file is not valid UTF-8 under strict verification.")
        if expected.expected_content is not None and actual.decoded_content != expected.expected_content:
            return _item(expected, actual, "CONTENT_MISMATCH", "Actual UTF-8 content does not match expected content.")
        if expected_size is not None and actual.size_bytes != expected_size:
            return _item(expected, actual, "HASH_MISMATCH", "Actual byte size does not match expected size.")
        if expected_hash is not None and actual.sha256 != expected_hash:
            return _item(expected, actual, "HASH_MISMATCH", "Actual SHA-256 does not match expected hash.")
        if expected.before_snapshot is not None and expected.before_snapshot.exists and expected.before_snapshot.content_hash == actual.sha256 and expected.before_snapshot.size_bytes == actual.size_bytes and expected.expected_state in {"modified", "unchanged"}:
            if expected.expected_state == "modified":
                return _item(expected, actual, "UNEXPECTED_MODIFICATION", "Expected an edit, but the target still matches its pre-mutation snapshot.")
        if expected.expected_state == "created" and expected.before_snapshot is not None and expected.before_snapshot.exists:
            return _item(expected, actual, "UNEXPECTED_MODIFICATION", "Creation expectation had an existing pre-mutation snapshot.")
        if expected.expected_state == "modified" and expected.before_snapshot is not None and not expected.before_snapshot.exists:
            return _item(expected, actual, "UNEXPECTED_CREATION", "Modification expectation had a missing pre-mutation snapshot.")
        if expected.expected_state == "unchanged" and expected.before_snapshot is not None:
            if expected.before_snapshot.content_hash != actual.sha256 or expected.before_snapshot.size_bytes != actual.size_bytes:
                return _item(expected, actual, "UNEXPECTED_MODIFICATION", "Unchanged expectation differs from its pre-mutation snapshot.")
        return _item(expected, actual, "VERIFIED", "Expected regular-file state verified.")

    def _compare_baseline(
        self,
        root: Path,
        baseline: Mapping[str, FileSnapshot],
        target_paths: set[str],
    ) -> tuple[tuple[ModificationVerificationItem, ...], list[str], bool, bool, str | None]:
        unexpected: list[ModificationVerificationItem] = []
        warnings: list[str] = []
        current_listing = list_files(root, max_files=self.max_files, max_directories=self.max_directories, max_depth=self.max_depth, include_hidden=True)
        complete = not current_listing.truncated
        truncated = current_listing.truncated
        reason = current_listing.truncation_reason
        current_paths = {item.relative_path for item in current_listing.files}
        baseline_paths = set(baseline)
        for relative in sorted(baseline_paths | current_paths, key=lambda value: (value.casefold(), value)):
            if relative in target_paths:
                continue
            expected_snapshot = baseline.get(relative)
            actual_snapshot = _safe_snapshot_for_baseline(root, relative, self.max_file_bytes)
            if expected_snapshot is None and relative in current_paths:
                unexpected.append(_baseline_item(relative, "unchanged", actual_snapshot, "UNEXPECTED_CREATION", "File was created outside the intended targets."))
            elif expected_snapshot is not None and not actual_snapshot.exists:
                unexpected.append(_baseline_item(relative, "unchanged", actual_snapshot, "UNEXPECTED_DELETION", "Baseline file was deleted outside the intended targets."))
            elif expected_snapshot is not None and not expected_snapshot.same_identity(actual_snapshot):
                unexpected.append(_baseline_item(relative, "unchanged", actual_snapshot, "UNEXPECTED_MODIFICATION", "Baseline file changed outside the intended targets."))
        if baseline_paths - current_paths:
            warnings.append("Baseline comparison could not enumerate every filesystem entry.")
        return tuple(unexpected), warnings, complete, truncated, reason


@dataclass(frozen=True, slots=True)
class _ActualEntry:
    actual_state: str
    size_bytes: int | None
    sha256: str | None
    file_type: str
    error: str | None
    decoded_content: str | None


def verify_modification(
    project_root: Path | str,
    expected_changes: Iterable[ExpectedModification],
    *,
    baseline: Mapping[str, FileSnapshot] | None = None,
    detect_unexpected: bool = True,
    max_file_bytes: int = DEFAULT_MAX_VERIFICATION_FILE_BYTES,
    max_files: int = DEFAULT_MAX_VERIFICATION_FILES,
    max_directories: int = DEFAULT_MAX_DIRECTORIES,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> ModificationVerificationResult:
    """Verify explicit expected changes without mutating or exposing contents."""

    return ModificationVerifier(
        max_file_bytes=max_file_bytes,
        max_files=max_files,
        max_directories=max_directories,
        max_depth=max_depth,
    ).verify(project_root, expected_changes, baseline=baseline, detect_unexpected=detect_unexpected)


def _safe_verification_path(root: Path, requested: str | Path) -> tuple[str, Path]:
    relative, lexical = _resolve_requested_path(root, requested)
    current = root
    parts = Path(relative).parts
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "Verification path contains a symlink parent.", path=Path(relative))
    return relative, lexical


def _inspect_entry(root: Path, relative: str, max_file_bytes: int) -> _ActualEntry:
    _, lexical = _safe_verification_path(root, relative)
    try:
        metadata = lexical.lstat()
    except FileNotFoundError:
        return _ActualEntry(ACTUAL_MISSING, None, None, "missing", None, None)
    except PermissionError:
        return _ActualEntry(ACTUAL_UNREADABLE, None, None, "unknown", "Permission denied while inspecting target.", None)
    except OSError as exc:
        return _ActualEntry(ACTUAL_UNREADABLE, None, None, "unknown", "Unable to inspect target.", None)
    file_type = _file_type(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        return _ActualEntry(ACTUAL_SYMLINK, metadata.st_size, None, file_type, None, None)
    if stat.S_ISDIR(metadata.st_mode):
        return _ActualEntry(ACTUAL_DIRECTORY, metadata.st_size, None, file_type, None, None)
    if not stat.S_ISREG(metadata.st_mode):
        return _ActualEntry(ACTUAL_SPECIAL, metadata.st_size, None, file_type, None, None)
    if metadata.st_size > max_file_bytes:
        return _ActualEntry(ACTUAL_UNREADABLE, metadata.st_size, None, file_type, "Target exceeds verification size limit.", None)
    try:
        with lexical.open("rb") as stream:
            raw = stream.read(max_file_bytes + 1)
    except PermissionError:
        return _ActualEntry(ACTUAL_UNREADABLE, metadata.st_size, None, file_type, "Permission denied while reading target.", None)
    except OSError:
        return _ActualEntry(ACTUAL_UNREADABLE, metadata.st_size, None, file_type, "Unable to read target for verification.", None)
    if len(raw) > max_file_bytes:
        return _ActualEntry(ACTUAL_UNREADABLE, len(raw), None, file_type, "Target exceeded verification size limit while reading.", None)
    digest = hashlib.sha256(raw).hexdigest()
    decoded: str | None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _ActualEntry(ACTUAL_INVALID_UTF8, len(raw), digest, file_type, "Target is not valid UTF-8 under strict verification.", None)
    return _ActualEntry(ACTUAL_REGULAR, len(raw), digest, file_type, None, decoded)


def _safe_snapshot_for_baseline(root: Path, relative: str, max_file_bytes: int) -> FileSnapshot:
    actual = _inspect_entry(root, relative, max_file_bytes)
    _, lexical = _safe_verification_path(root, relative)
    try:
        metadata = lexical.lstat()
    except FileNotFoundError:
        return FileSnapshot(relative, False, 0, None, None, None, None, "missing", None)
    return FileSnapshot(relative, actual.actual_state != ACTUAL_MISSING, actual.size_bytes or 0, metadata.st_mtime_ns, metadata.st_dev, metadata.st_ino, actual.sha256, actual.file_type, stat.S_IMODE(metadata.st_mode))


def _item(expected: ExpectedModification, actual: _ActualEntry, status: VerificationStatus, message: str) -> ModificationVerificationItem:
    return ModificationVerificationItem(expected.relative_path, expected.expected_state, actual.actual_state, status, expected.expected_size, actual.size_bytes, expected.expected_sha256, actual.sha256, actual.file_type, message)


def _baseline_item(relative: str, expected_state: ExpectedState, actual: FileSnapshot, status: VerificationStatus, message: str) -> ModificationVerificationItem:
    return ModificationVerificationItem(relative, expected_state, "missing" if not actual.exists else "present_regular_file", status, None, actual.size_bytes if actual.exists else None, None, actual.content_hash, actual.file_type, message)


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


__all__ = [
    "ACTUAL_DIRECTORY",
    "ACTUAL_INVALID_UTF8",
    "ACTUAL_MISSING",
    "ACTUAL_REGULAR",
    "ACTUAL_SPECIAL",
    "ACTUAL_SYMLINK",
    "ACTUAL_UNREADABLE",
    "DEFAULT_MAX_VERIFICATION_FILE_BYTES",
    "DEFAULT_MAX_VERIFICATION_FILES",
    "ExpectedModification",
    "ModificationVerificationItem",
    "ModificationVerificationResult",
    "ModificationVerifier",
    "verify_modification",
]
