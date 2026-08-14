"""Conservative single-operation modification transactions and recovery."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
from typing import Any, Literal

from backend_ai.tools.base import ToolError, ToolErrorCode
from backend_ai.tools.edit_file import edit_file
from backend_ai.tools.modification_verification import (
    ExpectedModification,
    ModificationVerificationResult,
    verify_modification,
)
from backend_ai.tools.safe_editing import (
    BackupResult,
    FileSnapshot,
    SafeEditPolicy,
    SafeEditResult,
    SafeEditSession,
    _read_bounded_bytes,
    _safe_target,
)

OperationKind = Literal["create", "edit", "delete"]
TransactionStatus = Literal[
    "planned",
    "snapshotted",
    "executing",
    "verified",
    "committed",
    "failed",
    "recovery_required",
    "recovered",
    "recovery_unavailable",
]
RecoveryStatus = Literal[
    "not_required",
    "recovered",
    "recovery_required",
    "recovery_unavailable",
    "recovery_failed",
    "user_change_preserved",
]


@dataclass(frozen=True, slots=True)
class ModificationOperation:
    """Immutable operation plan and final lifecycle metadata; content is never serialized."""

    operation: OperationKind
    relative_path: str
    status: TransactionStatus = "planned"
    previous_snapshot: FileSnapshot | None = None
    actual_snapshot: FileSnapshot | None = None
    verification: ModificationVerificationResult | None = None
    recovery: "RecoveryResult | None" = None
    error: str | None = None
    _content: str | None = field(default=None, repr=False, compare=False)
    _old_content: str | None = field(default=None, repr=False, compare=False)
    _new_content: str | None = field(default=None, repr=False, compare=False)

    @classmethod
    def create(cls, path: str, content: str) -> "ModificationOperation":
        return cls("create", path, _content=content)

    @classmethod
    def edit(cls, path: str, old_content: str, new_content: str) -> "ModificationOperation":
        return cls("edit", path, _old_content=old_content, _new_content=new_content)

    @classmethod
    def delete(cls, path: str) -> "ModificationOperation":
        return cls("delete", path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "relative_path": self.relative_path,
            "status": self.status,
            "previous_snapshot": self.previous_snapshot.to_dict() if self.previous_snapshot else None,
            "actual_snapshot": self.actual_snapshot.to_dict() if self.actual_snapshot else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "recovery": self.recovery.to_dict() if self.recovery else None,
            "error": self.error,
            "has_content": self._content is not None or self._new_content is not None,
        }


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Immutable structured recovery outcome without source content."""

    status: RecoveryStatus
    attempted: bool
    succeeded: bool
    relative_path: str
    backup_used: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "relative_path": self.relative_path,
            "backup_used": self.backup_used,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ModificationTransactionResult:
    """Immutable deterministic result for one controlled mutation transaction."""

    project_root: Path
    operations: tuple[ModificationOperation, ...]
    committed_operations: tuple[str, ...]
    failed_operations: tuple[str, ...]
    recovered_operations: tuple[str, ...]
    unexpected_changes: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    status: TransactionStatus
    complete: bool
    recoverable: bool
    recovery_attempted: bool
    recovery_succeeded: bool
    verification: ModificationVerificationResult | None
    recovery: RecoveryResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "operations": [item.to_dict() for item in self.operations],
            "committed_operations": list(self.committed_operations),
            "failed_operations": list(self.failed_operations),
            "recovered_operations": list(self.recovered_operations),
            "unexpected_changes": list(self.unexpected_changes),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "status": self.status,
            "complete": self.complete,
            "recoverable": self.recoverable,
            "recovery_attempted": self.recovery_attempted,
            "recovery_succeeded": self.recovery_succeeded,
            "verification": self.verification.to_dict() if self.verification else None,
            "recovery": self.recovery.to_dict() if self.recovery else None,
        }


class ModificationTransaction:
    """Execute one safe create/edit/delete operation with conservative recovery.

    Multi-file rollback is intentionally unsupported. A transaction has exactly
    one operation so that it never claims filesystem-wide atomicity it cannot
    guarantee.
    """

    def __init__(
        self,
        project_root: Path | str,
        operation: ModificationOperation,
        *,
        policy: SafeEditPolicy | None = None,
    ) -> None:
        if not isinstance(operation, ModificationOperation):
            raise ToolError(ToolErrorCode.TRANSACTION_FAILED, "A transaction requires one ModificationOperation.")
        self.project_root = project_root
        self.operation = operation
        self.policy = policy or SafeEditPolicy.for_modification(backup_enabled=True)
        self._backup: BackupResult | None = None
        self._result: ModificationTransactionResult | None = None

    def execute(self) -> ModificationTransactionResult:
        """Run the planned operation once and return an immutable result."""

        if self._result is not None:
            return self._result
        session = SafeEditSession(self.policy)
        mutation_session = SafeEditSession(replace(self.policy, backup_enabled=False, retain_backup_on_success=False))
        try:
            root, relative, target = _safe_target(self.project_root, self.operation.relative_path, self.policy)
            if relative != self.operation.relative_path:
                self.operation = replace(self.operation, relative_path=relative)
            before = session.snapshot(root, relative)
            self.operation = replace(self.operation, status="snapshotted", previous_snapshot=before)
            self._backup = session._backup_if_enabled(root, relative, target, before) if before.exists else None
            self.operation = replace(self.operation, status="executing")
            mutation_result = self._mutate(mutation_session, root, relative)
            expected = self._expected(before, mutation_result)
            verification = verify_modification(root, [expected], detect_unexpected=False, max_file_bytes=self.policy.max_file_size)
            after = session.snapshot(root, relative)
            self.operation = replace(self.operation, status="verified", actual_snapshot=after, verification=verification)
            if not verification.success:
                raise ToolError(ToolErrorCode.VERIFICATION_FAILED, "Transaction post-mutation verification failed.", path=Path(relative))
            try:
                self._cleanup_backup(session, root, self._backup)
            except Exception as exc:
                self.operation = replace(self.operation, status="recovery_required", error="Backup cleanup failed after mutation.")
                self._result = self._result_for("recovery_required", verification, RecoveryResult("recovery_required", False, False, relative, True, "Mutation verified, but backup cleanup failed; recovery remains available."), errors=(str(exc),))
                return self._result
            self.operation = replace(self.operation, status="committed")
            self._result = self._result_for("committed", verification, RecoveryResult("not_required", False, False, relative, self._backup is not None, "Mutation verified and committed."))
            return self._result
        except Exception as exc:
            self._result = self._handle_failure(session, exc)
            return self._result

    def recover(self) -> ModificationTransactionResult:
        """Recover only a provably transaction-generated state; never force-overwrite user changes."""

        if self._result is None:
            return self.execute()
        if self._result.status != "recovery_required":
            return self._result
        if self._backup is None or not self._backup.relative_path:
            self._result = self._with_recovery(RecoveryResult("recovery_unavailable", True, False, self.operation.relative_path, False, "No controlled backup is available for safe recovery."), "recovery_unavailable")
            return self._result
        session = SafeEditSession(self.policy)
        root, relative, target = _safe_target(self.project_root, self.operation.relative_path, self.policy)
        current = session.snapshot(root, relative)
        if self.operation.operation == "edit":
            expected_after = self.operation.verification
            if expected_after is None or not expected_after.success:
                return self._with_recovery(RecoveryResult("recovery_unavailable", True, False, relative, True, "Expected transaction-generated state is unavailable."), "recovery_unavailable")
            expected_item = expected_after.verified_targets[0]
            generated_snapshot = self.operation.actual_snapshot
            if generated_snapshot is None or not current.same_identity(generated_snapshot):
                return self._with_recovery(RecoveryResult("user_change_preserved", True, False, relative, True, "Current file differs from the transaction-generated state; user changes were preserved."), "recovery_required")
            try:
                backup_bytes = _read_bounded_bytes(_backup_path(root, self._backup), relative, self.policy.max_file_size)
                if self.operation.previous_snapshot is None or self.operation.previous_snapshot.content_hash is None or self.operation.previous_snapshot.size_bytes != len(backup_bytes) or hashlib.sha256(backup_bytes).hexdigest() != self.operation.previous_snapshot.content_hash:
                    raise ToolError(ToolErrorCode.RECOVERY_FAILED, "Controlled backup identity does not match the pre-mutation snapshot.", path=Path(relative))
                backup_text = backup_bytes.decode("utf-8")
                current_bytes = _read_bounded_bytes(target, relative, self.policy.max_file_size)
                current_text = current_bytes.decode("utf-8")
                edit_file(root, relative, current_text, backup_text, max_file_bytes=self.policy.max_file_size, max_old_content_bytes=self.policy.max_content_size, max_new_content_bytes=self.policy.max_content_size, max_result_bytes=self.policy.max_content_size)
                restored = session.snapshot(root, relative)
                if not self.operation.previous_snapshot or restored.content_hash != self.operation.previous_snapshot.content_hash:
                    raise ToolError(ToolErrorCode.RECOVERY_FAILED, "Restored file does not match the pre-mutation snapshot.", path=Path(relative))
                self._cleanup_backup(session, root, self._backup)
                recovery = RecoveryResult("recovered", True, True, relative, True, "Transaction-generated edit was safely restored from the controlled backup.")
                self.operation = replace(self.operation, status="recovered", recovery=recovery, actual_snapshot=restored)
                self._result = self._with_recovery(recovery, "recovered")
                return self._result
            except Exception as exc:
                recovery = RecoveryResult("recovery_failed", True, False, relative, True, "Safe recovery failed without forcing an overwrite.")
                self._result = self._with_recovery(recovery, "recovery_required", errors=(str(exc),))
                return self._result
        recovery = RecoveryResult("recovery_unavailable", True, False, relative, True, "This operation cannot be safely restored without proving the current state is transaction-generated.")
        self._result = self._with_recovery(recovery, "recovery_unavailable")
        return self._result

    def _mutate(self, session: SafeEditSession, root: Path, relative: str) -> SafeEditResult:
        if self.operation.operation == "create":
            return session.create(root, relative, self.operation._content or "")
        if self.operation.operation == "edit":
            return session.edit(root, relative, self.operation._old_content or "", self.operation._new_content or "")
        return session.delete(root, relative)

    def _expected(self, before: FileSnapshot, mutation_result: SafeEditResult) -> ExpectedModification:
        if self.operation.operation == "create":
            return ExpectedModification.created(self.operation.relative_path, expected_sha256=mutation_result.new_hash, expected_size=mutation_result.new_size_bytes)
        if self.operation.operation == "edit":
            return ExpectedModification.modified(self.operation.relative_path, expected_sha256=mutation_result.new_hash, expected_size=mutation_result.new_size_bytes, before_snapshot=before)
        return ExpectedModification.deleted(self.operation.relative_path, before_snapshot=before)

    def _handle_failure(self, session: SafeEditSession, error: Exception) -> ModificationTransactionResult:
        relative = self.operation.relative_path
        verification = self.operation.verification
        try:
            root, _, _ = _safe_target(self.project_root, relative, self.policy)
            expected = self._expected_after_failure()
            if expected is not None:
                verification = verify_modification(root, [expected], detect_unexpected=False, max_file_bytes=self.policy.max_file_size)
            current = session.snapshot(root, relative)
            before = self.operation.previous_snapshot
            if verification is not None and verification.success:
                self.operation = replace(self.operation, status="committed", actual_snapshot=current, verification=verification, error="Mutation raised after the expected state was published; finalized without rollback.")
                self._result = self._result_for("committed", verification, RecoveryResult("not_required", False, False, relative, self._backup is not None, "Expected post-state was present; no rollback was attempted."), errors=(str(error),))
                return self._result
            if before is not None and current.same_identity(before):
                self._cleanup_backup(session, root, self._backup)
                recovery = RecoveryResult("not_required", False, False, relative, self._backup is not None, "Mutation failed before changing the target; original state preserved.")
                self.operation = replace(self.operation, status="failed", actual_snapshot=current, verification=verification, error=str(error), recovery=recovery)
                self._result = self._result_for("failed", verification, recovery, errors=(str(error),))
                return self._result
            recovery = RecoveryResult("user_change_preserved", False, False, relative, self._backup is not None, "Unexpected target identity detected; no recovery overwrite was attempted.")
            self.operation = replace(self.operation, status="recovery_required", actual_snapshot=current, verification=verification, error=str(error), recovery=recovery)
            self._result = self._result_for("recovery_required", verification, recovery, errors=(str(error), "Concurrent or unexpected user change was preserved."))
            return self._result
        except Exception as secondary:
            recovery = RecoveryResult("recovery_unavailable", False, False, relative, self._backup is not None, "Failure state could not be safely inspected; no recovery was attempted.")
            self.operation = replace(self.operation, status="recovery_unavailable", error=str(error), recovery=recovery)
            self._result = self._result_for("recovery_unavailable", verification, recovery, errors=(str(error), str(secondary)))
            return self._result

    def _expected_after_failure(self) -> ExpectedModification | None:
        if self.operation.operation == "create":
            content = self.operation._content or ""
            return ExpectedModification.created(self.operation.relative_path, expected_content=content)
        if self.operation.operation == "edit":
            return ExpectedModification.modified(self.operation.relative_path, expected_content=self.operation._new_content or "", before_snapshot=self.operation.previous_snapshot)
        return ExpectedModification.deleted(self.operation.relative_path, before_snapshot=self.operation.previous_snapshot)

    def _cleanup_backup(self, session: SafeEditSession, root: Path, backup: BackupResult | None) -> None:
        if backup is not None and not self.policy.retain_backup_on_success:
            session._finish_backup(root, backup)

    def _result_for(self, status: TransactionStatus, verification: ModificationVerificationResult | None, recovery: RecoveryResult, *, errors: tuple[str, ...] = ()) -> ModificationTransactionResult:
        path = self.operation.relative_path
        committed = (path,) if status == "committed" else ()
        recovered = (path,) if status == "recovered" else ()
        failed = () if status in {"committed", "recovered"} else (path,)
        unexpected = tuple(item.relative_path for item in verification.unexpected_changes) if verification else ()
        warnings = ("This transaction supports one filesystem mutation only; multi-file rollback is intentionally unavailable.",)
        return ModificationTransactionResult(self._root(), (self.operation,), committed, failed, recovered, unexpected, warnings, errors, status, True, self._backup is not None, recovery.attempted, recovery.succeeded, verification, recovery)

    def _with_recovery(self, recovery: RecoveryResult, status: TransactionStatus, *, errors: tuple[str, ...] = ()) -> ModificationTransactionResult:
        verification = self._result.verification if self._result else self.operation.verification
        return self._result_for(status, verification, recovery, errors=errors)

    def _root(self) -> Path:
        return Path(self.project_root).expanduser().resolve(strict=False)


def _backup_path(root: Path, backup: BackupResult) -> Path:
    if not backup.relative_path:
        raise ToolError(ToolErrorCode.RECOVERY_UNAVAILABLE, "Controlled backup has no path.")
    path = root / Path(backup.relative_path)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "Controlled backup path escaped project root.") from exc
    return path


__all__ = [
    "ModificationOperation",
    "ModificationTransaction",
    "ModificationTransactionResult",
    "RecoveryResult",
]
