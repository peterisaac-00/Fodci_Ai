from __future__ import annotations

from pathlib import Path

import pytest

from backend_ai.tools import (
    ModificationOperation,
    ModificationTransaction,
    SafeEditPolicy,
    ToolError,
    ToolErrorCode,
)
import backend_ai.tools.modification_transaction as transaction_module


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def _policy() -> SafeEditPolicy:
    return SafeEditPolicy.for_modification(backup_enabled=True, retain_backup_on_success=False)


def _backup_files(root: Path) -> list[Path]:
    return sorted((root / ".fodci" / "backups").glob("*.bak")) if (root / ".fodci" / "backups").exists() else []


def test_create_edit_delete_transactions_are_committed_and_structured(tmp_path: Path) -> None:
    root = _project(tmp_path)

    created = ModificationTransaction(root, ModificationOperation.create("src/app.py", "مرحبا\n"), policy=_policy()).execute()
    assert created.status == "committed"
    assert created.committed_operations == ("src/app.py",)
    assert created.operations[0].status == "committed"
    assert created.recovery is not None and created.recovery.status == "not_required"
    assert (root / "src" / "app.py").read_text(encoding="utf-8") == "مرحبا\n"

    edited = ModificationTransaction(root, ModificationOperation.edit("src/app.py", "مرحبا", "hello"), policy=_policy()).execute()
    assert edited.status == "committed"
    assert edited.operations[0].verification is not None
    assert edited.operations[0].verification.success is True
    assert (root / "src" / "app.py").read_text(encoding="utf-8") == "hello\n"

    deleted = ModificationTransaction(root, ModificationOperation.delete("src/app.py"), policy=_policy()).execute()
    assert deleted.status == "committed"
    assert deleted.committed_operations == ("src/app.py",)
    assert not (root / "src" / "app.py").exists()
    assert _backup_files(root) == []
    assert "hello" not in str(deleted.to_dict())


def test_transaction_does_not_claim_multi_file_atomicity(tmp_path: Path) -> None:
    root = _project(tmp_path)
    operation = ModificationOperation.create("one.txt", "one")
    result = ModificationTransaction(root, operation, policy=_policy()).execute()

    assert len(result.operations) == 1
    assert result.warnings
    assert "one.txt" in result.committed_operations


def test_failure_before_publish_preserves_original_and_cleans_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("original", encoding="utf-8")
    original_edit = transaction_module.SafeEditSession.edit

    def fail_edit(self, project_root, path, old_content, new_content):
        raise ToolError(ToolErrorCode.ATOMIC_PUBLISH_FAILED, "simulated publish failure")

    monkeypatch.setattr(transaction_module.SafeEditSession, "edit", fail_edit)
    result = ModificationTransaction(root, ModificationOperation.edit("app.py", "original", "changed"), policy=_policy()).execute()

    assert result.status == "failed"
    assert result.recovery is not None and result.recovery.status == "not_required"
    assert target.read_text(encoding="utf-8") == "original"
    assert _backup_files(root) == []
    monkeypatch.setattr(transaction_module.SafeEditSession, "edit", original_edit)


def test_cleanup_failure_can_recover_edit_from_controlled_backup_without_overwrite(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("original", encoding="utf-8")
    transaction = ModificationTransaction(root, ModificationOperation.edit("app.py", "original", "changed"), policy=_policy())
    original_cleanup = transaction._cleanup_backup

    def fail_cleanup(*args, **kwargs):
        raise OSError("simulated cleanup failure")

    transaction._cleanup_backup = fail_cleanup  # type: ignore[method-assign]
    failed = transaction.execute()
    assert failed.status == "recovery_required"
    assert target.read_text(encoding="utf-8") == "changed"
    assert _backup_files(root)

    transaction._cleanup_backup = original_cleanup  # type: ignore[method-assign]
    recovered = transaction.recover()
    assert recovered.status == "recovered"
    assert recovered.recovery_succeeded is True
    assert target.read_text(encoding="utf-8") == "original"
    assert _backup_files(root) == []


def test_recovery_preserves_user_change_after_generated_state_is_lost(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("original", encoding="utf-8")
    transaction = ModificationTransaction(root, ModificationOperation.edit("app.py", "original", "changed"), policy=_policy())
    original_cleanup = transaction._cleanup_backup

    def fail_cleanup(*args, **kwargs):
        raise OSError("simulated cleanup failure")

    transaction._cleanup_backup = fail_cleanup  # type: ignore[method-assign]
    failed = transaction.execute()
    assert failed.status == "recovery_required"
    target.write_text("user change", encoding="utf-8")
    transaction._cleanup_backup = original_cleanup  # type: ignore[method-assign]

    preserved = transaction.recover()
    assert preserved.status == "recovery_required"
    assert preserved.recovery is not None
    assert preserved.recovery.status == "user_change_preserved"
    assert target.read_text(encoding="utf-8") == "user change"
    assert _backup_files(root)


def test_corrupt_backup_reports_recovery_failed_without_forced_restore(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("original", encoding="utf-8")
    transaction = ModificationTransaction(root, ModificationOperation.edit("app.py", "original", "changed"), policy=_policy())
    original_cleanup = transaction._cleanup_backup

    def fail_cleanup(*args, **kwargs):
        raise OSError("simulated cleanup failure")

    transaction._cleanup_backup = fail_cleanup  # type: ignore[method-assign]
    failed = transaction.execute()
    assert failed.status == "recovery_required"
    backup = _backup_files(root)[0]
    backup.write_bytes(b"corrupt-valid")
    transaction._cleanup_backup = original_cleanup  # type: ignore[method-assign]

    result = transaction.recover()
    assert result.status == "recovery_required"
    assert result.recovery is not None and result.recovery.status == "recovery_failed"
    assert target.read_text(encoding="utf-8") == "changed"


def test_delete_cleanup_failure_is_recovery_unavailable_and_never_restores_user_file(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("original", encoding="utf-8")
    transaction = ModificationTransaction(root, ModificationOperation.delete("app.py"), policy=_policy())
    transaction._cleanup_backup = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup"))  # type: ignore[method-assign]

    failed = transaction.execute()
    assert failed.status == "recovery_required"
    assert not target.exists()
    result = transaction.recover()
    assert result.status == "recovery_unavailable"
    assert not target.exists()


def test_invalid_path_and_symlink_parent_are_rejected_without_mutation(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    for path in ("../outside", str(outside), r"C:\outside", r"\\server\share\outside"):
        result = ModificationTransaction(root, ModificationOperation.create(path, "x"), policy=_policy()).execute()
        assert result.status in {"failed", "recovery_unavailable"}
        assert result.errors
    link = root / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    result = ModificationTransaction(root, ModificationOperation.create("link/file.txt", "x"), policy=_policy()).execute()
    assert result.status in {"failed", "recovery_unavailable"}
    assert result.errors
    assert outside.read_text(encoding="utf-8") == "outside"


def test_result_serialization_is_deterministic_and_excludes_operation_content(tmp_path: Path) -> None:
    root = _project(tmp_path)
    operation = ModificationOperation.create("file.txt", "secret-content")
    first = ModificationTransaction(root, operation, policy=_policy()).execute().to_dict()
    second = ModificationTransaction(root, ModificationOperation.delete("file.txt"), policy=_policy()).execute().to_dict()

    assert first == first.copy()
    assert "secret-content" not in str(first)
    assert second["status"] == "committed"
