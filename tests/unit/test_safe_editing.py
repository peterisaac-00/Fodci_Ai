from __future__ import annotations

from pathlib import Path

import pytest

from backend_ai.agent import AgentLoop, ToolRegistry
from backend_ai.tools import (
    BackupResult,
    DiffResult,
    FileSnapshot,
    SafeEditPolicy,
    SafeEditResult,
    SafeEditSession,
    ToolError,
    ToolErrorCode,
    read_file,
)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def _error(callable_obj, code: ToolErrorCode, *args, **kwargs) -> ToolError:
    with pytest.raises(ToolError) as raised:
        callable_obj(*args, **kwargs)
    assert raised.value.code is code
    return raised.value


def _modification_session(**kwargs) -> SafeEditSession:
    return SafeEditSession(SafeEditPolicy.for_modification(**kwargs))


def test_policy_defaults_are_conservative_and_cannot_weaken_required_safety() -> None:
    policy = SafeEditPolicy()
    assert policy.require_project_root is True
    assert policy.allow_create is False
    assert policy.allow_edit is False
    assert policy.allow_delete is False
    assert policy.backup_enabled is False
    assert policy.diff_enabled is True
    assert policy.reject_symlinks is True
    assert policy.atomic_write is True
    assert policy.verify_after_write is True
    with pytest.raises(ValueError):
        SafeEditPolicy(require_project_root=False)
    with pytest.raises(ValueError):
        SafeEditPolicy(reject_symlinks=False)
    with pytest.raises(ValueError):
        SafeEditPolicy(backup_directory="../outside")


def test_snapshot_is_immutable_bounded_and_hashes_deterministically(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("print('ok')\n", encoding="utf-8")
    session = SafeEditSession(SafeEditPolicy(max_file_size=100))

    first = session.snapshot(root, "app.py")
    second = session.snapshot(root, "app.py")

    assert isinstance(first, FileSnapshot)
    assert first == second
    assert first.exists is True
    assert first.file_type == "regular"
    assert first.size_bytes == len(b"print('ok')\n")
    assert first.content_hash is not None
    assert len(first.content_hash) == 64
    assert first.to_dict()["relative_path"] == "app.py"
    with pytest.raises(Exception):
        first.size_bytes = 99  # type: ignore[misc]

    missing = session.snapshot(root, "missing.py")
    assert missing.exists is False
    assert missing.file_type == "missing"
    assert missing.content_hash is None


def test_snapshot_detects_content_and_metadata_changes(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("old", encoding="utf-8")
    session = SafeEditSession(SafeEditPolicy())
    before = session.snapshot(root, "app.py")
    target.write_text("new", encoding="utf-8")
    after = session.snapshot(root, "app.py")

    assert not before.same_identity(after)
    assert before.content_hash != after.content_hash


def test_diff_generation_is_deterministic_relative_and_bounded(tmp_path: Path) -> None:
    root = _project(tmp_path)
    session = _modification_session(max_diff_bytes=10_000, max_diff_lines=100)

    created = session.diff_for_create(root, "app.py", "print('ok')\n")
    edited = session.diff_for_edit(root, "app.py", "old\n", "new\n")
    assert isinstance(created, DiffResult)
    assert isinstance(edited, DiffResult)
    assert created.operation == "create"
    assert edited.operation == "edit"
    assert "--- a/app.py" in created.text
    assert "+++ b/app.py" in created.text
    assert "-old" in edited.text
    assert "+new" in edited.text
    assert str(root) not in created.text
    assert created == session.diff_for_create(root, "app.py", "print('ok')\n")

    limited = SafeEditSession(SafeEditPolicy.for_modification(max_diff_bytes=20, max_diff_lines=2))
    diff = limited.diff_for_edit(root, "app.py", "old\n" * 20, "new\n" * 20)
    assert diff is not None
    assert diff.truncated is True
    assert "[diff truncated]" in diff.text
    assert diff.size_bytes <= 20 + len(b"\n[diff truncated]\n")


def test_diff_can_be_disabled_without_affecting_mutation_capabilities(tmp_path: Path) -> None:
    root = _project(tmp_path)
    session = _modification_session(diff_enabled=False)
    assert session.diff_for_create(root, "app.py", "x") is None
    assert session.diff_for_edit(root, "app.py", "old", "new") is None


def test_create_wrapper_verifies_and_returns_consistent_result(tmp_path: Path) -> None:
    root = _project(tmp_path)
    session = _modification_session()

    result = session.create(root, "src/app.py", "print('ok')\n")

    assert isinstance(result, SafeEditResult)
    assert result.operation == "create"
    assert result.success is True
    assert result.created is True
    assert result.changed is True
    assert result.deleted is False
    assert result.old_size_bytes == 0
    assert result.new_size_bytes == len(b"print('ok')\n")
    assert result.verification_passed is True
    assert result.diff is not None
    assert read_file(root, "src/app.py").content == "print('ok')\n"


def test_create_wrapper_requires_opt_in_and_preserves_existing_create_semantics(tmp_path: Path) -> None:
    root = _project(tmp_path)
    session = SafeEditSession(SafeEditPolicy())

    _error(session.create, ToolErrorCode.INVALID_ARGUMENT, root, "new.py", "x")
    with pytest.raises(ToolError) as raised:
        session.snapshot(root, "../outside.py")
    assert raised.value.code is ToolErrorCode.PATH_OUTSIDE_ROOT


def test_noop_edit_does_not_create_backup(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("value = 'old'\n", encoding="utf-8")
    session = _modification_session(backup_enabled=True, retain_backup_on_success=True)

    result = session.edit(root, "app.py", "'old'", "'old'")

    assert result.changed is False
    assert result.backup is None
    assert not list(root.glob(".fodci/backups/*.bak"))
    assert target.read_text(encoding="utf-8") == "value = 'old'\n"


def test_edit_wrapper_creates_diff_verifies_and_cleans_nonretained_backup(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("value = 'old'\n", encoding="utf-8")
    session = _modification_session(backup_enabled=True, retain_backup_on_success=False)

    result = session.edit(root, "app.py", "'old'", "'new'")

    assert result.operation == "edit"
    assert result.success is True
    assert result.changed is True
    assert result.backup is not None
    assert result.backup.created is True
    assert result.backup.retained is False
    assert result.backup.relative_path is None
    assert result.diff is not None
    assert result.verification_passed is True
    assert target.read_text(encoding="utf-8") == "value = 'new'\n"
    assert not list(root.glob(".fodci/backups/*.bak"))


def test_edit_wrapper_can_retain_backup_inside_root_and_preserves_original(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    original = "value = 'old'\n"
    target.write_text(original, encoding="utf-8")
    session = _modification_session(backup_enabled=True, retain_backup_on_success=True)

    result = session.edit(root, "app.py", "'old'", "'new'")

    assert result.backup == BackupResult(True, result.backup.relative_path, len(original.encode("utf-8")), True)
    backup_path = root / (result.backup.relative_path or "")
    assert backup_path.is_file()
    assert backup_path.relative_to(root)
    assert backup_path.read_text(encoding="utf-8") == original


def test_edit_wrapper_rejects_conservative_size_limits_before_mutation(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("old", encoding="utf-8")
    session = _modification_session(max_file_size=2)
    before = target.read_bytes()

    _error(session.edit, ToolErrorCode.FILE_TOO_LARGE, root, "app.py", "old", "new")
    assert target.read_bytes() == before


def test_delete_wrapper_verifies_absence_and_returns_diff(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "tests" / "test_app.py"
    target.parent.mkdir()
    target.write_text("assert True\n", encoding="utf-8")
    unrelated = root / "app.py"
    unrelated.write_text("keep", encoding="utf-8")
    session = _modification_session()

    result = session.delete(root, "tests/test_app.py")

    assert result.operation == "delete"
    assert result.deleted is True
    assert result.changed is True
    assert result.old_size_bytes == len(b"assert True\n")
    assert result.new_size_bytes == 0
    assert result.old_hash is not None
    assert result.new_hash is None
    assert result.diff is not None
    assert result.verification_passed is True
    assert not target.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_delete_wrapper_backup_is_retained_on_success_when_requested(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "delete.txt"
    target.write_text("recoverable", encoding="utf-8")
    session = _modification_session(backup_enabled=True, retain_backup_on_success=True)

    result = session.delete(root, "delete.txt")

    assert result.backup is not None
    assert result.backup.retained is True
    backup_path = root / (result.backup.relative_path or "")
    assert backup_path.is_file()
    assert backup_path.read_text(encoding="utf-8") == "recoverable"


def test_backup_failure_does_not_mutate_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("old", encoding="utf-8")
    session = _modification_session(backup_enabled=True)
    backup_module = __import__("backend_ai.tools.safe_editing", fromlist=["_atomic_create_bytes"])

    def fail_backup(path: Path, content: bytes) -> None:
        raise OSError("backup failed")

    monkeypatch.setattr(backup_module, "_atomic_create_bytes", fail_backup)
    before = target.read_bytes()
    _error(session.edit, ToolErrorCode.BACKUP_FAILED, root, "app.py", "old", "new")
    assert target.read_bytes() == before


def test_verification_failure_is_structured_after_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    session = _modification_session()
    safe_module = __import__("backend_ai.tools.safe_editing", fromlist=["FileSnapshot"])
    real_snapshot = safe_module.SafeEditSession.snapshot
    calls = {"count": 0}

    def fail_after_create(self, project_root, path):
        calls["count"] += 1
        snapshot = real_snapshot(self, project_root, path)
        if calls["count"] == 2:
            return FileSnapshot(snapshot.relative_path, snapshot.exists, snapshot.size_bytes, snapshot.mtime_ns, snapshot.device, snapshot.inode, "0" * 64, snapshot.file_type, snapshot.mode)
        return snapshot

    monkeypatch.setattr(safe_module.SafeEditSession, "snapshot", fail_after_create)
    _error(session.create, ToolErrorCode.VERIFICATION_FAILED, root, "app.py", "x")
    assert (root / "app.py").read_text(encoding="utf-8") == "x"


def test_safe_session_rejects_traversal_and_symlink_without_mutation(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = root / "link.txt"
    link.symlink_to(outside)
    session = _modification_session()

    _error(session.snapshot, ToolErrorCode.PATH_OUTSIDE_ROOT, root, "../outside.txt")
    _error(session.snapshot, ToolErrorCode.PATH_OUTSIDE_ROOT, root, "link.txt")
    assert outside.read_text(encoding="utf-8") == "outside"
    assert link.is_symlink()


def test_registry_remains_read_only_by_default_and_agent_default_is_unchanged() -> None:
    default = ToolRegistry.default()
    modification = ToolRegistry.with_file_modification()
    assert "write_file" not in default.names()
    assert "edit_file" not in default.names()
    assert "delete_file" not in default.names()
    assert {"write_file", "edit_file", "delete_file"}.issubset(modification.names())
    assert "safe_editing" not in default.names()


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class _FakeEngine:
    tokenizer = _FakeTokenizer()

    def generate(self, prompt: str):
        class Output:
            generated_text = "FINAL: no mutation"

        return Output()


def test_agent_loop_default_does_not_use_safe_editing_or_modify_files(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("old", encoding="utf-8")

    result = AgentLoop(_FakeEngine()).run("Inspect only", root)

    assert result.final_answer == "no mutation"
    assert target.read_text(encoding="utf-8") == "old"
