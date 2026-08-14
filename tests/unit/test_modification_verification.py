from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from backend_ai.tools import (
    ExpectedModification,
    FileSnapshot,
    ModificationVerificationResult,
    SafeEditPolicy,
    SafeEditSession,
    ToolError,
    ToolErrorCode,
    verify_modification,
)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def _find(result: ModificationVerificationResult, relative: str):
    return next(item for item in result.verified_targets if item.relative_path == relative)


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_created_file_verifies_by_content_hash_and_size_without_leaking_content(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "src" / "api.py"
    target.parent.mkdir()
    content = "محتوى عربي\nvalue = 1\n"
    target.write_text(content, encoding="utf-8")

    result = verify_modification(
        root,
        [ExpectedModification.created("src/api.py", expected_content=content)],
        detect_unexpected=False,
    )

    item = _find(result, "src/api.py")
    assert result.success is True
    assert result.complete is True
    assert item.status == "VERIFIED"
    assert item.actual_state == "present_regular_file"
    assert item.actual_size == len(content.encode("utf-8"))
    assert item.actual_sha256 == _hash(content.encode("utf-8"))
    assert content not in str(result.to_dict())
    with pytest.raises(Exception):
        result.success = False  # type: ignore[misc]


def test_created_file_missing_wrong_hash_and_wrong_content_are_explicit(tmp_path: Path) -> None:
    root = _project(tmp_path)
    missing = verify_modification(root, [ExpectedModification.created("missing.py", expected_content="x")], detect_unexpected=False)
    assert missing.success is False
    assert _find(missing, "missing.py").status == "MISSING"

    target = root / "created.py"
    target.write_text("actual", encoding="utf-8")
    wrong_hash = verify_modification(root, [ExpectedModification.created("created.py", expected_sha256="0" * 64)], detect_unexpected=False)
    wrong_content = verify_modification(root, [ExpectedModification.created("created.py", expected_content="expected")], detect_unexpected=False)
    assert _find(wrong_hash, "created.py").status == "HASH_MISMATCH"
    assert _find(wrong_content, "created.py").status == "CONTENT_MISMATCH"


def test_created_path_replaced_by_directory_symlink_or_fifo_is_type_changed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    directory = root / "directory"
    directory.mkdir()
    symlink_target = tmp_path / "outside.txt"
    symlink_target.write_text("outside", encoding="utf-8")
    symlink = root / "symlink"
    symlink.symlink_to(symlink_target)

    directory_result = verify_modification(root, [ExpectedModification.created("directory", expected_content="x")], detect_unexpected=False)
    symlink_result = verify_modification(root, [ExpectedModification.created("symlink", expected_content="x")], detect_unexpected=False)
    assert _find(directory_result, "directory").status == "TYPE_CHANGED"
    assert _find(directory_result, "directory").actual_state == "directory"
    assert _find(symlink_result, "symlink").status == "TYPE_CHANGED"
    assert _find(symlink_result, "symlink").actual_state == "symlink"

    fifo = root / "pipe"
    if hasattr(os, "mkfifo"):
        os.mkfifo(fifo)
        fifo_result = verify_modification(root, [ExpectedModification.created("pipe", expected_content="x")], detect_unexpected=False)
        assert _find(fifo_result, "pipe").actual_state == "special_file"
        assert _find(fifo_result, "pipe").status == "TYPE_CHANGED"


def test_edit_verification_checks_expected_transition_unicode_tabs_line_endings_and_noop(tmp_path: Path) -> None:
    root = _project(tmp_path)
    original = "قبل\tvalue\r\n"
    updated = "بعد\tvalue\r\n"
    target = root / "edit.py"
    target.write_bytes(updated.encode("utf-8"))
    session = SafeEditSession(SafeEditPolicy())
    before = FileSnapshot("edit.py", True, len(original.encode()), 1, 1, 1, _hash(original.encode()), "regular", 0o644)

    result = verify_modification(
        root,
        [ExpectedModification.modified("edit.py", expected_content=updated, before_snapshot=before)],
        detect_unexpected=False,
    )
    assert result.success is True
    assert _find(result, "edit.py").status == "VERIFIED"

    noop = verify_modification(
        root,
        [ExpectedModification.unchanged("edit.py", expected_content=updated, before_snapshot=FileSnapshot("edit.py", True, len(updated.encode()), 1, 1, 1, _hash(updated.encode()), "regular", 0o644))],
        detect_unexpected=False,
    )
    assert noop.success is True


def test_edit_verification_detects_wrong_hash_content_and_no_actual_change(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "edit.py"
    target.write_text("old", encoding="utf-8")
    session = SafeEditSession(SafeEditPolicy())
    before = session.snapshot(root, "edit.py")

    wrong_hash = verify_modification(root, [ExpectedModification.modified("edit.py", expected_sha256="0" * 64, before_snapshot=before)], detect_unexpected=False)
    wrong_content = verify_modification(root, [ExpectedModification.modified("edit.py", expected_content="new", before_snapshot=before)], detect_unexpected=False)
    no_change = verify_modification(root, [ExpectedModification.modified("edit.py", expected_content="old", before_snapshot=before)], detect_unexpected=False)

    assert _find(wrong_hash, "edit.py").status == "HASH_MISMATCH"
    assert _find(wrong_content, "edit.py").status == "CONTENT_MISMATCH"
    assert _find(no_change, "edit.py").status == "UNEXPECTED_MODIFICATION"


def test_deleted_file_verification_distinguishes_missing_directory_symlink_and_existing_file(tmp_path: Path) -> None:
    root = _project(tmp_path)
    missing = verify_modification(root, [ExpectedModification.deleted("deleted.py")], detect_unexpected=False)
    assert missing.success is True
    assert _find(missing, "deleted.py").status == "VERIFIED"

    target = root / "still.py"
    target.write_text("still", encoding="utf-8")
    existing = verify_modification(root, [ExpectedModification.deleted("still.py")], detect_unexpected=False)
    assert _find(existing, "still.py").status == "UNEXPECTED_CREATION"

    directory = root / "directory"
    directory.mkdir()
    directory_result = verify_modification(root, [ExpectedModification.deleted("directory")], detect_unexpected=False)
    assert _find(directory_result, "directory").status == "TYPE_CHANGED"

    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    link = root / "link"
    link.symlink_to(outside)
    link_result = verify_modification(root, [ExpectedModification.deleted("link")], detect_unexpected=False)
    assert _find(link_result, "link").actual_state == "symlink"
    assert _find(link_result, "link").status == "TYPE_CHANGED"
    assert outside.exists()


def test_invalid_utf8_is_strict_verification_error_without_content_fallback(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "binary.bin"
    target.write_bytes(b"\xff\x00")

    result = verify_modification(root, [ExpectedModification.created("binary.bin", expected_content="x")], detect_unexpected=False)

    item = _find(result, "binary.bin")
    assert item.actual_state == "invalid_utf8"
    assert item.status == "VERIFICATION_ERROR"
    assert "\ufffd" not in str(result.to_dict())


def test_baseline_detects_unexpected_modification_creation_and_deletion(tmp_path: Path) -> None:
    root = _project(tmp_path)
    app = root / "app.py"
    other = root / "other.py"
    app.write_text("old", encoding="utf-8")
    other.write_text("other", encoding="utf-8")
    session = SafeEditSession(SafeEditPolicy())
    baseline = {
        "app.py": session.snapshot(root, "app.py"),
        "other.py": session.snapshot(root, "other.py"),
    }
    app.write_text("expected", encoding="utf-8")
    other.write_text("unexpected", encoding="utf-8")
    (root / "new.py").write_text("new", encoding="utf-8")

    result = verify_modification(
        root,
        [ExpectedModification.modified("app.py", expected_content="expected", before_snapshot=baseline["app.py"])],
        baseline=baseline,
    )

    assert result.success is False
    unexpected = {item.relative_path: item.status for item in result.unexpected_changes}
    assert unexpected["other.py"] == "UNEXPECTED_MODIFICATION"
    assert unexpected["new.py"] == "UNEXPECTED_CREATION"

    (root / "other.py").unlink()
    deletion_result = verify_modification(
        root,
        [ExpectedModification.modified("app.py", expected_content="expected", before_snapshot=baseline["app.py"])],
        baseline=baseline,
    )
    deletion_statuses = {item.relative_path: item.status for item in deletion_result.unexpected_changes}
    assert deletion_statuses["other.py"] == "UNEXPECTED_DELETION"


def test_no_baseline_is_explicitly_incomplete_not_false_complete(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "app.py").write_text("ok", encoding="utf-8")

    result = verify_modification(root, [ExpectedModification.created("app.py", expected_content="ok")])

    assert result.success is False
    assert result.complete is False
    assert result.warnings


def test_bounded_baseline_reports_truncation(tmp_path: Path) -> None:
    root = _project(tmp_path)
    for index in range(5):
        (root / f"file-{index}.py").write_text(str(index), encoding="utf-8")
    session = SafeEditSession(SafeEditPolicy())
    baseline = {f"file-{index}.py": session.snapshot(root, f"file-{index}.py") for index in range(5)}

    result = verify_modification(
        root,
        [ExpectedModification.unchanged("file-0.py", before_snapshot=baseline["file-0.py"])],
        baseline=baseline,
        max_files=2,
    )

    assert result.truncated is True
    assert result.truncation_reason == "max_files"
    assert result.complete is False


def test_security_rejects_traversal_absolute_windows_and_symlink_parent(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("outside", encoding="utf-8")
    for path in ("../outside.py", str(outside), r"C:\outside.py", r"\\server\share\outside.py", "x\x00.py"):
        with pytest.raises(ToolError) as raised:
            verify_modification(root, [ExpectedModification.created(path, expected_content="x")], detect_unexpected=False)
        assert raised.value.code in {ToolErrorCode.PATH_OUTSIDE_ROOT, ToolErrorCode.INVALID_ARGUMENT}
    link_dir = root / "link-dir"
    link_dir.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ToolError) as raised:
        verify_modification(root, [ExpectedModification.created("link-dir/file.py", expected_content="x")], detect_unexpected=False)
    assert raised.value.code is ToolErrorCode.PATH_OUTSIDE_ROOT


def test_safe_edit_session_exposes_verification_metadata_for_create_edit_delete(tmp_path: Path) -> None:
    root = _project(tmp_path)
    session = SafeEditSession(SafeEditPolicy.for_modification())

    created = session.create(root, "created.py", "created")
    edited = session.edit(root, "created.py", "created", "edited")
    deleted = session.delete(root, "created.py")

    assert created.verification is not None and created.verification.success is True
    assert edited.verification is not None and edited.verification.success is True
    assert deleted.verification is not None and deleted.verification.success is True
    assert created.to_dict()["verification"]["verified_targets"][0]["status"] == "VERIFIED"
    assert (root / "created.py").exists() is False
