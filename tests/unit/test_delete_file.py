from __future__ import annotations

import os
from pathlib import Path
import socket

import pytest

from backend_ai.agent import ToolRegistry
from backend_ai.tools import (
    DeleteFileResult,
    DeleteFileTool,
    ToolError,
    ToolErrorCode,
    delete_file,
)


def _error(callable_obj, code: ToolErrorCode, *args, **kwargs) -> ToolError:
    with pytest.raises(ToolError) as raised:
        callable_obj(*args, **kwargs)
    assert raised.value.code is code
    return raised.value


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def test_delete_file_removes_one_regular_file_and_preserves_parent_and_unrelated(tmp_path: Path) -> None:
    root = _project(tmp_path)
    parent = root / "src"
    parent.mkdir()
    target = parent / "main.py"
    unrelated = parent / "keep.py"
    target.write_text("print('delete me')\n", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")

    result = delete_file(root, "src/main.py")

    assert isinstance(result, DeleteFileResult)
    assert result.relative_path == "src/main.py"
    assert result.file_name == "main.py"
    assert result.size_bytes == len(b"print('delete me')\n")
    assert result.deleted is True
    assert result.to_dict() == {
        "relative_path": "src/main.py",
        "file_name": "main.py",
        "size_bytes": len(b"print('delete me')\n"),
        "deleted": True,
    }
    assert not target.exists()
    assert parent.is_dir()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert root.is_dir()


def test_delete_file_handles_empty_and_unicode_arabic_filenames(tmp_path: Path) -> None:
    root = _project(tmp_path)
    empty = root / "empty.txt"
    arabic = root / "ملف_تجربة.txt"
    empty.write_bytes(b"")
    arabic.write_text("محتوى", encoding="utf-8")

    empty_result = delete_file(root, empty.name)
    arabic_result = delete_file(root, arabic.name)

    assert empty_result.size_bytes == 0
    assert arabic_result.file_name == "ملف_تجربة.txt"
    assert not empty.exists()
    assert not arabic.exists()


def test_delete_file_missing_target_and_parent_are_structured_not_found(tmp_path: Path) -> None:
    root = _project(tmp_path)

    missing = _error(delete_file, ToolErrorCode.FILE_NOT_FOUND, root, "missing.py")
    missing_parent = _error(delete_file, ToolErrorCode.FILE_NOT_FOUND, root, "missing/target.py")

    assert missing.path == Path("missing.py")
    assert missing_parent.path == Path("missing/target.py")


def test_delete_file_rejects_directory_and_root_without_mutation(tmp_path: Path) -> None:
    root = _project(tmp_path)
    directory = root / "folder"
    directory.mkdir()
    child = directory / "child.txt"
    child.write_text("keep", encoding="utf-8")

    _error(delete_file, ToolErrorCode.NOT_A_FILE, root, "folder")
    _error(delete_file, ToolErrorCode.NOT_A_FILE, root, ".")

    assert directory.is_dir()
    assert child.read_text(encoding="utf-8") == "keep"


def test_delete_file_rejects_all_symlink_variants_without_following_or_deleting(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    internal = root / "internal.txt"
    external = outside / "external.txt"
    internal.write_text("internal", encoding="utf-8")
    external.write_text("external", encoding="utf-8")
    internal_link = root / "internal-link"
    external_link = root / "external-link"
    directory_link = root / "directory-link"
    broken_link = root / "broken-link"
    internal_link.symlink_to(internal)
    external_link.symlink_to(external)
    directory_link.symlink_to(outside, target_is_directory=True)
    broken_link.symlink_to(outside / "missing.txt")

    for path in ("internal-link", "external-link", "directory-link", "broken-link"):
        _error(delete_file, ToolErrorCode.PATH_OUTSIDE_ROOT, root, path)

    assert internal.exists()
    assert external.exists()
    assert internal_link.is_symlink()
    assert external_link.is_symlink()
    assert directory_link.is_symlink()
    assert broken_link.is_symlink()


def test_delete_file_rejects_special_files_without_deletion(tmp_path: Path) -> None:
    root = _project(tmp_path)
    fifo = root / "pipe"
    if hasattr(os, "mkfifo"):
        os.mkfifo(fifo)
        _error(delete_file, ToolErrorCode.NOT_A_FILE, root, "pipe")
        assert fifo.exists()

    server = root / "socket"
    try:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(server))
    except (AttributeError, OSError):
        pytest.skip("Unix sockets are not available on this platform")
    try:
        _error(delete_file, ToolErrorCode.NOT_A_FILE, root, "socket")
        assert server.exists()
    finally:
        listener.close()
        server.unlink(missing_ok=True)


def test_delete_file_rejects_traversal_absolute_windows_and_mixed_paths(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("must remain", encoding="utf-8")

    for path in (
        "../outside.txt",
        "../../outside.txt",
        "nested/../../outside.txt",
        str(outside),
        r"C:\outside.txt",
        r"\\server\share\outside.txt",
        r"nested\..\..\outside.txt",
    ):
        _error(delete_file, ToolErrorCode.PATH_OUTSIDE_ROOT, root, path)

    assert outside.read_text(encoding="utf-8") == "must remain"


def test_delete_file_requires_explicit_valid_root_and_arguments(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "target.txt"
    target.write_text("keep", encoding="utf-8")

    _error(delete_file, ToolErrorCode.PATH_NOT_FOUND, tmp_path / "missing", "target.txt")
    _error(delete_file, ToolErrorCode.INVALID_ARGUMENT, root, "")
    _error(delete_file, ToolErrorCode.INVALID_ARGUMENT, root, "target\x00.txt")
    _error(delete_file, ToolErrorCode.INVALID_ARGUMENT, root, 123)


def test_delete_file_permission_failure_preserves_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    target = root / "target.txt"
    target.write_text("keep", encoding="utf-8")
    real_unlink = os.unlink

    def deny_unlink(path: str, *, dir_fd: int | None = None) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(os, "unlink", deny_unlink)
    _error(delete_file, ToolErrorCode.PERMISSION_DENIED, root, "target.txt")
    monkeypatch.setattr(os, "unlink", real_unlink)
    assert target.read_text(encoding="utf-8") == "keep"


def test_delete_file_detects_target_change_before_unlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    target = root / "target.txt"
    target.write_text("old", encoding="utf-8")
    delete_module = __import__("backend_ai.tools.delete_file", fromlist=["_stat_at_parent"])
    real_stat_at_parent = delete_module._stat_at_parent

    def mutate_then_stat(parent_fd: int | None, file_name: str, relative_path: str):
        metadata = real_stat_at_parent(parent_fd, file_name, relative_path)
        values = list(metadata)
        values[6] += 1
        return os.stat_result(values)

    monkeypatch.setattr(delete_module, "_stat_at_parent", mutate_then_stat)
    _error(delete_file, ToolErrorCode.CONCURRENT_MODIFICATION, root, "target.txt")
    assert target.read_text(encoding="utf-8") == "old"


def test_delete_file_does_not_print_contents_or_leave_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _project(tmp_path)
    target = root / "target.txt"
    target.write_text("secret-looking content", encoding="utf-8")

    delete_file(root, "target.txt")

    assert capsys.readouterr().out == ""
    assert not list(root.glob(".*"))


def test_delete_file_tool_protocol_and_registry_are_opt_in(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "main.py"
    target.write_text("pass", encoding="utf-8")
    tool = DeleteFileTool()

    assert tool.name == "delete_file"
    assert tool.metadata.input_schema["required"] == ["project_root", "path"]
    result = tool.run({"project_root": root, "path": "main.py"})
    assert result.deleted is True
    _error(tool.run, ToolErrorCode.INVALID_ARGUMENT, {})
    _error(tool.run, ToolErrorCode.INVALID_ARGUMENT, [])

    default = ToolRegistry.default()
    create_only = ToolRegistry.with_write_file()
    modification = ToolRegistry.with_file_modification()
    assert "delete_file" not in default.names()
    assert "delete_file" not in create_only.names()
    assert {"write_file", "edit_file", "delete_file"}.issubset(modification.names())
