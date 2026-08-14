from __future__ import annotations

import os
from pathlib import Path
import stat
from importlib import import_module

import pytest

from backend_ai.agent import ToolRegistry
from backend_ai.tools import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_NEW_CONTENT_BYTES,
    DEFAULT_MAX_OLD_CONTENT_BYTES,
    DEFAULT_MAX_RESULT_BYTES,
    EditFileResult,
    EditFileTool,
    ToolError,
    ToolErrorCode,
    edit_file,
    read_file,
    write_file,
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


def test_edit_file_replaces_one_exact_match_and_returns_metadata(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "app.py"
    target.write_text("def greet():\n    return 'old'\n", encoding="utf-8")
    original_mode = stat.S_IMODE(target.stat().st_mode)

    result = edit_file(root, "app.py", "return 'old'", "return 'new'")

    assert isinstance(result, EditFileResult)
    assert result.relative_path == "app.py"
    assert result.file_name == "app.py"
    assert result.original_size_bytes == len(b"def greet():\n    return 'old'\n")
    assert result.new_size_bytes == len(b"def greet():\n    return 'new'\n")
    assert result.bytes_changed == 0
    assert result.match_count == 1
    assert result.occurrence == 1
    assert result.changed is True
    assert result.to_dict()["changed"] is True
    assert target.read_text(encoding="utf-8") == "def greet():\n    return 'new'\n"
    assert stat.S_IMODE(target.stat().st_mode) == original_mode


def test_edit_file_preserves_exact_formats_and_unicode(tmp_path: Path) -> None:
    root = _project(tmp_path)
    samples = {
        "python.py": ("def f():\n\treturn 'old'\n", "return 'old'", "return 'عربي'") ,
        "script.js": ("const old = 1;", "old", "new"),
        "types.ts": ("const value: string = 'old';", "'old'", "'new'"),
        "data.json": ('{\n  "name": "old"\n}\n', '"old"', '"new"'),
        "query.sql": ("SELECT old\r\nFROM users;\r\n", "SELECT old", "SELECT new"),
        "README.md": ("# old\n\nno final marker", "# old", "# new"),
        "Dockerfile": ("FROM old:latest\n", "old:latest", "python:3.12-slim"),
    }

    for relative_path, (original, old, new) in samples.items():
        target = root / relative_path
        target.write_bytes(original.encode("utf-8"))
        edit_file(root, relative_path, old, new)
        assert target.read_bytes() == original.replace(old, new, 1).encode("utf-8")


def test_edit_file_replaces_at_beginning_middle_and_end(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "values.txt"
    target.write_text("old middle old-end", encoding="utf-8")

    edit_file(root, "values.txt", "old middle", "beginning")
    edit_file(root, "values.txt", "old-end", "end")

    assert target.read_text(encoding="utf-8") == "beginning end"


def test_edit_file_rejects_zero_or_ambiguous_matches_without_mutation(tmp_path: Path) -> None:
    root = _project(tmp_path)
    missing = root / "missing.txt"
    missing.write_text("stable", encoding="utf-8")
    before_missing = missing.read_bytes()
    ambiguous = root / "ambiguous.txt"
    ambiguous.write_text("old\nold\nold", encoding="utf-8")
    before_ambiguous = ambiguous.read_bytes()

    _error(edit_file, ToolErrorCode.MATCH_NOT_FOUND, root, "missing.txt", "absent", "new")
    _error(edit_file, ToolErrorCode.AMBIGUOUS_MATCH, root, "ambiguous.txt", "old", "new")

    assert missing.read_bytes() == before_missing
    assert ambiguous.read_bytes() == before_ambiguous


def test_edit_file_noop_does_not_rewrite_and_is_structured(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "noop.txt"
    target.write_text("keep old", encoding="utf-8")
    before_stat = target.stat()
    before_bytes = target.read_bytes()

    result = edit_file(root, "noop.txt", "old", "old")

    assert result.changed is False
    assert result.bytes_changed == 0
    assert result.match_count == 1
    assert target.read_bytes() == before_bytes
    after_stat = target.stat()
    assert after_stat.st_ino == before_stat.st_ino
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


def test_edit_file_requires_existing_regular_utf8_file(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "folder").mkdir()
    (root / "invalid.bin").write_bytes(b"\xff\xfe")
    fifo = root / "pipe"
    if hasattr(os, "mkfifo"):
        os.mkfifo(fifo)

    _error(edit_file, ToolErrorCode.FILE_NOT_FOUND, root, "missing.txt", "old", "new")
    _error(edit_file, ToolErrorCode.NOT_A_FILE, root, "folder", "old", "new")
    _error(edit_file, ToolErrorCode.INVALID_UTF8, root, "invalid.bin", "old", "new")
    if fifo.exists():
        _error(edit_file, ToolErrorCode.NOT_A_FILE, root, "pipe", "old", "new")


def test_edit_file_rejects_symlink_target_parent_and_broken_link(tmp_path: Path) -> None:
    root = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "target.txt").write_text("old", encoding="utf-8")
    (root / "target-link.txt").symlink_to(outside / "target.txt")
    (root / "parent-link").symlink_to(outside, target_is_directory=True)
    (root / "broken.txt").symlink_to(outside / "missing.txt")

    for path in ("target-link.txt", "parent-link/new.txt", "broken.txt"):
        _error(edit_file, ToolErrorCode.PATH_OUTSIDE_ROOT, root, path, "old", "new")
    assert (outside / "target.txt").read_text(encoding="utf-8") == "old"


def test_edit_file_rejects_traversal_windows_paths_and_empty_old_content(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "file.txt").write_text("old", encoding="utf-8")

    for path in ("../file.txt", "/tmp/file.txt", r"C:\file.txt", r"\\server\share\file.txt", r"nested\..\..\file.txt"):
        _error(edit_file, ToolErrorCode.PATH_OUTSIDE_ROOT, root, path, "old", "new")
    _error(edit_file, ToolErrorCode.INVALID_ARGUMENT, root, "file.txt", "", "new")
    _error(edit_file, ToolErrorCode.INVALID_ARGUMENT, root, "file" + chr(0) + ".txt", "old", "new")


def test_edit_file_rejects_invalid_arguments_and_content_encoding(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "file.txt").write_text("old", encoding="utf-8")

    _error(edit_file, ToolErrorCode.INVALID_ARGUMENT, root, "file.txt", 1, "new")
    _error(edit_file, ToolErrorCode.INVALID_ARGUMENT, root, "file.txt", "old", 1)
    _error(edit_file, ToolErrorCode.INVALID_UTF8, root, "file.txt", "\ud800", "new")
    _error(edit_file, ToolErrorCode.INVALID_UTF8, root, "file.txt", "old", "\ud800")
    for name in ("max_file_bytes", "max_old_content_bytes", "max_new_content_bytes", "max_result_bytes"):
        kwargs = {name: -1}
        _error(edit_file, ToolErrorCode.INVALID_ARGUMENT, root, "file.txt", "old", "new", **kwargs)


def test_edit_file_enforces_existing_old_new_and_result_size_limits(tmp_path: Path) -> None:
    root = _project(tmp_path)
    target = root / "file.txt"
    target.write_text("old", encoding="utf-8")
    before = target.read_bytes()

    _error(edit_file, ToolErrorCode.FILE_TOO_LARGE, root, "file.txt", "old", "new", max_file_bytes=2)
    _error(edit_file, ToolErrorCode.FILE_TOO_LARGE, root, "file.txt", "old", "new", max_old_content_bytes=2)
    _error(edit_file, ToolErrorCode.FILE_TOO_LARGE, root, "file.txt", "old", "new", max_new_content_bytes=2)
    _error(edit_file, ToolErrorCode.FILE_TOO_LARGE, root, "file.txt", "old", "newer", max_result_bytes=3)
    assert target.read_bytes() == before
    assert DEFAULT_MAX_FILE_BYTES == DEFAULT_MAX_NEW_CONTENT_BYTES == DEFAULT_MAX_OLD_CONTENT_BYTES == DEFAULT_MAX_RESULT_BYTES


def test_edit_file_cleans_temporary_file_and_preserves_original_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    target = root / "file.txt"
    target.write_text("old content", encoding="utf-8")
    before = target.read_bytes()

    def fail_replace(source: str, destination: str) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    _error(edit_file, ToolErrorCode.FILESYSTEM_ERROR, root, "file.txt", "old", "new")

    assert target.read_bytes() == before
    assert not list(root.glob(".*.fodci-edit-*"))


def test_edit_file_detects_target_change_before_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    target = root / "file.txt"
    target.write_text("old content", encoding="utf-8")
    edit_module = import_module("backend_ai.tools.edit_file")
    real_snapshot = edit_module._read_snapshot
    calls = {"count": 0}

    def mutate_on_second_snapshot(path: Path, relative_path: str, max_file_bytes: int):
        calls["count"] += 1
        if calls["count"] == 2:
            target.write_text("newer user content", encoding="utf-8")
        return real_snapshot(path, relative_path, max_file_bytes)

    monkeypatch.setattr(edit_module, "_read_snapshot", mutate_on_second_snapshot)
    _error(edit_file, ToolErrorCode.CONCURRENT_MODIFICATION, root, "file.txt", "old", "new")
    assert target.read_text(encoding="utf-8") == "newer user content"


def test_edit_file_preserves_unrelated_files_and_does_not_print_content(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _project(tmp_path)
    target = root / "target.py"
    unrelated = root / "unrelated.py"
    target.write_text("old", encoding="utf-8")
    unrelated.write_text("unchanged", encoding="utf-8")
    before_unrelated = unrelated.read_bytes()

    edit_file(root, "target.py", "old", "new")

    assert unrelated.read_bytes() == before_unrelated
    assert capsys.readouterr().out == ""


def test_edit_file_tool_protocol_and_registry_are_opt_in(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "main.py").write_text("old", encoding="utf-8")
    tool = EditFileTool()

    assert tool.name == "edit_file"
    assert tool.metadata.input_schema["required"] == ["project_root", "path", "old_content", "new_content"]
    result = tool.run({"project_root": root, "path": "main.py", "old_content": "old", "new_content": "new"})
    assert result.changed is True
    _error(tool.run, ToolErrorCode.INVALID_ARGUMENT, {})
    _error(tool.run, ToolErrorCode.INVALID_ARGUMENT, [])

    default = ToolRegistry.default()
    create_only = ToolRegistry.with_write_file()
    modification = ToolRegistry.with_file_modification()
    assert "edit_file" not in default.names()
    assert "edit_file" not in create_only.names()
    assert "edit_file" in modification.names()
    assert "write_file" in modification.names()
