from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend_ai.agent import ToolRegistry
from backend_ai.tools import (
    DEFAULT_MAX_PARENT_DIRECTORIES,
    DEFAULT_MAX_WRITE_BYTES,
    ToolError,
    ToolErrorCode,
    WriteFileResult,
    WriteFileTool,
    read_file,
    write_file,
)


def _error(callable_obj, code: ToolErrorCode, *args, **kwargs) -> ToolError:
    with pytest.raises(ToolError) as raised:
        callable_obj(*args, **kwargs)
    assert raised.value.code is code
    return raised.value


def test_write_file_creates_utf8_unicode_file_and_returns_structured_result(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    content = "from fastapi import APIRouter\n\n# عربي backend\n"

    result = write_file(root, "app/routes/users.py", content)

    assert isinstance(result, WriteFileResult)
    assert result.relative_path == "app/routes/users.py"
    assert result.file_name == "users.py"
    assert result.size_bytes == len(content.encode("utf-8"))
    assert result.encoding == "utf-8"
    assert result.created is True
    assert result.to_dict() == {
        "relative_path": "app/routes/users.py",
        "file_name": "users.py",
        "size_bytes": len(content.encode("utf-8")),
        "encoding": "utf-8",
        "created": True,
    }
    assert read_file(root, "app/routes/users.py").content == content
    assert not list((root / "app" / "routes").glob(".*.fodci-*"))


def test_write_file_creates_nested_missing_parents_with_bounded_depth(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    result = write_file(root, "app/routes/users.py", "x")

    assert result.relative_path == "app/routes/users.py"
    assert (root / "app" / "routes" / "users.py").read_text(encoding="utf-8") == "x"
    assert (root / "app").is_dir()
    assert (root / "app" / "routes").is_dir()


def test_write_file_allows_empty_content_and_exact_byte_boundary(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    empty = write_file(root, "empty.txt", "", max_bytes=0)
    exact = write_file(root, "exact.txt", "abc", max_bytes=3)

    assert empty.size_bytes == 0
    assert exact.size_bytes == 3
    assert (root / "empty.txt").read_bytes() == b""
    assert (root / "exact.txt").read_bytes() == b"abc"


def test_write_file_preserves_code_formats_and_line_endings(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    samples = {
        "main.py": "def main():\n\treturn 'ok'\n",
        "app.js": "export default { value: 1 };",
        "query.sql": "SELECT *\r\nFROM users;\r\n",
        "README.md": "# Title\n\nArabic: عربي",
        "Dockerfile": "FROM python:3.12-slim\n",
        "config.json": '{\n  "enabled": true\n}\n',
    }

    for relative_path, content in samples.items():
        write_file(root, relative_path, content)
        assert (root / relative_path).read_bytes() == content.encode("utf-8")


def test_write_file_never_overwrites_existing_file_or_directory(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    existing = root / "existing.py"
    existing.write_text("original", encoding="utf-8")
    (root / "folder").mkdir()

    file_error = _error(write_file, ToolErrorCode.FILE_EXISTS, root, "existing.py", "changed")
    directory_error = _error(write_file, ToolErrorCode.FILE_EXISTS, root, "folder", "changed")

    assert "existing.py" in file_error.message
    assert "folder" in directory_error.message
    assert existing.read_text(encoding="utf-8") == "original"


def test_write_file_requires_existing_root_but_creates_missing_parent(tmp_path: Path) -> None:
    root = tmp_path / "project"

    _error(write_file, ToolErrorCode.PATH_NOT_FOUND, root, "a.txt", "x")
    root.mkdir()
    depth_error = _error(
        write_file,
        ToolErrorCode.INVALID_ARGUMENT,
        root,
        "a/b/c.txt",
        "x",
        max_parent_directories=1,
    )
    assert "depth" in depth_error.message
    assert not (root / "a").exists()


def test_write_file_rejects_invalid_root_path_content_and_limits(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("x", encoding="utf-8")

    _error(write_file, ToolErrorCode.NOT_DIRECTORY, file_path, "a.txt", "x")
    _error(write_file, ToolErrorCode.INVALID_ARGUMENT, root, "a.txt", 123)
    _error(write_file, ToolErrorCode.INVALID_ARGUMENT, root, "a.txt", "x", max_bytes=-1)
    _error(write_file, ToolErrorCode.INVALID_ARGUMENT, root, "a.txt", "x", max_bytes=True)
    _error(write_file, ToolErrorCode.INVALID_ARGUMENT, root, "a.txt", "x", max_parent_directories=-1)
    _error(write_file, ToolErrorCode.FILE_TOO_LARGE, root, "a.txt", "1234", max_bytes=3)
    _error(
        write_file,
        ToolErrorCode.FILE_TOO_LARGE,
        root,
        "a.txt",
        "x" * DEFAULT_MAX_WRITE_BYTES,
        max_bytes=DEFAULT_MAX_WRITE_BYTES - 1,
    )
    _error(write_file, ToolErrorCode.INVALID_UTF8, root, "a.txt", "\ud800")


def test_write_file_rejects_traversal_absolute_windows_and_nul_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()

    for path in ("../outside.txt", "/tmp/outside.txt", r"C:\outside.txt", r"\\server\share\outside.txt"):
        _error(write_file, ToolErrorCode.PATH_OUTSIDE_ROOT, root, path, "x")
    _error(write_file, ToolErrorCode.INVALID_ARGUMENT, root, "a\x00.txt", "x")
    _error(write_file, ToolErrorCode.INVALID_ARGUMENT, root, "", "x")
    _error(write_file, ToolErrorCode.NOT_A_FILE, root, ".", "x")
    _error(write_file, ToolErrorCode.PATH_OUTSIDE_ROOT, root, r"nested\..\..\outside.txt", "x")


def test_write_file_rejects_symlink_target_and_parent(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "linked-parent").symlink_to(outside, target_is_directory=True)
    (root / "broken.txt").symlink_to(outside / "missing.txt")

    _error(write_file, ToolErrorCode.PATH_OUTSIDE_ROOT, root, "link.txt", "changed")
    _error(write_file, ToolErrorCode.PATH_OUTSIDE_ROOT, root, "linked-parent/new.txt", "changed")
    _error(write_file, ToolErrorCode.PATH_OUTSIDE_ROOT, root, "broken.txt", "changed")
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "secret"
    assert not (outside / "new.txt").exists()


def test_write_file_rejects_special_existing_path(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    fifo = root / "pipe"
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is not available on this platform")
    os.mkfifo(fifo)

    _error(write_file, ToolErrorCode.FILE_EXISTS, root, "pipe", "x")


def test_write_file_cleans_temporary_file_when_atomic_publish_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "project"
    root.mkdir()

    def fail_link(source: str, destination: str) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(os, "link", fail_link)
    _error(write_file, ToolErrorCode.FILESYSTEM_ERROR, root, "new.txt", "content")

    assert not (root / "new.txt").exists()
    assert not list(root.glob(".*.fodci-*"))


def test_write_file_tool_protocol_and_validation(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = WriteFileTool()

    assert tool.name == "write_file"
    assert tool.metadata.input_schema["required"] == ["project_root", "path", "content"]
    assert tool.metadata.input_schema["properties"]["max_parent_directories"]["default"] == DEFAULT_MAX_PARENT_DIRECTORIES
    result = tool.run({"project_root": root, "path": "main.py", "content": "print('ok')"})
    assert result.relative_path == "main.py"
    _error(tool.run, ToolErrorCode.INVALID_ARGUMENT, {})
    _error(tool.run, ToolErrorCode.INVALID_ARGUMENT, {"project_root": root, "path": "x.py"})
    _error(tool.run, ToolErrorCode.INVALID_ARGUMENT, [])


def test_registry_write_file_is_opt_in_and_default_stays_read_only() -> None:
    default = ToolRegistry.default()
    phase41 = ToolRegistry.with_write_file()

    assert "write_file" not in default.names()
    assert "write_file" in phase41.names()
    assert len(phase41.names()) == len(default.names()) + 1
