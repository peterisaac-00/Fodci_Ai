from __future__ import annotations

from pathlib import Path
import os
import stat

import pytest

from backend_ai.tools import (
    DEFAULT_MAX_READ_BYTES,
    ReadFileResult,
    ReadFileTool,
    Tool,
    ToolError,
    ToolErrorCode,
    read_file,
)


def _write_text(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return path


def _write_bytes(root: Path, relative_path: str, content: bytes) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _assert_error(callable_object, code: ToolErrorCode) -> ToolError:
    with pytest.raises(ToolError) as raised:
        callable_object()
    assert raised.value.code == code
    assert raised.value.to_dict()["code"] == code.value
    return raised.value


def test_read_utf8_python_and_nested_file_exactly(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    content = "def greet(name):\r\n\treturn f\"مرحبا, {name}!\"\r\n"
    file_path = _write_text(root, "app/services/greet.py", content)

    result = read_file(root, "app/services/greet.py")

    assert isinstance(result, ReadFileResult)
    assert result.relative_path == "app/services/greet.py"
    assert result.file_name == "greet.py"
    assert result.content == content
    assert result.encoding == "utf-8"
    assert result.size_bytes == len(content.encode("utf-8"))
    assert file_path.read_bytes() == content.encode("utf-8")


def test_read_preserves_empty_content_and_final_newline_behavior(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "empty.txt", "")
    _write_text(root, "without-final-newline.md", "line 1\nline 2")
    _write_text(root, "with-final-newline.sql", "SELECT 1;\r\n")

    assert read_file(root, "empty.txt").content == ""
    assert read_file(root, "without-final-newline.md").content == "line 1\nline 2"
    assert read_file(root, "with-final-newline.sql").content == "SELECT 1;\r\n"


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("config.json", '{"name": "Fodci", "enabled": true}'),
        ("schema.sql", "CREATE TABLE users (id INTEGER);"),
        ("README.md", "# Backend\n\nUse tabs\tand spaces.\n"),
        ("server.js", "export default function handler() { return 'ok'; }"),
        ("config.yaml", "service:\n  name: api\n"),
        ("Dockerfile", "FROM python:3.12-slim\n"),
    ],
)
def test_read_common_backend_text_formats(
    tmp_path: Path,
    relative_path: str,
    content: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, relative_path, content)

    result = read_file(root, relative_path)

    assert result.content == content


def test_tool_metadata_and_runtime_protocol(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "app.py", "print('ok')")
    tool = ReadFileTool()

    assert isinstance(tool, Tool)
    assert tool.name == "read_file"
    assert tool.metadata.name == "read_file"
    assert "project_root" in tool.metadata.input_schema["required"]
    assert "path" in tool.metadata.input_schema["required"]
    assert tool.run({"project_root": root, "path": "app.py"}).content == "print('ok')"


def test_tool_input_validation_is_structured(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = ReadFileTool()

    _assert_error(lambda: tool.run({}), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: read_file(root, "app.py", max_bytes=-1), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: read_file(root, "app.py", max_bytes=True), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: read_file(root, ""), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: read_file(root, "\x00bad"), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: read_file(root, 123), ToolErrorCode.INVALID_ARGUMENT)  # type: ignore[arg-type]


def test_missing_directory_and_directory_path_errors(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "app").mkdir()

    _assert_error(lambda: read_file(root, "missing.py"), ToolErrorCode.PATH_NOT_FOUND)
    _assert_error(lambda: read_file(root, "app"), ToolErrorCode.NOT_A_FILE)
    _assert_error(lambda: read_file(tmp_path / "missing-root", "app.py"), ToolErrorCode.PATH_NOT_FOUND)
    _assert_error(lambda: read_file(root / "app", "app.py"), ToolErrorCode.PATH_NOT_FOUND)


def test_relative_normalized_and_absolute_inside_paths_are_safe(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    file_path = _write_text(root, "app/main.py", "main()")

    assert read_file(root, "./app/nested/../main.py").content == "main()"
    assert read_file(root, "app\\nested/../main.py").content == "main()"
    assert read_file(root, file_path).relative_path == "app/main.py"


def test_path_traversal_absolute_and_windows_paths_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "app/main.py", "main()")
    _write_text(tmp_path, "secret.txt", "secret")

    for requested in (
        "../secret.txt",
        "../../secret.txt",
        "app/../../secret.txt",
        str(tmp_path / "secret.txt"),
        r"C:\Users\Peter\secret.txt",
        r"\\server\share\secret.txt",
    ):
        error = _assert_error(lambda requested=requested: read_file(root, requested), ToolErrorCode.PATH_OUTSIDE_ROOT)
        assert "outside" in error.message.lower() or "absolute" in error.message.lower()


def test_file_size_limit_rejects_above_limit_without_partial_content(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    content = "0123456789"
    _write_text(root, "data.txt", content)

    exact = read_file(root, "data.txt", max_bytes=len(content.encode("utf-8")))
    assert exact.content == content
    error = _assert_error(
        lambda: read_file(root, "data.txt", max_bytes=len(content.encode("utf-8")) - 1),
        ToolErrorCode.FILE_TOO_LARGE,
    )
    assert "data.txt" in error.message
    assert str(len(content.encode("utf-8")) - 1) in error.message


def test_default_limit_is_explicit_and_custom_limit_supports_unicode_bytes(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    content = "مرحبا" * 20
    path = _write_text(root, "arabic.txt", content)

    assert DEFAULT_MAX_READ_BYTES > 0
    result = read_file(root, "arabic.txt", max_bytes=path.stat().st_size)
    assert result.content == content
    _assert_error(lambda: read_file(root, "arabic.txt", max_bytes=1), ToolErrorCode.FILE_TOO_LARGE)


def test_invalid_utf8_is_rejected_without_replacement(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_bytes(root, "invalid.bin", b"valid-prefix\xff\xfe")

    error = _assert_error(lambda: read_file(root, "invalid.bin"), ToolErrorCode.INVALID_UTF8)

    assert "not valid UTF-8" in error.message


def test_symlinks_are_rejected_consistently(tmp_path: Path) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    _write_text(root, "inside.py", "inside")
    outside_file = _write_text(outside, "secret.py", "secret")
    outside_directory = outside / "package"
    outside_directory.mkdir()
    _write_text(outside_directory, "module.py", "module")

    try:
        (root / "external_file.py").symlink_to(outside_file)
        (root / "external_dir").symlink_to(outside_directory, target_is_directory=True)
        (root / "internal_link.py").symlink_to(root / "inside.py")
        (root / "broken.py").symlink_to(root / "missing.py")
        (root / "loop").symlink_to(root, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    for requested in ("external_file.py", "external_dir/module.py", "internal_link.py", "broken.py", "loop/inside.py"):
        _assert_error(lambda requested=requested: read_file(root, requested), ToolErrorCode.PATH_OUTSIDE_ROOT)


def test_special_files_are_not_read_when_supported(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO is not supported on this platform")
    root = tmp_path / "project"
    root.mkdir()
    fifo = root / "events.pipe"
    try:
        os.mkfifo(fifo)
    except (AttributeError, NotImplementedError, OSError) as exc:
        pytest.skip(f"FIFO unavailable: {exc}")

    _assert_error(lambda: read_file(root, "events.pipe"), ToolErrorCode.NOT_A_FILE)


def test_permission_error_is_structured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = _write_text(root, "secret.py", "secret")
    original_open = Path.open

    def deny_target(self: Path, *args: object, **kwargs: object):
        if self == target:
            raise PermissionError("denied for test")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_target)

    _assert_error(lambda: read_file(root, "secret.py"), ToolErrorCode.PERMISSION_DENIED)


def test_read_does_not_mutate_file_state_or_print_content(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "project"
    root.mkdir()
    path = _write_text(root, "source.py", "password = 'do-not-log'\n")
    before = path.stat()

    result = read_file(root, "source.py")

    after = path.stat()
    assert result.content == "password = 'do-not-log'\n"
    assert before.st_size == after.st_size
    assert before.st_mtime_ns == after.st_mtime_ns
    assert capsys.readouterr().out == ""
