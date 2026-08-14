from __future__ import annotations

from pathlib import Path
import os

import pytest

from backend_ai.tools import (
    DEFAULT_MAX_FILE_BYTES,
    MAX_MAX_FILE_BYTES,
    MAX_MAX_RESULTS,
    SearchCodeResult,
    SearchCodeTool,
    Tool,
    ToolError,
    ToolErrorCode,
    search_code,
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


def test_basic_search_returns_structured_matches_with_1_based_lines_and_0_based_columns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "app/auth.py", "class Auth:\n    def authenticate(user):\n        return user\n")
    _write_text(root, "app/other.py", "def authenticate_admin(user):\n    return user\n")

    result = search_code(root, "authenticate")

    assert isinstance(result, SearchCodeResult)
    assert [(match.relative_path, match.line_number) for match in result.matches] == [
        ("app/auth.py", 2),
        ("app/other.py", 1),
    ]
    first = result.matches[0]
    assert first.line == "    def authenticate(user):"
    assert first.column_start == 8
    assert first.column_end == 20
    assert result.total_matches == 2
    assert result.files_searched == 2
    assert not result.truncated


def test_multiple_matches_on_one_line_and_literal_regex_metacharacters(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "main.py", "x = foo + foo\nvalue = a.b\n")

    repeated = search_code(root, "foo")
    literal_dot = search_code(root, ".", use_regex=False)
    regex_dot = search_code(root, r"a\.b", use_regex=True)

    assert [(match.column_start, match.column_end) for match in repeated.matches] == [(4, 7), (10, 13)]
    assert literal_dot.total_matches == 1
    assert literal_dot.matches[0].line == "value = a.b"
    assert regex_dot.total_matches == 1


def test_case_sensitive_and_case_insensitive_search(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "messages.txt", "Auth\nauth\nAUTH\n")

    sensitive = search_code(root, "auth", case_sensitive=True)
    insensitive = search_code(root, "auth", case_sensitive=False)

    assert [match.line_number for match in sensitive.matches] == [2]
    assert [match.line_number for match in insensitive.matches] == [1, 2, 3]


def test_regex_mode_is_explicit_and_invalid_regex_is_structured(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "routes.py", "GET /users\nPOST /users\n")

    result = search_code(root, r"^(GET|POST) /", use_regex=True)
    assert result.total_matches == 2
    error = _assert_error(lambda: search_code(root, "[" , use_regex=True), ToolErrorCode.INVALID_REGEX)
    assert "regular expression" in error.message


def test_search_supports_common_backend_text_and_unicode(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "app/main.py", "def handler():\n    return 'مرحبا'\n")
    _write_text(root, "server.ts", "export function handler(): string { return 'ok'; }\n")
    _write_text(root, "config.json", '{"handler": true}\n')
    _write_text(root, "schema.sql", "CREATE TABLE handler (id INTEGER);\n")
    _write_text(root, "README.md", "# handler\n")
    _write_text(root, "Dockerfile", "CMD handler\n")

    result = search_code(root, "handler", case_sensitive=False)

    assert result.total_matches == 6
    assert {match.relative_path for match in result.matches} == {
        "app/main.py",
        "server.ts",
        "config.json",
        "schema.sql",
        "README.md",
        "Dockerfile",
    }


def test_no_matches_are_distinct_from_truncated_search(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "main.py", "def main():\n    return 1\n")

    no_match = search_code(root, "does-not-exist")
    limited = search_code(root, "main", max_results=1)

    assert no_match.total_matches == 0
    assert not no_match.truncated
    assert limited.total_matches == 1
    assert limited.truncated
    assert limited.truncation_reason == "max_results"


def test_max_results_is_exact_and_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for index in range(3):
        _write_text(root, f"src/file_{index}.py", "target\ntarget\n")

    first = search_code(root, "target", max_results=4)
    second = search_code(root, "target", max_results=4)

    assert first.to_dict() == second.to_dict()
    assert len(first.matches) == 4
    assert first.total_matches == 4
    assert first.truncated
    assert first.truncation_reason == "max_results"


def test_max_file_bytes_skips_oversized_files_without_partial_results(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "small.py", "target\n")
    _write_text(root, "large.py", "target\n" * 10)

    result = search_code(root, "target", max_file_bytes=8)

    assert [match.relative_path for match in result.matches] == ["small.py"]
    assert result.files_skipped == 1
    assert "max_file_bytes" in result.skipped_reasons
    assert result.truncated
    assert result.truncation_reason == "max_file_bytes"


def test_invalid_utf8_is_skipped_explicitly(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "valid.py", "target\n")
    _write_bytes(root, "binary.bin", b"target\xff\xfe")

    result = search_code(root, "target")

    assert [match.relative_path for match in result.matches] == ["valid.py"]
    assert result.files_skipped == 1
    assert result.skipped_reasons == ("invalid_utf8",)
    assert not result.truncated


def test_project_scope_nested_directory_and_single_file(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "app/main.py", "target\n")
    _write_text(root, "tests/test_main.py", "target\n")
    _write_text(root, "README.md", "target\n")

    nested = search_code(root, "target", path="app")
    single = search_code(root, "target", path="tests\\test_main.py")

    assert [match.relative_path for match in nested.matches] == ["app/main.py"]
    assert [match.relative_path for match in single.matches] == ["tests/test_main.py"]
    assert nested.files_searched == 1
    assert single.files_searched == 1


def test_default_exclusions_and_hidden_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, ".env.example", "target\n")
    _write_text(root, "src/main.py", "target\n")
    _write_text(root, ".git/hidden.py", "target\n")
    _write_text(root, "node_modules/ignored.js", "target\n")
    _write_text(root, "__pycache__/ignored.pyc", "target\n")
    _write_text(root, ".venv/ignored.py", "target\n")

    result = search_code(root, "target")

    assert {match.relative_path for match in result.matches} == {".env.example", "src/main.py"}


def test_path_traversal_absolute_windows_and_unc_paths_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "app/main.py", "target\n")
    _write_text(tmp_path, "secret.py", "target\n")

    for requested in (
        "../secret.py",
        "../../secret.py",
        "app/../../secret.py",
        str(tmp_path / "secret.py"),
        r"C:\\Users\\Peter\\secret.py",
        r"\\\\server\\share\\secret.py",
    ):
        _assert_error(lambda requested=requested: search_code(root, "target", path=requested), ToolErrorCode.PATH_OUTSIDE_ROOT)


def test_missing_scope_directory_and_special_scope_errors(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "main.py", "target\n")
    _assert_error(lambda: search_code(root, "target", path="missing"), ToolErrorCode.PATH_NOT_FOUND)
    _assert_error(lambda: search_code(root, "target", path="main.py/child"), ToolErrorCode.PATH_NOT_FOUND)
    if hasattr(os, "mkfifo"):
        fifo = root / "events.pipe"
        try:
            os.mkfifo(fifo)
        except (AttributeError, NotImplementedError, OSError):
            pass
        else:
            _assert_error(lambda: search_code(root, "target", path="events.pipe"), ToolErrorCode.NOT_A_FILE)


def test_symlink_scope_and_tree_entries_are_rejected_or_skipped(tmp_path: Path) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    _write_text(root, "inside.py", "target\n")
    outside_file = _write_text(outside, "secret.py", "target\n")
    try:
        (root / "external.py").symlink_to(outside_file)
        (root / "external_dir").symlink_to(outside, target_is_directory=True)
        (root / "loop").symlink_to(root, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    result = search_code(root, "target")
    assert [match.relative_path for match in result.matches] == ["inside.py"]
    _assert_error(lambda: search_code(root, "target", path="external.py"), ToolErrorCode.PATH_OUTSIDE_ROOT)
    _assert_error(lambda: search_code(root, "target", path="external_dir"), ToolErrorCode.PATH_OUTSIDE_ROOT)


def test_invalid_arguments_and_bounds_are_structured(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "main.py", "target\n")

    _assert_error(lambda: SearchCodeTool().run({}), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: search_code(root, "   "), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: search_code(root, "target", max_results=0), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: search_code(root, "target", max_results=MAX_MAX_RESULTS + 1), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: search_code(root, "target", max_file_bytes=-1), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: search_code(root, "target", max_file_bytes=MAX_MAX_FILE_BYTES + 1), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: search_code(root, "target", case_sensitive=1), ToolErrorCode.INVALID_ARGUMENT)  # type: ignore[arg-type]
    _assert_error(lambda: search_code(root, "target", use_regex=1), ToolErrorCode.INVALID_ARGUMENT)  # type: ignore[arg-type]
    _assert_error(lambda: search_code(root, "target", path=123), ToolErrorCode.INVALID_ARGUMENT)  # type: ignore[arg-type]
    _assert_error(lambda: search_code(root, "x" * 4_097), ToolErrorCode.INVALID_ARGUMENT)


def test_tool_protocol_metadata_and_no_stdout_or_mutation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "project"
    root.mkdir()
    path = _write_text(root, "main.py", "target\n")
    before = path.stat()
    tool = SearchCodeTool()

    assert isinstance(tool, Tool)
    assert tool.name == "search_code"
    assert tool.metadata.name == "search_code"
    assert "project_root" in tool.metadata.input_schema["required"]
    assert "query" in tool.metadata.input_schema["required"]
    assert tool.run({"project_root": root, "query": "target"}).total_matches == 1
    after = path.stat()
    assert before.st_mtime_ns == after.st_mtime_ns
    assert capsys.readouterr().out == ""


def test_search_skips_generated_directory_when_scope_is_explicit(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_text(root, "node_modules/ignored.js", "target\n")

    result = search_code(root, "target", path="node_modules")

    assert result.total_matches == 0
    assert result.files_searched == 0
