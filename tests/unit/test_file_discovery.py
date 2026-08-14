from __future__ import annotations

from pathlib import Path
import os

import pytest

from backend_ai.tools import (
    FileDiscoveryResult,
    ListFilesTool,
    Tool,
    ToolError,
    ToolErrorCode,
    list_files,
)


def _write(root: Path, relative_path: str, content: str = "content") -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_empty_project_returns_structured_empty_result(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    result = list_files(root)

    assert isinstance(result, FileDiscoveryResult)
    assert result.root == root.resolve()
    assert result.files == ()
    assert result.directories == ()
    assert result.total_files == 0
    assert result.total_directories == 0
    assert not result.truncated
    assert result.to_dict()["root"] == str(root.resolve())


def test_nested_discovery_returns_relative_paths_and_cheap_metadata(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "manage.py", "123")
    _write(root, "app/views.py", "view")
    _write(root, "app/__init__.py", "init")
    _write(root, "tests/test_api.py", "test")

    result = list_files(root)

    assert [item.relative_path for item in result.files] == [
        "app/__init__.py",
        "app/views.py",
        "manage.py",
        "tests/test_api.py",
    ]
    assert [item.relative_path for item in result.directories] == ["app", "tests"]
    assert result.total_files == 4
    assert result.total_directories == 2
    views = next(item for item in result.files if item.relative_path == "app/views.py")
    assert views.name == "views.py"
    assert views.extension == ".py"
    assert views.size == 4
    assert all("\\" not in item.relative_path for item in result.files)


def test_ordering_is_deterministic_across_repeated_runs(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for name in ("z.py", "A.py", "sub/B.py", "sub/a.py", "sub/z.py"):
        _write(root, name)

    first = list_files(root).to_dict()
    second = list_files(root).to_dict()

    assert first == second


def test_default_ignore_rules_exclude_generated_directories_but_keep_hidden_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, ".env.example")
    _write(root, ".gitignore")
    _write(root, ".dockerignore")
    _write(root, "src/main.py")
    for directory in (
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".eggs",
    ):
        _write(root, f"{directory}/ignored.txt")

    result = list_files(root)
    files = {item.relative_path for item in result.files}
    directories = {item.relative_path for item in result.directories}

    assert {".env.example", ".gitignore", ".dockerignore", "src/main.py"} <= files
    assert all(not path.startswith(tuple(f"{name}/" for name in (".git", "node_modules", ".venv", "dist"))) for path in files)
    assert ".git" not in directories
    assert "src" in directories


def test_include_hidden_false_excludes_dotfiles_but_default_keeps_them(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, ".env.example")
    _write(root, "visible.py")

    included = list_files(root)
    excluded = list_files(root, include_hidden=False)

    assert ".env.example" in {item.relative_path for item in included.files}
    assert ".env.example" not in {item.relative_path for item in excluded.files}
    assert "visible.py" in {item.relative_path for item in excluded.files}


def test_custom_ignore_rules_extend_defaults(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "generated/result.py")
    _write(root, "src/main.py")

    result = list_files(root, ignored_directories=("generated",))

    assert [item.relative_path for item in result.files] == ["src/main.py"]


def test_limits_are_explicit_and_report_truncation(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for index in range(5):
        _write(root, f"files/{index}.txt")
    _write(root, "level1/level2/file.py")

    by_files = list_files(root, max_files=2)
    by_directories = list_files(root, max_directories=1)
    by_depth = list_files(root, max_depth=1)

    assert by_files.truncated
    assert by_files.truncation_reason == "max_files"
    assert len(by_files.files) == 2
    assert by_directories.truncated
    assert by_directories.truncation_reason == "max_directories"
    assert len(by_directories.directories) == 1
    assert by_depth.truncated
    assert by_depth.truncation_reason == "max_depth"
    assert all("/" not in item.relative_path for item in by_depth.files)


@pytest.mark.parametrize(
    ("argument", "code"),
    [
        ("missing", ToolErrorCode.PATH_NOT_FOUND),
        ("file", ToolErrorCode.NOT_DIRECTORY),
    ],
)
def test_invalid_roots_return_structured_errors(
    tmp_path: Path,
    argument: str,
    code: ToolErrorCode,
) -> None:
    if argument == "missing":
        root = tmp_path / "missing"
    else:
        root = _write(tmp_path, "file", "not a directory")

    with pytest.raises(ToolError) as raised:
        list_files(root)

    assert raised.value.code == code
    assert raised.value.to_dict()["code"] == code.value
    assert raised.value.to_dict()["path"] is not None


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ToolError, match="explicit 'project_root'"):
        ListFilesTool().run({})
    with pytest.raises(ToolError, match="must not be empty"):
        list_files("   ")
    with pytest.raises(ToolError, match="max_files"):
        list_files(Path.cwd(), max_files=-1)
    with pytest.raises(ToolError, match="max_depth"):
        list_files(Path.cwd(), max_depth=True)  # type: ignore[arg-type]
    with pytest.raises(ToolError, match="max_directories"):
        list_files(Path.cwd(), max_directories=-1)
    with pytest.raises(ToolError, match="not a string"):
        list_files(Path.cwd(), ignored_directories="build")


def test_tool_metadata_and_runtime_protocol_are_available(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = ListFilesTool()

    assert isinstance(tool, Tool)
    assert tool.name == "list_files"
    assert tool.metadata.name == "list_files"
    assert "project_root" in tool.metadata.input_schema["required"]
    assert tool.run({"project_root": root}).root == root.resolve()


def test_symlink_files_directories_and_loops_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    _write(outside, "secret.txt", "secret")
    _write(root, "inside.txt", "inside")

    try:
        (root / "outside_link").symlink_to(outside, target_is_directory=True)
        (root / "file_link").symlink_to(outside / "secret.txt")
        (root / "loop").symlink_to(root, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    result = list_files(root)
    paths = {item.relative_path for item in result.files}
    directories = {item.relative_path for item in result.directories}

    assert paths == {"inside.txt"}
    assert "outside_link" not in directories
    assert "loop" not in directories


def test_permission_errors_are_structured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "project"
    blocked = root / "blocked"
    root.mkdir()
    blocked.mkdir()
    _write(blocked, "file.py")

    import backend_ai.tools.filesystem as filesystem_module

    original_scandir = filesystem_module.os.scandir

    def deny_blocked(path: object):
        if Path(path) == blocked:
            raise PermissionError("denied for test")
        return original_scandir(path)

    monkeypatch.setattr(filesystem_module.os, "scandir", deny_blocked)

    with pytest.raises(ToolError) as raised:
        list_files(root)

    assert raised.value.code == ToolErrorCode.PERMISSION_DENIED


def test_moderate_synthetic_tree_is_discovered_with_a_bound(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for directory_index in range(10):
        for file_index in range(20):
            _write(root, f"service_{directory_index}/file_{file_index}.py")

    result = list_files(root, max_files=200, max_directories=20)

    assert result.total_files == 200
    assert result.total_directories == 10
    assert not result.truncated


def test_path_normalization_does_not_escape_requested_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "inside.txt")

    result = list_files(root / "nested" / "..")

    assert result.root == root.resolve()
    assert [item.relative_path for item in result.files] == ["inside.txt"]
