from __future__ import annotations

from pathlib import Path

import pytest

from backend_ai.tools import (
    ProjectContext,
    ProjectContextBuilder,
    ProjectContextTool,
    Tool,
    ToolError,
    ToolErrorCode,
    project_context,
)


def _write(root: Path, relative: str, content: str = "x") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _assert_error(callable_object, code: ToolErrorCode) -> None:
    with pytest.raises(ToolError) as raised:
        callable_object()
    assert raised.value.code == code


def test_context_is_immutable_serializable_and_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "pyproject.toml", "[project]\ndependencies=['fastapi']\n")
    _write(root, "app/main.py", "from fastapi import FastAPI\napp=FastAPI()\n")
    _write(root, "tests/test_api.py", "def test_api(): pass\n")

    first = project_context(root)
    second = project_context(root)

    assert isinstance(first, ProjectContext)
    assert first.to_dict() == second.to_dict()
    assert first.project_type == "python"
    assert first.stack_summary == "Python + FastAPI"
    assert first.completeness == "complete"
    assert not first.truncated
    assert first.project_files == ("app/main.py", "pyproject.toml", "tests/test_api.py")
    with pytest.raises(AttributeError):
        first.project_type = "node"  # type: ignore[misc]


def test_context_preserves_structural_categories_and_entry_point_evidence(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "requirements.txt", "pytest\n")
    _write(root, "src/server.py", "def serve(): pass\n")
    _write(root, "tests/test_server.py", "def test_server(): pass\n")
    _write(root, "docs/architecture.md", "# Architecture\n")
    _write(root, ".env.example", "TOKEN=example\n")

    context = ProjectContextBuilder().build(root)

    assert context.source_directories == ("src",)
    assert context.test_directories == ("tests",)
    assert context.documentation_directories == ("docs",)
    assert context.config_files == (".env.example",)
    assert context.dependency_files == ("requirements.txt",)
    assert ".env.example" in context.important_files
    entry = next(item for item in context.entry_points if item.name == "src/server.py")
    assert entry.confidence == "medium"
    assert entry.evidence
    assert any("requirements.txt" in evidence for evidence in context.evidence)


def test_stack_summary_is_evidence_derived_and_does_not_hallucinate(tmp_path: Path) -> None:
    root = tmp_path / "ambiguous"
    root.mkdir()
    (root / "django").mkdir()
    _write(root, "flask.py", "print('plain file')\n")
    _write(root, "README.md", "generic project\n")

    context = project_context(root)

    assert context.project_type == "python"
    assert context.stack_summary == "Python"
    assert not any(item.name in {"Django", "Flask"} for item in context.frameworks)


def test_sensitive_content_never_enters_context(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    hidden_name = "hidden_" + "payload"
    key_content = "key" + "_material"
    _write(root, ".env", hidden_name + "=not-for-context\n")
    _write(root, "private.key", key_content + "\n")
    _write(root, "pyproject.toml", "[project]\nname='safe'\n")

    context = project_context(root)
    serialized = str(context.to_dict())

    assert "not-for-context" not in serialized
    assert "_material" not in serialized
    assert any("Sensitive files" in warning for warning in context.warnings)


def test_partial_discovery_is_explicitly_preserved(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for index in range(5):
        _write(root, f"src/file_{index}.py", "print('x')\n")

    context = project_context(root, max_files=2)

    assert context.truncated
    assert context.truncation_reason == "max_files"
    assert context.completeness == "partial"
    assert any("partial" in warning.lower() for warning in context.warnings)


def test_targeted_inspection_limit_is_preserved_as_partial(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "pyproject.toml", "[project]\ndependencies=['fastapi']\n")
    _write(root, "requirements.txt", "pytest\n")
    _write(root, "app/main.py", "from fastapi import FastAPI\n")

    context = project_context(root, max_inspected_files=1)

    assert context.truncated
    assert context.truncation_reason == "max_inspected_files"
    assert context.completeness == "partial"
    assert any("max_inspected_files" in warning for warning in context.warnings)


def test_tool_protocol_metadata_and_no_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")
    tool = ProjectContextTool()

    assert isinstance(tool, Tool)
    assert tool.name == "project_context"
    assert tool.metadata.name == "project_context"
    assert tool.metadata.input_schema["required"] == ["project_root"]
    assert tool.run({"project_root": root}).root == root.resolve()
    assert capsys.readouterr().out == ""


def test_invalid_roots_and_arguments_use_shared_errors(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")

    _assert_error(lambda: ProjectContextTool().run({}), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: project_context(tmp_path / "missing"), ToolErrorCode.PATH_NOT_FOUND)
    _assert_error(lambda: project_context(root / "main.py"), ToolErrorCode.NOT_DIRECTORY)
    _assert_error(lambda: project_context(root, max_inspected_files=0), ToolErrorCode.INVALID_ARGUMENT)


def test_context_does_not_expose_absolute_file_paths_except_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "src/main.py", "print('ok')\n")

    context = project_context(root)
    serialized = context.to_dict()

    assert serialized["root"] == str(root.resolve())
    assert all(not Path(path).is_absolute() for path in serialized["project_files"])
    assert all(not Path(path).is_absolute() for path in serialized["important_files"])
