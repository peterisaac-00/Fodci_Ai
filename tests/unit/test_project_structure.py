from __future__ import annotations

from pathlib import Path
import os

import pytest

from backend_ai.tools import (
    ProjectStructureResult,
    ProjectStructureTool,
    Tool,
    ToolError,
    ToolErrorCode,
    project_structure,
)


def _write(root: Path, relative: str, content: str = "x") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return path


def _assert_error(callable_object, code: ToolErrorCode) -> ToolError:
    with pytest.raises(ToolError) as raised:
        callable_object()
    assert raised.value.code == code
    return raised.value


def _names(items: tuple[object, ...]) -> set[str]:
    return {getattr(item, "name") for item in items}


def test_empty_project_is_low_confidence_and_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    first = project_structure(root)
    second = project_structure(root)

    assert isinstance(first, ProjectStructureResult)
    assert first.to_dict() == second.to_dict()
    assert first.project_type == "empty"
    assert first.confidence == "low"
    assert first.frameworks == ()
    assert first.languages == ()
    assert "No project files were discovered." in first.warnings


def test_generic_python_project_detects_languages_files_dirs_and_pip(tmp_path: Path) -> None:
    root = tmp_path / "python-project"
    root.mkdir()
    _write(root, "pyproject.toml", "[project]\nname='demo'\n")
    _write(root, "requirements.txt", "requests==2\n")
    _write(root, "src/main.py", "def main():\n    return 1\n")
    _write(root, "tests/test_main.py", "def test_main():\n    assert True\n")
    _write(root, "README.md", "# Demo\n")

    result = project_structure(root)

    assert result.project_type == "python"
    assert "Python" in {item.name for item in result.languages}
    assert {item.name for item in result.package_managers} == {"pip"}
    assert "pyproject.toml" in result.important_files
    assert "requirements.txt" in result.dependency_files
    assert "src" in result.source_directories
    assert "tests" in result.test_directories
    assert "src/main.py" in {item.name for item in result.entry_points}
    assert any(item.name == "generic tests" for item in result.test_frameworks)


def test_django_detection_requires_meaningful_evidence(tmp_path: Path) -> None:
    root = tmp_path / "django-project"
    root.mkdir()
    _write(root, "manage.py", "import django\n")
    _write(root, "requirements.txt", "Django>=5\n")
    _write(root, "config/settings.py", "DJANGO_SETTINGS_MODULE='config.settings'\n")

    result = project_structure(root)
    django = next(item for item in result.frameworks if item.name == "Django")

    assert django.confidence == "high"
    assert any("manage.py" in evidence for evidence in django.evidence)
    assert any("requirements.txt" in evidence for evidence in django.evidence)
    assert "manage.py" in {item.name for item in result.entry_points}


def test_fastapi_and_flask_detection_uses_dependency_or_source_evidence(tmp_path: Path) -> None:
    root = tmp_path / "python-frameworks"
    root.mkdir()
    _write(root, "pyproject.toml", "[project]\ndependencies=['fastapi','flask']\n")
    _write(root, "app/main.py", "from fastapi import FastAPI\napp = FastAPI()\n")
    _write(root, "app/flask_app.py", "from flask import Flask\napp = Flask(__name__)\n")

    result = project_structure(root)

    assert {item.name for item in result.frameworks} >= {"FastAPI", "Flask", "Python"}
    assert all(item.evidence for item in result.frameworks)


def test_node_express_and_typescript_package_managers(tmp_path: Path) -> None:
    root = tmp_path / "node-project"
    root.mkdir()
    _write(root, "package.json", '{"main":"src/server.js","dependencies":{"express":"4"}}')
    _write(root, "package-lock.json", "{}")
    _write(root, "tsconfig.json", "{}")
    _write(root, "pnpm-lock.yaml", "lockfileVersion: 9\n")
    _write(root, "src/server.js", "const express = require('express')\n")
    _write(root, "src/index.ts", "export const app: string = 'ok'\n")

    result = project_structure(root)

    assert result.project_type == "node"
    assert {item.name for item in result.frameworks} >= {"Node.js", "Express", "JavaScript", "TypeScript"}
    assert {item.name for item in result.package_managers} >= {"npm", "pnpm"}
    assert any(item.name == "src/server.js" and item.confidence == "high" for item in result.entry_points)
    assert any(item.name == "src/index.ts" for item in result.entry_points)
    assert "tsconfig.json" in result.config_files


def test_mixed_python_node_docker_compose_and_ci_are_detected(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    root.mkdir()
    _write(root, "pyproject.toml", "[project]\nname='mixed'\n")
    _write(root, "package.json", '{"dependencies":{"express":"4"}}')
    _write(root, "backend/main.py", "print('python')\n")
    _write(root, "frontend/app.tsx", "export default () => null\n")
    _write(root, "Dockerfile", "FROM python:3.12\n")
    _write(root, "compose.yaml", "services: {}\n")
    _write(root, ".github/workflows/test.yml", "name: test\n")

    result = project_structure(root)

    assert result.project_type == "mixed"
    assert {item.name for item in result.infrastructure} >= {"Docker", "Docker Compose", "CI"}
    assert {item.name for item in result.languages} >= {"Python", "TypeScript", "YAML"}
    assert {item.category for item in result.directories} >= {"source"}


def test_database_detection_uses_dependency_configuration_and_extensions(tmp_path: Path) -> None:
    root = tmp_path / "database-project"
    root.mkdir()
    _write(
        root,
        "requirements.txt",
        "psycopg2-binary\npymysql\nmariadb\npymongo\n",
    )
    _write(root, "app/db.py", "import sqlite3\npostgresql_url = 'postgresql://db'\n")
    (root / "data.sqlite3").write_bytes(b"SQLite format 3\x00")

    result = project_structure(root)

    assert {item.name for item in result.databases} >= {"PostgreSQL", "MySQL", "MariaDB", "MongoDB", "SQLite"}
    assert all(item.evidence for item in result.databases)


def test_testing_frameworks_and_directory_classification(tmp_path: Path) -> None:
    root = tmp_path / "tests-project"
    root.mkdir()
    _write(root, "requirements.txt", "pytest\n")
    _write(root, "package.json", '{"devDependencies":{"jest":"1","vitest":"1"}}')
    _write(root, "tests/test_api.py", "import unittest\n")
    _write(root, "specs/api.test.js", "describe('api', () => {})\n")
    _write(root, "src/api.py", "def api(): pass\n")

    result = project_structure(root)

    assert {item.name for item in result.test_frameworks} >= {"pytest", "unittest", "Jest", "Vitest", "generic tests"}
    assert {item.category for item in result.directories} >= {"tests", "source"}


def test_language_counts_cover_common_file_types(tmp_path: Path) -> None:
    root = tmp_path / "languages"
    root.mkdir()
    for relative in (
        "a.py", "b.js", "c.ts", "d.sql", "e.json", "f.yaml", "g.md", "h.html",
        "i.css", "j.sh", "Dockerfile", "Makefile",
    ):
        _write(root, relative)

    result = project_structure(root)
    counts = {item.name: item.files for item in result.languages}

    assert counts["Python"] == 1
    assert counts["JavaScript"] == 1
    assert counts["TypeScript"] == 1
    assert counts["SQL"] == 1
    assert counts["JSON"] == 1
    assert counts["YAML"] == 1
    assert counts["Markdown"] == 1
    assert counts["HTML"] == 1
    assert counts["CSS"] == 1
    assert counts["Shell"] == 2
    assert counts["Dockerfile"] == 1


def test_ambiguous_names_alone_do_not_claim_frameworks(tmp_path: Path) -> None:
    root = tmp_path / "ambiguous"
    root.mkdir()
    (root / "django").mkdir()
    _write(root, "flask.py", "print('not a framework import')\n")
    _write(root, "app.py", "print('generic')\n")

    result = project_structure(root)

    assert "Django" not in {item.name for item in result.frameworks}
    assert "Flask" not in {item.name for item in result.frameworks}
    assert result.project_type == "python"


def test_sensitive_files_are_classified_but_not_read_or_exposed(tmp_path: Path) -> None:
    root = tmp_path / "sensitive"
    root.mkdir()
    _write(root, ".env", "sensitive_value=do-not-read\n")
    _write(root, ".env.example", "sensitive_value=example\n")
    _write(root, "private.key", "private_material\n")
    _write(root, "README.md", "safe\n")

    result = project_structure(root)
    serialized = str(result.to_dict())

    assert ".env.example" in result.important_files
    assert "sensitive_value=do-not-read" not in serialized
    assert "private_material" not in serialized
    assert "Sensitive files were excluded" in " ".join(result.warnings)


def test_limits_are_reported_without_silent_incomplete_result(tmp_path: Path) -> None:
    root = tmp_path / "large"
    root.mkdir()
    for index in range(5):
        _write(root, f"src/file_{index}.py", "def value(): return 1\n")
    _write(root, "pyproject.toml", "[project]\nname='large'\n")

    result = project_structure(root, max_files=2)

    assert result.truncated
    assert result.truncation_reason == "max_files"
    assert any("incomplete" in warning for warning in result.warnings)


def test_structural_inspection_has_file_byte_and_count_limits(tmp_path: Path) -> None:
    root = tmp_path / "bounded"
    root.mkdir()
    _write(root, "pyproject.toml", "[project]\ndependencies=['django']\n" + "x" * 100)
    _write(root, "requirements.txt", "django\n")

    result = project_structure(root, max_file_bytes=10, max_inspected_files=1)

    assert result.project_type == "empty" or result.project_type == "python"
    assert any("byte limit" in warning or "max_inspected_files" in warning for warning in result.warnings)


def test_path_errors_and_tool_protocol_are_structured(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "print('ok')\n")
    tool = ProjectStructureTool()

    assert isinstance(tool, Tool)
    assert tool.name == "project_structure"
    assert tool.metadata.name == "project_structure"
    assert "project_root" in tool.metadata.input_schema["required"]
    assert tool.run({"project_root": root}).root == root.resolve()
    _assert_error(lambda: tool.run({}), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: project_structure(tmp_path / "missing"), ToolErrorCode.PATH_NOT_FOUND)
    file_root = root / "main.py"
    _assert_error(lambda: project_structure(file_root), ToolErrorCode.NOT_DIRECTORY)
    _assert_error(lambda: project_structure(root, max_file_bytes=-1), ToolErrorCode.INVALID_ARGUMENT)
    _assert_error(lambda: project_structure(root, max_inspected_files=0), ToolErrorCode.INVALID_ARGUMENT)


def test_symlinked_entries_are_not_used_and_result_is_read_only(tmp_path: Path) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    target = _write(outside, "secret.py", "import django\n")
    _write(root, "main.py", "print('ok')\n")
    try:
        (root / "external.py").symlink_to(target)
        (root / "external_dir").symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    before = (root / "main.py").stat()
    result = project_structure(root)
    after = (root / "main.py").stat()

    assert "Django" not in {item.name for item in result.frameworks}
    assert before.st_mtime_ns == after.st_mtime_ns


def test_no_stdout_and_no_target_project_imports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write(root, "main.py", "import os\n")

    project_structure(root)

    assert capsys.readouterr().out == ""
