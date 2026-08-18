from pathlib import Path

import pytest

from backend_ai.agent.codebase_understanding import (
    CodebaseUnderstanding,
    CodebaseUnderstandingBuilder,
    UnderstandingCompleteness,
    UnderstandingConfidence,
)
from backend_ai.tools.project_structure import project_structure
from backend_ai.tools.read_file import read_file
from backend_ai.tools.search_code import search_code


def _write(root: Path, relative: str, content: str = "x") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return path


def _python_backend(root: Path) -> None:
    _write(root, "pyproject.toml", "[project]\nname='demo'\ndependencies=['fastapi']\n")
    _write(root, "requirements.txt", "fastapi\npytest\n")
    _write(root, "app/main.py", "from fastapi import FastAPI\nfrom app.auth import login\n\napp = FastAPI()\n\nclass ApiServer:\n    def handle(self):\n        return login('user')\n")
    _write(root, "app/auth.py", "def login(user: str) -> str:\n    return user\n\nclass AuthService:\n    pass\n")
    _write(root, "app/db.py", "import sqlite3\n")
    _write(root, "tests/test_auth.py", "from app.auth import login\n\ndef test_login():\n    assert login('user')\n")


def test_empty_and_unknown_project_are_explicitly_uncertain(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    result = CodebaseUnderstandingBuilder().build("understand repository", root)

    assert isinstance(result, CodebaseUnderstanding)
    assert result.project_type == "empty"
    assert result.confidence is UnderstandingConfidence.UNKNOWN
    assert result.completeness is UnderstandingCompleteness.COMPLETE
    assert result.relevant_files == ()
    assert result.evidence == ()
    assert "No project files were discovered." in result.warnings


def test_python_framework_entry_point_and_architecture_evidence(tmp_path: Path) -> None:
    root = tmp_path / "python"
    root.mkdir()
    _python_backend(root)

    result = CodebaseUnderstandingBuilder().build("add authentication endpoint", root)

    assert result.project_type == "python"
    assert "Python" in result.frameworks
    assert "FastAPI" in result.frameworks
    assert "app/main.py" in result.entry_points
    assert "app" in result.important_directories
    assert result.architecture
    assert all(item.evidence for item in result.architecture)
    assert result.confidence in {UnderstandingConfidence.HIGH, UnderstandingConfidence.MEDIUM}


def test_symbol_reference_and_dependency_analysis_is_bounded_and_evidence_backed(tmp_path: Path) -> None:
    root = tmp_path / "python"
    root.mkdir()
    _python_backend(root)

    result = CodebaseUnderstandingBuilder(max_symbols=4, max_references=8, max_dependencies=8).build("fix auth bug", root)

    names = {item.name for item in result.symbols}
    assert {"login", "AuthService"} <= names
    assert any(item.target == "app.auth" and item.relation == "imports" for item in result.references)
    assert any(item.target == "sqlite3" and item.kind == "module_import" for item in result.dependencies)
    assert len(result.symbols) <= 4
    assert all(item.evidence for item in result.symbols)
    assert all(item.evidence for item in result.references)
    assert all(item.evidence for item in result.dependencies)


def test_node_express_typescript_symbols_and_imports(tmp_path: Path) -> None:
    root = tmp_path / "node"
    root.mkdir()
    _write(root, "package.json", '{"main":"src/server.js","dependencies":{"express":"4"}}')
    _write(root, "src/server.js", "const express = require('express');\nclass Server {}\nfunction createApp() {}\n")
    _write(root, "src/routes.ts", "import { createApp } from './server';\nexport function route() { return createApp(); }\n")

    result = CodebaseUnderstandingBuilder().build("add api endpoint", root)

    assert result.project_type == "node"
    assert {"Node.js", "Express", "JavaScript", "TypeScript"} <= set(result.frameworks)
    assert "src/server.js" in result.entry_points
    assert any(item.name == "Server" and item.kind == "class" for item in result.symbols)
    assert any(item.name == "route" and item.kind == "function" for item in result.symbols)
    assert any(item.target == "express" and item.relation == "imports" for item in result.references)
    assert any(item.target == "./server" and item.relation == "imports" for item in result.references)
    assert any(item.target == "express" for item in result.dependencies)


def test_task_relevance_prioritizes_matching_backend_files_and_preserves_reasons(tmp_path: Path) -> None:
    root = tmp_path / "ranking"
    root.mkdir()
    _write(root, "app/auth.py", "def authenticate(): pass\n")
    _write(root, "app/database.py", "def connect(): pass\n")
    _write(root, "docs/auth.md", "authentication notes\n")
    _write(root, "tests/test_auth.py", "def test_auth(): pass\n")

    result = CodebaseUnderstandingBuilder().build("fix auth bug", root)
    paths = [item.path for item in result.relevant_files]

    assert paths
    assert paths.index("app/auth.py") < paths.index("docs/auth.md")
    auth = next(item for item in result.relevant_files if item.path == "app/auth.py")
    assert auth.relevance in {"high", "medium"}
    assert auth.reasons
    assert auth.evidence


def test_unicode_and_source_files_are_read_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "unicode"
    root.mkdir()
    path = _write(root, "app/main.py", "def مرحبا():\n    return 'أهلاً'\n")
    before = path.read_bytes()

    result = CodebaseUnderstandingBuilder().build("inspect Arabic backend", root)

    assert any(item.name == "مرحبا" for item in result.symbols)
    assert path.read_bytes() == before
    assert "مرحبا" in result.compact_summary()


def test_evidence_and_serialization_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "deterministic"
    root.mkdir()
    _python_backend(root)

    first = CodebaseUnderstandingBuilder().build("fix auth bug", root)
    second = CodebaseUnderstandingBuilder().build("fix auth bug", root)

    assert first.to_dict() == second.to_dict()
    assert first.compact_summary() == second.compact_summary()
    assert first.evidence
    assert all(item.path for item in first.evidence)
    assert all(item.detail for item in first.evidence)


def test_truncation_is_explicit_and_limits_are_enforced(tmp_path: Path) -> None:
    root = tmp_path / "bounded"
    root.mkdir()
    for index in range(8):
        _write(root, f"app/module_{index}.py", f"def function_{index}(): return {index}\n")

    result = CodebaseUnderstandingBuilder(max_files=2, max_inspected_files=1).build("inspect modules", root)

    assert result.truncated
    assert result.completeness is UnderstandingCompleteness.PARTIAL
    assert result.truncation_reason
    with pytest.raises(ValueError):
        CodebaseUnderstandingBuilder(max_symbols=0)
    with pytest.raises(ValueError):
        CodebaseUnderstandingBuilder(max_files=10_000)


def test_incremental_tool_results_merge_new_evidence_without_rescan(tmp_path: Path) -> None:
    root = tmp_path / "updates"
    root.mkdir()
    _write(root, "app/main.py", "def existing(): pass\n")
    understanding = CodebaseUnderstandingBuilder().build("inspect app", root)
    original_symbols = understanding.symbols

    new_path = _write(root, "app/new_service.py", "def new_service(): pass\n")
    search_result = search_code(root, "new_service", max_results=8)
    updated = CodebaseUnderstandingBuilder().update_from_tool_result(understanding, "search_code", search_result)
    assert "app/new_service.py" in {item.path for item in updated.relevant_files}

    read_result = read_file(root, new_path.relative_to(root))
    updated = CodebaseUnderstandingBuilder().update_from_tool_result(updated, "read_file", read_result)
    assert len(updated.symbols) > len(original_symbols)
    assert any(item.name == "new_service" for item in updated.symbols)
    assert len(updated.evidence) >= len(understanding.evidence)

    structure_result = project_structure(root)
    updated_again = CodebaseUnderstandingBuilder().update_from_tool_result(updated, "project_structure", structure_result)
    assert updated_again.root == root.resolve()
    assert updated_again.important_files == updated.important_files
