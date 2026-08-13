from __future__ import annotations

from pathlib import Path

import pytest

from backend_ai.config import Settings
from backend_ai.core import InvalidProjectRootError, ProjectContext, resolve_project_context


def test_default_project_root_is_current_working_directory(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path.resolve(), log_level="INFO")

    context = resolve_project_context(settings)

    assert isinstance(context, ProjectContext)
    assert context.root == tmp_path.resolve()
    assert context.root.is_absolute()


def test_explicit_project_root_is_normalized_and_not_scanned(tmp_path: Path) -> None:
    nested = tmp_path / "project"
    nested.mkdir()
    (nested / "secret.txt").write_text("content", encoding="utf-8")
    equivalent = nested / ".." / "project" / "."

    context = resolve_project_context(Settings(project_root=equivalent, log_level="INFO"))

    assert context.root == nested.resolve()
    assert not hasattr(context, "files")


def test_missing_project_root_fails_without_fallback(tmp_path: Path) -> None:
    missing = tmp_path / "missing-project"

    with pytest.raises(InvalidProjectRootError, match="Invalid project root") as error:
        resolve_project_context(Settings(project_root=missing, log_level="INFO"))

    assert str(missing) in str(error.value)


def test_file_project_root_is_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("not a project root", encoding="utf-8")

    with pytest.raises(InvalidProjectRootError, match="not a directory"):
        resolve_project_context(Settings(project_root=file_path, log_level="INFO"))
