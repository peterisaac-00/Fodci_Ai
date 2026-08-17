from __future__ import annotations

from pathlib import Path

from backend_ai.agent.project_memory import (
    FactCategory,
    FactConfidence,
    FactEvidence,
    FactSource,
    ProjectMemoryLoadStatus,
    ProjectMemoryStore,
)
from backend_ai.tools.project_context import project_context


def test_cross_task_persistence_and_project_isolation(tmp_path: Path) -> None:
    project_a = tmp_path / "project-a"
    (project_a / "app").mkdir(parents=True)
    (project_a / "tests").mkdir()
    (project_a / "manage.py").write_text("import django\n", encoding="utf-8")
    (project_a / "pyproject.toml").write_text(
        "[project]\ndependencies=['Django>=5', 'psycopg2-binary', 'pytest']\n",
        encoding="utf-8",
    )
    (project_a / "app" / "settings.py").write_text("DATABASES = {'default': {'ENGINE': 'postgresql'}}\n", encoding="utf-8")
    (project_a / "tests" / "test_auth.py").write_text("import pytest\n", encoding="utf-8")
    (project_a / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (project_a / ".env").write_text("DATABASE_PASSWORD=do-not-persist\nAPI_KEY=secret-value\n", encoding="utf-8")

    context = project_context(project_a)
    store_a = ProjectMemoryStore.for_project(project_a)
    assert store_a.load().status is ProjectMemoryLoadStatus.MEMORY_MISSING
    task_a_memory = store_a.empty()
    task_a_memory.add_project_context(context)
    task_a_memory.add_fact(
        category=FactCategory.TESTING,
        key="testing.framework",
        value="pytest",
        source=FactSource.CONFIGURATION,
        confidence=FactConfidence.VERIFIED,
        evidence=(FactEvidence(FactSource.CONFIGURATION, "pytest.ini", "pytest configuration exists", verified=True),),
    )
    task_a_memory.add_fact(
        category=FactCategory.AUTHENTICATION,
        key="auth.mechanism",
        value="JWT",
        source=FactSource.USER_PROVIDED,
        confidence=FactConfidence.USER_CONFIRMED,
        evidence=(FactEvidence(FactSource.USER_PROVIDED, "task-a", "user confirmed the project uses JWT", verified=True),),
    )
    store_a.save(task_a_memory)
    assert (project_a / ".fodci" / "project_memory.json").is_file()

    # Task B starts later and loads the same persistent project memory.
    task_b_result = ProjectMemoryStore.for_project(project_a).load()
    assert task_b_result.status is ProjectMemoryLoadStatus.LOADED
    assert task_b_result.memory is not None
    task_b_snapshot = task_b_result.memory.snapshot()
    values = {(fact.category, fact.key, fact.value) for fact in task_b_snapshot.active_facts}
    assert (FactCategory.FRAMEWORK, "framework.name", "Django") in values
    assert (FactCategory.DATABASE, "database.name", "PostgreSQL") in values
    assert (FactCategory.TESTING, "testing.framework", "pytest") in values
    assert (FactCategory.AUTHENTICATION, "auth.mechanism", "JWT") in values

    # A separate project receives a separate identity and starts empty.
    project_b = tmp_path / "project-b"
    project_b.mkdir()
    project_b_result = ProjectMemoryStore.for_project(project_b).load()
    assert project_b_result.status is ProjectMemoryLoadStatus.MEMORY_MISSING
    assert project_b_result.memory is None
    assert ProjectMemoryStore.for_project(project_a).identity.project_id != ProjectMemoryStore.for_project(project_b).identity.project_id

    persisted = (project_a / ".fodci" / "project_memory.json").read_text(encoding="utf-8")
    assert "do-not-persist" not in persisted
    assert "secret-value" not in persisted
    assert "DATABASE_PASSWORD" not in persisted


def test_project_context_update_is_bounded_and_does_not_store_raw_files(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("print('safe')\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=never-store\n", encoding="utf-8")
    context = project_context(root)
    memory = ProjectMemoryStore.for_project(root).empty()
    snapshot = memory.add_project_context(context)
    assert snapshot.identity.project_root == str(root.resolve())
    assert all("print('safe')" not in snapshot.to_json() for _ in (0,))
    assert "never-store" not in snapshot.to_json()
