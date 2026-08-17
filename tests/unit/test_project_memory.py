from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend_ai.agent.project_memory import (
    PROJECT_MEMORY_SCHEMA_VERSION,
    FactCategory,
    FactConfidence,
    FactEvidence,
    FactSource,
    FactStatus,
    ProjectMemory,
    ProjectMemoryClosedError,
    ProjectMemoryConflictError,
    ProjectMemoryLimits,
    ProjectMemoryLoadStatus,
    ProjectMemoryStore,
    ProjectMemoryValidationError,
)


def _evidence(source: FactSource = FactSource.PROJECT_CONTEXT, reference: str = "context") -> FactEvidence:
    return FactEvidence(source, reference, "bounded structured evidence", verified=True)


def test_project_identity_is_stable_and_project_scoped(tmp_path: Path) -> None:
    first = ProjectMemory.for_project(tmp_path / "project-a")
    repeat = ProjectMemory.for_project(tmp_path / "project-a")
    other = ProjectMemory.for_project(tmp_path / "project-b")
    assert first.project_id == repeat.project_id
    assert first.project_id != other.project_id
    assert first.project_root != other.project_root


def test_fact_insertion_snapshot_and_canonical_serialization() -> None:
    memory = ProjectMemory.for_project("/tmp/fodci-project")
    snapshot = memory.add_fact(
        category=FactCategory.FRAMEWORK,
        key="framework.name",
        value="Django",
        source=FactSource.PROJECT_CONTEXT,
        confidence=FactConfidence.VERIFIED,
        evidence=(_evidence(),),
    )
    assert snapshot.active_facts[0].value == "Django"
    assert snapshot.active_facts[0].status is FactStatus.ACTIVE
    encoded = snapshot.to_json()
    assert json.dumps(json.loads(encoded), ensure_ascii=False, sort_keys=True, separators=(",", ":")) == encoded
    assert memory.to_json() == encoded


def test_same_fact_merges_evidence_without_duplicate_records() -> None:
    memory = ProjectMemory.for_project("/tmp/fodci-project")
    memory.add_fact(category="TESTING", key="testing.framework", value="pytest", source="PROJECT_CONTEXT", confidence="OBSERVED", evidence=(_evidence(reference="context-1"),))
    snapshot = memory.add_fact(category="TESTING", key="testing.framework", value="pytest", source="VERIFICATION_RESULT", confidence="VERIFIED", evidence=(_evidence(FactSource.VERIFICATION_RESULT, "test-1"),))
    assert len(snapshot.active_facts) == 1
    assert snapshot.active_facts[0].confidence is FactConfidence.VERIFIED
    assert len(snapshot.active_facts[0].evidence) == 2


def test_lower_confidence_conflict_cannot_override_verified_fact() -> None:
    memory = ProjectMemory.for_project("/tmp/fodci-project")
    first = memory.add_fact(category=FactCategory.DATABASE, key="database.engine", value="PostgreSQL", source=FactSource.CONFIGURATION, confidence=FactConfidence.VERIFIED, evidence=(_evidence(FactSource.CONFIGURATION, "settings"),))
    snapshot = memory.add_fact(category=FactCategory.DATABASE, key="database.engine", value="MySQL", source=FactSource.STRUCTURED_TOOL_RESULT, confidence=FactConfidence.INFERRED, evidence=(_evidence(FactSource.STRUCTURED_TOOL_RESULT, "tool"),))
    assert snapshot.active_facts[0].value == "PostgreSQL"
    assert any(item.status is FactStatus.REJECTED for item in snapshot.conflicts)
    assert first.active_facts[0].fact_id == snapshot.active_facts[0].fact_id


def test_user_confirmed_correction_supersedes_old_fact_and_preserves_trace() -> None:
    memory = ProjectMemory.for_project("/tmp/fodci-project")
    memory.add_fact(category=FactCategory.DATABASE, key="database.engine", value="MySQL", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.OBSERVED, evidence=(_evidence(),))
    snapshot = memory.add_fact(category=FactCategory.DATABASE, key="database.engine", value="PostgreSQL", source=FactSource.USER_PROVIDED, confidence=FactConfidence.USER_CONFIRMED, evidence=(_evidence(FactSource.USER_PROVIDED, "user"),))
    assert snapshot.active_facts[0].value == "PostgreSQL"
    assert any(item.status is FactStatus.SUPERSEDED for item in snapshot.conflicts)


def test_invalidation_is_explicit_and_closed_memory_rejects_writes() -> None:
    memory = ProjectMemory.for_project("/tmp/fodci-project")
    snapshot = memory.add_fact(category=FactCategory.TESTING, key="testing.framework", value="pytest", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(_evidence(),))
    invalidated = memory.invalidate_fact(snapshot.active_facts[0].fact_id, reason="configuration was removed")
    assert invalidated.active_facts == ()
    assert any(item.status is FactStatus.INVALID for item in invalidated.facts)
    memory.close()
    with pytest.raises(ProjectMemoryClosedError):
        memory.add_fact(category=FactCategory.FRAMEWORK, key="framework.name", value="Django", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(_evidence(),))


def test_bounds_evict_and_reject_oversized_values() -> None:
    limits = ProjectMemoryLimits(max_facts=2, max_fact_value_length=16, max_total_memory_bytes=8_192, max_evidence_per_fact=2, max_metadata_size=128)
    memory = ProjectMemory.for_project("/tmp/fodci-project", limits=limits)
    for index in range(4):
        memory.add_fact(category=FactCategory.CONVENTION, key=f"convention.item{index}", value=f"value-{index}", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.OBSERVED, evidence=(_evidence(reference=str(index)),))
    snapshot = memory.snapshot()
    assert len(snapshot.facts) <= 2
    assert snapshot.evictions > 0
    with pytest.raises(ProjectMemoryValidationError):
        memory.add_fact(category=FactCategory.FRAMEWORK, key="framework.name", value="x" * 100, source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(_evidence(),))


def test_secrets_are_redacted_and_raw_logs_are_not_stored() -> None:
    memory = ProjectMemory.for_project("/tmp/fodci-project")
    snapshot = memory.add_fact(category=FactCategory.CONFIGURATION, key="configuration.environment", value={"uses_env": True, "API_KEY": "secret-value", "password": "secret-password"}, source=FactSource.CONFIGURATION, confidence=FactConfidence.VERIFIED, evidence=(FactEvidence(FactSource.CONFIGURATION, ".env", "DATABASE_PASSWORD=secret-password and token=abc"),))
    encoded = snapshot.to_json()
    assert "secret-value" not in encoded
    assert "secret-password" not in encoded
    assert "abc" not in encoded
    assert "uses_env" in encoded
    assert "DATABASE_PASSWORD" in encoded


def test_invalid_inputs_and_malformed_evidence_are_rejected() -> None:
    memory = ProjectMemory.for_project("/tmp/fodci-project")
    with pytest.raises(ProjectMemoryValidationError):
        memory.add_fact(category="NOT_A_CATEGORY", key="framework.name", value="Django", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(_evidence(),))
    with pytest.raises(ProjectMemoryValidationError):
        memory.add_fact(category=FactCategory.FRAMEWORK, key="../unsafe", value="Django", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(_evidence(),))
    with pytest.raises(ProjectMemoryValidationError):
        memory.add_fact(category=FactCategory.FRAMEWORK, key="framework.name", value="Django", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=())


def test_store_missing_save_reload_and_corruption(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    store = ProjectMemoryStore.for_project(root)
    missing = store.load()
    assert missing.status is ProjectMemoryLoadStatus.MEMORY_MISSING
    memory = store.empty()
    memory.add_fact(category=FactCategory.FRAMEWORK, key="framework.name", value="Django", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(_evidence(),))
    path = store.save(memory)
    assert path == root / ".fodci" / "project_memory.json"
    loaded = ProjectMemoryStore.for_project(root).load()
    assert loaded.status is ProjectMemoryLoadStatus.LOADED
    assert loaded.memory is not None
    assert loaded.memory.to_json() == memory.to_json()
    path.write_text("{not json", encoding="utf-8")
    corrupted = ProjectMemoryStore.for_project(root).load()
    assert corrupted.status is ProjectMemoryLoadStatus.MEMORY_CORRUPTED
    assert corrupted.memory is None


def test_store_detects_stale_writes_and_schema_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    first = ProjectMemoryStore.for_project(root)
    memory = first.empty()
    memory.add_fact(category=FactCategory.FRAMEWORK, key="framework.name", value="Django", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(_evidence(),))
    first.save(memory)
    second = ProjectMemoryStore.for_project(root)
    loaded = second.load()
    assert loaded.memory is not None
    loaded.memory.add_fact(category=FactCategory.TESTING, key="testing.framework", value="pytest", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(_evidence(),))
    second.save(loaded.memory)
    memory.add_fact(category=FactCategory.DATABASE, key="database.engine", value="PostgreSQL", source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(_evidence(),))
    with pytest.raises(ProjectMemoryConflictError):
        first.save(memory)
    payload = json.loads((root / ".fodci" / "project_memory.json").read_text(encoding="utf-8"))
    payload["schema_version"] = "99.0"
    (root / ".fodci" / "project_memory.json").write_text(json.dumps(payload), encoding="utf-8")
    invalid = ProjectMemoryStore.for_project(root).load()
    assert invalid.status is ProjectMemoryLoadStatus.MEMORY_INVALID
    assert invalid.memory is None
    assert PROJECT_MEMORY_SCHEMA_VERSION == "9.2"


def test_snapshot_does_not_expose_mutable_fact_values() -> None:
    memory = ProjectMemory.for_project("/tmp/fodci-project")
    snapshot = memory.add_fact(category=FactCategory.CONFIGURATION, key="configuration.flags", value={"safe": True}, source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(_evidence(),))
    with pytest.raises(TypeError):
        snapshot.active_facts[0].value["safe"] = False  # type: ignore[index]
