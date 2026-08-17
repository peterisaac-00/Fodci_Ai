from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend_ai.agent.long_term_memory import (
    LONG_TERM_MEMORY_SCHEMA_VERSION,
    LongTermMemory,
    LongTermMemoryCategory,
    LongTermMemoryClosedError,
    LongTermMemoryConfidence,
    LongTermMemoryConflictError,
    LongTermMemoryLimits,
    LongTermMemoryLoadStatus,
    LongTermMemorySource,
    LongTermMemoryStatus,
    LongTermMemoryStore,
    LongTermMemoryValidationError,
)


class Clock:
    def __init__(self) -> None:
        self.index = 0

    def __call__(self) -> str:
        self.index += 1
        return f"2026-08-17T00:00:{self.index:02d}Z"


def _memory(*, limits: LongTermMemoryLimits | None = None) -> LongTermMemory:
    return LongTermMemory(limits=limits, clock=Clock())


def _add(memory: LongTermMemory, content: str = "Django authentication uses explicit token validation", *, category: LongTermMemoryCategory = LongTermMemoryCategory.SOLUTION, topic: str | None = None):
    metadata = {"topic": topic} if topic is not None else {}
    return memory.add(content=content, category=category, source=LongTermMemorySource.VERIFIED_TASK, confidence=LongTermMemoryConfidence.VERIFIED, metadata=metadata)


def test_entry_creation_validation_and_immutable_metadata() -> None:
    entry = _add(_memory(), metadata if False else "Django authentication uses explicit token validation")
    assert entry.entry_id.startswith("ltm-")
    assert entry.category is LongTermMemoryCategory.SOLUTION
    assert entry.status is LongTermMemoryStatus.ACTIVE
    assert entry.access_count == 0
    memory = _memory()
    stored = memory.add(content="Use pytest markers for bounded integration tests", category="pattern", source="USER_PROVIDED", confidence="USER_CONFIRMED", metadata={"tags": ["pytest", "integration"]})
    with pytest.raises(TypeError):
        stored.metadata["tags"] = ("changed",)  # type: ignore[index]


def test_crud_and_explicit_lifecycle() -> None:
    memory = _memory()
    entry = _add(memory)
    assert memory.get(entry.entry_id, track_access=False) == entry
    updated = memory.update(entry.entry_id, content="Django authentication requires explicit token validation", status="archived")
    assert updated.status is LongTermMemoryStatus.ARCHIVED
    assert memory.list(status="archived") == (updated,)
    assert memory.delete(entry.entry_id) is True
    assert memory.get(entry.entry_id, track_access=False) is None
    assert memory.delete(entry.entry_id) is False
    with pytest.raises(LongTermMemoryValidationError):
        memory.update(entry.entry_id, content="missing entry")


def test_search_category_limit_ranking_and_access_tracking() -> None:
    memory = _memory()
    low = memory.add(content="Django authentication overview", category="knowledge", source="STRUCTURED_OBSERVATION", confidence="OBSERVED", metadata={"topic": "authentication"})
    high = memory.add(content="Django authentication verified solution with token validation", category="solution", source="VERIFIED_TASK", confidence="USER_CONFIRMED", metadata={"topic": "authentication"})
    memory.add(content="Database migrations should be reviewed", category="lesson", source="VERIFIED_TASK", confidence="VERIFIED", metadata={"topic": "database"})
    results = memory.search("Django authentication", category="solution", limit=1)
    assert [item.entry_id for item in results] == [high.entry_id]
    assert results[0].access_count == 1
    assert results[0].last_accessed_at is not None
    assert low.access_count == 0
    assert memory.list(category="solution")[0].entry_id == high.entry_id
    with pytest.raises(LongTermMemoryValidationError):
        memory.search("Django", limit=0)
    with pytest.raises(LongTermMemoryValidationError):
        memory.search("   ")


def test_conflicting_topics_are_preserved_and_marked() -> None:
    memory = _memory()
    first = _add(memory, "Use PostgreSQL for the service", category=LongTermMemoryCategory.KNOWLEDGE, topic="database")
    second = _add(memory, "Use SQLite for the service", category=LongTermMemoryCategory.KNOWLEDGE, topic="database")
    assert first.entry_id != second.entry_id
    assert memory.get(first.entry_id, track_access=False).status is LongTermMemoryStatus.CONFLICTED  # type: ignore[union-attr]
    assert memory.get(second.entry_id, track_access=False).status is LongTermMemoryStatus.CONFLICTED  # type: ignore[union-attr]
    assert second.entry_id in memory.get(first.entry_id, track_access=False).conflict_with  # type: ignore[union-attr]
    assert len(memory.list(status="conflicted")) == 2


def test_closed_memory_rejects_explicit_writes_but_snapshot_is_available() -> None:
    memory = _memory()
    _add(memory)
    snapshot = memory.close()
    assert snapshot.entries
    with pytest.raises(LongTermMemoryClosedError):
        memory.add(content="A later lesson", category="lesson", source="VERIFIED_TASK", confidence="VERIFIED")
    with pytest.raises(LongTermMemoryClosedError):
        memory.update(snapshot.entries[0].entry_id, status="archived")
    with pytest.raises(LongTermMemoryClosedError):
        memory.delete(snapshot.entries[0].entry_id)
    with pytest.raises(LongTermMemoryClosedError):
        memory.search("Django")


def test_bounds_are_strict_and_failed_mutations_are_atomic() -> None:
    limits = LongTermMemoryLimits(max_memories=2, max_content_length=20, max_metadata_size=96, max_total_memory_bytes=8_192)
    memory = _memory(limits=limits)
    _add(memory, "one")
    _add(memory, "two")
    before = memory.snapshot()
    with pytest.raises(LongTermMemoryValidationError):
        _add(memory, "this content is intentionally too long")
    assert memory.snapshot() == before
    with pytest.raises(LongTermMemoryValidationError):
        memory.add(content="three", category="lesson", source="VERIFIED_TASK", confidence="VERIFIED", metadata={"large": "x" * 200})
    assert memory.snapshot() == before


def test_redaction_removes_secrets_from_entry_and_json() -> None:
    memory = _memory()
    entry = memory.add(
        content="Use API_KEY=super-secret and password=hidden only as a warning",
        category="warning",
        source="USER_PROVIDED",
        confidence="USER_CONFIRMED",
        metadata={"api_key": "secret-value", "safe": "keep-this"},
    )
    encoded = memory.to_json()
    assert "super-secret" not in entry.content
    assert "hidden" not in entry.content
    assert "secret-value" not in encoded
    assert "keep-this" in encoded


def test_persistence_missing_reload_corruption_and_future_schema(tmp_path: Path) -> None:
    path = tmp_path / "global" / "long_term_memory.json"
    store = LongTermMemoryStore(path, limits=LongTermMemoryLimits())
    assert store.load().status is LongTermMemoryLoadStatus.MEMORY_MISSING
    memory = store.empty(clock=Clock())
    entry = _add(memory)
    store.save(memory)
    loaded = LongTermMemoryStore(path).load(clock=Clock())
    assert loaded.status is LongTermMemoryLoadStatus.LOADED
    assert loaded.memory is not None
    assert loaded.memory.get(entry.entry_id, track_access=False).content == entry.content  # type: ignore[union-attr]
    path.write_text("{not json", encoding="utf-8")
    assert LongTermMemoryStore(path).load().status is LongTermMemoryLoadStatus.MEMORY_CORRUPTED
    store.save(memory) if False else None
    payload = {"format": "fodci.long_term_memory", "schema_version": "99.0", "entries": [], "status": "LOADED", "sequence": 0, "warnings": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    invalid = LongTermMemoryStore(path).load()
    assert invalid.status is LongTermMemoryLoadStatus.MEMORY_INVALID
    assert invalid.memory is None
    assert LONG_TERM_MEMORY_SCHEMA_VERSION == "9.3"


def test_stale_writes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "global.json"
    first = LongTermMemoryStore(path)
    memory = first.empty(clock=Clock())
    _add(memory, "first validated lesson")
    first.save(memory)
    second = LongTermMemoryStore(path)
    loaded = second.load(clock=Clock())
    assert loaded.memory is not None
    _add(loaded.memory, "second validated lesson")
    second.save(loaded.memory)
    _add(memory, "stale process lesson")
    with pytest.raises(LongTermMemoryConflictError):
        first.save(memory)
