from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from backend_ai.agent.short_term_memory import (
    MemoryClosedError,
    MemoryImportance,
    MemoryInformationKind,
    MemoryLifecycle,
    MemoryStatus,
    MemoryValidationError,
    ShortTermMemory,
    ShortTermMemoryLimits,
)


def test_lifecycle_snapshot_is_immutable_and_closes() -> None:
    memory = ShortTermMemory("task-a", "Fix the authentication endpoint", session_id="session-a")
    assert memory.lifecycle is MemoryLifecycle.CREATED
    snapshot = memory.record_observation("pytest is configured", source="user", authoritative=True)
    assert snapshot.lifecycle is MemoryLifecycle.UPDATED
    assert snapshot.observations[0].information_kind is MemoryInformationKind.AUTHORITATIVE
    assert isinstance(snapshot.observations[0].metadata, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot.observations[0].metadata["secret"] = "value"  # type: ignore[index]

    nested = memory.record_observation("nested", metadata={"details": {"value": "fixed"}})
    with pytest.raises(TypeError):
        nested.observations[-1].metadata["details"]["value"] = "changed"  # type: ignore[index]
    closed = memory.close(MemoryLifecycle.COMPLETED, reason="verification passed")
    assert closed.lifecycle is MemoryLifecycle.CLOSED
    assert closed.status is MemoryStatus.CLOSED
    assert closed.closed_reason == "verification passed"
    with pytest.raises(MemoryClosedError):
        memory.record_observation("late update")


def test_task_isolation_and_deterministic_identity() -> None:
    first = ShortTermMemory.for_task("Fix auth", "/tmp/project")
    second = ShortTermMemory.for_task("Fix auth", "/tmp/project")
    other = ShortTermMemory.for_task("Fix auth", "/tmp/other-project")
    assert first.task_id == second.task_id
    assert first.task_id != other.task_id
    first.record_observation("task A only", source="test")
    assert "task A only" not in other.snapshot().to_json()


def test_authoritative_objective_and_constraints_cannot_be_overwritten_by_derived_data() -> None:
    memory = ShortTermMemory(
        "task-a",
        "Fix authentication without changing the database schema",
        constraints=("Do not change the database schema",),
    )
    memory.record_observation("Changing the schema may solve the issue", source="agent", authoritative=False)
    snapshot = memory.snapshot()
    assert snapshot.objective == "Fix authentication without changing the database schema"
    assert snapshot.constraints == ("Do not change the database schema",)
    assert snapshot.observations[0].information_kind is MemoryInformationKind.DERIVED


def test_category_and_global_bounds_use_deterministic_eviction() -> None:
    limits = ShortTermMemoryLimits(
        max_observations=3,
        max_tool_records=2,
        max_failure_records=2,
        max_test_records=2,
        max_fix_records=2,
        max_verification_records=2,
        max_memory_entries=6,
        max_text_length_per_entry=64,
        max_total_memory_bytes=4_096,
    )
    memory = ShortTermMemory("task-a", "bounded task", limits=limits)
    for index in range(8):
        memory.record_observation(f"observation-{index}", source="test", importance=MemoryImportance.NORMAL)
    snapshot = memory.snapshot()
    assert len(snapshot.observations) <= 3
    assert snapshot.total_entries <= 6
    assert snapshot.evictions > 0
    assert "observation-7" in snapshot.to_json()

    repeat = ShortTermMemory("task-a", "bounded task", limits=limits)
    for index in range(8):
        repeat.record_observation(f"observation-{index}", source="test", importance=MemoryImportance.NORMAL)
    assert repeat.to_json() == memory.to_json()


def test_long_unicode_text_is_bounded_and_canonical_json_is_stable() -> None:
    memory = ShortTermMemory("task-unicode", "تعامل مع Arabic وEnglish", limits=ShortTermMemoryLimits(max_text_length_per_entry=64, max_total_memory_bytes=4_096))
    memory.record_observation("مرحبا " + "x" * 500, source="اختبار")
    encoded = memory.to_json()
    assert "مرحبا" in encoded
    assert len(encoded.encode("utf-8")) <= 4_096
    assert json.dumps(json.loads(encoded), ensure_ascii=False, sort_keys=True, separators=(",", ":")) == encoded


def test_secret_redaction_applies_to_tool_failures_and_metadata() -> None:
    memory = ShortTermMemory("task-secret", "Inspect safely")
    memory.record_tool_result(
        {
            "success": False,
            "message": "password=super-secret token=abc123",
            "authorization": "Bearer private-token",
            "nested": {"api_key": "key-value"},
        },
        tool_name="read_file",
        metadata={"private_key": "-----BEGIN PRIVATE KEY-----abc-----END PRIVATE KEY-----"},
    )
    encoded = memory.to_json()
    assert "super-secret" not in encoded
    assert "abc123" not in encoded
    assert "key-value" not in encoded
    assert "BEGIN PRIVATE KEY" not in encoded
    assert "[REDACTED]" in encoded


def test_structured_records_and_plan_are_bounded() -> None:
    memory = ShortTermMemory("task-flow", "Fix a failing authentication test")
    memory.update_plan_state(
        {"steps": ["inspect", "fix", "verify"]},
        current_step="inspect",
        completed_steps=("context",),
        next_step="fix",
    )
    memory.record_test_result({"status": "FAIL", "stdout": "x" * 10_000}, status="FAIL", tests_executed=4, important_failures=("401",))
    memory.record_failure({"error": "401"}, classification="ASSERTION", location="tests/test_auth.py", hypothesis="token validation")
    memory.record_fix({"target": "auth.py", "result": "changed"}, target="auth.py", result="applied", verification_status="PENDING")
    snapshot = memory.record_verification({"state": "PASSED"}, tests="PASS", regression="FREE", final="VERIFIED", completion="COMPLETE", status="PASSED")
    assert snapshot.plan_state is not None
    assert len(snapshot.test_records) == 1
    assert len(snapshot.failure_records) == 1
    assert len(snapshot.fix_records) == 1
    assert len(snapshot.verification_records) == 1
    assert len(snapshot.to_json().encode("utf-8")) <= memory.limits.max_total_memory_bytes


def test_invalid_inputs_are_rejected_without_mutation() -> None:
    with pytest.raises(MemoryValidationError):
        ShortTermMemory("", "objective")
    with pytest.raises(MemoryValidationError):
        ShortTermMemoryLimits(max_observations=0)
    memory = ShortTermMemory("task", "objective")
    before = memory.to_json()
    with pytest.raises(MemoryValidationError):
        memory.record_observation("value", source="")
    assert memory.to_json() == before


def test_project_root_is_normalized_without_persistence(tmp_path: Path) -> None:
    memory = ShortTermMemory("task", "objective", project_root=tmp_path / "nested" / "..")
    snapshot = memory.snapshot()
    assert snapshot.project_root == str((tmp_path / ".").resolve())
    assert not hasattr(memory, "retrieve")
    assert not hasattr(memory, "store")
