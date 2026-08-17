from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend_ai.agent.experience_records import (
    EXPERIENCE_RECORD_SCHEMA_VERSION,
    ExperienceErrorStatus,
    ExperienceEvaluation,
    ExperienceLifecycleStatus,
    ExperienceOutcomeStatus,
    ExperienceProjectIdentity,
    ExperienceRecordClosedError,
    ExperienceRecordConflictError,
    ExperienceRecordLimits,
    ExperienceRecordLoadStatus,
    ExperienceRecordStore,
    ExperienceRecordValidationError,
    ExperienceRecords,
    ExperienceVerification,
)


class Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"2026-08-17T00:00:{self.value:02d}Z"


def _verification() -> ExperienceVerification:
    return ExperienceVerification(2, 2, 0, "PASS", "Both tests passed", "2026-08-17T00:00:50Z")


def _success(manager: ExperienceRecords, task: str = "Fix authentication"):
    session = manager.start_experience(task, project_identity=ExperienceProjectIdentity("project-a", "/tmp/project-a"), metadata={"topic": "auth"})
    attempt = session.start_attempt()
    error = session.record_error("ASSERTION", "authentication test returned 401", source="test_runner")
    session.record_correction("Adjusted JWT authentication configuration", "authentication tests passed", error_id=error.error_id)
    session.record_action("run_tests", "run authentication tests", status="success")
    session.record_observation("pytest reported PASS", source="run_tests")
    session.record_attempt_result("tests passed", attempt_id=attempt)
    session.record_verification(_verification())
    session.record_evaluation(ExperienceEvaluation(1.0, "PASS", "Verified", ({"criterion": "tests", "status": "PASS"},), {"evaluator": "existing"}))
    record = session.finalize(status="completed", outcome="success", final_solution="JWT validation is explicit", final_summary="Authentication tests passed")
    return record, session


def test_creation_lifecycle_attempt_events_and_nested_types() -> None:
    manager = ExperienceRecords(clock=Clock())
    record, _ = _success(manager)
    assert record.experience_id.startswith("exp-")
    assert record.status is ExperienceLifecycleStatus.COMPLETED
    assert record.outcome is ExperienceOutcomeStatus.SUCCESS
    assert record.project_identity is not None
    assert len(record.attempts) == 1
    attempt = record.attempts[0]
    assert attempt.actions[0].name == "run_tests"
    assert attempt.observations[0].source == "run_tests"
    assert attempt.errors[0].status is ExperienceErrorStatus.UNRESOLVED
    assert attempt.corrections[0].error_id == attempt.errors[0].error_id
    assert record.verification == _verification()
    assert record.evaluation is not None


def test_multiple_attempts_are_explicit_and_bounded() -> None:
    manager = ExperienceRecords(limits=ExperienceRecordLimits(max_attempts_per_experience=2), clock=Clock())
    session = manager.start_experience("Try two fixes")
    first = session.start_attempt()
    session.record_error("TEST", "first attempt failed", attempt_id=first)
    session.record_attempt_result("failed", status="failed", attempt_id=first)
    second = session.start_attempt()
    session.record_observation("second attempt corrected the issue", attempt_id=second)
    session.record_attempt_result("passed", attempt_id=second)
    session.record_verification(_verification())
    record = session.finalize(status="completed", outcome="success")
    assert len(record.attempts) == 2
    with pytest.raises(ExperienceRecordClosedError):
        session.start_attempt()


def test_success_requires_verification_and_finalized_record_is_immutable() -> None:
    manager = ExperienceRecords(clock=Clock())
    session = manager.start_experience("No evidence task")
    session.start_attempt()
    with pytest.raises(ExperienceRecordValidationError):
        session.finalize(status="completed", outcome="success")
    record, _ = _success(manager, "Immutable task")
    with pytest.raises(ExperienceRecordClosedError):
        session = manager.start_experience("closed")
        session.start_attempt()
        session.record_verification(_verification())
        session.finalize(status="completed", outcome="success")
        session.record_observation("late observation")
    assert manager.get(record.experience_id) == record
    assert record.to_json() if hasattr(record, "to_json") else record.to_dict()


def test_failed_and_cancelled_outcomes_do_not_claim_success() -> None:
    manager = ExperienceRecords(clock=Clock())
    failed = manager.start_experience("Failed task")
    failed.start_attempt()
    failed_record = failed.finalize(status="failed", outcome="failure", final_summary="verification failed")
    cancelled = manager.start_experience("Cancelled task")
    cancelled.start_attempt()
    cancelled_record = cancelled.finalize(status="cancelled", outcome="cancelled")
    assert failed_record.outcome is ExperienceOutcomeStatus.FAILURE
    assert cancelled_record.outcome is ExperienceOutcomeStatus.CANCELLED


def test_redaction_and_recursive_metadata_immutability() -> None:
    manager = ExperienceRecords(clock=Clock())
    session = manager.start_experience("Store API_KEY=super-secret", metadata={"nested": {"password": "hidden", "safe": "value"}})
    session.start_attempt()
    session.record_observation("Authorization: Bearer secret-token", metadata={"api_key": "another-secret"})
    session.record_verification(_verification())
    record = session.finalize(status="completed", outcome="success")
    encoded = manager.to_json()
    assert "super-secret" not in encoded
    assert "hidden" not in encoded
    assert "secret-token" not in encoded
    assert "another-secret" not in encoded
    assert record.metadata["nested"]["safe"] == "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.metadata["x"] = "y"  # type: ignore[index]


def test_limits_reject_without_partial_event_mutation() -> None:
    manager = ExperienceRecords(limits=ExperienceRecordLimits(max_actions_per_attempt=1, max_content_length=20), clock=Clock())
    session = manager.start_experience("bounded")
    session.start_attempt()
    session.record_action("read_file", "one action")
    with pytest.raises(ExperienceRecordValidationError):
        session.record_action("read_file", "second action")
    with pytest.raises(ExperienceRecordValidationError):
        session.record_observation("this summary is intentionally too long")
    session.record_verification(_verification())
    record = session.finalize(status="completed", outcome="success")
    assert len(record.attempts[0].actions) == 1
    assert not record.attempts[0].observations


def test_listing_filters_project_status_and_date() -> None:
    manager = ExperienceRecords(clock=Clock())
    first, _ = _success(manager, "first")
    second_session = manager.start_experience("second", project_identity=ExperienceProjectIdentity("project-b", "/tmp/project-b"))
    second_session.start_attempt()
    second = second_session.finalize(status="failed", outcome="failure")
    assert manager.get(first.experience_id) == first
    assert manager.list(project_id="project-a") == (first,)
    assert manager.list(project_id="project-b") == (second,)
    assert manager.list(status="failed") == (second,)
    assert manager.list(started_after=first.started_at, started_before=second.started_at) == (first, second)


def test_persistence_reload_corruption_future_schema_and_stale_writes(tmp_path: Path) -> None:
    path = tmp_path / ".fodci" / "experience_records.json"
    store = ExperienceRecordStore(path)
    manager = store.empty(clock=Clock())
    record, _ = _success(manager)
    store.save(manager)
    load_result = ExperienceRecordStore(path).load(clock=Clock())
    assert load_result.status is ExperienceRecordLoadStatus.LOADED
    assert load_result.error is None
    assert load_result.records is not None
    assert load_result.records.get(record.experience_id) == record
    path.write_text("{bad", encoding="utf-8")
    assert ExperienceRecordStore(path).load().status is ExperienceRecordLoadStatus.MEMORY_CORRUPTED
    future = {"format": "fodci.experience_records", "schema_version": "99.0", "records": [], "status": "LOADED", "sequence": 0, "warnings": []}
    path.write_text(json.dumps(future), encoding="utf-8")
    assert ExperienceRecordStore(path).load().status is ExperienceRecordLoadStatus.MEMORY_INVALID
    fresh = ExperienceRecordStore(path)
    manager2 = fresh.empty(clock=Clock())
    _success(manager2)
    with pytest.raises(ExperienceRecordConflictError):
        fresh.save(manager2)


def test_future_schema_and_unknown_fields_are_not_fabricated(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"format": "fodci.experience_records", "schema_version": EXPERIENCE_RECORD_SCHEMA_VERSION, "records": [], "status": "LOADED", "sequence": 0, "warnings": [], "unexpected": True}), encoding="utf-8")
    load_result = ExperienceRecordStore(path).load()
    assert load_result.status is ExperienceRecordLoadStatus.MEMORY_INVALID
    assert load_result.records is None
    assert load_result.error


def test_store_is_separate_from_memory_paths() -> None:
    from backend_ai.agent.experience_records import default_experience_record_path
    from backend_ai.agent.long_term_memory import LongTermMemoryStore
    assert default_experience_record_path() != LongTermMemoryStore.default().path
    assert default_experience_record_path().name == "experience_records.json"
