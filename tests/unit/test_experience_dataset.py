from __future__ import annotations

from pathlib import Path

from backend_ai.agent.experience_dataset import (
    DATASET_CANDIDATE_SOURCE_TYPE,
    DatasetExtractionReason,
    DatasetExtractionLimits,
    ExperienceDatasetExtractor,
)
from backend_ai.agent.experience_records import (
    ExperienceEvaluation,
    ExperienceLifecycleStatus,
    ExperienceProjectIdentity,
    ExperienceRecordStore,
    ExperienceRecords,
    ExperienceVerification,
)


STAMP = "2026-08-17T00:00:00Z"


def _records() -> ExperienceRecords:
    return ExperienceRecords(clock=lambda: STAMP)


def _complete(*, task: str = "Fix the API", project_id: str = "project-a", with_details: bool = True):
    records = _records()
    session = records.start_experience(task, project_identity=ExperienceProjectIdentity(project_id, "/project-a"), metadata={"kind": "backend"})
    attempt_id = session.start_attempt()
    if with_details:
        session.record_action("inspect", "Inspect API route", status="completed", attempt_id=attempt_id)
        session.record_observation("Found missing validation", source="test", attempt_id=attempt_id)
        error = session.record_error("validation", "Request was rejected", source="test", attempt_id=attempt_id)
        session.record_correction("Add explicit validation", "verified", error_id=error.error_id, attempt_id=attempt_id)
    session.record_attempt_result("API validation fixed", attempt_id=attempt_id)
    session.record_verification(ExperienceVerification(2, 2, 0, "PASS", "all tests passed", STAMP))
    session.record_evaluation(ExperienceEvaluation(1.0, "accepted", "verified candidate", ({"criterion": "tests", "passed": True},), {"source": "local"}))
    record = session.finalize(status="completed", outcome="success", final_solution="Validate request payload", final_summary="API validation fixed")
    return records, record


def test_finalized_successful_experience_preserves_full_trajectory_and_provenance() -> None:
    records, record = _complete()
    before = record.to_dict()
    candidate = ExperienceDatasetExtractor().extract(record)
    assert candidate.experience_id == record.experience_id
    assert candidate.task == record.task
    assert candidate.project_identity["project_id"] == "project-a"
    assert len(candidate.attempts) == 1
    assert len(candidate.actions) == 1
    assert len(candidate.observations) == 1
    assert len(candidate.errors) == 1
    assert len(candidate.corrections) == 1
    assert candidate.verification["test_status"] == "PASS"
    assert candidate.evaluation["status"] == "accepted"
    assert candidate.final_solution == "Validate request payload"
    assert candidate.outcome == "success"
    assert candidate.provenance.source_type == DATASET_CANDIDATE_SOURCE_TYPE
    assert candidate.provenance.experience_id == record.experience_id
    assert candidate.provenance.source_schema_version == record.schema_version
    assert candidate.provenance.verification_present is True
    assert record.to_dict() == before
    assert records.get(record.experience_id).to_dict() == before


def test_repeated_extraction_is_deterministic_and_read_only() -> None:
    _, record = _complete()
    extractor = ExperienceDatasetExtractor()
    first = extractor.extract(record).to_dict()
    second = extractor.extract(record).to_dict()
    assert first == second


def test_batch_ordering_is_stable_and_valid_records_survive_invalid_inputs() -> None:
    records_a, first = _complete(task="First")
    records_b, second = _complete(task="Second")
    unfinished = _records().start_experience("unfinished", project_identity=ExperienceProjectIdentity("project-a", "/project-a")).record
    result = ExperienceDatasetExtractor().extract_many((second, unfinished, "not a record", first))  # type: ignore[arg-type]
    assert result.inspected_count == 4
    assert result.extracted_count == 2
    assert result.skipped_count == 2
    assert [candidate.experience_id for candidate in result.candidates] == sorted((first.experience_id, second.experience_id))
    assert {item.reason for item in result.diagnostics} == {DatasetExtractionReason.INCOMPLETE_EXPERIENCE, DatasetExtractionReason.INVALID_RECORD}
    assert records_a.get(first.experience_id) is first
    assert records_b.get(second.experience_id) is second


def test_cancelled_without_final_result_is_not_extracted() -> None:
    records = _records()
    session = records.start_experience("cancelled task")
    session.start_attempt()
    record = session.finalize(status=ExperienceLifecycleStatus.CANCELLED, outcome="cancelled")
    result = ExperienceDatasetExtractor().extract_many((record,))
    assert not result.candidates
    assert result.diagnostics[0].reason is DatasetExtractionReason.INCOMPLETE_EXPERIENCE


def test_governance_invalidated_experience_is_excluded_without_mutating_source() -> None:
    records, record = _complete()
    before = record.to_dict()
    records.invalidate(record.experience_id, reason="known invalid evidence")
    invalidated = records.get(record.experience_id)
    result = ExperienceDatasetExtractor().extract_many((invalidated,))
    assert not result.candidates
    assert result.diagnostics[0].reason is DatasetExtractionReason.SECURITY_VIOLATION
    assert records.get(record.experience_id).metadata["governance_invalidated"] is True
    assert records.get(record.experience_id).to_dict()["task"] == before["task"]


def test_existing_redaction_prevents_secrets_in_candidate_or_diagnostic() -> None:
    records = _records()
    session = records.start_experience("Use token=super-secret", metadata={"api_key": "super-secret"})
    session.start_attempt()
    session.record_observation("Authorization: Bearer super-secret")
    session.record_attempt_result("redacted")
    session.record_verification(ExperienceVerification(1, 1, 0, "PASS", "safe", STAMP))
    record = session.finalize(status="completed", outcome="success", final_summary="safe")
    candidate = ExperienceDatasetExtractor().extract(record)
    serialized = str(candidate.to_dict())
    assert "super-secret" not in serialized
    assert "[REDACTED]" in serialized


def test_store_api_is_used_and_missing_store_is_diagnostic(tmp_path: Path) -> None:
    records, record = _complete()
    path = tmp_path / "experience_records.json"
    store = ExperienceRecordStore(path)
    store.save(records)
    loaded_store = ExperienceRecordStore(path)
    result = ExperienceDatasetExtractor().extract_from_store(loaded_store)
    assert result.extracted_count == 1
    assert result.candidates[0].experience_id == record.experience_id

    missing = ExperienceRecordStore(tmp_path / "missing.json")
    missing_result = ExperienceDatasetExtractor().extract_from_store(missing)
    assert missing_result.extracted_count == 0
    assert missing_result.diagnostics[0].reason is DatasetExtractionReason.UNAVAILABLE_SOURCE


def test_candidate_and_batch_resource_bounds_are_deterministic() -> None:
    _, record = _complete()
    extractor = ExperienceDatasetExtractor(limits=DatasetExtractionLimits(max_records=1, max_candidate_bytes=1, max_total_bytes=1))
    result = extractor.extract_many((record,))
    assert result.extracted_count == 0
    assert result.diagnostics[0].reason is DatasetExtractionReason.RESOURCE_LIMIT
