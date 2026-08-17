from __future__ import annotations

import json
from dataclasses import replace

import pytest

from backend_ai.agent.dataset_schema import (
    DATASET_RECORD_FORMAT,
    DATASET_RECORD_SCHEMA_VERSION,
    DatasetOutcome,
    DatasetRecord,
    DatasetRecordLimits,
    DatasetRecordValidationError,
    DatasetSchemaValidationResult,
    derive_dataset_record_id,
    validate_dataset_record,
)
from backend_ai.agent.experience_dataset import ExperienceDatasetExtractor
from backend_ai.agent.experience_records import ExperienceEvaluation, ExperienceProjectIdentity, ExperienceRecords, ExperienceVerification


STAMP = "2026-08-17T00:00:00Z"


def _candidate():
    records = ExperienceRecords(clock=lambda: STAMP)
    session = records.start_experience("Fix the API", project_identity=ExperienceProjectIdentity("project-a", "/project-a"))
    attempt_id = session.start_attempt()
    action = session.record_action("inspect", "Inspect route", attempt_id=attempt_id)
    session.record_observation("Found validation gap", source="test", attempt_id=attempt_id)
    error = session.record_error("validation", "Request rejected", source="test", attempt_id=attempt_id)
    session.record_correction("Add validation", "verified", error_id=error.error_id, attempt_id=attempt_id)
    session.record_attempt_result("Validation added", attempt_id=attempt_id)
    session.record_verification(ExperienceVerification(2, 2, 0, "PASS", "all tests passed", STAMP))
    session.record_evaluation(ExperienceEvaluation(1.0, "accepted", "verified", ({"criterion": "tests", "passed": True},), {"source": "local"}))
    record = session.finalize(status="completed", outcome="success", final_solution="Add validation", final_summary="Validation added")
    return ExperienceDatasetExtractor().extract(record)


def test_candidate_conversion_preserves_canonical_schema_fields_and_identity() -> None:
    candidate = _candidate()
    record = DatasetRecord.from_candidate(candidate)
    assert record.format == DATASET_RECORD_FORMAT
    assert record.schema_version == DATASET_RECORD_SCHEMA_VERSION
    assert record.record_id == derive_dataset_record_id(candidate.experience_id, candidate.source_schema_version)
    assert record.experience_id == candidate.experience_id
    assert record.task == candidate.task
    assert record.project_context is not None
    assert record.trajectory.actions[0]["action_id"] == candidate.actions[0]["action_id"]
    assert record.trajectory.observations[0]["summary"] == "Found validation gap"
    assert record.trajectory.errors[0]["summary"] == "Request rejected"
    assert record.trajectory.corrections[0]["error_id"] == candidate.corrections[0]["error_id"]
    assert record.solution.solution == candidate.final_solution
    assert record.solution.final_result == "Validation added"
    assert record.verification.present is True
    assert record.evaluation.present is True
    assert record.outcome is DatasetOutcome.SUCCESS
    assert record.provenance.experience_id == candidate.experience_id


def test_record_id_is_reproducible_and_does_not_change_on_serialization() -> None:
    candidate = _candidate()
    first = DatasetRecord.from_candidate(candidate)
    second = DatasetRecord.from_candidate(candidate)
    assert first.record_id == second.record_id
    assert first.to_json() == second.to_json()


def test_strict_json_round_trip_is_semantically_identical() -> None:
    record = DatasetRecord.from_candidate(_candidate())
    payload = record.to_dict()
    rebuilt = DatasetRecord.from_json(record.to_json())
    assert rebuilt.to_dict() == payload
    assert validate_dataset_record(payload) == DatasetSchemaValidationResult(True, ())


def test_schema_requires_fields_and_rejects_unknown_or_future_version() -> None:
    payload = DatasetRecord.from_candidate(_candidate()).to_dict()
    missing = dict(payload)
    del missing["provenance"]
    with pytest.raises(DatasetRecordValidationError):
        DatasetRecord.from_dict(missing)
    unknown = dict(payload)
    unknown["unknown_field"] = True
    with pytest.raises(DatasetRecordValidationError):
        DatasetRecord.from_dict(unknown)
    future = dict(payload)
    future["schema_version"] = "2.0"
    with pytest.raises(DatasetRecordValidationError):
        DatasetRecord.from_dict(future)


def test_invalid_nested_timestamp_enum_and_duplicate_identifier_are_rejected() -> None:
    payload = DatasetRecord.from_candidate(_candidate()).to_dict()
    bad_timestamp = json.loads(json.dumps(payload))
    bad_timestamp["trajectory"]["actions"][0]["timestamp"] = "not-a-timestamp"
    with pytest.raises(DatasetRecordValidationError):
        DatasetRecord.from_dict(bad_timestamp)
    bad_outcome = json.loads(json.dumps(payload))
    bad_outcome["outcome"] = "unknown"
    with pytest.raises(DatasetRecordValidationError):
        DatasetRecord.from_dict(bad_outcome)
    duplicate = json.loads(json.dumps(payload))
    duplicate["trajectory"]["actions"].append(dict(duplicate["trajectory"]["actions"][0]))
    with pytest.raises(DatasetRecordValidationError):
        DatasetRecord.from_dict(duplicate)


def test_absent_verification_and_evaluation_are_explicit_not_invented() -> None:
    records = ExperienceRecords(clock=lambda: STAMP)
    session = records.start_experience("failure without verification")
    session.start_attempt()
    session.record_attempt_result("failed")
    source = session.finalize(status="failed", outcome="failure", final_summary="failed")
    candidate = ExperienceDatasetExtractor().extract(source)
    record = DatasetRecord.from_candidate(candidate)
    assert record.verification.present is False
    assert record.evaluation.present is False
    assert record.outcome is DatasetOutcome.FAILURE


def test_provenance_is_mandatory_and_must_match_identity() -> None:
    payload = DatasetRecord.from_candidate(_candidate()).to_dict()
    payload["provenance"] = None
    result = validate_dataset_record(payload)
    assert result.valid is False
    assert "provenance" in result.errors[0]
    mismatch = DatasetRecord.from_candidate(_candidate()).to_dict()
    mismatch["provenance"]["experience_id"] = "exp-other"
    with pytest.raises(DatasetRecordValidationError):
        DatasetRecord.from_dict(mismatch)


def test_security_rejects_secret_in_untrusted_schema_payload() -> None:
    payload = DatasetRecord.from_candidate(_candidate()).to_dict()
    payload["task"] = "Use token=super-secret"
    with pytest.raises(DatasetRecordValidationError):
        DatasetRecord.from_dict(payload)


def test_resource_limits_are_enforced_without_filtering_quality() -> None:
    record = DatasetRecord.from_candidate(_candidate())
    with pytest.raises(DatasetRecordValidationError):
        DatasetRecord.from_dict(record.to_dict(), limits=DatasetRecordLimits(max_task_length=3))
    # A failed historical outcome remains a schema-valid record; this layer does not score it.
    records = ExperienceRecords(clock=lambda: STAMP)
    session = records.start_experience("failed task")
    session.start_attempt()
    session.record_attempt_result("failed")
    failed = session.finalize(status="failed", outcome="failure", final_summary="failure")
    assert DatasetRecord.from_candidate(ExperienceDatasetExtractor().extract(failed)).outcome is DatasetOutcome.FAILURE
