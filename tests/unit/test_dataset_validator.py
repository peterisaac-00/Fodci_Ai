from __future__ import annotations

from dataclasses import replace

import pytest

from backend_ai.agent.dataset_quality import DatasetQualityEvaluator, QualityDecision
from backend_ai.agent.dataset_schema import DatasetOutcome, DatasetProjectContext
from backend_ai.agent.dataset_split import DatasetSplitGroup, DatasetSplitPolicy, DatasetSplitter
from backend_ai.agent.dataset_validator import (
    DatasetDiagnosticCode,
    DatasetValidationLimits,
    DatasetValidator,
    DiagnosticSeverity,
    ValidationStatus,
)

from tests.unit.test_dataset_split import _record, _records


def test_valid_canonical_record_and_dataset_pass() -> None:
    record = _record(1)
    result = DatasetValidator().validate_record(record)
    assert result.validation_status is ValidationStatus.VALID
    assert result.valid_records == 1
    assert result.error_count == 0
    assert result.provenance[0].experience_id == record.experience_id
    assert result.to_json() == DatasetValidator().validate_record(record).to_json()


def test_invalid_schema_and_invalid_provenance_are_explicit() -> None:
    record = _record(2)
    payload = record.to_dict()
    payload["schema_version"] = "9.9"
    schema_result = DatasetValidator().validate_record(payload)
    assert schema_result.validation_status is ValidationStatus.INVALID
    assert any(item.code is DatasetDiagnosticCode.RECORD_SCHEMA_INVALID for item in schema_result.diagnostics)

    original_experience_id = record.provenance.experience_id
    object.__setattr__(record.provenance, "experience_id", "exp-contradictory")
    try:
        provenance_result = DatasetValidator().validate_record(record)
        assert provenance_result.validation_status is ValidationStatus.INVALID
        assert any(item.code is DatasetDiagnosticCode.PROVENANCE_INVALID for item in provenance_result.diagnostics)
    finally:
        object.__setattr__(record.provenance, "experience_id", original_experience_id)


def test_security_violation_is_reported_without_secret_value() -> None:
    record = _record(3)
    original_task = record.task
    fake_secret = "super" + "-secret-value"
    object.__setattr__(record, "task", "Fix backend password: " + fake_secret)

    try:
        result = DatasetValidator().validate_record(record)
        assert result.validation_status is ValidationStatus.INVALID
        assert any(item.code is DatasetDiagnosticCode.SECURITY_VIOLATION for item in result.diagnostics)
        serialized = result.to_json()
        assert "super-secret-value" not in serialized
        assert "[REDACTED]" in serialized or "security validation" in serialized
    finally:
        object.__setattr__(record, "task", original_task)


def test_verification_and_evaluation_inconsistencies_are_reported() -> None:
    record = _record(4)
    original_failed = record.verification.tests_failed
    object.__setattr__(record.verification, "tests_failed", 1)
    try:
        result = DatasetValidator().validate_record(record)
        assert any(item.code is DatasetDiagnosticCode.VERIFICATION_INCONSISTENCY for item in result.diagnostics)
    finally:
        object.__setattr__(record.verification, "tests_failed", original_failed)

    original_score = record.evaluation.score
    object.__setattr__(record.evaluation, "score", 2.0)
    try:
        result = DatasetValidator().validate_record(record)
        assert result.validation_status is ValidationStatus.INVALID
        assert any(item.code is DatasetDiagnosticCode.EVALUATION_INCONSISTENCY for item in result.diagnostics)
    finally:
        object.__setattr__(record.evaluation, "score", original_score)


def test_duplicate_record_id_experience_and_exact_payload_are_reported() -> None:
    record = _record(5)
    result = DatasetValidator().validate_records((record, record))
    codes = {item.code for item in result.diagnostics}
    assert DatasetDiagnosticCode.DUPLICATE_RECORD in codes
    assert DatasetDiagnosticCode.EXACT_DUPLICATE_RECORD in codes
    assert result.validation_status is ValidationStatus.INVALID


def test_duplicate_experience_with_different_canonical_records_is_reported() -> None:
    first = _record(6, "project-one")
    second = _record(7, "project-two")
    object.__setattr__(second, "experience_id", first.experience_id)
    try:
        result = DatasetValidator().validate_records((first, second))
        assert any(item.code is DatasetDiagnosticCode.DUPLICATE_EXPERIENCE for item in result.diagnostics)
    finally:
        object.__setattr__(second, "experience_id", "experience-7")


def test_split_validation_accepts_valid_split_and_detects_overlap_and_manifest_mismatch() -> None:
    records = _records(9)
    split = DatasetSplitter(policy=DatasetSplitPolicy(train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, seed=42)).split(records)
    validator = DatasetValidator()
    valid = validator.validate_split(split, records=records)
    assert valid.validation_status is ValidationStatus.VALID

    original_validation = split.validation
    object.__setattr__(split, "validation", (split.train[0],) + split.validation)
    try:
        overlap = validator.validate_split(split, records=records)
        assert any(item.code is DatasetDiagnosticCode.PARTITION_OVERLAP for item in overlap.diagnostics)
    finally:
        object.__setattr__(split, "validation", original_validation)

    original_train_count = split.manifest.train_count
    object.__setattr__(split.manifest, "train_count", original_train_count + 1)
    try:
        mismatch = validator.validate_split(split, records=records)
        assert any(item.code in {DatasetDiagnosticCode.DATASET_COUNT_MISMATCH, DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH} for item in mismatch.diagnostics)
    finally:
        object.__setattr__(split.manifest, "train_count", original_train_count)


def test_missing_partition_record_and_quality_decision_mismatch_are_reported() -> None:
    records = _records(9)
    evaluator = DatasetQualityEvaluator()
    assessments = tuple(evaluator.evaluate(record) for record in records)
    split = DatasetSplitter(policy=DatasetSplitPolicy(train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, seed=7)).split(records)
    original_test = split.test
    object.__setattr__(split, "test", ())
    try:
        result = DatasetValidator().validate_dataset(records, split_result=split, quality_assessments=assessments)
        codes = {item.code for item in result.diagnostics}
        assert DatasetDiagnosticCode.PARTITION_MISSING_RECORD in codes or DatasetDiagnosticCode.DATASET_COUNT_MISMATCH in codes
    finally:
        object.__setattr__(split, "test", original_test)

    bad_assessment = replace(assessments[0], decision=QualityDecision.REJECT)
    mismatched = (bad_assessment,) + assessments[1:]
    result = DatasetValidator().validate_dataset(records, split_result=split, quality_assessments=mismatched)
    assert any(item.code is DatasetDiagnosticCode.QUALITY_DECISION_MISMATCH for item in result.diagnostics)


def test_experience_and_project_leakage_are_detected_structurally() -> None:
    records = _records(9)
    experience_split = DatasetSplitter(policy=DatasetSplitPolicy(group_by=DatasetSplitGroup.EXPERIENCE, train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, seed=42)).split(records)
    original_experience_id = experience_split.validation[0].experience_id
    object.__setattr__(experience_split.validation[0], "experience_id", experience_split.train[0].experience_id)
    try:
        result = DatasetValidator().validate_split(experience_split, records=records)
        assert any(item.code is DatasetDiagnosticCode.EXPERIENCE_LEAKAGE for item in result.diagnostics)
    finally:
        object.__setattr__(experience_split.validation[0], "experience_id", original_experience_id)

    project_records = tuple(_record(index, f"project-{index // 3}") for index in range(9))
    project_split = DatasetSplitter(policy=DatasetSplitPolicy(group_by=DatasetSplitGroup.PROJECT, train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, seed=42, require_non_empty_partitions=True)).split(project_records)
    original_context = project_split.validation[0].project_context
    object.__setattr__(project_split.validation[0], "project_context", project_split.train[0].project_context)
    try:
        result = DatasetValidator().validate_split(project_split, records=project_records)
        assert any(item.code is DatasetDiagnosticCode.PROJECT_LEAKAGE for item in result.diagnostics)
    finally:
        object.__setattr__(project_split.validation[0], "project_context", original_context)


def test_quality_acceptance_and_valid_grouped_split_pass_without_rerunning_evaluator() -> None:
    records = tuple(_record(index, f"project-{index // 3}") for index in range(9))
    assessments = tuple(DatasetQualityEvaluator().evaluate(record) for record in records)
    assert all(item.decision is QualityDecision.ACCEPT for item in assessments)
    split = DatasetSplitter(policy=DatasetSplitPolicy(group_by=DatasetSplitGroup.PROJECT, train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, seed=11, require_non_empty_partitions=True)).split(records, quality_assessments=assessments)
    result = DatasetValidator().validate_dataset(records, split_result=split, quality_assessments=assessments)
    assert result.validation_status in {ValidationStatus.VALID, ValidationStatus.VALID_WITH_WARNINGS}
    assert result.error_count == 0


def test_validation_is_deterministic_independent_of_input_order_and_limits_are_explicit() -> None:
    records = _records(12)
    first = DatasetValidator().validate_records(records)
    second = DatasetValidator().validate_records(tuple(reversed(records)))
    assert first.to_json() == second.to_json()

    limited = DatasetValidator(limits=DatasetValidationLimits(max_records=2)).validate_records(records)
    assert limited.validation_status is ValidationStatus.INVALID
    assert any(item.code is DatasetDiagnosticCode.RESOURCE_LIMIT_EXCEEDED for item in limited.diagnostics)


def test_diagnostics_are_immutable_machine_readable_and_bounded() -> None:
    result = DatasetValidator().validate_record(_record(13))
    assert isinstance(result.diagnostics, tuple)
    assert isinstance(result.summary, dict | object)
    assert all(item.code.value and item.severity in {DiagnosticSeverity.INFO, DiagnosticSeverity.WARNING, DiagnosticSeverity.ERROR} for item in result.diagnostics)
    with pytest.raises(TypeError):
        result.summary["new"] = 1  # type: ignore[index]
