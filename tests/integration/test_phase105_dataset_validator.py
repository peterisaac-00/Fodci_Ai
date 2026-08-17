from __future__ import annotations

from dataclasses import replace

from backend_ai.agent.dataset_quality import DatasetFilteringResult, DatasetQualityEvaluator, QualityDecision
from backend_ai.agent.dataset_schema import DatasetRecord
from backend_ai.agent.dataset_split import DatasetSplitPolicy, DatasetSplitter
from backend_ai.agent.dataset_validator import DatasetDiagnosticCode, DatasetValidator, ValidationStatus
from backend_ai.agent.experience_dataset import ExperienceDatasetExtractor
from backend_ai.agent.experience_records import ExperienceRecords

from tests.integration.test_phase104_dataset_split import STAMP, _experience


def test_phase105_real_pipeline_validates_accepted_split_and_detects_corruption() -> None:
    source_records = ExperienceRecords(clock=lambda: STAMP)
    experiences = (
        _experience(source_records, "Fix the FastAPI authentication timeout", "project-auth", verified=True),
        _experience(source_records, "Repair Redis connection retry handling", "project-redis", verified=True, recovery=True),
        _experience(source_records, "Improve the backend service", "project-weak", verified=False),
        _experience(source_records, "Fix the PostgreSQL migration", "project-failed", verified=False, failed=True),
    )
    extractor = ExperienceDatasetExtractor()
    dataset_records = tuple(DatasetRecord.from_candidate(extractor.extract(item)) for item in experiences)
    evaluator = DatasetQualityEvaluator()
    assessments = tuple(evaluator.evaluate(record) for record in dataset_records)
    accepted = tuple(record for record, assessment in zip(dataset_records, assessments) if assessment.decision is QualityDecision.ACCEPT)
    filtered = DatasetFilteringResult(
        accepted,
        tuple(assessment for assessment in assessments if assessment.decision is QualityDecision.REJECT),
        tuple(assessment for assessment in assessments if assessment.decision is QualityDecision.REVIEW),
        assessments,
        (),
    )
    splitter = DatasetSplitter(policy=DatasetSplitPolicy(seed=42, minimum_train_records=1, minimum_validation_records=1))
    split = splitter.split_accepted(filtered)
    validator = DatasetValidator()
    valid = validator.validate_dataset(dataset_records, split_result=split, quality_assessments=assessments)
    assert valid.validation_status is ValidationStatus.VALID
    assert valid.error_count == 0
    assert {record.record_id for partition in (split.train, split.validation, split.test) for record in partition} == {record.record_id for record in accepted}

    original_task = split.train[0].task
    fake_secret = "integration" + "-secret"
    object.__setattr__(split.train[0], "task", "password: " + fake_secret)
    try:
        corrupted = validator.validate_split(split, records=accepted, quality_assessments=tuple(assessment for assessment in assessments if assessment.decision is QualityDecision.ACCEPT))
        assert corrupted.validation_status is ValidationStatus.INVALID
        assert any(item.code is DatasetDiagnosticCode.SECURITY_VIOLATION for item in corrupted.diagnostics)
        assert "integration-secret" not in corrupted.to_json()
    finally:
        object.__setattr__(split.train[0], "task", original_task)

    original_count = split.manifest.test_count
    object.__setattr__(split.manifest, "test_count", original_count + 1)
    try:
        mismatch = validator.validate_split(split, records=accepted)
        assert mismatch.validation_status is ValidationStatus.INVALID
        assert any(item.code in {DatasetDiagnosticCode.DATASET_COUNT_MISMATCH, DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH} for item in mismatch.diagnostics)
    finally:
        object.__setattr__(split.manifest, "test_count", original_count)
