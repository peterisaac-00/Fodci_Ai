from __future__ import annotations

from dataclasses import replace

from backend_ai.agent.dataset_quality import DatasetFilteringResult, DatasetQualityEvaluator, QualityDecision
from backend_ai.agent.dataset_schema import DatasetRecord
from backend_ai.agent.dataset_split import DatasetSplitPolicy, DatasetSplitter
from backend_ai.agent.dataset_validator import DatasetValidator, ValidationStatus
from backend_ai.agent.dataset_versioning import DatasetVersionRegistry, DatasetVersioner, VersionComparisonStatus
from backend_ai.agent.experience_dataset import ExperienceDatasetExtractor
from backend_ai.agent.experience_records import ExperienceRecords

from tests.integration.test_phase104_dataset_split import STAMP, _experience


def _build_pipeline_records() -> tuple[DatasetRecord, ...]:
    source_records = ExperienceRecords(clock=lambda: STAMP)
    experiences = (
        _experience(source_records, "Fix the FastAPI authentication timeout", "project-auth", verified=True),
        _experience(source_records, "Repair Redis connection retry handling", "project-redis", verified=True, recovery=True),
        _experience(source_records, "Improve the backend service", "project-weak", verified=False),
        _experience(source_records, "Fix the PostgreSQL migration", "project-failed", verified=False, failed=True),
    )
    extractor = ExperienceDatasetExtractor()
    return tuple(DatasetRecord.from_candidate(extractor.extract(item)) for item in experiences)


def _accepted_pipeline(records: tuple[DatasetRecord, ...]):
    evaluator = DatasetQualityEvaluator()
    assessments = tuple(evaluator.evaluate(record) for record in records)
    accepted = tuple(record for record, assessment in zip(records, assessments) if assessment.decision is QualityDecision.ACCEPT)
    filtered = DatasetFilteringResult(
        accepted,
        tuple(assessment for assessment in assessments if assessment.decision is QualityDecision.REJECT),
        tuple(assessment for assessment in assessments if assessment.decision is QualityDecision.REVIEW),
        assessments,
        (),
    )
    splitter = DatasetSplitter(policy=DatasetSplitPolicy(seed=42, minimum_train_records=1, minimum_validation_records=1))
    split = splitter.split_accepted(filtered)
    accepted_assessments = tuple(assessment for assessment in assessments if assessment.decision is QualityDecision.ACCEPT)
    validation = DatasetValidator().validate_dataset(accepted, split_result=split, quality_assessments=accepted_assessments)
    assert validation.validation_status is ValidationStatus.VALID
    return accepted, split, validation, assessments


def test_phase106_full_pipeline_persist_reload_verify_and_lineage(tmp_path) -> None:
    records = _build_pipeline_records()
    accepted, split, validation, assessments = _accepted_pipeline(records)
    registry = DatasetVersionRegistry(tmp_path / ".fodci" / "datasets.json")
    versioner = DatasetVersioner(registry=registry)
    v1 = versioner.create_version("dataset-v1", accepted, split, validation, quality_policy_version="quality-1", metadata={"purpose": "evaluation"})
    assert registry.require_version("dataset-v1").dataset_fingerprint == v1.dataset_fingerprint

    reloaded = DatasetVersionRegistry(tmp_path / ".fodci" / "datasets.json")
    reloaded_versioner = DatasetVersioner(registry=reloaded)
    loaded_v1 = reloaded_versioner.get_version("dataset-v1")
    unchanged = reloaded_versioner.verify_version(loaded_v1, accepted, split, validation, quality_policy_version="quality-1")
    assert unchanged.valid is True

    changed = replace(accepted[0], task="Fix a different FastAPI authentication timeout")
    changed_records = (changed,) + accepted[1:]
    changed_assessments = tuple(DatasetQualityEvaluator().evaluate(record) for record in changed_records)
    changed_split = DatasetSplitter(policy=DatasetSplitPolicy(seed=43, minimum_train_records=1, minimum_validation_records=1)).split(changed_records, quality_assessments=changed_assessments)
    changed_validation = DatasetValidator().validate_dataset(changed_records, split_result=changed_split, quality_assessments=changed_assessments)
    failed = reloaded_versioner.verify_version(loaded_v1, changed_records, changed_split, changed_validation, quality_policy_version="quality-1")
    assert failed.valid is False
    assert changed.record_id in failed.changed_record_ids

    v2 = reloaded_versioner.create_version("dataset-v2", changed_records, changed_split, changed_validation, quality_policy_version="quality-1", parent_version="dataset-v1")
    assert reloaded_versioner.registry.lineage("dataset-v2")[0].version == "dataset-v1"
    assert reloaded_versioner.verify_version(v2, changed_records, changed_split, changed_validation, quality_policy_version="quality-1").valid is True
    comparison = reloaded_versioner.compare_versions("dataset-v1", "dataset-v2")
    assert comparison.status is VersionComparisonStatus.DIFFERENT
    assert changed.record_id in comparison.changed_record_ids
    assert [item.version for item in reloaded_versioner.list_versions()] == ["dataset-v1", "dataset-v2"]
