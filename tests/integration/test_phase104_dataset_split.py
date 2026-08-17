from __future__ import annotations

from pathlib import Path

from backend_ai.agent.dataset_quality import DatasetQualityEvaluator, QualityDecision
from backend_ai.agent.dataset_schema import DatasetRecord
from backend_ai.agent.dataset_split import DatasetSplitPolicy, DatasetSplitter
from backend_ai.agent.experience_dataset import ExperienceDatasetExtractor
from backend_ai.agent.experience_records import ExperienceEvaluation, ExperienceProjectIdentity, ExperienceRecords, ExperienceVerification


STAMP = "2026-08-17T00:00:00Z"


def _experience(records: ExperienceRecords, task: str, project: str, *, verified: bool, recovery: bool = False, failed: bool = False):
    session = records.start_experience(task, project_identity=ExperienceProjectIdentity(project, f"/{project}"))
    attempt_id = session.start_attempt()
    session.record_action("inspect", "Inspect backend behavior", attempt_id=attempt_id)
    session.record_observation("Observed service behavior", source="integration", attempt_id=attempt_id)
    if recovery:
        error = session.record_error("test_failure", "Endpoint test failed", source="pytest", attempt_id=attempt_id)
        session.record_correction("Fix validation branch", "tests pass", error_id=error.error_id, attempt_id=attempt_id)
    session.record_attempt_result("tests failed" if failed else "implementation complete", attempt_id=attempt_id)
    if verified:
        session.record_verification(ExperienceVerification(5, 5, 0, "PASS", "backend integration checks passed", STAMP))
        session.record_evaluation(ExperienceEvaluation(1.0, "accepted", "verified backend task", ({"criterion": "tests", "passed": True},), {"source": "local"}))
    return session.finalize(
        status="failed" if failed else ("cancelled" if not verified else "completed"),
        outcome="failure" if failed else ("cancelled" if not verified else "success"),
        final_solution=None if failed else "Update backend validation branch",
        final_summary="tests failed" if failed else "backend implementation complete",
    )


def test_phase104_real_pipeline_splits_only_quality_accepted_records(tmp_path: Path) -> None:
    source_records = ExperienceRecords(clock=lambda: STAMP)
    sources = (
        _experience(source_records, "Fix the FastAPI authentication timeout", "project-auth", verified=True),
        _experience(source_records, "Repair Redis connection retry handling", "project-redis", verified=True, recovery=True),
        _experience(source_records, "Improve the backend service", "project-weak", verified=False),
        _experience(source_records, "Fix the PostgreSQL migration", "project-failed", verified=False, failed=True),
    )
    extractor = ExperienceDatasetExtractor()
    dataset_records = tuple(DatasetRecord.from_candidate(extractor.extract(source)) for source in sources)
    assessments = tuple(DatasetQualityEvaluator().evaluate(record) for record in dataset_records)
    assert [assessment.decision for assessment in assessments] == [QualityDecision.ACCEPT, QualityDecision.ACCEPT, QualityDecision.REVIEW, QualityDecision.REJECT]

    from backend_ai.agent.dataset_quality import DatasetFilteringResult

    accepted = tuple(record for record, assessment in zip(dataset_records, assessments) if assessment.decision is QualityDecision.ACCEPT)
    filtered = DatasetFilteringResult(
        accepted,
        tuple(assessment for assessment in assessments if assessment.decision is QualityDecision.REJECT),
        tuple(assessment for assessment in assessments if assessment.decision is QualityDecision.REVIEW),
        assessments,
        tuple(),
    )
    before = tuple(record.to_json() for record in dataset_records)
    splitter = DatasetSplitter(policy=DatasetSplitPolicy(seed=42, minimum_train_records=1, minimum_validation_records=1))
    result = splitter.split_accepted(filtered)

    split_ids = {record.record_id for partition in (result.train, result.validation, result.test) for record in partition}
    accepted_ids = {record.record_id for record in accepted}
    assert split_ids == accepted_ids
    assert result.excluded_record_ids == tuple(sorted({dataset_records[2].record_id, dataset_records[3].record_id}))
    assert result.quality_decisions[dataset_records[2].record_id] == "REVIEW"
    assert result.quality_decisions[dataset_records[3].record_id] == "REJECT"
    assert tuple(record.to_json() for record in dataset_records) == before
    assert all(record.provenance.source_type == "experience_record" for partition in (result.train, result.validation, result.test) for record in partition)
    assert str(tmp_path) not in result.manifest.to_json()
