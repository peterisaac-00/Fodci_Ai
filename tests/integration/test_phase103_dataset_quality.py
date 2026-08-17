from __future__ import annotations

from pathlib import Path

from backend_ai.agent import AutonomousLoopRequest, AutonomousToolLoop, ExperienceRecords, ProjectMemory, ShortTermMemory, ToolRegistry
from backend_ai.agent.dataset_quality import DatasetQualityEvaluator, QualityDecision
from backend_ai.agent.dataset_schema import DatasetOutcome, DatasetRecord
from backend_ai.agent.experience_dataset import ExperienceDatasetExtractor

from tests.integration.test_phase94_experience_records import _checkpoint_plan
from tests.integration.test_phase91_short_term_memory import _Engine, _Planner, _Selector, _Tool, _context


def _real_record(tmp_path: Path):
    root = tmp_path / "backend-project"
    root.mkdir()
    task = "Fix a failing authentication test"
    records = ExperienceRecords()
    short_term = ShortTermMemory.for_task(task, root)
    project_memory = ProjectMemory.for_project(root)
    observation_tool = _Tool("record_observation", {"observation": "pytest is configured", "password": "not persisted"})
    test_tool = _Tool("parse_test_result", {"overall_status": "PASS", "tests": 1})
    loop = AutonomousToolLoop(_Engine(), registry=ToolRegistry((observation_tool, test_tool)), planner=_Planner(_checkpoint_plan()), selector=_Selector())
    result = loop.run(AutonomousLoopRequest(task, root, _context(root), short_term, project_memory, experience_records=records))
    source = result.experience_record
    assert source is not None
    candidate = ExperienceDatasetExtractor().extract(source)
    return source, DatasetRecord.from_candidate(candidate)


def test_real_pipeline_accepts_successful_backend_experience(tmp_path: Path) -> None:
    source, record = _real_record(tmp_path)
    before = record.to_dict()
    assessment = DatasetQualityEvaluator().evaluate(record)
    assert assessment.decision is QualityDecision.ACCEPT
    assert assessment.record_id == record.record_id
    assert assessment.experience_id == source.experience_id
    assert assessment.provenance.experience_id == source.experience_id
    assert record.to_dict() == before


def test_real_pipeline_rejects_failed_canonical_record_without_mutating_it(tmp_path: Path) -> None:
    _, record = _real_record(tmp_path)
    failed = record.__class__(
        record.format,
        record.schema_version,
        record.record_id,
        record.experience_id,
        record.task,
        record.project_context,
        record.trajectory,
        record.solution,
        record.verification,
        record.evaluation,
        DatasetOutcome.FAILURE,
        record.provenance.__class__(
            record.provenance.source_type,
            record.provenance.experience_id,
            record.provenance.source_schema_version,
            record.provenance.source_created_at,
            record.provenance.completed_at,
            record.provenance.project_identity,
            record.provenance.original_status,
            "failure",
            record.provenance.verification_present,
        ),
        record.metadata,
    )
    before = failed.to_dict()
    assessment = DatasetQualityEvaluator().evaluate(failed)
    assert assessment.decision is QualityDecision.REJECT
    assert failed.to_dict() == before


def test_real_pipeline_reviews_success_without_verification(tmp_path: Path) -> None:
    _, record = _real_record(tmp_path)
    from backend_ai.agent.dataset_schema import DatasetVerification

    weak = record.__class__(
        record.format,
        record.schema_version,
        record.record_id,
        record.experience_id,
        record.task,
        record.project_context,
        record.trajectory,
        record.solution,
        DatasetVerification(False, None, None, None, None, None, None, {}),
        record.evaluation,
        record.outcome,
        record.provenance,
        record.metadata,
    )
    assessment = DatasetQualityEvaluator().evaluate(weak)
    assert assessment.decision is QualityDecision.REVIEW
    assert "verification_missing" in assessment.warnings
