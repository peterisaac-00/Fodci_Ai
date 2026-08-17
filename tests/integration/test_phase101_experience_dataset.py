from __future__ import annotations

from pathlib import Path

from backend_ai.agent import (
    AutonomousLoopRequest,
    AutonomousToolLoop,
    ExperienceRecordLoadStatus,
    ExperienceRecordStore,
    ExperienceRecords,
    ProjectMemory,
    ShortTermMemory,
    ToolRegistry,
)
from backend_ai.agent.experience_dataset import DatasetExtractionReason, ExperienceDatasetExtractor

from tests.integration.test_phase94_experience_records import _checkpoint_plan
from tests.integration.test_phase91_short_term_memory import _Engine, _Planner, _Selector, _Tool, _context


def test_real_loop_store_and_dataset_extraction_preserve_historical_experience(tmp_path: Path) -> None:
    root = tmp_path / "project-a"
    root.mkdir()
    observation_tool = _Tool("record_observation", {"observation": "pytest is configured", "password": "not persisted"})
    test_tool = _Tool("parse_test_result", {"overall_status": "PASS", "tests": 1})
    records = ExperienceRecords()
    task = "Fix a failing authentication test"
    short_term = ShortTermMemory.for_task(task, root)
    project_memory = ProjectMemory.for_project(root)
    loop = AutonomousToolLoop(_Engine(), registry=ToolRegistry((observation_tool, test_tool)), planner=_Planner(_checkpoint_plan()), selector=_Selector())
    result = loop.run(AutonomousLoopRequest(task, root, _context(root), short_term, project_memory, experience_records=records))

    assert result.experience_record is not None
    original = result.experience_record
    original_json = original.to_dict()
    assert original.attempts
    assert original.verification is not None

    store = ExperienceRecordStore(tmp_path / ".fodci" / "experience_records.json")
    store.save(records)
    loaded = ExperienceRecordStore(store.path).load()
    assert loaded.status is ExperienceRecordLoadStatus.LOADED
    assert loaded.records is not None

    extraction = ExperienceDatasetExtractor().extract_from_store(ExperienceRecordStore(store.path))
    assert extraction.extracted_count == 1
    candidate = extraction.candidates[0]
    assert candidate.experience_id == original.experience_id
    assert candidate.task == task
    assert candidate.attempts
    assert candidate.actions
    assert candidate.observations
    assert candidate.verification is not None
    assert candidate.provenance.source_type == "experience_record"
    assert candidate.provenance.experience_id == original.experience_id
    assert "not persisted" not in str(candidate.to_dict())
    assert original.to_dict() == original_json


def test_invalid_and_valid_records_batch_without_fail_fast() -> None:
    records = ExperienceRecords()
    valid_session = records.start_experience("valid task")
    valid_session.start_attempt()
    valid = valid_session.finalize(status="failed", outcome="failure", final_summary="failure preserved")
    unfinished = records.start_experience("unfinished task").record

    extraction = ExperienceDatasetExtractor().extract_many((unfinished, valid))
    assert extraction.extracted_count == 1
    assert extraction.skipped_count == 1
    assert extraction.candidates[0].experience_id == valid.experience_id
    assert extraction.diagnostics[0].reason is DatasetExtractionReason.INCOMPLETE_EXPERIENCE
