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

from tests.integration.test_phase91_short_term_memory import (
    ExecutionPlan,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    _Engine,
    _Planner,
    _Selector,
    _Tool,
    _context,
)


def _checkpoint_plan() -> ExecutionPlan:
    return ExecutionPlan(
        task="Fix a failing authentication test",
        normalized_task="Fix a failing authentication test",
        goal="Preserve historical task experience.",
        task_type=PlannerTaskType.INVESTIGATION,
        steps=(
            PlanStep("s1", "Observe", "Record safe observation", "context exists", "observation exists", (), PlanRiskLevel.LOW, False),
            PlanStep("s2", "Test", "Record passing test evidence", "observation exists", "verification exists", ("s1",), PlanRiskLevel.LOW),
        ),
        assumptions=(), constraints=("Do not persist secrets",), risks=(), expected_changes=(), verification_strategy=("tests pass",), confidence=PlannerConfidence.HIGH, warnings=(), completeness=PlanCompleteness.COMPLETE,
    )


def test_real_loop_experience_is_explicit_persistent_and_reloaded(tmp_path: Path) -> None:
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
    experience = result.experience_record
    assert experience.project_identity is not None
    assert experience.project_identity.project_id == project_memory.project_id
    assert experience.attempts
    assert experience.attempts[0].actions
    assert experience.attempts[0].observations
    assert experience.verification is not None
    assert experience.status.value in {"completed", "failed"}
    assert "not persisted" not in records.to_json()

    store = ExperienceRecordStore(tmp_path / ".fodci" / "experience_records.json")
    store.save(records)
    load_result = ExperienceRecordStore(store.path).load()
    assert load_result.status is ExperienceRecordLoadStatus.LOADED
    assert load_result.error is None
    assert load_result.records is not None
    historical = load_result.records.get(experience.experience_id)
    assert historical == experience
    assert historical.to_dict() == experience.to_dict()  # type: ignore[union-attr]


def test_experience_project_scope_is_distinct(tmp_path: Path) -> None:
    records = ExperienceRecords()
    first = records.start_experience("Task A", project_identity=None)
    first.start_attempt()
    first_record = first.finalize(status="failed", outcome="failure")
    second = records.start_experience("Task B", project_identity=None)
    second.start_attempt()
    second_record = second.finalize(status="cancelled", outcome="cancelled")
    assert first_record.experience_id != second_record.experience_id
    assert records.list(status="failed") == (first_record,)
    assert records.list(status="cancelled") == (second_record,)
