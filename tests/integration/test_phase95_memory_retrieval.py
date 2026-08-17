from __future__ import annotations

from pathlib import Path

from backend_ai.agent import (
    AutonomousLoopRequest,
    AutonomousToolLoop,
    ExperienceProjectIdentity,
    ExperienceRecords,
    ExperienceVerification,
    MemoryRetrieval,
    MemoryRetrievalRequest,
    ProjectMemory,
    RetrievalSource,
    ShortTermMemory,
    ToolRegistry,
)
from backend_ai.agent.long_term_memory import LongTermMemory
from backend_ai.agent.project_memory import FactCategory, FactConfidence, FactEvidence, FactSource

from tests.integration.test_phase91_short_term_memory import (
    _Engine,
    _Planner,
    _Selector,
    _Tool,
    _context,
)
from tests.integration.test_phase94_experience_records import _checkpoint_plan


class _RecordingEngine(_Engine):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    def generate(self, prompt: str):
        self.prompts.append(prompt)
        return super().generate(prompt)


def _project(root: Path, framework: str) -> ProjectMemory:
    memory = ProjectMemory.for_project(root)
    memory.add_fact(category=FactCategory.FRAMEWORK, key="framework.name", value=framework, source=FactSource.PROJECT_CONTEXT, confidence=FactConfidence.VERIFIED, evidence=(FactEvidence(FactSource.PROJECT_CONTEXT, "integration", f"verified {framework}", verified=True),))
    return memory


def _global_memory() -> LongTermMemory:
    memory = LongTermMemory()
    memory.add(content="Django authentication uses explicit JWT validation", category="solution", source="VERIFIED_TASK", confidence="VERIFIED", metadata={})
    return memory


def _history(project_id: str, project_root: str) -> ExperienceRecords:
    records = ExperienceRecords()
    session = records.start_experience("Django authentication verification", project_identity=ExperienceProjectIdentity(project_id, project_root))
    session.start_attempt()
    session.record_observation("Django authentication tests passed", source="test_runner")
    session.record_verification(ExperienceVerification(1, 1, 0, "PASS", "authentication passed", "2026-08-17T00:00:01Z"))
    session.finalize(status="completed", outcome="success", final_solution="Validate JWT explicitly", final_summary="verified historical solution")
    return records


def test_project_global_historical_and_partial_failure_scenarios(tmp_path: Path) -> None:
    root_a = tmp_path / "project-a"
    root_b = tmp_path / "project-b"
    root_a.mkdir()
    root_b.mkdir()
    project_a = _project(root_a, "Django")
    project_b = _project(root_b, "Flask")
    global_memory = _global_memory()
    history = _history(project_a.project_id, project_a.project_root)
    retrieval = MemoryRetrieval()

    project_result = retrieval.retrieve(MemoryRetrievalRequest("Django", (RetrievalSource.PROJECT_MEMORY,), project_id=project_a.project_id, project_memory=project_a.snapshot()))
    assert project_result.items and project_result.items[0].source is RetrievalSource.PROJECT_MEMORY
    assert "Django" in project_result.items[0].content
    isolated = retrieval.retrieve(MemoryRetrievalRequest("Django", (RetrievalSource.PROJECT_MEMORY,), project_id=project_a.project_id, project_memory=project_b.snapshot()))
    assert isolated.items == ()
    assert isolated.diagnostics[0].status == "FAILED"

    global_a = retrieval.retrieve(MemoryRetrievalRequest("JWT authentication", (RetrievalSource.LONG_TERM_MEMORY,), long_term_memory=global_memory))
    global_b = retrieval.retrieve(MemoryRetrievalRequest("JWT authentication", (RetrievalSource.LONG_TERM_MEMORY,), long_term_memory=global_memory))
    assert global_a.items and global_b.items
    assert global_a.items[0].memory_id == global_b.items[0].memory_id

    historical = retrieval.retrieve(MemoryRetrievalRequest("Django authentication", (RetrievalSource.EXPERIENCE_RECORDS,), project_id=project_a.project_id, experience_records=history))
    assert historical.items and historical.items[0].source is RetrievalSource.EXPERIENCE_RECORDS
    assert "verified historical solution" in historical.items[0].content

    unified = retrieval.retrieve(MemoryRetrievalRequest("Django authentication", (RetrievalSource.PROJECT_MEMORY, RetrievalSource.LONG_TERM_MEMORY, RetrievalSource.EXPERIENCE_RECORDS), project_id=project_a.project_id, project_memory=project_a.snapshot(), long_term_memory=global_memory, experience_records=history, max_results=8, max_results_per_source=4))
    assert {item.source for item in unified.items} == {RetrievalSource.PROJECT_MEMORY, RetrievalSource.LONG_TERM_MEMORY, RetrievalSource.EXPERIENCE_RECORDS}
    assert unified.context_characters == len(unified.context)

    partial = retrieval.retrieve(MemoryRetrievalRequest("JWT authentication", (RetrievalSource.PROJECT_MEMORY, RetrievalSource.LONG_TERM_MEMORY), project_memory=None, long_term_memory=global_memory))
    assert partial.items and partial.items[0].source is RetrievalSource.LONG_TERM_MEMORY
    assert any(item.source is RetrievalSource.PROJECT_MEMORY and item.status == "FAILED" for item in partial.diagnostics)


def test_loop_receives_explicit_unified_retrieval_context(tmp_path: Path) -> None:
    root = tmp_path / "project-a"
    root.mkdir()
    project = _project(root, "Django")
    short = ShortTermMemory.for_task("Django authentication", root)
    short.record_observation("Current authentication task", source="integration")
    global_memory = _global_memory()
    history = _history(project.project_id, project.project_root)
    retrieval_request = MemoryRetrievalRequest("Django authentication", (RetrievalSource.PROJECT_MEMORY, RetrievalSource.LONG_TERM_MEMORY, RetrievalSource.EXPERIENCE_RECORDS), project_id=project.project_id, project_memory=project.snapshot(), long_term_memory=global_memory, experience_records=history)
    engine = _RecordingEngine()
    loop = AutonomousToolLoop(engine, registry=ToolRegistry((_Tool("record_observation", {"observation": "safe"}), _Tool("parse_test_result", {"overall_status": "PASS", "tests": 1}))), planner=_Planner(_checkpoint_plan()), selector=_Selector())
    result = loop.run(AutonomousLoopRequest("Django authentication", root, _context(root), short, project, experience_records=history, memory_retrieval_request=retrieval_request))
    assert result.memory_retrieval is not None
    assert result.state.memory_retrieval is not None
    assert "[PROJECT_MEMORY]" in result.memory_retrieval.context
    assert any("[PROJECT_MEMORY]" in prompt for prompt in engine.prompts)
    assert result.to_dict()["memory_retrieval"]["items"]
