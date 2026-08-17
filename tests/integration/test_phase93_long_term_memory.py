from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import ExecutionPlan, PlanCompleteness, PlanRiskLevel, PlanStep, PlannerConfidence, PlannerTaskType, ToolRegistry
from backend_ai.agent.long_term_memory import (
    LongTermMemoryCategory,
    LongTermMemoryConfidence,
    LongTermMemoryLoadStatus,
    LongTermMemorySource,
    LongTermMemoryStore,
)
from backend_ai.agent.project_memory import FactCategory, FactConfidence, FactEvidence, FactSource, ProjectMemoryStore
from backend_ai.agent.short_term_memory import ShortTermMemory
from backend_ai.agent.autonomous_tool_loop import AutonomousLoopRequest, AutonomousToolLoop

from tests.integration.test_phase91_short_term_memory import _Selector, _Tool, _context


class _RecordingEngine:
    class tokenizer:
        @staticmethod
        def encode(text: str) -> list[int]:
            return list(text.encode("utf-8"))

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(generated_text='ACTION: FINAL\\nARGS: {"message":"context received"}')


class _OneStepPlanner:
    def plan(self, request):
        return SimpleNamespace(
            plan=ExecutionPlan(
                request.task,
                request.task,
                "Use explicit retrieved context.",
                PlannerTaskType.INVESTIGATION,
                (PlanStep("s1", "Observe", "Observe safely", "context", "observation", (), PlanRiskLevel.LOW),),
                (), (), (), (), (), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE,
            ),
            warnings=(), errors=(), status="CREATED",
        )


def test_global_memory_survives_restart_and_is_retrievable_from_another_project(tmp_path: Path) -> None:
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    global_path = tmp_path / "global" / "long_term_memory.json"

    # Project A keeps project-specific facts in Project Memory only.
    project_store = ProjectMemoryStore.for_project(project_a)
    project_memory = project_store.empty()
    project_memory.add_fact(
        category=FactCategory.FRAMEWORK,
        key="framework.name",
        value="Django",
        source=FactSource.PROJECT_CONTEXT,
        confidence=FactConfidence.VERIFIED,
        evidence=(FactEvidence(FactSource.PROJECT_CONTEXT, "project-a", "local project fact", verified=True),),
    )
    project_store.save(project_memory)

    # A validated reusable lesson is explicitly written to global Long-Term Memory.
    store_a = LongTermMemoryStore(global_path)
    memory_a = store_a.empty()
    lesson = memory_a.add(
        content="Django authentication solutions should validate tokens explicitly before loading a user.",
        category=LongTermMemoryCategory.SOLUTION,
        source=LongTermMemorySource.VERIFIED_TASK,
        confidence=LongTermMemoryConfidence.VERIFIED,
        metadata={"topic": "Django authentication", "validated_by": "task-A"},
    )
    store_a.save(memory_a)

    # A later task in Project B reloads the same global store and tracks access.
    store_b = LongTermMemoryStore(global_path)
    loaded = store_b.load()
    assert loaded.status is LongTermMemoryLoadStatus.LOADED
    assert loaded.memory is not None
    results = loaded.memory.search("Django authentication", category="solution", limit=5)
    assert [item.entry_id for item in results] == [lesson.entry_id]
    assert results[0].access_count == 1
    store_b.save(loaded.memory)
    reloaded = LongTermMemoryStore(global_path).load()
    assert reloaded.memory is not None
    assert reloaded.memory.get(lesson.entry_id, track_access=False).access_count == 1  # type: ignore[union-attr]

    # Project A's local fact is not present in the global store.
    assert all(item.content != "Django" for item in reloaded.memory.list())
    assert ProjectMemoryStore.for_project(project_b).load().status.name == "MEMORY_MISSING"


def test_autonomous_loop_receives_explicit_retrieved_context_without_new_capabilities(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    global_path = tmp_path / "global.json"
    store = LongTermMemoryStore(global_path)
    memory = store.empty()
    lesson = memory.add(
        content="Use a bounded pytest integration checkpoint for authentication fixes.",
        category="pattern",
        source="VERIFIED_TASK",
        confidence="VERIFIED",
        metadata={"topic": "authentication"},
    )
    store.save(memory)
    loaded = LongTermMemoryStore(global_path).load()
    assert loaded.memory is not None
    request = AutonomousLoopRequest(
        "Fix an authentication test",
        root,
        _context(root),
        ShortTermMemory.for_task("Fix an authentication test", root),
        project_memory=None,
        long_term_memory=loaded.memory,
        long_term_query="authentication",
        long_term_memory_limit=2,
    )
    engine = _RecordingEngine()
    loop = AutonomousToolLoop(engine, registry=ToolRegistry((_Tool("record_observation", {"ok": True}),)), planner=_OneStepPlanner(), selector=_Selector())
    result = loop.run(request)
    assert result.project_memory is not None
    assert result.long_term_memories
    assert result.long_term_memories[0].entry_id == lesson.entry_id
    assert result.state.long_term_memories[0].entry_id == lesson.entry_id
    assert any(lesson.content in prompt for prompt in engine.prompts)
