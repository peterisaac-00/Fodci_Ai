from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import (
    AutonomousLoopRequest,
    AutonomousToolLoop,
    ExecutionPlan,
    MemoryLifecycle,
    MemoryImportance,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    ShortTermMemory,
    ProjectMemory,
    ToolRegistry,
    ToolSelectionRisk,
    ToolSelectionStatus,
)
from backend_ai.agent.planner import PlannerResultStatus
from backend_ai.tools import ToolMetadata
from backend_ai.tools.project_context import ProjectContext


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class _Engine:
    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.outputs = iter(
            (
                'ACTION: TOOL\nARGS: {"tool":"record_observation","arguments":{}}',
                'ACTION: TOOL\nARGS: {"tool":"parse_test_result","arguments":{}}',
                'ACTION: FINAL\nARGS: {"message":"task checkpoint complete"}',
            )
        )

    def generate(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(generated_text=next(self.outputs))


class _Tool:
    def __init__(self, name: str, result: object) -> None:
        self.name = name
        self.description = f"Run bounded checkpoint tool {name}."
        self.metadata = ToolMetadata(name, self.description, {"type": "object", "properties": {}})
        self.result = result
        self.calls = 0

    def run(self, arguments):
        self.calls += 1
        return self.result


class _Planner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan_value = plan

    def plan(self, request):
        return SimpleNamespace(plan=self.plan_value, warnings=(), errors=(), status=PlannerResultStatus.CREATED)


class _Selector:
    def select(self, request):
        tool_name = {"s1": "record_observation", "s2": "parse_test_result"}[request.selected_step_ids[0]]
        return SimpleNamespace(
            status=ToolSelectionStatus.SELECTED,
            decisions=(SimpleNamespace(
                selected_tool=tool_name,
                selection_reason="checkpoint fixture",
                risk_level=ToolSelectionRisk.READ_ONLY,
                prerequisites=(),
                expected_output="bounded observation",
            ),),
            errors=(),
        )


def _context(root: Path) -> ProjectContext:
    return ProjectContext(
        root=root,
        project_type="python",
        stack_summary="Python",
        languages=(),
        frameworks=(),
        package_managers=(),
        databases=(),
        test_frameworks=(),
        infrastructure=(),
        source_directories=("src",),
        test_directories=("tests",),
        documentation_directories=(),
        config_files=(),
        dependency_files=(),
        important_files=(),
        entry_points=(),
        project_files=("src/main.py",),
        confidence="high",
        evidence=("temporary integration project",),
        warnings=(),
        truncated=False,
        truncation_reason=None,
        completeness="complete",
    )


def test_real_task_checkpoint_uses_bounded_memory_until_close(tmp_path: Path) -> None:
    root = tmp_path / "task-project"
    root.mkdir()
    plan = ExecutionPlan(
        task="Fix a failing authentication test",
        normalized_task="Fix a failing authentication test",
        goal="Preserve a bounded engineering-task checkpoint.",
        task_type=PlannerTaskType.INVESTIGATION,
        steps=(
            PlanStep("s1", "Observe the temporary project", "Record one safe observation", "context exists", "observation exists", (), PlanRiskLevel.LOW, False),
            PlanStep("s2", "Record passing test evidence", "Record a bounded PASS result", "observation exists", "verification exists", ("s1",), PlanRiskLevel.LOW),
        ),
        assumptions=(),
        constraints=("Do not persist secrets",),
        risks=(),
        expected_changes=(),
        verification_strategy=("final response is recorded",),
        confidence=PlannerConfidence.HIGH,
        warnings=(),
        completeness=PlanCompleteness.COMPLETE,
    )
    observation_tool = _Tool("record_observation", {"observation": "pytest is configured", "password": "not persisted"})
    test_tool = _Tool("parse_test_result", {"overall_status": "PASS", "tests": 1})
    memory = ShortTermMemory.for_task("Fix a failing authentication test", root)
    project_memory = ProjectMemory.for_project(root)
    memory.record_failure({"error": "401 from authentication endpoint"}, classification="ASSERTION", message="authentication test failed", location="tests/test_auth.py")
    memory.record_fix({"target": "auth.py", "result": "token validation corrected"}, target="auth.py", result="applied", verification_status="PENDING")
    memory.record_test_result({"status": "PASS", "tests": 1}, status="PASS", tests_executed=1)
    loop = AutonomousToolLoop(_Engine(), registry=ToolRegistry((observation_tool, test_tool)), planner=_Planner(plan), selector=_Selector())
    result = loop.run(AutonomousLoopRequest("Fix a failing authentication test", root, _context(root), memory, project_memory))

    assert result.short_term_memory is not None
    assert result.project_memory is not None
    assert result.project_memory.identity.project_id == project_memory.project_id
    snapshot = result.short_term_memory
    assert result.status.value in {"COMPLETED", "BLOCKED"}
    assert snapshot.lifecycle is MemoryLifecycle.CLOSED
    assert snapshot.task_id == memory.task_id
    assert snapshot.plan_state is not None
    assert snapshot.observations
    assert snapshot.tool_records
    assert snapshot.test_records
    assert snapshot.failure_records
    assert snapshot.fix_records
    assert snapshot.verification_records
    assert snapshot.total_entries <= memory.limits.max_memory_entries
    assert "not persisted" not in snapshot.to_json()
    assert "password" in snapshot.to_json()
    assert result.to_dict()["short_term_memory"]["lifecycle"] == "CLOSED"
    assert observation_tool.calls == 1
    assert test_tool.calls == 1


def test_new_task_gets_a_new_memory_owner(tmp_path: Path) -> None:
    root = tmp_path / "task-project"
    root.mkdir()
    first = ShortTermMemory.for_task("Task A", root)
    second = ShortTermMemory.for_task("Task B", root)
    first.record_observation("A-only", source="test", importance=MemoryImportance.CRITICAL)
    assert first.task_id != second.task_id
    assert "A-only" not in second.to_json()
