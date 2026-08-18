from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import (
    AutonomousLoopRequest,
    AutonomousToolLoop,
    ExecutionPlan,
    LoopStatus,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    PlanValidationResult,
    ToolRegistry,
)
from backend_ai.tools import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.project_context import ProjectContext


class Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class Engine:
    tokenizer = Tokenizer()

    def __init__(self) -> None:
        self.outputs = iter([
            'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}',
            'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}',
            'ACTION: FINAL\nARGS: {"message":"recovered"}',
        ])

    def generate(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(generated_text=next(self.outputs))


class FlakyStructureTool:
    name = "project_structure"
    description = "flaky bounded structure inspection"
    metadata = ToolMetadata(name, description, {"type": "object", "properties": {}})

    def __init__(self) -> None:
        self.calls = 0

    def run(self, arguments):
        self.calls += 1
        if self.calls == 1:
            raise ToolError(ToolErrorCode.FILE_NOT_FOUND, "fixture structure was not ready")
        return {"ok": True, "recovered": True}


def context(root: Path) -> ProjectContext:
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
        evidence=("fixture",),
        warnings=(),
        truncated=False,
        truncation_reason=None,
        completeness="complete",
    )


def test_failure_triggers_bounded_replan_and_updates_plan_state(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    plan = ExecutionPlan(
        "Inspect the project",
        "Inspect the project",
        "Inspect safely",
        PlannerTaskType.INVESTIGATION,
        (PlanStep("s1", "Inspect project structure", "inspect", "fixture", "observed", risk_level=PlanRiskLevel.LOW),),
        (),
        (),
        (),
        (),
        (),
        PlannerConfidence.HIGH,
        (),
        PlanCompleteness.COMPLETE,
    )
    tool = FlakyStructureTool()
    static_result = SimpleNamespace(plan=plan, warnings=(), errors=(), status="CREATED", validation=PlanValidationResult(True))
    loop = AutonomousToolLoop(Engine(), registry=ToolRegistry((tool,)), planner=type("StaticPlanner", (), {"plan": lambda self, request: static_result})())
    result = loop.run(AutonomousLoopRequest("Inspect the project", root, context(root)))
    assert result.status is LoopStatus.COMPLETED
    assert tool.calls == 2
    assert result.plan_execution is not None
    assert result.plan_execution.replan_count == 1
    assert any("REPLAN" in warning or "replanned" in warning for warning in result.warnings)
    assert result.plan_execution.statuses["s1"].value == "completed"
