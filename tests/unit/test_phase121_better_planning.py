from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import (
    AutonomousLoopConfig,
    AutonomousLoopRequest,
    AutonomousToolLoop,
    ExecutionPlan,
    PlanCompleteness,
    PlanExecutionState,
    PlanReplanner,
    PlanRiskLevel,
    PlanStep,
    PlanStepStatus,
    Planner,
    PlannerConfidence,
    PlannerRequest,
    PlannerTaskType,
    ReplanRequest,
    TaskAnalyzer,
)
from backend_ai.tools import Detection, ProjectContext, ToolMetadata
from backend_ai.agent.registry import ToolRegistry


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class FakeEngine:
    tokenizer = FakeTokenizer()

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)

    def generate(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(generated_text=next(self.outputs))


class StaticPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan_value = plan

    def plan(self, request):
        return SimpleNamespace(plan=self.plan_value, warnings=(), errors=(), status="CREATED")


class RecordingTool:
    name = "project_structure"
    description = "bounded project structure inspection"

    def __init__(self) -> None:
        self.metadata = ToolMetadata(self.name, self.description, {"type": "object", "properties": {}})
        self.calls = 0

    def run(self, arguments):
        self.calls += 1
        return {"ok": True, "calls": self.calls}


def context(root: Path) -> ProjectContext:
    return ProjectContext(
        root=root,
        project_type="python",
        stack_summary="Python + FastAPI + pytest",
        languages=(),
        frameworks=(Detection("FastAPI", "high", ("fixture",)),),
        package_managers=(),
        databases=(),
        test_frameworks=(Detection("pytest", "high", ("fixture",)),),
        infrastructure=(),
        source_directories=("src",),
        test_directories=("tests",),
        documentation_directories=("docs",),
        config_files=("pyproject.toml",),
        dependency_files=("pyproject.toml",),
        important_files=("pyproject.toml",),
        entry_points=(),
        project_files=("src/main.py", "tests/test_main.py"),
        confidence="high",
        evidence=("fixture",),
        warnings=(),
        truncated=False,
        truncation_reason=None,
        completeness="complete",
    )


def make_plan(*steps: PlanStep) -> ExecutionPlan:
    return ExecutionPlan(
        "Inspect the project",
        "Inspect the project",
        "Inspect the project safely.",
        PlannerTaskType.INVESTIGATION,
        tuple(steps),
        (),
        (),
        (),
        (),
        ("verify the observed result",),
        PlannerConfidence.HIGH,
        (),
        PlanCompleteness.COMPLETE,
    )


def test_task_analyzer_extracts_typed_requirements_and_context_areas(tmp_path: Path) -> None:
    analysis = TaskAnalyzer().analyze(PlannerRequest("Add a users endpoint", context(tmp_path)))
    assert analysis.task_type is PlannerTaskType.FEATURE
    assert analysis.requirements
    assert "src" in analysis.relevant_project_areas
    assert analysis.verification_criteria
    assert analysis.completeness is PlanCompleteness.COMPLETE


def test_plan_execution_state_respects_dependencies_and_transitions() -> None:
    plan = make_plan(
        PlanStep("inspect", "Inspect project", "inspect", "needed first", "context", risk_level=PlanRiskLevel.LOW),
        PlanStep("verify", "Verify result", "verify", "evidence", "verified", ("inspect",), risk_level=PlanRiskLevel.LOW),
    )
    state = PlanExecutionState.from_plan(plan)
    assert state.next_ready_step(plan) == "inspect"
    state = state.transition(plan, "inspect", PlanStepStatus.IN_PROGRESS, reason="started")
    state = state.transition(plan, "inspect", PlanStepStatus.COMPLETED, reason="verification passed", evidence=("verification passed",))
    assert state.next_ready_step(plan) == "verify"
    assert state.completed_step_ids == ("inspect",)
    assert state.events[-1].event_type.value == "verification_passed"


def test_replanner_is_deterministic_and_preserves_completed_steps() -> None:
    plan = Planner().create_plan("Add a users endpoint", project_context=None)
    state = PlanExecutionState.from_plan(plan).transition(plan, "step-1", PlanStepStatus.COMPLETED, reason="bounded inspection")
    result = PlanReplanner().replan(ReplanRequest("Add a users endpoint", plan, state, None, "missing project evidence"))
    assert result.plan is not None
    assert result.validation.valid
    assert result.changed is False
    assert result.execution_state.replan_count == 1
    assert result.execution_state.statuses["step-1"] is PlanStepStatus.COMPLETED


def test_autonomous_loop_exposes_plan_execution_state(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = RecordingTool()
    plan = make_plan(PlanStep("s1", "Inspect project structure", "inspect", "fixture", "observed", risk_level=PlanRiskLevel.LOW))
    loop = AutonomousToolLoop(
        FakeEngine([
            'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}',
            'ACTION: FINAL\nARGS: {"message":"done"}',
        ]),
        registry=ToolRegistry((tool,)),
        planner=StaticPlanner(plan),
        config=AutonomousLoopConfig(max_replans=1),
    )
    result = loop.run(AutonomousLoopRequest("Inspect the project", root, context(root)))
    assert result.plan_execution is not None
    assert result.plan_execution.statuses["s1"] is PlanStepStatus.COMPLETED
    assert result.state.to_dict()["plan_execution"]["completed_step_ids"] == ["s1"]
    assert result.plan is not None and result.plan.steps[0].status is PlanStepStatus.COMPLETED
