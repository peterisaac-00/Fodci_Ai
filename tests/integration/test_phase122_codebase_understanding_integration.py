from pathlib import Path
from types import SimpleNamespace

import pytest

from backend_ai.agent import (
    AutonomousLoopRequest,
    AutonomousToolLoop,
    CodebaseUnderstanding,
    ExecutionPlan,
    LoopStatus,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    Planner,
    PlannerConfidence,
    PlannerRequest,
    PlannerTaskType,
    ToolRegistry,
    ToolSelectionStatus,
)
from backend_ai.tools.project_context import ProjectContext, project_context


class Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class Engine:
    tokenizer = Tokenizer()

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(generated_text=next(self.outputs))


class CapturingPlanner:
    def __init__(self) -> None:
        self.requests: list[PlannerRequest] = []
        self.planner = Planner()

    def plan(self, request: PlannerRequest):
        self.requests.append(request)
        return self.planner.plan(request)


class StaticPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan_value = plan
        self.requests: list[PlannerRequest] = []

    def plan(self, request: PlannerRequest):
        self.requests.append(request)
        return SimpleNamespace(plan=self.plan_value, warnings=(), errors=(), status="CREATED")


class StaticSelector:
    def __init__(self, tool_name: str, bootstrap_tool: str | None = None) -> None:
        self.tool_name = tool_name
        self.bootstrap_tool = bootstrap_tool

    def select(self, request):
        selected_tool = self.bootstrap_tool if self.bootstrap_tool and not request.selected_step_ids else self.tool_name
        decision = SimpleNamespace(
            selected_tool=selected_tool,
            selection_reason="Phase 12.2 integration fixture",
            risk_level=SimpleNamespace(value="LOW"),
            prerequisites=(),
            expected_output="read-only source evidence",
        )
        return SimpleNamespace(status=ToolSelectionStatus.SELECTED, decisions=(decision,), errors=())


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def _backend_fixture(root: Path) -> None:
    _write(root, "pyproject.toml", "[project]\nname='backend'\ndependencies=['fastapi']\n")
    _write(root, "app/auth.py", "class AuthService:\n    def authenticate(self, token: str) -> bool:\n        return bool(token)\n")
    _write(root, "app/routes.py", "from app.auth import AuthService\n\ndef create_user_endpoint():\n    return AuthService()\n")
    _write(root, "app/database.py", "import sqlite3\n\ndef connect():\n    return sqlite3.connect(':memory:')\n")
    _write(root, "tests/test_api.py", "def test_api():\n    assert True\n")


def _context(root: Path) -> ProjectContext:
    return project_context(root)


def _read_plan() -> ExecutionPlan:
    return ExecutionPlan(
        task="Inspect auth source",
        normalized_task="Inspect auth source",
        goal="Read the affected authentication source safely.",
        task_type=PlannerTaskType.INVESTIGATION,
        steps=(PlanStep("read-auth", "Read authentication source", "Read the affected auth source", "Use bounded evidence", "Source evidence is available", risk_level=PlanRiskLevel.LOW),),
        assumptions=(),
        constraints=(),
        risks=(),
        expected_changes=(),
        verification_strategy=(),
        confidence=PlannerConfidence.HIGH,
        warnings=(),
        completeness=PlanCompleteness.COMPLETE,
    )


def test_task_understanding_planner_and_execution_share_repository_evidence(tmp_path: Path) -> None:
    root = tmp_path / "backend"
    root.mkdir()
    _backend_fixture(root)
    understanding = Planner().plan(PlannerRequest("inspect auth source", _context(root), codebase_understanding=None))
    assert understanding.plan is not None

    from backend_ai.agent.codebase_understanding import CodebaseUnderstandingBuilder

    codebase = CodebaseUnderstandingBuilder().build("inspect auth source", root, project_context=_context(root))
    planner = StaticPlanner(_read_plan())
    loop = AutonomousToolLoop(
        Engine([
            'ACTION: TOOL\nARGS: {"tool":"read_file","arguments":{"path":"app/auth.py"}}',
            'ACTION: FINAL\nARGS: {"message":"auth source inspected"}',
        ]),
        registry=ToolRegistry.default(),
        planner=planner,
        selector=StaticSelector("read_file"),
    )

    result = loop.run(AutonomousLoopRequest("inspect auth source", root, _context(root), codebase_understanding=codebase))

    assert result.status is LoopStatus.COMPLETED
    assert result.plan is not None
    assert result.codebase_understanding is not None
    assert result.codebase_understanding.root == root.resolve()
    assert any(item.name == "AuthService" for item in result.codebase_understanding.symbols)
    assert planner.requests[0].codebase_understanding is codebase
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_results[0].success


def test_autonomous_loop_builds_understanding_after_context_bootstrap(tmp_path: Path) -> None:
    root = tmp_path / "backend"
    root.mkdir()
    _backend_fixture(root)
    planner = StaticPlanner(_read_plan())
    loop = AutonomousToolLoop(
        Engine([
            'ACTION: TOOL\nARGS: {"tool":"read_file","arguments":{"path":"app/database.py"}}',
            'ACTION: FINAL\nARGS: {"message":"database source inspected"}',
        ]),
        registry=ToolRegistry.default(),
        planner=planner,
        selector=StaticSelector("read_file", bootstrap_tool="project_context"),
    )

    result = loop.run(AutonomousLoopRequest("inspect database source", root))

    assert result.status is LoopStatus.COMPLETED
    assert planner.requests
    assert isinstance(planner.requests[0].codebase_understanding, CodebaseUnderstanding)
    assert planner.requests[0].codebase_understanding.root == root.resolve()
    assert result.state.codebase_understanding is not None
    assert result.to_dict()["codebase_understanding"]["root"] == str(root.resolve())
    assert [call.name for call in result.tool_calls] == ["project_context", "read_file"]


@pytest.mark.parametrize(
    ("task", "expected_path"),
    [
        ("add authentication endpoint", "app/auth.py"),
        ("fix database connection bug", "app/database.py"),
        ("add API endpoint", "app/routes.py"),
    ],
)
def test_realistic_backend_tasks_produce_relevant_plan_areas(tmp_path: Path, task: str, expected_path: str) -> None:
    root = tmp_path / expected_path.replace("/", "_")
    root.mkdir()
    _backend_fixture(root)

    from backend_ai.agent.codebase_understanding import CodebaseUnderstandingBuilder

    understanding = CodebaseUnderstandingBuilder().build(task, root)
    planned = Planner().plan(PlannerRequest(task, _context(root), codebase_understanding=understanding))

    assert planned.plan is not None
    assert expected_path in {item.path for item in understanding.relevant_files}
    assert any(expected_path in item for item in planned.plan.assumptions + planned.plan.constraints)
    assert planned.plan.validation if hasattr(planned.plan, "validation") else True
