from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import (
    AgentMessage,
    AgentMessageRole,
    AutonomousLoopRequest,
    AutonomousToolLoop,
    CodebaseUnderstandingBuilder,
    ContextItem,
    ContextManager,
    ContextManagerConfig,
    ContextPriority,
    ContextType,
    CompressionStatus,
    ExecutionPlan,
    LoopStatus,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    ToolRegistry,
    ToolSelectionStatus,
    TokenCountKind,
)
from backend_ai.tools.project_context import project_context


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


class StaticPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan_value = plan
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        return SimpleNamespace(plan=self.plan_value, warnings=(), errors=(), status="CREATED")


class StaticSelector:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def select(self, request):
        decision = SimpleNamespace(selected_tool=self.tool_name, selection_reason="Phase 12.3 integration fixture", risk_level=SimpleNamespace(value="LOW"), prerequisites=(), expected_output="bounded evidence")
        return SimpleNamespace(status=ToolSelectionStatus.SELECTED, decisions=(decision,), errors=())


class RecordingTool:
    def __init__(self, name: str, result: object) -> None:
        self.name = name
        self.description = name
        self.metadata = SimpleNamespace(name=name, input_schema={"type": "object", "properties": {}})
        self.result = result

    def run(self, arguments):
        return self.result


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        "inspect authentication",
        "inspect authentication",
        "Inspect the authentication evidence safely.",
        PlannerTaskType.INVESTIGATION,
        (PlanStep("s1", "Inspect authentication", "Read authentication evidence", "bounded evidence", "source observation", risk_level=PlanRiskLevel.LOW),),
        (), (), (), (), (), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE,
    )


def test_large_repository_selects_relevant_context_without_loading_every_file(tmp_path: Path) -> None:
    root = tmp_path / "large-repository"
    root.mkdir()
    _write(root, "pyproject.toml", "[project]\nname='large'\n")
    _write(root, "app/auth.py", "def authenticate(token):\n    return token\n")
    for index in range(120):
        _write(root, f"app/unrelated_{index}.py", f"def unrelated_{index}(): return {index}\n")
    understanding = CodebaseUnderstandingBuilder(max_relevant_files=12, max_inspected_files=8).build("fix authentication", root, project_context=project_context(root))
    manager = ContextManager(tokenizer=Tokenizer(), config=ContextManagerConfig(max_context_tokens=700, reserve_output_tokens=80, max_items=24))

    assembly = manager.assemble(task="fix authentication", system_prompt="system", instruction="inspect", project_context=project_context(root), codebase_understanding=understanding)

    assert "app/auth.py" in assembly.prompt
    assert len(assembly.items) < 30
    assert "unrelated_119.py" not in assembly.prompt
    assert assembly.metrics.dropped_count > 0 or len(understanding.relevant_files) < 120


def test_large_file_is_compressed_around_task_relevant_sections(tmp_path: Path) -> None:
    root = tmp_path / "large-file"
    root.mkdir()
    manager = ContextManager(tokenizer=Tokenizer(), config=ContextManagerConfig(max_context_tokens=500, reserve_output_tokens=64, summary_threshold_characters=200, max_summary_characters=180))
    large = "\n".join([f"def unrelated_{index}(): return {index}" for index in range(80)] + ["def authenticate_user(token):", "    return validate_token(token)"])
    item = ContextItem("auth-file", "read_file", large, ContextType.FILE_CONTENT, 0.9, ContextPriority.HIGH, len(large), TokenCountKind.ESTIMATED if False else TokenCountKind.ESTIMATED, metadata={"path": "app/auth.py"})

    assembly = manager.assemble(task="inspect authenticate_user", system_prompt="system", instruction="inspect", extra_items=(item,))

    assert "authenticate_user" in assembly.prompt
    assert "unrelated_0" not in assembly.prompt
    assert any(candidate.compression is not CompressionStatus.ORIGINAL for candidate in assembly.items)


def test_huge_tool_output_keeps_failure_summary_not_the_entire_log(tmp_path: Path) -> None:
    root = tmp_path / "tool-output"
    root.mkdir()
    plan = _plan()
    huge = "noise line\n" * 10_000 + "Traceback: IntegrityError in authenticate_user\nFAILED tests/test_auth.py::test_login"
    tool = RecordingTool("project_structure", {"output": huge})
    loop = AutonomousToolLoop(
        Engine([
            'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}',
            'ACTION: FINAL\nARGS: {"message":"observed"}',
        ]),
        registry=ToolRegistry((tool,)),
        planner=StaticPlanner(plan),
        selector=StaticSelector("project_structure"),
    )

    result = loop.run(AutonomousLoopRequest("inspect authentication failure", root, project_context(root)))

    assert result.status in {LoopStatus.COMPLETED, LoopStatus.CONTINUE}
    assert len(loop.engine.prompts) >= 2
    assert all(len(prompt) <= 2_048 * 4 for prompt in loop.engine.prompts)
    assert any("Traceback" in prompt or "FAILED" in prompt for prompt in loop.engine.prompts[1:])
    assert "noise line\n" * 100 not in loop.engine.prompts[-1]


def test_context_refresh_invalidates_old_file_evidence_after_modification(tmp_path: Path) -> None:
    manager = ContextManager(tokenizer=Tokenizer(), config=ContextManagerConfig(max_context_tokens=400, reserve_output_tokens=64))
    old = ContextItem("auth-old", "read_file", "old authenticate implementation", ContextType.FILE_CONTENT, 0.9, ContextPriority.HIGH, 4, TokenCountKind.ESTIMATED, metadata={"path": "app/auth.py"})
    refreshed = manager.refresh((old,), invalidated_paths=("app/auth.py",))
    assembly = manager.assemble(task="inspect authentication", system_prompt="system", instruction="inspect", extra_items=refreshed)

    assert refreshed[0].validity.value == "invalidated"
    assert "old authenticate implementation" not in assembly.prompt


def test_progressive_expansion_and_replanning_receive_context_updates(tmp_path: Path) -> None:
    root = tmp_path / "replan"
    root.mkdir()
    _write(root, "pyproject.toml", "[project]\nname='replan'\n")
    _write(root, "app/auth.py", "def authenticate(): pass\n")
    understanding = CodebaseUnderstandingBuilder().build("inspect authentication", root, project_context=project_context(root))
    planner = StaticPlanner(_plan())
    loop = AutonomousToolLoop(
        Engine([
            'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}',
            'ACTION: FINAL\nARGS: {"message":"context updated"}',
        ]),
        registry=ToolRegistry((RecordingTool("project_structure", {"new": "authentication evidence"}),)),
        planner=planner,
        selector=StaticSelector("project_structure"),
    )

    result = loop.run(AutonomousLoopRequest("inspect authentication", root, project_context(root), codebase_understanding=understanding))

    assert result.status in {LoopStatus.COMPLETED, LoopStatus.CONTINUE}
    assert planner.requests[0].codebase_understanding is understanding
    assert result.context_assembly is not None
    assert result.context_assembly.metrics.input_budget_tokens > 0
    assert result.to_dict()["context_assembly"]["metrics"]["selected_count"] >= 1
