from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend_ai.agent import (
    AgentMessage,
    AgentUsage,
    AutonomousLoopConfig,
    ExecutionBudget,
    AutonomousLoopRequest,
    AutonomousToolLoop,
    ExecutionPlan,
    LoopActionParseError,
    LoopActionType,
    LoopFailureCode,
    LoopLifecycleState,
    LoopStateError,
    LoopStateMachine,
    LoopStatus,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    ToolRegistry,
    ToolSelectionStatus,
    parse_loop_action,
)
from backend_ai.agent.planner import PlannerResultStatus
from backend_ai.agent.protocol import ParsedFinalAnswer
from backend_ai.tools import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.project_context import ProjectContext


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class FakeEngine:
    tokenizer = FakeTokenizer()

    def __init__(self, outputs: list[str], *, fail: Exception | None = None) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []
        self.fail = fail

    def generate(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if self.fail is not None:
            raise self.fail
        return SimpleNamespace(generated_text=next(self.outputs))


class StaticPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan_value = plan

    def plan(self, request):
        return SimpleNamespace(plan=self.plan_value, warnings=(), errors=(), status=PlannerResultStatus.CREATED)


class RecordingTool:
    def __init__(self, name: str, *, result: object = None, schema: dict | None = None, failure: Exception | None = None) -> None:
        self.name = name
        self.description = f"recording {name}"
        self.metadata = ToolMetadata(name, self.description, schema or {"type": "object", "properties": {}})
        self.result = result if result is not None else {"ok": True}
        self.failure = failure
        self.calls: list[dict] = []

    def run(self, arguments):
        self.calls.append(dict(arguments))
        if self.failure is not None:
            raise self.failure
        return self.result


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
        evidence=("supplied fixture",),
        warnings=(),
        truncated=False,
        truncation_reason=None,
        completeness="complete",
    )


def _plan(*steps: PlanStep) -> ExecutionPlan:
    return ExecutionPlan(
        task="Inspect the project",
        normalized_task="Inspect the project",
        goal="Inspect the supplied project safely.",
        task_type=PlannerTaskType.INVESTIGATION,
        steps=tuple(steps),
        assumptions=(),
        constraints=(),
        risks=(),
        expected_changes=(),
        verification_strategy=(),
        confidence=PlannerConfidence.HIGH,
        warnings=(),
        completeness=PlanCompleteness.COMPLETE,
    )


def _step(step_id: str, title: str, objective: str | None = None, dependencies: tuple[str, ...] = ()) -> PlanStep:
    return PlanStep(step_id, title, objective or title, "supplied plan", "observed result", dependencies, PlanRiskLevel.LOW)


def _loop(root: Path, outputs: list[str], tools: tuple[RecordingTool, ...], plan: ExecutionPlan, *, config: AutonomousLoopConfig | None = None):
    registry = ToolRegistry(tools)
    return AutonomousToolLoop(FakeEngine(outputs), registry=registry, planner=StaticPlanner(plan), config=config), registry


class StaticSelector:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def select(self, request):
        step_id = request.selected_step_ids[0]
        tool_name = self.mapping[step_id]
        decision = SimpleNamespace(
            selected_tool=tool_name,
            selection_reason="fixture capability mapping",
            risk_level=SimpleNamespace(value="LOW"),
            prerequisites=(),
            expected_output="structured evidence",
        )
        return SimpleNamespace(status=ToolSelectionStatus.SELECTED, decisions=(decision,), errors=())


def test_action_protocol_requires_explicit_json_and_rejects_prose() -> None:
    final = parse_loop_action('ACTION: FINAL\nARGS: {"message":"done"}')
    assert final.action_type is LoopActionType.FINAL
    assert final.message == "done"
    tool = parse_loop_action('ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{"path":"x"}}')
    assert tool.action_type is LoopActionType.TOOL
    assert tool.tool_name == "project_structure"
    with pytest.raises(LoopActionParseError) as prose:
        parse_loop_action("please run ls")
    assert prose.value.code == LoopFailureCode.INVALID_ACTION.value
    with pytest.raises(LoopActionParseError):
        parse_loop_action('ACTION: TOOL\nARGS: {"tool":"x","arguments":[] }')


def test_state_machine_rejects_invalid_transitions() -> None:
    machine = LoopStateMachine()
    assert machine.transition(LoopLifecycleState.PLANNING) is LoopLifecycleState.PLANNING
    with pytest.raises(LoopStateError):
        machine.transition(LoopLifecycleState.COMPLETED)


def test_single_read_only_tool_then_final_updates_context_and_history(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = RecordingTool("project_structure", result={"finding": "safe"})
    plan = _plan(_step("s1", "Inspect project structure"))
    loop, registry = _loop(root, [
        'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}',
        'ACTION: FINAL\nARGS: {"message":"inspection complete"}',
    ], (tool,), plan)
    result = loop.run(AutonomousLoopRequest("Inspect the project", root, _context(root)))
    assert result.status is LoopStatus.COMPLETED
    assert result.final_answer == "inspection complete"
    assert [call.name for call in result.tool_calls] == ["project_structure"]
    assert result.tool_results[0].success
    assert tool.calls == [{}]
    assert len(result.steps) == 2
    assert result.state.lifecycle is LoopLifecycleState.COMPLETED
    repeat_tool = RecordingTool("project_structure", result={"finding": "safe"})
    repeat_loop, _ = _loop(root, [
        'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}',
        'ACTION: FINAL\nARGS: {"message":"inspection complete"}',
    ], (repeat_tool,), plan)
    first_dict = result.to_dict()
    second_dict = repeat_loop.run(AutonomousLoopRequest("Inspect the project", root, _context(root))).to_dict()
    first_dict.pop("execution_budget", None)
    second_dict.pop("execution_budget", None)
    assert first_dict == second_dict


def test_multiple_sequential_tools_and_second_model_action(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    first = RecordingTool("project_structure", result={"next": "read_file"})
    second = RecordingTool("read_file", result={"contents": "bounded"})
    plan = _plan(_step("s1", "Inspect project structure"), _step("s2", "Inspect exact file contents", dependencies=("s1",)))
    loop, _ = _loop(root, [
        'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}',
        'ACTION: TOOL\nARGS: {"tool":"read_file","arguments":{"path":"src/main.py"}}',
        'ACTION: FINAL\nARGS: {"message":"done after observation"}',
    ], (first, second), plan)
    result = loop.run(AutonomousLoopRequest("Inspect the project", root, _context(root)))
    assert result.status is LoopStatus.COMPLETED
    assert [call.name for call in result.tool_calls] == ["project_structure", "read_file"]
    assert [item["contents"] if "contents" in item else item for item in [first.result, second.result]]
    assert len(result.state.history) == 4
    assert "bounded" in result.state.history[-1].content
    assert result.usage.tool_calls == 2


def test_final_action_terminates_without_tool_execution(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = RecordingTool("project_structure")
    plan = _plan(_step("s1", "Inspect project structure"))
    loop, _ = _loop(root, ['ACTION: FINAL\nARGS: {"message":"nothing to do"}'], (tool,), plan)
    result = loop.run(AutonomousLoopRequest("Inspect", root, _context(root)))
    assert result.status is LoopStatus.CONTINUE
    assert result.stop_evaluation is not None
    assert result.stop_evaluation.decision.value == "CONTINUE"
    assert result.tool_calls == ()
    assert tool.calls == []


def test_unknown_or_mismatched_tool_is_rejected_before_dispatch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = RecordingTool("project_structure")
    plan = _plan(_step("s1", "Inspect project structure"))
    loop, _ = _loop(root, ['ACTION: TOOL\nARGS: {"tool":"unknown","arguments":{}}'], (tool,), plan)
    result = loop.run(AutonomousLoopRequest("Inspect", root, _context(root)))
    assert result.status is LoopStatus.INVALID_ACTION
    assert tool.calls == []


def test_tool_execution_failure_is_preserved_without_retry(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = RecordingTool("project_structure", failure=ToolError(ToolErrorCode.PERMISSION_DENIED, "denied"))
    plan = _plan(_step("s1", "Inspect project structure"))
    loop, _ = _loop(root, ['ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}', 'ACTION: FINAL\nARGS: {"message":"should not run"}'], (tool,), plan)
    result = loop.run(AutonomousLoopRequest("Inspect", root, _context(root)))
    assert result.status is LoopStatus.TOOL_EXECUTION_FAILED
    assert len(result.tool_calls) == 1
    assert result.tool_results[0].error_code == ToolErrorCode.PERMISSION_DENIED.value
    assert len(loop.engine.prompts) == 1


def test_mutation_is_unavailable_in_read_only_registry_and_opt_in_when_supplied(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    plan = _plan(_step("s1", "Create a new file"))
    read_only_loop, _ = _loop(root, ['ACTION: TOOL\nARGS: {"tool":"write_file","arguments":{"path":"new.txt","content":"x"}}'], (), plan)
    unavailable = read_only_loop.run(AutonomousLoopRequest("Create a new file", root, _context(root)))
    assert unavailable.status is LoopStatus.TOOL_UNAVAILABLE

    writer = RecordingTool("write_file", schema={"type": "object", "required": ["project_root", "path", "content"], "properties": {"project_root": {}, "path": {}, "content": {}}})
    write_loop, _ = _loop(root, ['ACTION: TOOL\nARGS: {"tool":"write_file","arguments":{"path":"new.txt","content":"x"}}', 'ACTION: FINAL\nARGS: {"message":"created"}'], (writer,), plan)
    created = write_loop.run(AutonomousLoopRequest("Create a new file", root, _context(root)))
    assert created.status is LoopStatus.CONTINUE
    assert created.stop_evaluation is not None
    assert created.stop_evaluation.reason.value == "VERIFICATION_REQUIRED"
    assert writer.calls[0]["project_root"] == str(root)
    assert created.steps[0].mutated_project


def test_project_root_escape_is_rejected_before_dispatch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = RecordingTool("read_file", schema={"type": "object", "properties": {"project_root": {}, "path": {}}})
    plan = _plan(_step("s1", "Inspect exact file contents"))
    loop, _ = _loop(root, ['ACTION: TOOL\nARGS: {"tool":"read_file","arguments":{"project_root":"/outside","path":"x"}}'], (tool,), plan)
    result = loop.run(AutonomousLoopRequest("Inspect", root, _context(root)))
    assert result.status is LoopStatus.INVALID_ACTION
    assert tool.calls == []


def test_command_policy_remains_boundary_for_shell_bypass(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    from backend_ai.agent import ToolRegistry as Registry

    plan = _plan(_step("s1", "Run an approved command"))
    loop = AutonomousToolLoop(
        FakeEngine(['ACTION: TOOL\nARGS: {"tool":"run_command_with_policy","arguments":{"argv":["sh","-c","echo unsafe"],"working_directory":"."}}']),
        registry=Registry.with_command_policy(),
        planner=StaticPlanner(plan),
    )
    result = loop.run(AutonomousLoopRequest("Run an approved command", root, _context(root)))
    assert result.status is LoopStatus.TOOL_EXECUTION_FAILED
    assert result.tool_results[0].error_code == ToolErrorCode.SHELL_BYPASS_ATTEMPT.value


def test_context_limits_and_output_truncation_are_recorded(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = RecordingTool("project_structure", result={"text": "x" * 10_000})
    plan = _plan(_step("s1", "Inspect project structure"))
    loop, _ = _loop(root, ['ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}', 'ACTION: FINAL\nARGS: {"message":"done"}'], (tool,), plan, config=AutonomousLoopConfig(max_context_tokens=1_024, max_tool_result_chars=128))
    result = loop.run(AutonomousLoopRequest("Inspect", root, _context(root)))
    assert result.status is LoopStatus.COMPLETED
    assert result.state.context_truncated or result.usage.context_truncations >= 0
    assert len(result.state.history[-1].content) <= 128


def test_emergency_bound_prevents_infinite_model_tool_cycle_without_retry(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = RecordingTool("project_structure", result={"ok": True})
    plan = _plan(*[_step(f"s{i}", "Inspect project structure") for i in range(1, 12)])
    outputs = ['ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}'] * 12
    loop, _ = _loop(root, outputs, (tool,), plan)
    result = loop.run(AutonomousLoopRequest("Inspect", root, _context(root)))
    assert result.status is LoopStatus.LOOP_BOUND_REACHED
    assert len(result.tool_calls) <= 8
    assert len(tool.calls) == len(result.tool_calls)


def test_determinism_and_default_agent_loop_remain_separate(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    plan = _plan(_step("s1", "Inspect project structure"))
    outputs = ['ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}', 'ACTION: FINAL\nARGS: {"message":"done"}']
    first, _ = _loop(root, outputs, (RecordingTool("project_structure", result={"ok": True}),), plan)
    second, _ = _loop(root, outputs, (RecordingTool("project_structure", result={"ok": True}),), plan)
    request = AutonomousLoopRequest("Inspect", root, _context(root))
    first_dict = first.run(request).to_dict()
    second_dict = second.run(request).to_dict()
    first_dict.pop("execution_budget", None)
    second_dict.pop("execution_budget", None)
    assert first_dict == second_dict
    assert ToolRegistry.default().names() == ("list_files", "project_context", "project_structure", "read_file", "search_code")


def test_missing_context_bootstraps_only_through_registered_project_context(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    context_tool = RecordingTool(
        "project_context",
        result=_context(root),
        schema={"type": "object", "required": ["project_root"], "properties": {"project_root": {}}},
    )
    inspector = RecordingTool("project_structure", result={"safe": True})
    plan = _plan(_step("s1", "Inspect project structure"))
    loop = AutonomousToolLoop(
        FakeEngine([
            'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}',
            'ACTION: FINAL\nARGS: {"message":"bootstrapped"}',
        ]),
        registry=ToolRegistry((context_tool, inspector)),
        planner=StaticPlanner(plan),
    )
    result = loop.run(AutonomousLoopRequest("Inspect", root))
    assert result.status is LoopStatus.COMPLETED
    assert [call.name for call in result.tool_calls] == ["project_context", "project_structure"]
    assert result.usage.tool_calls == 2
    assert result.steps[0].plan_step_id == "context-1"
    assert context_tool.calls[0]["project_root"] == str(root)


def test_secret_like_arguments_are_redacted_in_loop_serialization(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = RecordingTool("project_structure", result={"ok": True})
    plan = _plan(_step("s1", "Inspect project structure"))
    loop, _ = _loop(root, [
        'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{"api_key":"secret-value"}}',
        'ACTION: FINAL\nARGS: {"message":"done"}',
    ], (tool,), plan)
    result = loop.run(AutonomousLoopRequest("Inspect", root, _context(root)))
    serialized = str(result.to_dict())
    assert "secret-value" not in serialized
    assert "[REDACTED]" in serialized
    with pytest.raises(TypeError):
        result.steps[0].sanitized_arguments["new"] = "value"


def test_explicitly_tiny_context_budget_fails_structured_without_dispatch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    tool = RecordingTool("project_structure")
    plan = _plan(_step("s1", "Inspect project structure"))
    loop, _ = _loop(root, ['ACTION: FINAL\nARGS: {"message":"not reached"}'], (tool,), plan, config=AutonomousLoopConfig(max_context_tokens=64, reserve_response_tokens=16))
    result = loop.run(AutonomousLoopRequest("x" * 500, root, _context(root)))
    assert result.status is LoopStatus.CONTEXT_LIMIT_REACHED
    assert tool.calls == []


def test_mutation_plus_structured_verification_reaches_done(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    writer = RecordingTool("write_file", schema={"type": "object", "required": ["project_root", "path", "content"], "properties": {"project_root": {}, "path": {}, "content": {}}})
    verifier = RecordingTool("verify_modification", result=SimpleNamespace(success=True, complete=True))
    plan = _plan(_step("s1", "Create a new file"), _step("s2", "Verify the modification", dependencies=("s1",)))
    loop = AutonomousToolLoop(
        FakeEngine([
            'ACTION: TOOL\nARGS: {"tool":"write_file","arguments":{"path":"new.txt","content":"x"}}',
            'ACTION: TOOL\nARGS: {"tool":"verify_modification","arguments":{}}',
            'ACTION: FINAL\nARGS: {"message":"created and verified"}',
        ]),
        registry=ToolRegistry((writer, verifier)),
        planner=StaticPlanner(plan),
        selector=StaticSelector({"s1": "write_file", "s2": "verify_modification"}),
    )
    result = loop.run(AutonomousLoopRequest("Create and verify a new file", root, _context(root)))
    assert result.status is LoopStatus.COMPLETED
    assert result.stop_evaluation is not None
    assert result.stop_evaluation.decision.value == "DONE"
    assert result.stop_evaluation.reason.value == "VERIFICATION_PASSED"
    assert len(result.tool_calls) == 2


def test_budget_blocks_second_tool_before_dispatch(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    first_tool = RecordingTool("project_structure", result={"ok": True})
    second_tool = RecordingTool("read_file", result={"content": "x"})
    plan = _plan(_step("s1", "Inspect project structure"), _step("s2", "Read the known file", dependencies=("s1",)))
    config = AutonomousLoopConfig(execution_budget=ExecutionBudget(max_tool_calls=1, max_iterations=8, max_action_steps=8))
    loop = AutonomousToolLoop(
        FakeEngine([
            'ACTION: TOOL\nARGS: {"tool":"project_structure","arguments":{}}',
            'ACTION: TOOL\nARGS: {"tool":"read_file","arguments":{"path":"x.py"}}',
        ]),
        registry=ToolRegistry((first_tool, second_tool)),
        planner=StaticPlanner(plan),
        config=config,
    )
    result = loop.run(AutonomousLoopRequest("Inspect and read", root, _context(root)))
    assert result.status is LoopStatus.BLOCKED
    assert result.stop_evaluation is not None
    assert result.stop_evaluation.decision.value == "BUDGET_EXHAUSTED"
    assert result.execution_budget is not None
    assert result.execution_budget.usage.tool_calls_attempted == 1
    assert len(first_tool.calls) == 1
    assert second_tool.calls == []
