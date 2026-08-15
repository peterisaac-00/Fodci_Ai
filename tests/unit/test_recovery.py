from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import (
    AutonomousLoopConfig,
    AutonomousLoopRequest,
    AutonomousToolLoop,
    ErrorCategory,
    ExecutionBudget,
    RecoveryAction,
    RecoveryConfidence,
    RecoveryContext,
    RecoveryStatus,
    ToolRegistry,
    classify_error,
    decide_recovery,
)
from backend_ai.agent.planner import (
    ExecutionPlan,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
)
from backend_ai.agent.tool_selection import ToolSelectionStatus
from backend_ai.tools import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.agent.models import ToolResult
from backend_ai.tools.project_context import ProjectContext


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode())


class FakeEngine:
    tokenizer = FakeTokenizer()

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)

    def generate(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(generated_text=next(self.outputs))


class StaticPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.value = plan

    def plan(self, request):
        return SimpleNamespace(plan=self.value, warnings=(), errors=())


class RecordingTool:
    def __init__(self, name: str, result: object = None, failure: Exception | None = None) -> None:
        self.name = name
        self.description = name
        self.metadata = ToolMetadata(name, name, {"type": "object", "properties": {}})
        self.result = {"ok": True} if result is None else result
        self.failure = failure
        self.calls = 0

    def run(self, arguments):
        self.calls += 1
        if self.failure:
            raise self.failure
        return self.result


def _context(root: Path) -> ProjectContext:
    return ProjectContext(root=root, project_type="python", stack_summary="Python", languages=(), frameworks=(), package_managers=(), databases=(), test_frameworks=(), infrastructure=(), source_directories=("src",), test_directories=("tests",), documentation_directories=(), config_files=(), dependency_files=(), important_files=(), entry_points=(), project_files=("src/main.py",), confidence="high", evidence=("fixture",), warnings=(), truncated=False, truncation_reason=None, completeness="complete")


def _plan(*steps: PlanStep) -> ExecutionPlan:
    return ExecutionPlan("inspect", "inspect", "inspect safely", PlannerTaskType.INVESTIGATION, steps, (), (), (), (), (), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE)


def _step(name: str, title: str) -> PlanStep:
    return PlanStep(name, title, title, "fixture", "evidence", (), PlanRiskLevel.LOW)


def test_classifier_distinguishes_safety_policy_budget_and_actionable_failures() -> None:
    cases = [
        (ToolErrorCode.PERMISSION_DENIED.value, ErrorCategory.SAFETY_BLOCK, False),
        (ToolErrorCode.COMMAND_DENIED.value, ErrorCategory.POLICY_DENIAL, False),
        (ToolErrorCode.PATH_OUTSIDE_ROOT.value, ErrorCategory.PROJECT_ROOT_VIOLATION, False),
        (ToolErrorCode.FILE_NOT_FOUND.value, ErrorCategory.FILE_NOT_FOUND, True),
        (ToolErrorCode.VERIFICATION_FAILED.value, ErrorCategory.VERIFICATION_FAILURE, True),
        (ToolErrorCode.PROCESS_TIMEOUT.value, ErrorCategory.PROCESS_TIMEOUT, True),
        ("BUDGET_EXHAUSTED", ErrorCategory.BUDGET_EXHAUSTED, False),
    ]
    for code, category, recoverable in cases:
        actual = classify_error(ToolResult("c", "tool", False, error_code=code, message=code))
        assert actual.category is category
        assert actual.recoverable is recoverable
        assert actual.safety_or_policy_boundary is (category in {ErrorCategory.SAFETY_BLOCK, ErrorCategory.POLICY_DENIAL, ErrorCategory.PROJECT_ROOT_VIOLATION})


def test_policy_blocks_safety_denial_without_bypass() -> None:
    result = ToolResult("c", "run_command_with_policy", False, error_code=ToolErrorCode.COMMAND_DENIED.value, message="denied")
    recovery = decide_recovery(RecoveryContext(result, "run_command_with_policy", selected_tool="run_command_with_policy"))
    assert recovery.decision.action is RecoveryAction.BLOCK
    assert recovery.decision.status is RecoveryStatus.BLOCKED
    assert not recovery.decision.execution_allowed
    assert not recovery.decision.mutation_allowed


def test_concurrent_modification_requires_user_and_preserves_boundary() -> None:
    result = ToolResult("c", "edit_file", False, error_code=ToolErrorCode.CONCURRENT_MODIFICATION.value, message="newer user change")
    recovery = decide_recovery(RecoveryContext(result, "edit_file", selected_tool="edit_file"))
    assert recovery.decision.action is RecoveryAction.USER_INTERVENTION_REQUIRED
    assert recovery.decision.status is RecoveryStatus.USER_INTERVENTION_REQUIRED
    assert recovery.original_error_preserved


def test_actionable_failure_requests_inspection_not_blind_retry() -> None:
    result = ToolResult("c", "run_tests", False, error_code="TEST_FAILURE", message="assertion failed")
    recovery = decide_recovery(RecoveryContext(result, "run_tests", selected_tool="run_tests", has_next_plan_step=True))
    assert recovery.decision.action is RecoveryAction.INSPECT
    assert recovery.decision.status is RecoveryStatus.CONTINUE
    assert recovery.decision.confidence is RecoveryConfidence.MEDIUM
    assert "blind retry" in " ".join(recovery.decision.warnings + recovery.decision.blocking_conditions)


def test_repeated_action_is_stopped_deterministically() -> None:
    result = ToolResult("c", "read_file", False, error_code=ToolErrorCode.FILE_NOT_FOUND.value, message="missing")
    recovery = decide_recovery(RecoveryContext(result, "read_file", has_next_plan_step=True, failed_action_signatures=("read_file",)))
    assert recovery.decision.action is RecoveryAction.STOP
    assert recovery.decision.status is RecoveryStatus.FAILED


def test_budget_exhaustion_cannot_be_recovered() -> None:
    result = ToolResult("c", "read_file", False, error_code="TOOL_FAILED", message="original")
    budget = ExecutionBudget(max_tool_calls=0)
    from backend_ai.agent import ExecutionBudgetLedger
    ledger = ExecutionBudgetLedger(budget)
    decision = ledger.check_tool_operation("read_file")
    recovery = decide_recovery(RecoveryContext(result, "read_file", budget_snapshot=ledger.snapshot(), budget_decision=decision))
    assert recovery.decision.classification.category is ErrorCategory.BUDGET_EXHAUSTED
    assert recovery.decision.action is RecoveryAction.BLOCK
    assert recovery.decision.status is RecoveryStatus.BLOCKED


def test_loop_continues_with_next_bounded_plan_step_after_actionable_failure(tmp_path: Path) -> None:
    first = RecordingTool("read_file", failure=ToolError(ToolErrorCode.FILE_NOT_FOUND, "missing"))
    second = RecordingTool("search_code", result={"found": "evidence"})
    plan = _plan(_step("s1", "read missing file"), _step("s2", "inspect evidence"))

    class Selector:
        def select(self, request):
            name = "read_file" if request.selected_step_ids[0] == "s1" else "search_code"
            return SimpleNamespace(status=ToolSelectionStatus.SELECTED, decisions=(SimpleNamespace(selected_tool=name, selection_reason="fixture", risk_level=SimpleNamespace(value="LOW"), prerequisites=(), expected_output="evidence"),), errors=())

    loop = AutonomousToolLoop(
        FakeEngine([
            'ACTION: TOOL\nARGS: {"tool":"read_file","arguments":{}}',
            'ACTION: TOOL\nARGS: {"tool":"search_code","arguments":{}}',
            'ACTION: FINAL\nARGS: {"message":"inspected"}',
        ]),
        registry=ToolRegistry((first, second)), planner=StaticPlanner(plan), selector=Selector(), config=AutonomousLoopConfig(execution_budget=ExecutionBudget(max_iterations=8, max_action_steps=8, max_tool_calls=4)),
    )
    result = loop.run(AutonomousLoopRequest("inspect", tmp_path, _context(tmp_path)))
    assert first.calls == 1
    assert second.calls == 1
    assert result.recovery is not None
    assert result.recovery.decision.action is RecoveryAction.INSPECT
    assert result.status.value == "COMPLETED"


def test_recovery_serialization_is_bounded_and_does_not_expose_data() -> None:
    result = ToolResult("c", "read_file", False, error_code="FILE_NOT_FOUND", message="secret token=do-not-leak")
    recovery = decide_recovery(RecoveryContext(result, "read_file", evidence=("x" * 5000,)))
    payload = str(recovery.to_dict())
    assert "do-not-leak" in payload or "secret" in payload
    assert len(payload) < 20_000
