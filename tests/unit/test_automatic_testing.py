from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import (
    AutomaticTestConfig,
    AutomaticTestRequest,
    AutomaticTestStatus,
    ExecutionBudget,
    ToolRegistry,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    PlannerConfidence,
    PlannerTaskType,
    ExecutionPlan,
    run_automatic_tests,
    decide_automatic_tests,
)
from backend_ai.tools import ToolMetadata
from backend_ai.tools.test_runner import TestRunPlan as _TestRunPlan, TestRunResult as _TestRunResult, TestRunStatus as _TestRunStatus


class FakeTestTool:
    name = "run_tests"
    description = "fixture test runner"
    metadata = ToolMetadata(name, description, {"type": "object", "required": ["project_root"], "properties": {"project_root": {"type": "string"}, "test_target": {"type": "string"}, "test_args": {"type": "array"}}})

    def __init__(self, result: _TestRunResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    def run(self, arguments):
        self.calls.append(dict(arguments))
        return self.result


def _plan(kind: PlannerTaskType = PlannerTaskType.FEATURE, steps: tuple[PlanStep, ...] = ()) -> ExecutionPlan:
    return ExecutionPlan("implement feature", "implement feature", "goal", kind, steps, (), (), (), (), (), PlannerConfidence.HIGH, (), PlanCompleteness.COMPLETE)


def _result(status: _TestRunStatus = _TestRunStatus.COMPLETED) -> _TestRunResult:
    plan = _TestRunPlan(("python", "-m", "pytest"), ".", "pytest", "fixture", ("pytest.ini",), "HIGH", False)
    return _TestRunResult(status, plan, None, None, None, "python", "pytest", ("pytest",), ("fixture evidence",), ())


def _request(root: Path, **kwargs) -> AutomaticTestRequest:
    return AutomaticTestRequest("implement feature", root, **kwargs)


def test_implementation_at_verification_boundary_runs_when_capability_is_explicit(tmp_path: Path) -> None:
    tool = FakeTestTool(_result())
    request = _request(tmp_path, plan=_plan(), registry=ToolRegistry((tool,)), implementation_changed=True, verification_boundary=True)
    result = run_automatic_tests(request)
    assert result.decision.status is AutomaticTestStatus.RUN
    assert result.started
    assert tool.calls[0]["project_root"] == str(tmp_path.resolve())


def test_plan_test_step_triggers_without_implementation_flag(tmp_path: Path) -> None:
    tool = FakeTestTool(_result())
    step = PlanStep("tests", "Run tests", "execute tests", "required", "test result", (), PlanRiskLevel.LOW, True)
    result = run_automatic_tests(_request(tmp_path, plan=_plan(steps=(step,)), registry=ToolRegistry((tool,))))
    assert result.decision.status is AutomaticTestStatus.RUN


def test_completion_requirement_triggers_testing(tmp_path: Path) -> None:
    tool = FakeTestTool(_result())
    result = run_automatic_tests(_request(tmp_path, plan=_plan(), registry=ToolRegistry((tool,)), completion_requires_tests=True))
    assert result.decision.status is AutomaticTestStatus.RUN


def test_documentation_task_skips_by_default(tmp_path: Path) -> None:
    tool = FakeTestTool(_result())
    result = run_automatic_tests(_request(tmp_path, plan=_plan(PlannerTaskType.DOCUMENTATION_CHANGE), registry=ToolRegistry((tool,)), implementation_changed=True, verification_boundary=True))
    assert result.decision.status is AutomaticTestStatus.SKIP
    assert not result.started


def test_investigation_can_skip(tmp_path: Path) -> None:
    result = decide_automatic_tests(_request(tmp_path, plan=_plan(PlannerTaskType.INVESTIGATION), implementation_changed=False))
    assert result.status is AutomaticTestStatus.BLOCKED
    assert "ToolRegistry" in result.reason


def test_missing_test_capability_is_blocked_without_execution(tmp_path: Path) -> None:
    result = run_automatic_tests(_request(tmp_path, plan=_plan(), implementation_changed=True, verification_boundary=True))
    assert result.decision.status is AutomaticTestStatus.BLOCKED
    assert not result.started


def test_budget_exhaustion_blocks_before_run_tests(tmp_path: Path) -> None:
    tool = FakeTestTool(_result())
    result = run_automatic_tests(_request(tmp_path, plan=_plan(), registry=ToolRegistry((tool,)), implementation_changed=True, verification_boundary=True, budget=ExecutionBudget(max_tool_calls=0)))
    assert result.decision.status is AutomaticTestStatus.BUDGET_EXHAUSTED
    assert not result.started
    assert tool.calls == []


def test_target_and_structured_args_are_preserved(tmp_path: Path) -> None:
    tool = FakeTestTool(_result())
    result = run_automatic_tests(_request(tmp_path, plan=_plan(), registry=ToolRegistry((tool,)), user_requested=True, test_target="tests/test_api.py", test_args=("-q",)))
    assert result.decision.status is AutomaticTestStatus.RUN
    assert tool.calls[0]["test_target"] == "tests/test_api.py"
    assert tool.calls[0]["test_args"] == ["-q"]


def test_no_test_command_is_unavailable_and_not_analyzer(tmp_path: Path) -> None:
    tool = FakeTestTool(_result(_TestRunStatus.NO_TEST_COMMAND))
    result = run_automatic_tests(_request(tmp_path, plan=_plan(), registry=ToolRegistry((tool,)), user_requested=True))
    assert result.decision.status is AutomaticTestStatus.UNAVAILABLE
    assert result.test_run_result is not None
    assert result.test_run_result.status is _TestRunStatus.NO_TEST_COMMAND


def test_failure_result_is_preserved_without_retry_or_fix(tmp_path: Path) -> None:
    tool = FakeTestTool(_result(_TestRunStatus.EXECUTION_ERROR))
    result = run_automatic_tests(_request(tmp_path, plan=_plan(), registry=ToolRegistry((tool,)), user_requested=True))
    assert result.test_run_result is not None
    assert result.test_run_result.status is _TestRunStatus.EXECUTION_ERROR
    assert len(tool.calls) == 1


def test_disabled_config_skips(tmp_path: Path) -> None:
    tool = FakeTestTool(_result())
    result = run_automatic_tests(_request(tmp_path, plan=_plan(), registry=ToolRegistry((tool,)), config=AutomaticTestConfig(enabled=False), user_requested=True))
    assert result.decision.status is AutomaticTestStatus.SKIP
    assert tool.calls == []


def test_decision_is_deterministic(tmp_path: Path) -> None:
    request = _request(tmp_path, plan=_plan(), user_requested=True)
    assert decide_automatic_tests(request).to_dict() == decide_automatic_tests(request).to_dict()
