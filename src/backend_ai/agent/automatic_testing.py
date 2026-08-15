"""Bounded automatic test-execution orchestration for Phase 7.1."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from backend_ai.agent.execution_budget import BudgetDecision, BudgetDimension, ExecutionBudget, ExecutionBudgetLedger, ExecutionBudgetSnapshot
from backend_ai.agent.planner import ExecutionPlan, PlannerTaskType
from backend_ai.agent.registry import ToolRegistry, UnknownToolError
from backend_ai.tools.test_runner import TestRunResult, TestRunStatus

_MAX = 32


class AutomaticTestStatus(str, Enum):
    RUN = "RUN"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class AutomaticTestExecutionState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class AutomaticTestConfig:
    enabled: bool = True
    require_verification_boundary: bool = True
    implementation_tasks: tuple[PlannerTaskType, ...] = (PlannerTaskType.FEATURE, PlannerTaskType.BUG_FIX, PlannerTaskType.REFACTOR, PlannerTaskType.TEST_ADDITION, PlannerTaskType.CONFIGURATION_CHANGE)
    max_target_length: int = 512

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.require_verification_boundary, bool):
            raise ValueError("automatic test flags must be boolean")
        if self.max_target_length <= 0 or self.max_target_length > 16_384:
            raise ValueError("max_target_length must be positive and bounded")


@dataclass(frozen=True, slots=True)
class AutomaticTestRequest:
    task: str
    project_root: Path | str
    plan: ExecutionPlan | None = None
    registry: ToolRegistry | None = None
    budget_ledger: ExecutionBudgetLedger | None = None
    budget: ExecutionBudget | None = None
    config: AutomaticTestConfig = field(default_factory=AutomaticTestConfig)
    implementation_changed: bool = False
    verification_boundary: bool = False
    user_requested: bool = False
    completion_requires_tests: bool = False
    plan_requires_tests: bool = False
    test_target: str | None = None
    test_args: tuple[str, ...] = ()
    working_directory: str | None = None
    timeout_seconds: float | None = None
    max_stdout_bytes: int | None = None
    max_stderr_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must contain text")
        object.__setattr__(self, "project_root", Path(self.project_root).expanduser().resolve(strict=False))
        if self.test_target is not None and (not isinstance(self.test_target, str) or len(self.test_target) > self.config.max_target_length):
            raise ValueError("test_target is invalid or exceeds the configured bound")
        object.__setattr__(self, "test_args", tuple(self.test_args[:_MAX]))
        if any(not isinstance(item, str) or not item.strip() for item in self.test_args):
            raise ValueError("test_args must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class AutomaticTestDecision:
    status: AutomaticTestStatus
    reason: str
    trigger: str | None = None
    capability_available: bool = False
    budget_decision: BudgetDecision | None = None
    target: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "reason": self.reason, "trigger": self.trigger, "capability_available": self.capability_available, "budget_decision": self.budget_decision.to_dict() if self.budget_decision else None, "target": self.target, "warnings": list(self.warnings)}


@dataclass(frozen=True, slots=True)
class AutomaticTestExecution:
    state: AutomaticTestExecutionState
    tool_name: str = "run_tests"
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "tool_name": self.tool_name, "arguments": {str(k): v for k, v in self.arguments.items() if str(k) not in {"environment", "env"}}}


@dataclass(frozen=True, slots=True)
class AutomaticTestResult:
    decision: AutomaticTestDecision
    execution: AutomaticTestExecution
    test_run_result: TestRunResult | None = None
    budget: ExecutionBudgetSnapshot | None = None
    warnings: tuple[str, ...] = ()

    @property
    def started(self) -> bool:
        return self.execution.state in {AutomaticTestExecutionState.STARTED, AutomaticTestExecutionState.COMPLETED}

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision.to_dict(), "execution": self.execution.to_dict(), "test_run_result": self.test_run_result.to_dict() if self.test_run_result else None, "budget": self.budget.to_dict() if self.budget else None, "warnings": list(self.warnings)}


class AutomaticTestOrchestrator:
    """Decide and initiate one bounded test run through the existing registry only."""

    def decide(self, request: AutomaticTestRequest) -> AutomaticTestDecision:
        if not request.config.enabled:
            return AutomaticTestDecision(AutomaticTestStatus.SKIP, "automatic testing is disabled by configuration")
        registry = request.registry
        if registry is None or "run_tests" not in registry.names():
            return AutomaticTestDecision(AutomaticTestStatus.BLOCKED, "test capability is not available in the explicit ToolRegistry", capability_available=False)
        trigger = self._trigger(request)
        if trigger is None:
            return AutomaticTestDecision(AutomaticTestStatus.SKIP, "the task has not reached a bounded test-verification boundary", capability_available=True)
        if request.test_target is not None and not request.test_target.strip():
            return AutomaticTestDecision(AutomaticTestStatus.INVALID, "test target must contain text", trigger, True)
        return AutomaticTestDecision(AutomaticTestStatus.RUN, "structured task/plan evidence requires automatic test verification", trigger, True, target=request.test_target)

    def run(self, request: AutomaticTestRequest) -> AutomaticTestResult:
        decision = self.decide(request)
        ledger = request.budget_ledger or ExecutionBudgetLedger(request.budget or ExecutionBudget.conservative_defaults())
        if decision.status is not AutomaticTestStatus.RUN:
            return AutomaticTestResult(decision, AutomaticTestExecution(AutomaticTestExecutionState.NOT_STARTED), budget=ledger.snapshot())
        budget_decision = ledger.check_tool_operation("run_tests")
        if not budget_decision.allowed:
            blocked = AutomaticTestDecision(AutomaticTestStatus.BUDGET_EXHAUSTED, budget_decision.message, decision.trigger, True, budget_decision, decision.target)
            return AutomaticTestResult(blocked, AutomaticTestExecution(AutomaticTestExecutionState.BLOCKED), budget=ledger.snapshot())
        ledger.consume("run_tests", dimension=BudgetDimension.TOOL_CALLS)
        ledger.consume("run_tests", dimension=BudgetDimension.TEST_EXECUTIONS)
        arguments: dict[str, Any] = {"project_root": str(request.project_root)}
        if request.working_directory is not None: arguments["working_directory"] = request.working_directory
        if request.test_target is not None: arguments["test_target"] = request.test_target
        if request.test_args: arguments["test_args"] = list(request.test_args)
        if request.timeout_seconds is not None: arguments["timeout_seconds"] = request.timeout_seconds
        if request.max_stdout_bytes is not None: arguments["max_stdout_bytes"] = request.max_stdout_bytes
        if request.max_stderr_bytes is not None: arguments["max_stderr_bytes"] = request.max_stderr_bytes
        execution = AutomaticTestExecution(AutomaticTestExecutionState.STARTED, arguments=arguments)
        try:
            raw = request.registry.dispatch("run_tests", arguments) if request.registry is not None else None
            if not isinstance(raw, TestRunResult):
                raise TypeError("run_tests returned an unexpected result type")
            ledger.complete_tool()
            ledger.account_tool_result(tool_name="run_tests", stdout_bytes=len(raw.stdout.encode("utf-8")), stderr_bytes=len(raw.stderr.encode("utf-8")), tool_output_bytes=len(str(raw.to_dict()).encode("utf-8")), success=True)
            status = AutomaticTestStatus.UNAVAILABLE if raw.status in {TestRunStatus.NO_TEST_COMMAND, TestRunStatus.AMBIGUOUS_TEST_COMMAND, TestRunStatus.RESOLUTION_FAILED} else AutomaticTestStatus.RUN
            reason = "existing TestRunner returned a bounded TestRunResult" if status is AutomaticTestStatus.RUN else "existing TestRunner could not resolve a supported test command"
            return AutomaticTestResult(AutomaticTestDecision(status, reason, decision.trigger, True, target=decision.target), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED, arguments=arguments), raw, ledger.snapshot())
        except Exception as exc:
            ledger.complete_tool()
            failure = AutomaticTestDecision(AutomaticTestStatus.BLOCKED, f"automatic test dispatch failed at the existing tool boundary: {exc}", decision.trigger, True, target=decision.target)
            return AutomaticTestResult(failure, AutomaticTestExecution(AutomaticTestExecutionState.BLOCKED, arguments=arguments), budget=ledger.snapshot())

    @staticmethod
    def _trigger(request: AutomaticTestRequest) -> str | None:
        if request.user_requested: return "USER_REQUEST"
        if request.completion_requires_tests: return "COMPLETION_VERIFICATION"
        if request.plan_requires_tests: return "PLAN_TEST_REQUIREMENT"
        if request.plan and any(step.title.casefold().find("test") >= 0 or step.objective.casefold().find("test") >= 0 for step in request.plan.steps):
            return "PLAN_TEST_STEP"
        if request.implementation_changed and request.plan and request.plan.task_type in request.config.implementation_tasks and (request.verification_boundary or not request.config.require_verification_boundary):
            return "IMPLEMENTATION_VERIFICATION_BOUNDARY"
        return None


def decide_automatic_tests(request: AutomaticTestRequest) -> AutomaticTestDecision:
    return AutomaticTestOrchestrator().decide(request)


def run_automatic_tests(request: AutomaticTestRequest) -> AutomaticTestResult:
    return AutomaticTestOrchestrator().run(request)


__all__ = ["AutomaticTestConfig", "AutomaticTestDecision", "AutomaticTestExecution", "AutomaticTestExecutionState", "AutomaticTestOrchestrator", "AutomaticTestRequest", "AutomaticTestResult", "AutomaticTestStatus", "decide_automatic_tests", "run_automatic_tests"]
