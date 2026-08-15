"""Bounded opt-in autonomous tool execution loop for Phase 6.3.

The loop is deliberately separate from the Phase 3.6 read-only ``AgentLoop``.
It consumes explicit registry capabilities, uses the Planner and ToolSelector,
and dispatches only through ``ToolRegistry``. It has one fixed emergency bound;
configurable stop conditions, execution budgets, and bounded evidence-driven recovery
remain explicit safety layers over this loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import json
from pathlib import Path
from types import MappingProxyType
import re
from typing import Any

from backend_ai.agent.budget import ContextBudget, ContextBudgetError
from backend_ai.agent.execution_budget import (
    BudgetDimension,
    BudgetDecision,
    ExecutionBudget,
    ExecutionBudgetLedger,
    ExecutionBudgetSnapshot,
)
from backend_ai.agent.models import (
    AgentConfig,
    AgentMessage,
    AgentMessageRole,
    AgentUsage,
    ToolCall,
    ToolResult,
)
from backend_ai.agent.planner import (
    ExecutionPlan,
    PlanCompleteness,
    PlanRiskLevel,
    PlanStep,
    Planner,
    PlannerConfidence,
    PlannerRequest,
    PlannerTaskType,
)
from backend_ai.agent.registry import ToolRegistry, ToolRegistryError, UnknownToolError
from backend_ai.agent.automatic_testing import AutomaticTestRequest, AutomaticTestResult, AutomaticTestOrchestrator
from backend_ai.agent.test_failure_analysis import FailureAnalysisConfig, TestFailureAnalysis, TestFailureAnalysisRequest, TestFailureAnalyzer
from backend_ai.agent.root_cause_analysis import RootCauseAnalysis, RootCauseAnalysisConfig, RootCauseAnalysisRequest, RootCauseAnalyzer
from backend_ai.agent.completion import CompletionDecision, EvidenceStrength, TaskCompletionEvidence, TaskCompletionRequest, TaskCompletionResult, verify_task_completion
from backend_ai.agent.recovery import RecoveryContext, RecoveryResult, RecoveryStatus, decide_recovery
from backend_ai.agent.stop_conditions import (
    StopConditionRequest,
    StopDecision,
    StopEvaluation,
    StopReason,
    VerificationEvidence,
    VerificationState,
    evaluate_stop_condition,
)
from backend_ai.agent.tool_selection import (
    ToolSelectionDecision,
    ToolSelectionRequest,
    ToolSelectionResult,
    ToolSelectionStatus,
    ToolSelector,
)
from backend_ai.tools.base import ToolError, ToolErrorCode
from backend_ai.tools.project_context import ProjectContext


# This is intentionally a private, fixed safety boundary. Phase 6.5 will define
# a separate configurable iteration/budget feature; the model cannot override it.
_EMERGENCY_TOOL_EXECUTION_BOUND = 8
_MAX_TEXT_CHARS = 4_096
_SECRET_KEY_RE = re.compile(r"(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)", re.IGNORECASE)
_SECRET_TEXT_RE = re.compile(r'''((?:"|')?(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credential)(?:"|')?\s*(?:=|:)\s*)(?:"[^"]*"|'[^']*'|[^,\s}\]]+)''', re.IGNORECASE)
_SHELL_TEXT_RE = re.compile(r"(?:\b(?:bash|sh|cmd|powershell|pwsh)\s+-c\b|[;&|`$()]|\bshell\s*=)", re.IGNORECASE)


class LoopLifecycleState(str, Enum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    SELECTING_TOOL = "SELECTING_TOOL"
    VALIDATING_ACTION = "VALIDATING_ACTION"
    EXECUTING_TOOL = "EXECUTING_TOOL"
    OBSERVING_RESULT = "OBSERVING_RESULT"
    UPDATING_CONTEXT = "UPDATING_CONTEXT"
    REQUESTING_NEXT_ACTION = "REQUESTING_NEXT_ACTION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class LoopActionType(str, Enum):
    TOOL = "TOOL"
    FINAL = "FINAL"


class LoopStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INVALID_ACTION = "INVALID_ACTION"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    CONTEXT_LIMIT_REACHED = "CONTEXT_LIMIT_REACHED"
    LOOP_BOUND_REACHED = "LOOP_BOUND_REACHED"
    EXECUTION_ABORTED = "EXECUTION_ABORTED"
    CONTINUE = "CONTINUE"
    BLOCKED = "BLOCKED"


class LoopFailureCode(str, Enum):
    INVALID_ACTION = "INVALID_ACTION"
    INVALID_TOOL_CALL = "INVALID_TOOL_CALL"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    TOOL_ARGUMENT_ERROR = "TOOL_ARGUMENT_ERROR"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    CONTEXT_LIMIT_REACHED = "CONTEXT_LIMIT_REACHED"
    LOOP_BOUND_REACHED = "LOOP_BOUND_REACHED"
    INVALID_STATE = "INVALID_STATE"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    EXECUTION_ABORTED = "EXECUTION_ABORTED"


class LoopActionParseError(ValueError):
    """A model response did not use the strict Phase 6.3 action protocol."""

    def __init__(self, code: LoopFailureCode | str, message: str) -> None:
        self.code = code.value if isinstance(code, LoopFailureCode) else str(code)
        self.message = message
        super().__init__(message)


class LoopStateError(RuntimeError):
    """An invalid lifecycle transition was requested."""

    def __init__(self, state: LoopLifecycleState, target: LoopLifecycleState) -> None:
        self.state = state
        self.target = target
        super().__init__(f"Invalid autonomous-loop transition: {state.value} -> {target.value}.")


@dataclass(frozen=True, slots=True)
class LoopAction:
    """Strict parsed action; never contains a free-form executable command."""

    action_type: LoopActionType
    tool_name: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, LoopActionType):
            raise ValueError("action_type must be LoopActionType")
        if self.tool_name is not None and (not isinstance(self.tool_name, str) or not self.tool_name.strip()):
            raise ValueError("tool_name must be non-empty when present")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("arguments must be a mapping")
        if self.action_type is LoopActionType.TOOL and not self.tool_name:
            raise ValueError("TOOL action requires tool_name")
        if self.action_type is LoopActionType.FINAL and self.message is None:
            raise ValueError("FINAL action requires message")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "tool_name": self.tool_name,
            "arguments": _safe_value(dict(self.arguments)),
            "message": _sanitize_text(_bounded_text(self.message or "", _MAX_TEXT_CHARS)) if self.message is not None else None,
        }


@dataclass(frozen=True, slots=True)
class AutonomousLoopConfig:
    """Bounded context/history settings; no user-configurable iteration count."""

    max_context_tokens: int = 2_048
    reserve_response_tokens: int = 32
    max_tool_result_chars: int = 4_000
    max_history_items: int = 8
    max_task_prompt_chars: int = 2_048
    system_prompt: str = (
        "Fodci autonomous tool loop. Return exactly ACTION: TOOL with ARGS JSON "
        "or ACTION: FINAL with ARGS {\"message\": \"...\"}. Never emit prose commands."
    )
    execution_budget: ExecutionBudget = field(default_factory=ExecutionBudget.conservative_defaults)

    def __post_init__(self) -> None:
        for field_name in (
            "max_context_tokens",
            "reserve_response_tokens",
            "max_tool_result_chars",
            "max_history_items",
            "max_task_prompt_chars",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.max_context_tokens <= self.reserve_response_tokens:
            raise ValueError("max_context_tokens must exceed reserve_response_tokens")
        if self.max_tool_result_chars > 64_000 or self.max_history_items > 128 or self.max_task_prompt_chars > 16_384:
            raise ValueError("Autonomous loop budget exceeds the safety ceiling")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("system_prompt must contain text")
        if not isinstance(self.execution_budget, ExecutionBudget):
            raise ValueError("execution_budget must be ExecutionBudget")


@dataclass(frozen=True, slots=True)
class AutonomousLoopRequest:
    """Explicit inputs for one opt-in autonomous loop invocation."""

    task: str
    project_root: Path | str
    project_context: ProjectContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must contain text")
        root = Path(self.project_root).expanduser().resolve(strict=False)
        object.__setattr__(self, "project_root", root)
        if self.project_context is not None and self.project_context.root != root:
            raise ValueError("project_context.root must equal the explicit project_root")

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "project_root": str(self.project_root), "has_project_context": self.project_context is not None}


@dataclass(frozen=True, slots=True)
class AutonomousLoopStep:
    """Bounded chronological record of one loop decision/execution attempt."""

    index: int
    state_before: LoopLifecycleState
    state_after: LoopLifecycleState
    plan_step_id: str | None
    selected_tool: str | None
    sanitized_arguments: Mapping[str, Any]
    tool_result_status: str | None
    tool_result_error: str | None
    model_response: str
    context_truncated: bool
    mutated_project: bool
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sanitized_arguments", MappingProxyType(_safe_value(dict(self.sanitized_arguments))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "state_before": self.state_before.value,
            "state_after": self.state_after.value,
            "plan_step_id": self.plan_step_id,
            "selected_tool": self.selected_tool,
            "sanitized_arguments": dict(self.sanitized_arguments),
            "tool_result_status": self.tool_result_status,
            "tool_result_error": self.tool_result_error,
            "model_response": _sanitize_text(_bounded_text(self.model_response, _MAX_TEXT_CHARS)),
            "context_truncated": self.context_truncated,
            "mutated_project": self.mutated_project,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class AutonomousLoopState:
    """Immutable snapshot of the loop lifecycle and bounded working context."""

    lifecycle: LoopLifecycleState
    task: str
    plan_step_id: str | None
    selection: ToolSelectionDecision | None
    last_tool_call: ToolCall | None
    last_tool_result: ToolResult | None
    model_response: str
    history: tuple[AgentMessage, ...]
    context_truncated: bool
    truncation_reason: str | None
    preserved_sections: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    usage: AgentUsage
    recovery: RecoveryResult | None = None
    completion: TaskCompletionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lifecycle": self.lifecycle.value,
            "task": _bounded_text(self.task, _MAX_TEXT_CHARS),
            "plan_step_id": self.plan_step_id,
            "selection": self.selection.to_dict() if self.selection else None,
            "last_tool_call": _safe_value(self.last_tool_call.to_dict()) if self.last_tool_call else None,
            "last_tool_result": _safe_value(self.last_tool_result.to_dict()) if self.last_tool_result else None,
            "model_response": _sanitize_text(_bounded_text(self.model_response, _MAX_TEXT_CHARS)),
            "history": [_safe_value(message.to_dict()) | {"content": _sanitize_text(message.content)} for message in self.history],
            "context_truncated": self.context_truncated,
            "truncation_reason": self.truncation_reason,
            "preserved_sections": list(self.preserved_sections),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "usage": self.usage.to_dict(),
            "recovery": self.recovery.to_dict() if self.recovery else None,
            "completion": self.completion.to_dict() if self.completion else None,
        }


@dataclass(frozen=True, slots=True)
class AutonomousLoopResult:
    """Complete bounded outcome of one autonomous loop invocation."""

    task: str
    status: LoopStatus
    final_answer: str
    plan: ExecutionPlan | None
    project_context: ProjectContext | None
    state: AutonomousLoopState
    steps: tuple[AutonomousLoopStep, ...]
    tool_calls: tuple[ToolCall, ...]
    tool_results: tuple[ToolResult, ...]
    usage: AgentUsage
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    stop_evaluation: StopEvaluation | None = None
    execution_budget: ExecutionBudgetSnapshot | None = None
    recovery: RecoveryResult | None = None
    completion: TaskCompletionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status.value,
            "final_answer": _sanitize_text(_bounded_text(self.final_answer, _MAX_TEXT_CHARS)),
            "plan": self.plan.to_dict() if self.plan else None,
            "project_context": self.project_context.to_dict() if self.project_context else None,
            "state": self.state.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "tool_calls": [_safe_value(call.to_dict()) for call in self.tool_calls],
            "tool_results": [_safe_value(result.to_dict()) for result in self.tool_results],
            "usage": self.usage.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "stop_evaluation": self.stop_evaluation.to_dict() if self.stop_evaluation else None,
            "execution_budget": self.execution_budget.to_dict() if self.execution_budget else None,
            "recovery": self.recovery.to_dict() if self.recovery else None,
            "completion": self.completion.to_dict() if self.completion else None,
        }


_ALLOWED_TRANSITIONS: dict[LoopLifecycleState, frozenset[LoopLifecycleState]] = {
    LoopLifecycleState.CREATED: frozenset({LoopLifecycleState.PLANNING, LoopLifecycleState.FAILED}),
    LoopLifecycleState.PLANNING: frozenset({LoopLifecycleState.SELECTING_TOOL, LoopLifecycleState.FAILED}),
    LoopLifecycleState.SELECTING_TOOL: frozenset({LoopLifecycleState.VALIDATING_ACTION, LoopLifecycleState.REQUESTING_NEXT_ACTION, LoopLifecycleState.FAILED, LoopLifecycleState.BLOCKED, LoopLifecycleState.COMPLETED}),
    LoopLifecycleState.VALIDATING_ACTION: frozenset({LoopLifecycleState.EXECUTING_TOOL, LoopLifecycleState.REQUESTING_NEXT_ACTION, LoopLifecycleState.COMPLETED, LoopLifecycleState.FAILED, LoopLifecycleState.BLOCKED}),
    LoopLifecycleState.EXECUTING_TOOL: frozenset({LoopLifecycleState.OBSERVING_RESULT, LoopLifecycleState.FAILED}),
    LoopLifecycleState.OBSERVING_RESULT: frozenset({LoopLifecycleState.UPDATING_CONTEXT, LoopLifecycleState.FAILED, LoopLifecycleState.BLOCKED}),
    LoopLifecycleState.UPDATING_CONTEXT: frozenset({LoopLifecycleState.REQUESTING_NEXT_ACTION, LoopLifecycleState.SELECTING_TOOL, LoopLifecycleState.FAILED}),
    LoopLifecycleState.REQUESTING_NEXT_ACTION: frozenset({LoopLifecycleState.SELECTING_TOOL, LoopLifecycleState.VALIDATING_ACTION, LoopLifecycleState.COMPLETED, LoopLifecycleState.FAILED, LoopLifecycleState.BLOCKED}),
    LoopLifecycleState.COMPLETED: frozenset(),
    LoopLifecycleState.FAILED: frozenset(),
    LoopLifecycleState.BLOCKED: frozenset(),
}


class LoopStateMachine:
    """Explicit transition validator used by AutonomousToolLoop."""

    def __init__(self) -> None:
        self.state = LoopLifecycleState.CREATED

    def transition(self, target: LoopLifecycleState) -> LoopLifecycleState:
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise LoopStateError(self.state, target)
        self.state = target
        return self.state

    def fail(self) -> LoopLifecycleState:
        return self.transition(LoopLifecycleState.FAILED)

    def block(self) -> LoopLifecycleState:
        return self.transition(LoopLifecycleState.BLOCKED)


def parse_loop_action(text: str) -> LoopAction:
    """Parse only explicit ``ACTION``/``ARGS`` protocol; prose is rejected."""

    if not isinstance(text, str) or not text.strip():
        raise LoopActionParseError(LoopFailureCode.MODEL_OUTPUT_INVALID, "Model output must contain an explicit action.")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = lines[0]
    if first.upper().startswith("FINAL:"):
        message = first[len("FINAL:"):].strip()
        if len(lines) > 1:
            message = "\n".join([message, *lines[1:]]).strip()
        return LoopAction(LoopActionType.FINAL, message=message)
    if not first.upper().startswith("ACTION:"):
        raise LoopActionParseError(LoopFailureCode.INVALID_ACTION, "Model output must begin with ACTION: TOOL or ACTION: FINAL.")
    action_name = first[len("ACTION:"):].strip()
    if action_name.upper() not in {"TOOL", "FINAL"} and not _valid_name(action_name):
        raise LoopActionParseError(LoopFailureCode.INVALID_ACTION, "ACTION must be TOOL, FINAL, or one valid registered tool name.")
    if len(lines) != 2 or not lines[1].upper().startswith("ARGS:"):
        raise LoopActionParseError(LoopFailureCode.INVALID_ACTION, "An explicit ACTION must be followed by exactly one ARGS JSON object.")
    raw = lines[1][len("ARGS:"):].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LoopActionParseError(LoopFailureCode.INVALID_ACTION, f"ARGS is not valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise LoopActionParseError(LoopFailureCode.INVALID_ACTION, "ARGS must decode to a JSON object.")
    if action_name.upper() == "FINAL":
        message = payload.get("message")
        if not isinstance(message, str):
            raise LoopActionParseError(LoopFailureCode.INVALID_ACTION, "FINAL ARGS requires a string message.")
        return LoopAction(LoopActionType.FINAL, message=message)
    if action_name.upper() == "TOOL":
        tool_name = payload.get("tool")
        arguments = payload.get("arguments")
        if not isinstance(tool_name, str) or not _valid_name(tool_name):
            raise LoopActionParseError(LoopFailureCode.INVALID_TOOL_CALL, "TOOL ARGS requires one valid tool name.")
        if not isinstance(arguments, dict):
            raise LoopActionParseError(LoopFailureCode.INVALID_TOOL_CALL, "TOOL ARGS requires an arguments JSON object.")
        return LoopAction(LoopActionType.TOOL, tool_name=tool_name, arguments=arguments)
    return LoopAction(LoopActionType.TOOL, tool_name=action_name, arguments=payload)


class AutonomousToolLoop:
    """Explicitly opt-in bounded loop integrating planner, selector, and registry."""

    def __init__(
        self,
        engine: Any,
        *,
        registry: ToolRegistry | None = None,
        planner: Planner | None = None,
        selector: ToolSelector | None = None,
        config: AutonomousLoopConfig | None = None,
    ) -> None:
        if not callable(getattr(engine, "generate", None)):
            raise TypeError("AutonomousToolLoop requires an inference engine with generate().")
        tokenizer = getattr(engine, "tokenizer", None)
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("AutonomousToolLoop requires an inference engine exposing tokenizer.encode().")
        self.engine = engine
        self.registry = registry or ToolRegistry.default()
        self.planner = planner or Planner()
        self.selector = selector or ToolSelector()
        self.config = config or AutonomousLoopConfig()
        budget_config = AgentConfig(
            max_steps=_EMERGENCY_TOOL_EXECUTION_BOUND + 2,
            max_tool_calls=_EMERGENCY_TOOL_EXECUTION_BOUND + 1,
            max_context_tokens=self.config.max_context_tokens,
            reserve_response_tokens=self.config.reserve_response_tokens,
            max_tool_result_chars=self.config.max_tool_result_chars,
            max_history_items=self.config.max_history_items,
            system_prompt=self.config.system_prompt,
        )
        self._budget = ContextBudget(tokenizer, budget_config)

    def run(self, request: AutonomousLoopRequest | str, project_root: Path | str | None = None) -> AutonomousLoopResult:
        """Run one bounded task with explicit, evidence-driven recovery decisions."""

        try:
            request = self._coerce_request(request, project_root)
        except (TypeError, ValueError) as exc:
            return self._failure_result(str(exc), LoopStatus.FAILED, LoopFailureCode.INVALID_ACTION.value, task="")

        machine = LoopStateMachine()
        steps: list[AutonomousLoopStep] = []
        calls: list[ToolCall] = []
        results: list[ToolResult] = []
        history: list[AgentMessage] = []
        warnings: list[str] = []
        errors: list[str] = []
        usage = AgentUsage()
        context = request.project_context
        plan: ExecutionPlan | None = None
        current_selection: ToolSelectionDecision | None = None
        current_step_id: str | None = None
        last_call: ToolCall | None = None
        last_result: ToolResult | None = None
        model_response = ""
        final_answer = ""
        context_truncated = False
        truncation_reason: str | None = None
        preserved_sections: tuple[str, ...] = ("task", "plan", "current_step", "tool_observation")
        completed_step_ids: list[str] = []
        skipped_step_ids: list[str] = []
        pending_verification_steps: list[str] = []
        completion_evidence: list[str] = []
        verification = VerificationEvidence.not_required()
        blocked_capabilities: list[str] = []
        safety_blocked = False
        context_complete = context is not None
        stop_evaluation: StopEvaluation | None = None
        budget_ledger = ExecutionBudgetLedger(self.config.execution_budget)
        budget_decision: BudgetDecision | None = None
        recovery_result: RecoveryResult | None = None
        completion_result: TaskCompletionResult | None = None
        failed_action_signatures: list[str] = []

        def snapshot() -> AutonomousLoopState:
            return AutonomousLoopState(machine.state, request.task, current_step_id, current_selection, last_call, last_result, model_response, tuple(history[-self.config.max_history_items:]), context_truncated, truncation_reason, preserved_sections, tuple(_unique(warnings)), tuple(_unique(errors)), usage, recovery_result, completion_result)

        def finish(_request: AutonomousLoopRequest, status: LoopStatus, answer: str, finish_plan: ExecutionPlan | None, finish_context: ProjectContext | None, finish_state: AutonomousLoopState, finish_steps: Sequence[AutonomousLoopStep], finish_calls: Sequence[ToolCall], finish_results: Sequence[ToolResult], finish_usage: AgentUsage, finish_warnings: Sequence[str], finish_errors: Sequence[str]) -> AutonomousLoopResult:
            nonlocal stop_evaluation, budget_decision
            if stop_evaluation is None:
                stop_evaluation = evaluate_stop_condition(StopConditionRequest(
                    plan=finish_plan,
                    completed_step_ids=tuple(completed_step_ids),
                    skipped_step_ids=tuple(skipped_step_ids),
                    current_step_id=current_step_id,
                    final_action_valid=status in {LoopStatus.COMPLETED, LoopStatus.CONTINUE, LoopStatus.BLOCKED},
                    invalid_action=status in {LoopStatus.INVALID_ACTION},
                    last_tool_result=last_result,
                    last_tool_name=last_call.name if last_call else None,
                    missing_capabilities=tuple(blocked_capabilities),
                    safety_blocked=safety_blocked,
                    context_complete=context_complete,
                    completion_evidence=tuple(completion_evidence),
                    verification=verification,
                    emergency_bound_reached=status is LoopStatus.LOOP_BOUND_REACHED,
                    fatal_error=finish_errors[-1] if status is LoopStatus.FAILED and finish_errors else None,
                    budget_decision=budget_decision,
                    warning_messages=tuple(finish_warnings),
                    tool_result_recoverable=bool(recovery_result and recovery_result.decision.status is RecoveryStatus.CONTINUE),
                    completion_decision=(completion_result.decision.value if completion_result and completion_result.decision is not CompletionDecision.COMPLETE and verification.state not in {VerificationState.REQUIRED, VerificationState.PENDING} else None),
                ))
            return self._finalize(_request, status, answer, finish_plan, finish_context, finish_state, finish_steps, finish_calls, finish_results, finish_usage, finish_warnings, finish_errors, stop_evaluation=stop_evaluation, execution_budget=budget_ledger.snapshot(), recovery=recovery_result, completion=completion_result)

        try:
            machine.transition(LoopLifecycleState.PLANNING)
            if context is None:
                bootstrap = self._bootstrap_context(request.task)
                bootstrap_selection = self.selector.select(ToolSelectionRequest(bootstrap, registry=self.registry))
                if not bootstrap_selection.decisions or bootstrap_selection.decisions[0].selected_tool != "project_context":
                    return finish(request, LoopStatus.TOOL_UNAVAILABLE, "", None, None, snapshot(), steps, calls, results, usage, warnings, errors + ("project_context is unavailable for initial bounded context construction",))
                bootstrap_decision = bootstrap_selection.decisions[0]
                machine.transition(LoopLifecycleState.SELECTING_TOOL)
                machine.transition(LoopLifecycleState.VALIDATING_ACTION)
                bootstrap_call = ToolCall("call-0001", "project_context", {"project_root": str(request.project_root)})
                bootstrap_budget = budget_ledger.check_tool_operation("project_context")
                if not bootstrap_budget.allowed:
                    budget_decision = bootstrap_budget
                    errors.append(bootstrap_budget.message)
                    machine.block()
                    return finish(request, LoopStatus.BLOCKED, "", None, None, snapshot(), steps, calls, results, usage, warnings, errors)
                budget_ledger.consume("project_context", dimension=BudgetDimension.TOOL_CALLS)
                machine.transition(LoopLifecycleState.EXECUTING_TOOL)
                bootstrap_result = self._dispatch(bootstrap_call)
                calls.append(bootstrap_call)
                results.append(bootstrap_result)
                budget_ledger.complete_tool()
                last_call, last_result = bootstrap_call, bootstrap_result
                bootstrap_observation = self._safe_result_text(bootstrap_result)
                budget_ledger.account_tool_result(tool_name=bootstrap_call.name, tool_output_bytes=len(bootstrap_observation), success=bootstrap_result.success)
                usage = AgentUsage(1, 1, 0, len(bootstrap_observation), 0)
                machine.transition(LoopLifecycleState.OBSERVING_RESULT)
                if not bootstrap_result.success or not isinstance(bootstrap_result.data, ProjectContext):
                    message = bootstrap_result.message or "project_context failed"
                    errors.append(message)
                    steps.append(AutonomousLoopStep(0, LoopLifecycleState.EXECUTING_TOOL, LoopLifecycleState.FAILED, "context-1", bootstrap_call.name, bootstrap_call.arguments, bootstrap_result.error_code, bootstrap_result.error_code, "", False, False, errors=(message,)))
                    machine.fail()
                    return finish(request, LoopStatus.TOOL_EXECUTION_FAILED, "", None, None, snapshot(), steps, calls, results, usage, warnings, errors)
                context = bootstrap_result.data
                context_complete = True
                machine.transition(LoopLifecycleState.UPDATING_CONTEXT)
                history.append(AgentMessage(AgentMessageRole.TOOL, bootstrap_observation, name=bootstrap_call.name, call_id=bootstrap_call.call_id))
                steps.append(AutonomousLoopStep(0, LoopLifecycleState.EXECUTING_TOOL, LoopLifecycleState.UPDATING_CONTEXT, "context-1", bootstrap_call.name, bootstrap_call.arguments, "SUCCESS", None, "", False, False))
                machine.transition(LoopLifecycleState.REQUESTING_NEXT_ACTION)

            plan_result = self.planner.plan(PlannerRequest(request.task, context))
            if plan_result.plan is None:
                errors.extend(plan_result.errors or plan_result.warnings)
                return finish(request, LoopStatus.FAILED, "", plan_result.plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
            plan = plan_result.plan
            warnings.extend(plan_result.warnings)
            if not plan.steps:
                return finish(request, LoopStatus.FAILED, "", plan, context, snapshot(), steps, calls, results, usage, warnings, errors + ["Planner returned no executable plan steps"])

            plan_index = 0
            cycle_index = 0
            while True:
                iteration_decision = budget_ledger.consume("autonomous iteration", dimension=BudgetDimension.ITERATIONS)
                if not iteration_decision.allowed:
                    budget_decision = iteration_decision
                    errors.append(iteration_decision.message)
                    machine.block()
                    return finish(request, LoopStatus.BLOCKED, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                action_decision = budget_ledger.consume("model action step", dimension=BudgetDimension.ACTION_STEPS)
                if not action_decision.allowed:
                    budget_decision = action_decision
                    errors.append(action_decision.message)
                    machine.block()
                    return finish(request, LoopStatus.BLOCKED, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                if budget_ledger.snapshot().exhausted_dimensions:
                    exhausted = budget_ledger.snapshot().exhausted_dimensions[0]
                    budget_decision = BudgetDecision(False, "autonomous iteration", BudgetDimension(exhausted), None, None, 0, 0, False, "accumulated budget dimension is exhausted")
                    errors.append(budget_decision.message)
                    machine.block()
                    return finish(request, LoopStatus.BLOCKED, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                if len(calls) >= _EMERGENCY_TOOL_EXECUTION_BOUND:
                    errors.append("The fixed Phase 6.3 emergency tool-execution bound was reached.")
                    stop_evaluation = evaluate_stop_condition(StopConditionRequest(
                        plan=plan,
                        completed_step_ids=tuple(completed_step_ids),
                        skipped_step_ids=tuple(skipped_step_ids),
                        current_step_id=current_step_id,
                        last_tool_result=last_result,
                        last_tool_name=last_call.name if last_call else None,
                        completion_evidence=tuple(completion_evidence),
                        verification=verification,
                        context_complete=context_complete,
                        emergency_bound_reached=True,
                        warning_messages=tuple(warnings),
                    ))
                    machine.block()
                    return finish(request, LoopStatus.LOOP_BOUND_REACHED, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)

                plan_step = plan.steps[plan_index] if plan_index < len(plan.steps) else None
                current_step_id = plan_step.step_id if plan_step else (plan.steps[-1].step_id if plan.steps else None)
                if machine.state is not LoopLifecycleState.SELECTING_TOOL:
                    machine.transition(LoopLifecycleState.SELECTING_TOOL)
                current_selection = None
                selection_result = None
                selection_failure = False
                if plan_step is not None:
                    selection_result = self.selector.select(ToolSelectionRequest(plan, registry=self.registry, project_context=context, selected_step_ids=(plan_step.step_id,)))
                    current_selection = selection_result.decisions[0] if selection_result.decisions else None
                    if current_selection is None or selection_result.status is not ToolSelectionStatus.SELECTED or current_selection.selected_tool is None:
                        if selection_result.status is ToolSelectionStatus.NO_SUITABLE_TOOL:
                            message = current_selection.selection_reason if current_selection else "Declarative plan step requires no registered tool."
                            warnings.append(f"Skipped non-executable plan step {plan_step.step_id}: {message}")
                            skipped_step_ids.append(plan_step.step_id)
                            steps.append(AutonomousLoopStep(cycle_index + 1, LoopLifecycleState.SELECTING_TOOL, LoopLifecycleState.REQUESTING_NEXT_ACTION, plan_step.step_id, None, {}, None, None, "", False, False, warnings=(message,)))
                            plan_index += 1
                            cycle_index += 1
                            machine.transition(LoopLifecycleState.REQUESTING_NEXT_ACTION)
                            continue
                        selection_failure = True
                        blocked_capabilities.extend(selection_result.errors or ("required capability is unavailable in the supplied registry",))

                machine.transition(LoopLifecycleState.VALIDATING_ACTION)
                try:
                    if selection_failure and plan_step is not None and selection_result is not None:
                        prompt, prompt_tokens, prompt_truncated, prompt_warnings = self._render_selection_failure_prompt(request.task, plan_step, selection_result, context, tuple(history))
                    elif plan_step is not None and current_selection is not None:
                        prompt, prompt_tokens, prompt_truncated, prompt_warnings = self._render_prompt(request.task, plan, plan_step, current_selection, context, tuple(history))
                    else:
                        prompt, prompt_tokens, prompt_truncated, prompt_warnings = self._render_completion_prompt(request.task, plan, context, tuple(history))
                    context_truncated = prompt_truncated
                    if prompt_truncated:
                        truncation_reason = "ContextBudget compacted optional plan/history/tool context."
                    warnings.extend(prompt_warnings)
                    usage = AgentUsage(usage.steps + 1, usage.tool_calls, usage.prompt_tokens + prompt_tokens, usage.tool_result_chars, usage.context_truncations + int(prompt_truncated))
                    generated = self.engine.generate(prompt)
                    model_response = getattr(generated, "generated_text", None)
                    if not isinstance(model_response, str):
                        raise LoopActionParseError(LoopFailureCode.MODEL_OUTPUT_INVALID, "Inference result did not contain generated_text.")
                    action = parse_loop_action(model_response)
                except ContextBudgetError as exc:
                    errors.append(str(exc))
                    machine.fail()
                    return finish(request, LoopStatus.CONTEXT_LIMIT_REACHED, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                except LoopActionParseError as exc:
                    errors.append(exc.message)
                    steps.append(AutonomousLoopStep(cycle_index + 1, LoopLifecycleState.VALIDATING_ACTION, LoopLifecycleState.FAILED, current_step_id, None, {}, None, exc.code, model_response, context_truncated, False, errors=(exc.message,)))
                    machine.fail()
                    return finish(request, LoopStatus.INVALID_ACTION, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                except Exception as exc:
                    errors.append(str(exc))
                    machine.fail()
                    return finish(request, LoopStatus.FAILED, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)

                if action.action_type is LoopActionType.FINAL:
                    final_answer = action.message or ""
                    completion_result = verify_task_completion(TaskCompletionRequest(
                        task=request.task,
                        plan=plan,
                        completed_step_ids=tuple(completed_step_ids),
                        skipped_step_ids=tuple(skipped_step_ids),
                        tool_results=tuple(results),
                        verification=verification,
                        recovery=recovery_result,
                        budget=budget_ledger.snapshot(),
                        evidence=(TaskCompletionEvidence("final_action", "The model emitted FINAL; this is only a claim and not completion proof.", EvidenceStrength.INDIRECT),),
                        final_response=final_answer,
                    ))
                    stop_evaluation = evaluate_stop_condition(StopConditionRequest(
                        plan=plan,
                        completed_step_ids=tuple(completed_step_ids),
                        skipped_step_ids=tuple(skipped_step_ids),
                        current_step_id=current_step_id,
                        final_action_valid=True,
                        last_tool_result=last_result,
                        last_tool_name=last_call.name if last_call else None,
                        missing_capabilities=tuple(blocked_capabilities),
                        safety_blocked=safety_blocked,
                        context_complete=context_complete,
                        completion_evidence=tuple(completion_evidence),
                        verification=verification,
                        emergency_bound_reached=False,
                        completion_decision=(completion_result.decision.value if completion_result.decision is not CompletionDecision.COMPLETE and verification.state not in {VerificationState.REQUIRED, VerificationState.PENDING} else None),
                        warning_messages=tuple(warnings),
                    ))
                    if stop_evaluation.decision is StopDecision.DONE:
                        machine.transition(LoopLifecycleState.COMPLETED)
                        terminal_status = LoopStatus.COMPLETED
                    elif stop_evaluation.decision is StopDecision.BLOCKED:
                        machine.block()
                        terminal_status = LoopStatus.BLOCKED
                        errors.extend(stop_evaluation.blocking_conditions)
                    else:
                        machine.transition(LoopLifecycleState.REQUESTING_NEXT_ACTION)
                        terminal_status = LoopStatus.CONTINUE
                    steps.append(AutonomousLoopStep(cycle_index + 1, LoopLifecycleState.VALIDATING_ACTION, machine.state, current_step_id, None, {}, None, None, model_response, context_truncated, False))
                    return finish(request, terminal_status, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)

                if selection_failure:
                    message = "The selected capability is unavailable or invalid for this plan step. Only FINAL is accepted now."
                    errors.append(message)
                    steps.append(AutonomousLoopStep(cycle_index + 1, LoopLifecycleState.VALIDATING_ACTION, LoopLifecycleState.FAILED, current_step_id, action.tool_name, action.arguments, None, LoopFailureCode.TOOL_UNAVAILABLE.value, model_response, context_truncated, False, errors=(message,)))
                    machine.fail()
                    return finish(request, _loop_status_for_selection(selection_result.status if selection_result else ToolSelectionStatus.TOOL_UNAVAILABLE), final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                if current_selection is None or plan_step is None:
                    message = "A TOOL action is not allowed after all planned tool steps have been consumed. Return FINAL."
                    errors.append(message)
                    steps.append(AutonomousLoopStep(cycle_index + 1, LoopLifecycleState.VALIDATING_ACTION, LoopLifecycleState.FAILED, current_step_id, action.tool_name, action.arguments, None, LoopFailureCode.INVALID_TOOL_CALL.value, model_response, context_truncated, False, errors=(message,)))
                    machine.fail()
                    return finish(request, LoopStatus.INVALID_ACTION, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                if action.tool_name != current_selection.selected_tool:
                    message = "Model-selected tool does not match the ToolSelector decision for the current plan step."
                    errors.append(message)
                    steps.append(AutonomousLoopStep(cycle_index + 1, LoopLifecycleState.VALIDATING_ACTION, LoopLifecycleState.FAILED, current_step_id, action.tool_name, action.arguments, None, LoopFailureCode.INVALID_TOOL_CALL.value, model_response, context_truncated, False, errors=(message,)))
                    machine.fail()
                    return finish(request, LoopStatus.INVALID_ACTION, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                if action.tool_name not in self.registry.names():
                    message = f"Tool is not available in the supplied ToolRegistry: {action.tool_name}."
                    errors.append(message)
                    machine.fail()
                    return finish(request, LoopStatus.TOOL_UNAVAILABLE, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                if _contains_shell_payload(action.arguments):
                    message = "Shell-like or implicit command payloads are not accepted by the autonomous action boundary."
                    errors.append(message)
                    machine.fail()
                    return finish(request, LoopStatus.INVALID_ACTION, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)

                call_id = f"call-{len(calls) + 1:04d}"
                try:
                    call = self._bind_project_root(ToolCall(call_id, action.tool_name, action.arguments), request.project_root)
                except (ToolError, ValueError, TypeError) as exc:
                    message = getattr(exc, "message", str(exc))
                    errors.append(message)
                    steps.append(AutonomousLoopStep(cycle_index + 1, LoopLifecycleState.VALIDATING_ACTION, LoopLifecycleState.FAILED, current_step_id, action.tool_name, action.arguments, None, LoopFailureCode.INVALID_TOOL_CALL.value, model_response, context_truncated, current_selection.risk_level.value in {"MUTATING", "DESTRUCTIVE"}, errors=(message,)))
                    machine.fail()
                    return finish(request, LoopStatus.INVALID_ACTION, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)

                tool_budget = budget_ledger.check_tool_operation(call.name)
                if not tool_budget.allowed:
                    budget_decision = tool_budget
                    errors.append(tool_budget.message)
                    machine.block()
                    return finish(request, LoopStatus.BLOCKED, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                budget_ledger.consume("tool call", dimension=BudgetDimension.TOOL_CALLS)
                normalized_tool = call.name.casefold()
                specific_dimension = {"write_file": BudgetDimension.MUTATIONS, "edit_file": BudgetDimension.MUTATIONS, "delete_file": BudgetDimension.MUTATIONS, "run_command": BudgetDimension.COMMAND_EXECUTIONS, "run_command_with_policy": BudgetDimension.COMMAND_EXECUTIONS, "run_tests": BudgetDimension.TEST_EXECUTIONS, "run_application": BudgetDimension.APPLICATION_LAUNCHES}.get(normalized_tool)
                if specific_dimension is not None:
                    budget_ledger.consume(call.name, dimension=specific_dimension)
                machine.transition(LoopLifecycleState.EXECUTING_TOOL)
                tool_result = self._dispatch(call)
                calls.append(call)
                results.append(tool_result)
                last_call, last_result = call, tool_result
                budget_ledger.complete_tool()
                machine.transition(LoopLifecycleState.OBSERVING_RESULT)
                bounded_observation = self._safe_result_text(tool_result)
                budget_ledger.account_tool_result(tool_name=call.name, tool_output_bytes=len(bounded_observation), context_tokens=prompt_tokens, success=tool_result.success)
                usage = AgentUsage(usage.steps, usage.tool_calls + 1, usage.prompt_tokens, usage.tool_result_chars + len(bounded_observation), usage.context_truncations)
                if not tool_result.success:
                    message = tool_result.message or "Tool execution failed."
                    errors.append(message)
                    signature = f"{call.name}:{tool_result.error_code}:{json.dumps(dict(call.arguments), sort_keys=True, ensure_ascii=False)}"
                    recovery_result = decide_recovery(RecoveryContext(tool_result, call.name, current_step_id, call.name, plan_index + 1 < len(plan.steps), (message,), tuple(failed_action_signatures), budget_ledger.snapshot(), budget_decision, verification.state is VerificationState.REQUIRED))
                    failed_action_signatures.append(signature)
                    safety_blocked = recovery_result.decision.classification.safety_or_policy_boundary
                    recovery_action = recovery_result.decision.action.value
                    warnings.extend(recovery_result.decision.warnings)
                    if recovery_result.decision.status is RecoveryStatus.CONTINUE and plan_index + 1 < len(plan.steps):
                        if current_step_id:
                            completed_step_ids.append(current_step_id)
                        history.append(AgentMessage(AgentMessageRole.TOOL, _sanitize_text(_bounded_text(f"Recovery decision: {recovery_action}. {recovery_result.decision.reason}", self.config.max_tool_result_chars)), name="recovery", call_id=call.call_id))
                        steps.append(AutonomousLoopStep(cycle_index + 1, LoopLifecycleState.EXECUTING_TOOL, LoopLifecycleState.UPDATING_CONTEXT, current_step_id, call.name, call.arguments, tool_result.error_code, tool_result.error_code, model_response, context_truncated, current_selection.risk_level.value in {"MUTATING", "DESTRUCTIVE"}, warnings=(recovery_action,), errors=(message,)))
                        machine.transition(LoopLifecycleState.UPDATING_CONTEXT)
                        plan_index += 1
                        cycle_index += 1
                        machine.transition(LoopLifecycleState.REQUESTING_NEXT_ACTION)
                        continue
                    stop_evaluation = evaluate_stop_condition(StopConditionRequest(
                        plan=plan,
                        completed_step_ids=tuple(completed_step_ids),
                        skipped_step_ids=tuple(skipped_step_ids),
                        current_step_id=current_step_id,
                        last_tool_result=tool_result,
                        last_tool_name=call.name,
                        safety_blocked=safety_blocked,
                        completion_evidence=tuple(completion_evidence),
                        verification=verification,
                        context_complete=context_complete,
                        tool_result_recoverable=recovery_result.decision.status is RecoveryStatus.CONTINUE,
                        warning_messages=tuple(warnings),
                    ))
                    terminal = LoopStatus.TOOL_EXECUTION_FAILED if recovery_result.decision.status in {RecoveryStatus.BLOCKED, RecoveryStatus.USER_INTERVENTION_REQUIRED} else LoopStatus.CONTINUE if recovery_result.decision.status is RecoveryStatus.CONTINUE else LoopStatus.TOOL_EXECUTION_FAILED
                    steps.append(AutonomousLoopStep(cycle_index + 1, LoopLifecycleState.EXECUTING_TOOL, LoopLifecycleState.BLOCKED if recovery_result.decision.status in {RecoveryStatus.BLOCKED, RecoveryStatus.USER_INTERVENTION_REQUIRED} else LoopLifecycleState.FAILED, current_step_id, call.name, call.arguments, tool_result.error_code, tool_result.error_code, model_response, context_truncated, current_selection.risk_level.value in {"MUTATING", "DESTRUCTIVE"}, warnings=(recovery_action,), errors=(message,)))
                    if recovery_result.decision.status in {RecoveryStatus.BLOCKED, RecoveryStatus.USER_INTERVENTION_REQUIRED}:
                        machine.block()
                    else:
                        machine.fail()
                    return finish(request, terminal, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)
                result_verification = _verification_from_tool_result(call.name, tool_result)
                if result_verification is not None:
                    verification = result_verification
                    if verification.state is VerificationState.PASSED:
                        completed_step_ids.extend(pending_verification_steps)
                        pending_verification_steps.clear()
                        if current_step_id:
                            completed_step_ids.append(current_step_id)
                        completion_evidence.extend(verification.evidence or (verification.message or "verification passed",))
                    elif current_step_id:
                        pending_verification_steps.append(current_step_id)
                elif _tool_requires_verification(call.name, plan_step):
                    if current_step_id:
                        pending_verification_steps.append(current_step_id)
                    verification = VerificationEvidence.pending(call.name, "A post-operation verification step is still required.")
                elif current_step_id:
                    completed_step_ids.append(current_step_id)
                    completion_evidence.append(f"{call.name} returned a structured successful observation")
                machine.transition(LoopLifecycleState.UPDATING_CONTEXT)
                history.append(AgentMessage(AgentMessageRole.ASSISTANT, _sanitize_text(_bounded_text(model_response, _MAX_TEXT_CHARS)), call_id=call.call_id))
                history.append(AgentMessage(AgentMessageRole.TOOL, bounded_observation, name=call.name, call_id=call.call_id))
                history = history[-self.config.max_history_items:]
                steps.append(AutonomousLoopStep(cycle_index + 1, LoopLifecycleState.EXECUTING_TOOL, LoopLifecycleState.UPDATING_CONTEXT, current_step_id, call.name, call.arguments, "SUCCESS", None, model_response, context_truncated, current_selection.risk_level.value in {"MUTATING", "DESTRUCTIVE"}))
                plan_index += 1
                cycle_index += 1
                machine.transition(LoopLifecycleState.REQUESTING_NEXT_ACTION)
        except LoopStateError as exc:
            errors.append(str(exc))
            machine.fail()
            return finish(request, LoopStatus.FAILED, final_answer, plan, context, snapshot(), steps, calls, results, usage, warnings, errors)

    def run_automatic_tests(self, request: AutomaticTestRequest) -> AutomaticTestResult:
        """Run one explicit automatic-test orchestration using this loop's registry."""
        if not isinstance(request, AutomaticTestRequest):
            raise TypeError("request must be AutomaticTestRequest")
        return AutomaticTestOrchestrator().run(replace(request, registry=self.registry))

    def analyze_test_failure(self, test_result, parsed_result, *, config: FailureAnalysisConfig | None = None) -> TestFailureAnalysis:
        """Analyze existing test evidence without executing, mutating, or repairing anything."""
        return TestFailureAnalyzer(config=config).analyze(TestFailureAnalysisRequest(test_result, parsed_result, config or FailureAnalysisConfig()))

    def analyze_root_cause(self, failure_analysis: TestFailureAnalysis, *, project_context=None, evidence=(), config: RootCauseAnalysisConfig | None = None) -> RootCauseAnalysis:
        """Build bounded causal hypotheses from existing failure analysis only."""
        active = config or RootCauseAnalysisConfig()
        return RootCauseAnalyzer(config=active).analyze(RootCauseAnalysisRequest(failure_analysis, project_context, tuple(evidence), active))

    def _coerce_request(self, request: AutonomousLoopRequest | str, project_root: Path | str | None) -> AutonomousLoopRequest:
        if isinstance(request, AutonomousLoopRequest):
            return request
        if isinstance(request, str) and project_root is not None:
            return AutonomousLoopRequest(request, project_root)
        raise ValueError("run requires AutonomousLoopRequest or task plus project_root")

    def _bootstrap_context(self, task: str) -> ExecutionPlan:
        step = PlanStep("context-1", "Build canonical project context", "Build canonical project context before selecting implementation tools.", "Context is required for bounded loop prompts.", "A ProjectContext is available.", (), PlanRiskLevel.LOW)
        return ExecutionPlan(task, task, "Build bounded project context.", PlannerTaskType.INVESTIGATION, (step,), (), ("project root is explicit",), (), (), ("context is available",), PlannerConfidence.MEDIUM, (), PlanCompleteness.PARTIAL)

    def _render_prompt(self, task: str, plan: ExecutionPlan, step: PlanStep, selection: ToolSelectionDecision, context: ProjectContext, history: tuple[AgentMessage, ...]) -> tuple[str, int, bool, tuple[str, ...]]:
        plan_fragment = json.dumps({"task": task, "step": step.to_dict(), "selected_tool": selection.selected_tool, "prerequisites": list(selection.prerequisites), "expected_output": selection.expected_output}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        instruction = _bounded_text(
            f"Task: {task}\nCurrent plan step: {step.step_id}\nPlan step detail: {plan_fragment}\nSelected tool: {selection.selected_tool}\nReturn exactly ACTION: TOOL with ARGS {{\"tool\": \"{selection.selected_tool}\", \"arguments\": {{...}}}} or ACTION: FINAL with ARGS {{\"message\": \"...\"}}.",
            self.config.max_task_prompt_chars,
        )
        budgeted = self._budget.render(instruction, context, history)
        return budgeted.prompt, budgeted.token_count, budgeted.truncated, budgeted.warnings

    def _render_selection_failure_prompt(self, task: str, step: PlanStep, selection: ToolSelectionResult, context: ProjectContext, history: tuple[AgentMessage, ...]) -> tuple[str, int, bool, tuple[str, ...]]:
        reason = selection.errors[0] if selection.errors else "The required capability is unavailable in the supplied registry."
        instruction = _bounded_text(
            f"Task: {task}\nCurrent plan step: {step.step_id}\nSelection status: {selection.status.value}\nSelection reason: {reason}\nReturn exactly ACTION: FINAL with ARGS {{\"message\": \"...\"}}. Do not request an unavailable tool.",
            self.config.max_task_prompt_chars,
        )
        budgeted = self._budget.render(instruction, context, history)
        return budgeted.prompt, budgeted.token_count, budgeted.truncated, budgeted.warnings

    def _render_completion_prompt(self, task: str, plan: ExecutionPlan, context: ProjectContext, history: tuple[AgentMessage, ...]) -> tuple[str, int, bool, tuple[str, ...]]:
        instruction = _bounded_text(
            f"Task: {task}\nThe declarative plan steps have been observed. Return exactly ACTION: FINAL with ARGS {{\"message\": \"...\"}}. Do not request another tool.",
            self.config.max_task_prompt_chars,
        )
        budgeted = self._budget.render(instruction, context, history)
        return budgeted.prompt, budgeted.token_count, budgeted.truncated, budgeted.warnings

    def _bind_project_root(self, call: ToolCall, project_root: Path) -> ToolCall:
        registered = self.registry.metadata_for(call.name)
        arguments = dict(call.arguments)
        schema = registered.input_schema
        if isinstance(schema, Mapping):
            properties = schema.get("properties", {})
            required = schema.get("required", ())
        else:
            properties = {}
            required = ()
        if "project_root" in properties or "project_root" in required:
            supplied_root = arguments.get("project_root")
            if supplied_root is not None and Path(str(supplied_root)).expanduser().resolve(strict=False) != project_root:
                raise ToolError(ToolErrorCode.PATH_OUTSIDE_ROOT, "Autonomous tool calls must remain inside the explicit project root.", path=project_root)
            arguments["project_root"] = str(project_root)
        return ToolCall(call.call_id, call.name, arguments)

    def _dispatch(self, call: ToolCall) -> ToolResult:
        try:
            data = self.registry.dispatch(call.name, call.arguments)
            return ToolResult(call.call_id, call.name, True, data=data, truncated=bool(getattr(data, "truncated", False)))
        except UnknownToolError as exc:
            return ToolResult(call.call_id, call.name, False, error_code=LoopFailureCode.TOOL_UNAVAILABLE.value, message=str(exc))
        except ToolError as exc:
            return ToolResult(call.call_id, call.name, False, error_code=exc.code.value, message=exc.message)
        except (ToolRegistryError, ValueError, TypeError) as exc:
            return ToolResult(call.call_id, call.name, False, error_code=LoopFailureCode.TOOL_ARGUMENT_ERROR.value, message=str(exc))
        except Exception as exc:
            return ToolResult(call.call_id, call.name, False, error_code=LoopFailureCode.TOOL_EXECUTION_FAILED.value, message=str(exc))

    def _safe_result_text(self, result: ToolResult) -> str:
        payload = _safe_value(result.to_dict())
        return _bounded_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), self.config.max_tool_result_chars)

    def _finalize(self, request: AutonomousLoopRequest, status: LoopStatus, final_answer: str, plan: ExecutionPlan | None, context: ProjectContext | None, state: AutonomousLoopState, steps: Sequence[AutonomousLoopStep], calls: Sequence[ToolCall], results: Sequence[ToolResult], usage: AgentUsage, warnings: Sequence[str], errors: Sequence[str], *, stop_evaluation: StopEvaluation | None = None, execution_budget: ExecutionBudgetSnapshot | None = None, recovery: RecoveryResult | None = None, completion: TaskCompletionResult | None = None) -> AutonomousLoopResult:
        return AutonomousLoopResult(request.task, status, final_answer, plan, context, state, tuple(steps), tuple(calls), tuple(results), usage, _unique(warnings), _unique(errors), stop_evaluation, execution_budget, recovery, completion)

    def _failure_result(self, message: str, status: LoopStatus, code: str, *, task: str) -> AutonomousLoopResult:
        state = AutonomousLoopState(LoopLifecycleState.FAILED, task, None, None, None, None, "", (), False, None, (), (), (message,), AgentUsage())
        return AutonomousLoopResult(task, status, "", None, None, state, (), (), (), AgentUsage(), (), (message,))


def create_autonomous_tool_loop(engine: Any, *, registry: ToolRegistry | None = None, config: AutonomousLoopConfig | None = None) -> AutonomousToolLoop:
    """Create an explicitly opt-in autonomous loop; defaults remain read-only."""

    return AutonomousToolLoop(engine, registry=registry, config=config)


def _tool_requires_verification(tool_name: str, plan_step: PlanStep | None) -> bool:
    if tool_name in {"write_file", "edit_file", "delete_file", "run_tests", "run_command_with_policy", "run_command", "run_application"}:
        return True
    if plan_step is not None and plan_step.verification_required and tool_name in {"write_file", "edit_file", "delete_file", "run_tests", "parse_test_result"}:
        return True
    return False


def _verification_from_tool_result(tool_name: str, result: ToolResult) -> VerificationEvidence | None:
    data = result.data
    if tool_name == "verify_modification":
        success = bool(getattr(data, "success", data.get("success", False) if isinstance(data, Mapping) else False))
        complete = bool(getattr(data, "complete", data.get("complete", False) if isinstance(data, Mapping) else False))
        if success and complete:
            return VerificationEvidence.passed("verify_modification", "Modification targets and bounded baseline verification passed.", "all explicit targets verified")
        if not complete:
            return VerificationEvidence.pending("verify_modification", "Verification evidence is incomplete or bounded listing was truncated.")
        return VerificationEvidence.failed("verify_modification", "Modification verification did not pass.")
    if tool_name == "parse_test_result":
        status = getattr(getattr(data, "overall_status", None), "value", getattr(data, "overall_status", None))
        if status is None and isinstance(data, Mapping):
            status = data.get("overall_status")
        if status == "PASS":
            return VerificationEvidence.passed("parse_test_result", "Structured test result status is PASS.", "test parser reported PASS")
        if status in {"UNKNOWN", "NO_TESTS"}:
            return VerificationEvidence.pending("parse_test_result", f"Test parser status is {status}; completion is not proven.")
        return VerificationEvidence.failed("parse_test_result", f"Test parser status is {status or 'unavailable'}.")
    return None


def _is_safety_error(code: str | None) -> bool:
    normalized = (code or "").upper()
    return any(marker in normalized for marker in ("POLICY", "DENIED", "NOT_ALLOWED", "PATH_OUTSIDE_ROOT", "SAFETY", "PERMISSION"))


def _loop_status_for_selection(status: ToolSelectionStatus) -> LoopStatus:
    return {
        ToolSelectionStatus.TOOL_UNAVAILABLE: LoopStatus.TOOL_UNAVAILABLE,
        ToolSelectionStatus.MISSING_PREREQUISITES: LoopStatus.FAILED,
        ToolSelectionStatus.AMBIGUOUS_SELECTION: LoopStatus.FAILED,
        ToolSelectionStatus.NO_SUITABLE_TOOL: LoopStatus.FAILED,
    }.get(status, LoopStatus.FAILED)


def _valid_name(name: str) -> bool:
    return bool(name) and name[0].isalpha() and all(character.isalnum() or character == "_" for character in name)


def _contains_shell_payload(value: Any, key: str = "") -> bool:
    if isinstance(value, Mapping):
        for name, item in value.items():
            if str(name).casefold() in {"shell", "shell_command", "command_string"}:
                return True
            if _contains_shell_payload(item, str(name)):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_shell_payload(item, key) for item in value)
    if isinstance(value, str):
        return bool(_SHELL_TEXT_RE.search(value)) if key.casefold() in {"command", "cmd", "script", "shell_command"} else False
    return False


def _safe_value(value: Any, *, key: str = "") -> Any:
    if _SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(name): _safe_value(item, key=str(name)) for name, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, key=key) for item in value[:64]]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _bounded_text(value, _MAX_TEXT_CHARS) if isinstance(value, str) else value
    if hasattr(value, "to_dict"):
        return _safe_value(value.to_dict(), key=key)
    return _bounded_text(str(value), _MAX_TEXT_CHARS)


def _sanitize_text(value: str) -> str:
    return _SECRET_TEXT_RE.sub(r"\1[REDACTED]", value)


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = f"\n[truncated: kept_first_{limit}_chars]"
    return value[: max(0, limit - len(marker))] + marker


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in values if item))


__all__ = [
    "AutonomousLoopConfig",
    "AutonomousLoopRequest",
    "AutonomousLoopResult",
    "AutonomousLoopState",
    "AutonomousLoopStep",
    "AutonomousToolLoop",
    "LoopAction",
    "LoopActionParseError",
    "LoopActionType",
    "LoopFailureCode",
    "LoopLifecycleState",
    "LoopStateError",
    "LoopStateMachine",
    "LoopStatus",
    "create_autonomous_tool_loop",
    "parse_loop_action",
]
