"""Deterministic, side-effect-free stop-condition evaluation for Phase 6.4."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from backend_ai.agent.execution_budget import BudgetDecision
from backend_ai.agent.models import ToolResult
from backend_ai.agent.planner import ExecutionPlan


class StopDecision(str, Enum):
    DONE = "DONE"
    CONTINUE = "CONTINUE"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class StopReason(str, Enum):
    FINAL_RESPONSE = "FINAL_RESPONSE"
    TASK_COMPLETED = "TASK_COMPLETED"
    VERIFICATION_PASSED = "VERIFICATION_PASSED"
    MORE_WORK_REQUIRED = "MORE_WORK_REQUIRED"
    TOOL_RESULT_REQUIRES_FOLLOW_UP = "TOOL_RESULT_REQUIRES_FOLLOW_UP"
    PLAN_STEP_REMAINS = "PLAN_STEP_REMAINS"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    TOOL_FAILURE = "TOOL_FAILURE"
    INVALID_ACTION = "INVALID_ACTION"
    MISSING_CAPABILITY = "MISSING_CAPABILITY"
    SAFETY_BLOCK = "SAFETY_BLOCK"
    EMERGENCY_BOUND_REACHED = "EMERGENCY_BOUND_REACHED"
    INCOMPLETE_CONTEXT = "INCOMPLETE_CONTEXT"
    UNRESOLVED_STATE = "UNRESOLVED_STATE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    POLICY_DENIED = "POLICY_DENIED"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class VerificationState(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    REQUIRED = "REQUIRED"
    PENDING = "PENDING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """Bounded structured evidence used to decide whether completion is proven."""

    state: VerificationState = VerificationState.NOT_REQUIRED
    source: str | None = None
    message: str | None = None
    complete: bool = True
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, VerificationState):
            raise ValueError("state must be VerificationState")
        if self.source is not None and not isinstance(self.source, str):
            raise ValueError("source must be text or None")
        if self.message is not None and not isinstance(self.message, str):
            raise ValueError("message must be text or None")
        if not isinstance(self.complete, bool):
            raise ValueError("complete must be boolean")
        object.__setattr__(self, "evidence", _bounded_unique(self.evidence, 8, 512))

    @classmethod
    def not_required(cls, *evidence: str) -> "VerificationEvidence":
        return cls(VerificationState.NOT_REQUIRED, evidence=tuple(evidence))

    @classmethod
    def pending(cls, source: str | None = None, message: str | None = None) -> "VerificationEvidence":
        return cls(VerificationState.PENDING, source=source, message=message, complete=False)

    @classmethod
    def passed(cls, source: str, message: str | None = None, *evidence: str) -> "VerificationEvidence":
        return cls(VerificationState.PASSED, source=source, message=message, evidence=tuple(evidence))

    @classmethod
    def failed(cls, source: str, message: str | None = None, *, complete: bool = True) -> "VerificationEvidence":
        return cls(VerificationState.FAILED, source=source, message=message, complete=complete)

    @classmethod
    def unavailable(cls, source: str | None = None, message: str | None = None) -> "VerificationEvidence":
        return cls(VerificationState.UNAVAILABLE, source=source, message=message, complete=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "source": self.source,
            "message": _bounded(self.message or "", 1_024) if self.message is not None else None,
            "complete": self.complete,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class StopConditionRequest:
    """Pure evaluator input; all project facts must be supplied explicitly."""

    plan: ExecutionPlan | None = None
    completed_step_ids: tuple[str, ...] = ()
    skipped_step_ids: tuple[str, ...] = ()
    blocked_step_ids: tuple[str, ...] = ()
    current_step_id: str | None = None
    final_action_valid: bool = False
    invalid_action: bool = False
    last_tool_result: ToolResult | None = None
    last_tool_name: str | None = None
    tool_result_requires_follow_up: bool = False
    tool_result_recoverable: bool = False
    missing_capabilities: tuple[str, ...] = ()
    safety_blocked: bool = False
    context_complete: bool = True
    completion_evidence: tuple[str, ...] = ()
    verification: VerificationEvidence = field(default_factory=VerificationEvidence.not_required)
    emergency_bound_reached: bool = False
    fatal_error: str | None = None
    budget_decision: BudgetDecision | None = None
    warning_messages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("completed_step_ids", "skipped_step_ids", "blocked_step_ids", "missing_capabilities", "completion_evidence", "warning_messages"):
            object.__setattr__(self, name, _bounded_unique(getattr(self, name), 64, 512))
        if self.current_step_id is not None and not isinstance(self.current_step_id, str):
            raise ValueError("current_step_id must be text or None")
        if not isinstance(self.final_action_valid, bool) or not isinstance(self.invalid_action, bool):
            raise ValueError("action flags must be boolean")
        if self.last_tool_name is not None and not isinstance(self.last_tool_name, str):
            raise ValueError("last_tool_name must be text or None")
        if not isinstance(self.verification, VerificationEvidence):
            raise ValueError("verification must be VerificationEvidence")
        if self.fatal_error is not None and not isinstance(self.fatal_error, str):
            raise ValueError("fatal_error must be text or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict() if self.plan else None,
            "completed_step_ids": list(self.completed_step_ids),
            "skipped_step_ids": list(self.skipped_step_ids),
            "blocked_step_ids": list(self.blocked_step_ids),
            "current_step_id": self.current_step_id,
            "final_action_valid": self.final_action_valid,
            "invalid_action": self.invalid_action,
            "last_tool_result": _safe_result(self.last_tool_result),
            "last_tool_name": self.last_tool_name,
            "tool_result_requires_follow_up": self.tool_result_requires_follow_up,
            "tool_result_recoverable": self.tool_result_recoverable,
            "missing_capabilities": list(self.missing_capabilities),
            "safety_blocked": self.safety_blocked,
            "context_complete": self.context_complete,
            "completion_evidence": list(self.completion_evidence),
            "verification": self.verification.to_dict(),
            "emergency_bound_reached": self.emergency_bound_reached,
            "fatal_error": _bounded(self.fatal_error or "", 1_024) if self.fatal_error is not None else None,
            "budget_decision": self.budget_decision.to_dict() if self.budget_decision else None,
            "warning_messages": list(self.warning_messages),
        }


@dataclass(frozen=True, slots=True)
class StopEvaluation:
    """Immutable serializable stop decision with bounded evidence and explanations."""

    decision: StopDecision
    reason: StopReason
    evidence: tuple[str, ...] = ()
    blocking_conditions: tuple[str, ...] = ()
    remaining_required_steps: tuple[str, ...] = ()
    completed_step_ids: tuple[str, ...] = ()
    verification: VerificationEvidence = field(default_factory=VerificationEvidence.not_required)
    tool_state: str = "NO_TOOL_RESULT"
    confidence: str = "MEDIUM"
    emergency_bound_reached: bool = False
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.decision, StopDecision) or not isinstance(self.reason, StopReason):
            raise ValueError("decision and reason must use their enums")
        for name in ("evidence", "blocking_conditions", "remaining_required_steps", "completed_step_ids", "warnings"):
            object.__setattr__(self, name, _bounded_unique(getattr(self, name), 32, 1_024))
        if not isinstance(self.verification, VerificationEvidence):
            raise ValueError("verification must be VerificationEvidence")
        if not isinstance(self.tool_state, str) or not isinstance(self.confidence, str):
            raise ValueError("tool_state and confidence must be text")
        object.__setattr__(self, "tool_state", _bounded(self.tool_state, 256))
        object.__setattr__(self, "confidence", _bounded(self.confidence, 64))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason.value,
            "evidence": list(self.evidence),
            "blocking_conditions": list(self.blocking_conditions),
            "remaining_required_steps": list(self.remaining_required_steps),
            "completed_step_ids": list(self.completed_step_ids),
            "verification": self.verification.to_dict(),
            "tool_state": self.tool_state,
            "confidence": self.confidence,
            "emergency_bound_reached": self.emergency_bound_reached,
            "warnings": list(self.warnings),
        }


class StopConditionEvaluator:
    """Pure deterministic evaluator; it performs no I/O, dispatch, or model call."""

    def evaluate(self, request: StopConditionRequest) -> StopEvaluation:
        if not isinstance(request, StopConditionRequest):
            raise TypeError("request must be StopConditionRequest")

        completed = set(request.completed_step_ids)
        skipped = set(request.skipped_step_ids)
        blocked = set(request.blocked_step_ids)
        remaining = self._remaining_required(request.plan, completed, skipped)
        evidence = list(request.completion_evidence)
        warnings = list(request.warning_messages)

        if request.budget_decision is not None and not request.budget_decision.allowed:
            return self._result(StopDecision.BUDGET_EXHAUSTED, StopReason.BUDGET_EXHAUSTED, request, evidence + [request.budget_decision.message], remaining, "BUDGET_EXHAUSTED", "HIGH", warnings, blocking=[request.budget_decision.exhaustion.value if request.budget_decision.exhaustion else "budget limit reached"])
        if request.fatal_error:
            return self._result(StopDecision.FAILED, StopReason.INTERNAL_ERROR, request, evidence + [request.fatal_error], remaining, "FATAL_ERROR", "HIGH", warnings)
        if request.invalid_action:
            return self._result(StopDecision.FAILED, StopReason.INVALID_ACTION, request, evidence + ["invalid action"], remaining, "INVALID_ACTION", "HIGH", warnings)
        if blocked or request.missing_capabilities or request.safety_blocked:
            reasons = list(blocked) + list(request.missing_capabilities)
            if request.safety_blocked:
                reasons.append("safety policy blocked the required operation")
            reason = StopReason.SAFETY_BLOCK if request.safety_blocked else (StopReason.MISSING_CAPABILITY if request.missing_capabilities or blocked else StopReason.UNRESOLVED_STATE)
            return self._result(StopDecision.BLOCKED, reason, request, evidence + reasons, remaining, "BLOCKED", "HIGH", warnings, blocking=reasons)
        if request.emergency_bound_reached:
            return self._result(StopDecision.BLOCKED, StopReason.EMERGENCY_BOUND_REACHED, request, evidence + ["fixed emergency tool bound reached"], remaining, "EMERGENCY_BOUND", "HIGH", warnings, blocking=["no additional tool execution is permitted"])

        tool_failure = self._tool_failure(request.last_tool_result)
        if tool_failure is not None:
            if tool_failure[0] == "BLOCKED":
                return self._result(StopDecision.BLOCKED, tool_failure[1], request, evidence + [tool_failure[2]], remaining, "BLOCKED_TOOL_RESULT", "HIGH", warnings, blocking=[tool_failure[2]])
            if request.tool_result_recoverable:
                return self._result(StopDecision.CONTINUE, StopReason.TOOL_RESULT_REQUIRES_FOLLOW_UP, request, evidence + [tool_failure[2]], remaining, "RECOVERABLE_TOOL_FAILURE", "MEDIUM", warnings)
            return self._result(StopDecision.FAILED, StopReason.TOOL_FAILURE, request, evidence + [tool_failure[2]], remaining, "FAILED_TOOL_RESULT", "HIGH", warnings)

        if request.verification.state is VerificationState.UNAVAILABLE:
            return self._result(StopDecision.BLOCKED, StopReason.INCOMPLETE_CONTEXT, request, evidence + [request.verification.message or "verification unavailable"], remaining, "VERIFICATION_UNAVAILABLE", "HIGH", warnings, blocking=[request.verification.message or "verification evidence is unavailable"])
        if request.verification.state is VerificationState.FAILED:
            if request.verification.complete:
                return self._result(StopDecision.CONTINUE, StopReason.VERIFICATION_FAILED, request, evidence + [request.verification.message or "verification failed"], remaining, "VERIFICATION_FAILED", "MEDIUM", warnings)
            return self._result(StopDecision.BLOCKED, StopReason.INCOMPLETE_CONTEXT, request, evidence + [request.verification.message or "verification incomplete"], remaining, "VERIFICATION_INCOMPLETE", "HIGH", warnings, blocking=["verification evidence is incomplete"])
        if request.verification.state in {VerificationState.REQUIRED, VerificationState.PENDING, VerificationState.INCOMPLETE}:
            return self._result(StopDecision.CONTINUE, StopReason.VERIFICATION_REQUIRED, request, evidence + [request.verification.message or "required verification has not passed"], remaining, "VERIFICATION_PENDING", "HIGH", warnings)

        if request.final_action_valid:
            if remaining:
                return self._result(StopDecision.CONTINUE, StopReason.PLAN_STEP_REMAINS, request, evidence + ["valid FINAL action was received but required plan steps remain"], remaining, "FINAL_WITH_REMAINING_WORK", "HIGH", warnings)
            if not request.context_complete:
                return self._result(StopDecision.BLOCKED, StopReason.INCOMPLETE_CONTEXT, request, evidence + ["project context is incomplete"], remaining, "INCOMPLETE_CONTEXT", "HIGH", warnings, blocking=["completion evidence is incomplete"])
            reason = StopReason.VERIFICATION_PASSED if request.verification.state is VerificationState.PASSED else StopReason.FINAL_RESPONSE
            return self._result(StopDecision.DONE, reason, request, evidence + ["valid FINAL action"], remaining, "FINAL", "HIGH", warnings)

        if request.tool_result_requires_follow_up:
            return self._result(StopDecision.CONTINUE, StopReason.TOOL_RESULT_REQUIRES_FOLLOW_UP, request, evidence + ["tool result explicitly requires follow-up"], remaining, "FOLLOW_UP_REQUIRED", "MEDIUM", warnings)
        if remaining:
            return self._result(StopDecision.CONTINUE, StopReason.PLAN_STEP_REMAINS, request, evidence + ["required plan steps remain"], remaining, "WORK_REMAINS", "HIGH", warnings)
        if request.verification.state is VerificationState.PASSED and request.context_complete and evidence:
            return self._result(StopDecision.DONE, StopReason.VERIFICATION_PASSED, request, evidence + ["verification passed"], remaining, "VERIFIED", "HIGH", warnings)
        if request.context_complete and evidence and request.last_tool_result is not None and request.last_tool_result.success:
            return self._result(StopDecision.DONE, StopReason.TASK_COMPLETED, request, evidence + ["all required steps are complete"], remaining, "SUCCESSFUL_EVIDENCE", "MEDIUM", warnings)
        return self._result(StopDecision.CONTINUE, StopReason.MORE_WORK_REQUIRED, request, evidence + ["completion criteria are not yet proven"], remaining, "UNRESOLVED", "MEDIUM", warnings)

    @staticmethod
    def _remaining_required(plan: ExecutionPlan | None, completed: set[str], skipped: set[str]) -> tuple[str, ...]:
        if plan is None:
            return ()
        return tuple(step.step_id for step in plan.steps if step.step_id not in completed and step.step_id not in skipped)

    @staticmethod
    def _tool_failure(result: ToolResult | None) -> tuple[str, StopReason, str] | None:
        if result is None or result.success:
            return None
        code = (result.error_code or "TOOL_EXECUTION_FAILED").upper()
        message = result.message or code
        if any(marker in code for marker in ("UNKNOWN_TOOL", "TOOL_UNAVAILABLE", "COMMAND_DENIED", "COMMAND_NOT_ALLOWED", "POLICY", "PATH_OUTSIDE_ROOT", "PERMISSION_DENIED", "SAFETY")):
            reason = StopReason.POLICY_DENIED if any(marker in code for marker in ("POLICY", "COMMAND_DENIED", "COMMAND_NOT_ALLOWED", "SAFETY")) else StopReason.TOOL_UNAVAILABLE
            return "BLOCKED", reason, message
        if any(marker in code for marker in ("INTERNAL", "INFRASTRUCTURE", "CORRUPTED")):
            return "FAILED", StopReason.INTERNAL_ERROR, message
        return "FAILED", StopReason.TOOL_FAILURE, message

    @staticmethod
    def _result(decision: StopDecision, reason: StopReason, request: StopConditionRequest, evidence: Sequence[str], remaining: Sequence[str], tool_state: str, confidence: str, warnings: Sequence[str], *, blocking: Sequence[str] = ()) -> StopEvaluation:
        return StopEvaluation(decision, reason, _bounded_unique(evidence, 32, 1_024), _bounded_unique(blocking, 16, 1_024), _bounded_unique(remaining, 64, 256), _bounded_unique(request.completed_step_ids, 64, 256), request.verification, tool_state, confidence, request.emergency_bound_reached, _bounded_unique(warnings, 16, 512))


def evaluate_stop_condition(request: StopConditionRequest) -> StopEvaluation:
    """Pure convenience API for deterministic stop evaluation."""

    return StopConditionEvaluator().evaluate(request)


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 24)] + "\n[truncated]"


def _bounded_unique(values: Sequence[str], max_items: int, max_chars: int) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        bounded = _bounded(value, max_chars)
        if bounded not in result:
            result.append(bounded)
        if len(result) >= max_items:
            break
    return tuple(result)


def _safe_result(result: ToolResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    data = result.to_dict()
    if isinstance(data.get("data"), dict):
        data["data"] = {str(key): "[bounded]" for key in data["data"]}
    return data


__all__ = [
    "StopConditionEvaluator",
    "StopConditionRequest",
    "StopDecision",
    "StopEvaluation",
    "StopReason",
    "VerificationEvidence",
    "VerificationState",
    "evaluate_stop_condition",
]
