"""Bounded, evidence-driven error recovery for the explicit autonomous loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
import re

from backend_ai.agent.execution_budget import BudgetDecision, ExecutionBudgetSnapshot
from backend_ai.agent.models import ToolResult

_MAX_TEXT = 1_024
_MAX_ITEMS = 16


class ErrorCategory(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    SAFETY_BLOCK = "SAFETY_BLOCK"
    POLICY_DENIAL = "POLICY_DENIAL"
    PROJECT_ROOT_VIOLATION = "PROJECT_ROOT_VIOLATION"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_EXISTS = "FILE_EXISTS"
    MATCH_NOT_FOUND = "MATCH_NOT_FOUND"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    CONCURRENT_MODIFICATION = "CONCURRENT_MODIFICATION"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    COMMAND_FAILURE = "COMMAND_FAILURE"
    PROCESS_TIMEOUT = "PROCESS_TIMEOUT"
    OUTPUT_LIMIT = "OUTPUT_LIMIT"
    APPLICATION_FAILURE = "APPLICATION_FAILURE"
    TEST_FAILURE = "TEST_FAILURE"
    TEST_ERROR = "TEST_ERROR"
    PARSE_FAILURE = "PARSE_FAILURE"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIMEOUT = "TIMEOUT"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    RESOURCE_ERROR = "RESOURCE_ERROR"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    CONTEXT_LIMIT = "CONTEXT_LIMIT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class RecoverySeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecoveryConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(str, Enum):
    CONTINUE_WITH_NEW_ACTION = "CONTINUE_WITH_NEW_ACTION"
    REPLAN = "REPLAN"
    INSPECT = "INSPECT"
    VERIFY = "VERIFY"
    STOP = "STOP"
    BLOCK = "BLOCK"
    USER_INTERVENTION_REQUIRED = "USER_INTERVENTION_REQUIRED"


class RecoveryStatus(str, Enum):
    CONTINUE = "CONTINUE"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    USER_INTERVENTION_REQUIRED = "USER_INTERVENTION_REQUIRED"


@dataclass(frozen=True, slots=True)
class ErrorClassification:
    category: ErrorCategory
    recoverable: bool
    severity: RecoverySeverity
    another_action_allowed: bool
    user_intervention_required: bool
    terminate_task: bool
    safety_or_policy_boundary: bool
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category.value, "recoverable": self.recoverable, "severity": self.severity.value, "another_action_allowed": self.another_action_allowed, "user_intervention_required": self.user_intervention_required, "terminate_task": self.terminate_task, "safety_or_policy_boundary": self.safety_or_policy_boundary, "code": self.code, "message": _bounded(self.message)}


@dataclass(frozen=True, slots=True)
class RecoveryContext:
    tool_result: ToolResult
    operation: str
    plan_step_id: str | None = None
    selected_tool: str | None = None
    has_next_plan_step: bool = False
    evidence: tuple[str, ...] = ()
    failed_action_signatures: tuple[str, ...] = ()
    budget_snapshot: ExecutionBudgetSnapshot | None = None
    budget_decision: BudgetDecision | None = None
    verification_required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.tool_result, ToolResult):
            raise TypeError("tool_result must be ToolResult")
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must contain text")
        object.__setattr__(self, "evidence", _bounded_items(self.evidence))
        object.__setattr__(self, "failed_action_signatures", _bounded_items(self.failed_action_signatures))

    def to_dict(self) -> dict[str, Any]:
        return {"tool_result": _safe_result(self.tool_result), "operation": self.operation, "plan_step_id": self.plan_step_id, "selected_tool": self.selected_tool, "has_next_plan_step": self.has_next_plan_step, "evidence": list(self.evidence), "failed_action_signatures": list(self.failed_action_signatures), "budget_snapshot": self.budget_snapshot.to_dict() if self.budget_snapshot else None, "budget_decision": self.budget_decision.to_dict() if self.budget_decision else None, "verification_required": self.verification_required}


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    status: RecoveryStatus
    classification: ErrorClassification
    action: RecoveryAction
    reason: str
    confidence: RecoveryConfidence
    affected_tool: str
    affected_operation: str
    another_tool_call_allowed: bool
    mutation_allowed: bool
    execution_allowed: bool
    verification_required: bool
    budget_impact: str
    warnings: tuple[str, ...] = ()
    blocking_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", _bounded_items(self.warnings))
        object.__setattr__(self, "blocking_conditions", _bounded_items(self.blocking_conditions))

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "classification": self.classification.to_dict(), "action": self.action.value, "reason": _bounded(self.reason), "confidence": self.confidence.value, "affected_tool": self.affected_tool, "affected_operation": self.affected_operation, "another_tool_call_allowed": self.another_tool_call_allowed, "mutation_allowed": self.mutation_allowed, "execution_allowed": self.execution_allowed, "verification_required": self.verification_required, "budget_impact": _bounded(self.budget_impact), "warnings": list(self.warnings), "blocking_conditions": list(self.blocking_conditions)}


@dataclass(frozen=True, slots=True)
class RecoveryHistoryRecord:
    classification: ErrorClassification
    failed_operation: str
    decision: RecoveryDecision
    selected_next_action: str | None
    reason: str
    confidence: RecoveryConfidence
    budget_snapshot: ExecutionBudgetSnapshot | None
    outcome: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"classification": self.classification.to_dict(), "failed_operation": self.failed_operation, "decision": self.decision.to_dict(), "selected_next_action": self.selected_next_action, "reason": _bounded(self.reason), "confidence": self.confidence.value, "budget_snapshot": self.budget_snapshot.to_dict() if self.budget_snapshot else None, "outcome": _bounded(self.outcome), "warnings": list(self.warnings)}


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    decision: RecoveryDecision
    history: tuple[RecoveryHistoryRecord, ...] = ()
    original_error_preserved: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "history", tuple(self.history[-_MAX_ITEMS:]))

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision.to_dict(), "history": [item.to_dict() for item in self.history], "original_error_preserved": self.original_error_preserved}


class ErrorClassifier:
    """Pure classifier over already-structured tool evidence; never performs I/O."""

    def classify(self, result: ToolResult, *, operation: str | None = None) -> ErrorClassification:
        code = (result.error_code or "UNKNOWN_ERROR").upper()
        category = self._category(code, operation)
        boundary = category in {ErrorCategory.SAFETY_BLOCK, ErrorCategory.POLICY_DENIAL, ErrorCategory.PROJECT_ROOT_VIOLATION}
        user = category in {ErrorCategory.CONCURRENT_MODIFICATION, ErrorCategory.AMBIGUOUS_MATCH} or category is ErrorCategory.PROJECT_ROOT_VIOLATION
        unrecoverable = category in {ErrorCategory.VALIDATION_ERROR, ErrorCategory.INVALID_ARGUMENT, ErrorCategory.BUDGET_EXHAUSTED, ErrorCategory.CONTEXT_LIMIT, ErrorCategory.INTERNAL_ERROR, ErrorCategory.UNKNOWN_ERROR}
        recoverable = not boundary and not unrecoverable and category not in {ErrorCategory.TOOL_UNAVAILABLE}
        if category in {ErrorCategory.FILE_NOT_FOUND, ErrorCategory.MATCH_NOT_FOUND, ErrorCategory.VERIFICATION_FAILURE, ErrorCategory.TEST_FAILURE, ErrorCategory.TEST_ERROR, ErrorCategory.COMMAND_FAILURE, ErrorCategory.PROCESS_TIMEOUT, ErrorCategory.OUTPUT_LIMIT, ErrorCategory.APPLICATION_FAILURE}:
            recoverable = True
        severity = RecoverySeverity.CRITICAL if category in {ErrorCategory.SAFETY_BLOCK, ErrorCategory.POLICY_DENIAL, ErrorCategory.PROJECT_ROOT_VIOLATION, ErrorCategory.BUDGET_EXHAUSTED, ErrorCategory.INTERNAL_ERROR} else RecoverySeverity.HIGH if not recoverable else RecoverySeverity.MEDIUM
        return ErrorClassification(category, recoverable, severity, recoverable and not user, user, boundary or unrecoverable or user, boundary, code, result.message or code)

    @staticmethod
    def _category(code: str, operation: str | None) -> ErrorCategory:
        if "BUDGET" in code: return ErrorCategory.BUDGET_EXHAUSTED
        if "CONTEXT" in code: return ErrorCategory.CONTEXT_LIMIT
        if "PATH_OUTSIDE_ROOT" in code or "ROOT_VIOLATION" in code: return ErrorCategory.PROJECT_ROOT_VIOLATION
        if "PERMISSION" in code or "SAFETY" in code: return ErrorCategory.SAFETY_BLOCK
        if any(x in code for x in ("POLICY", "DENIED", "NOT_ALLOWED", "UNSAFE", "SHELL_BYPASS")): return ErrorCategory.POLICY_DENIAL
        if "CONCURRENT" in code: return ErrorCategory.CONCURRENT_MODIFICATION
        if "AMBIGUOUS" in code: return ErrorCategory.AMBIGUOUS_MATCH
        if "MATCH_NOT_FOUND" in code: return ErrorCategory.MATCH_NOT_FOUND
        if "FILE_NOT_FOUND" in code or "PATH_NOT_FOUND" in code: return ErrorCategory.FILE_NOT_FOUND
        if "FILE_EXISTS" in code: return ErrorCategory.FILE_EXISTS
        if "VERIFICATION" in code: return ErrorCategory.VERIFICATION_FAILURE
        if "TIMEOUT" in code: return ErrorCategory.PROCESS_TIMEOUT
        if "OUTPUT" in code: return ErrorCategory.OUTPUT_LIMIT
        if "APPLICATION" in code: return ErrorCategory.APPLICATION_FAILURE
        if "TEST" in code and any(x in code for x in ("FAIL", "NONZERO")): return ErrorCategory.TEST_FAILURE
        if "TEST" in code: return ErrorCategory.TEST_ERROR
        if "PARSE" in code: return ErrorCategory.PARSE_FAILURE
        if "DEPENDENCY" in code or "MODULE" in code or "IMPORT" in code: return ErrorCategory.DEPENDENCY_ERROR
        if "RUNTIME" in code or "TYPE" in code or "VALUE" in code or "KEY" in code or "ATTRIBUTE" in code or "TRACEBACK" in code: return ErrorCategory.RUNTIME_ERROR
        if "TIMEOUT" in code or "EXCEEDED" in code: return ErrorCategory.TIMEOUT
        if "PERMISSION" in code or "ACCES" in code: return ErrorCategory.PERMISSION_ERROR
        if "COMMAND" in code or "PROCESS" in code: return ErrorCategory.COMMAND_FAILURE
        if "ARGUMENT" in code or "INVALID" in code: return ErrorCategory.INVALID_ARGUMENT
        if "UNAVAILABLE" in code or "UNKNOWN_TOOL" in code: return ErrorCategory.TOOL_UNAVAILABLE
        if "INTERNAL" in code or "CORRUPT" in code: return ErrorCategory.INTERNAL_ERROR
        return ErrorCategory.UNKNOWN_ERROR


class RecoverabilityPolicy:
    """Strict deterministic policy; it proposes safe observation, never retries blindly."""

    def decide(self, context: RecoveryContext) -> RecoveryResult:
        classification = ErrorClassifier().classify(context.tool_result, operation=context.operation)
        if context.budget_decision is not None and not context.budget_decision.allowed:
            classification = ErrorClassification(ErrorCategory.BUDGET_EXHAUSTED, False, RecoverySeverity.CRITICAL, False, False, True, False, "BUDGET_EXHAUSTED", context.budget_decision.message)
        if classification.safety_or_policy_boundary:
            action = RecoveryAction.USER_INTERVENTION_REQUIRED if classification.user_intervention_required else RecoveryAction.BLOCK
            status = RecoveryStatus.USER_INTERVENTION_REQUIRED if classification.user_intervention_required else RecoveryStatus.BLOCKED
            return self._result(context, classification, action, status, RecoveryConfidence.HIGH, "Safety or policy boundaries are never bypassed.", "no recovery budget consumed", ("automatic recovery is prohibited at this boundary",))
        if classification.category is ErrorCategory.BUDGET_EXHAUSTED:
            return self._result(context, classification, RecoveryAction.BLOCK, RecoveryStatus.BLOCKED, RecoveryConfidence.HIGH, "Execution budget is authoritative and cannot be increased by recovery.", "recovery stopped; no retry", ("budget exhausted",))
        if classification.user_intervention_required:
            return self._result(context, classification, RecoveryAction.USER_INTERVENTION_REQUIRED, RecoveryStatus.USER_INTERVENTION_REQUIRED, RecoveryConfidence.HIGH, "Automatic recovery could overwrite or reinterpret a user-owned state.", "no recovery action", ("user intervention is required",))
        if not classification.recoverable:
            action = RecoveryAction.USER_INTERVENTION_REQUIRED if classification.user_intervention_required else RecoveryAction.STOP
            status = RecoveryStatus.USER_INTERVENTION_REQUIRED if classification.user_intervention_required else RecoveryStatus.FAILED
            return self._result(context, classification, action, status, RecoveryConfidence.HIGH, "Evidence is insufficient for a safe automatic action.", "no recovery action", (classification.category.value,))
        if not context.has_next_plan_step:
            return self._result(context, classification, RecoveryAction.REPLAN, RecoveryStatus.CONTINUE, RecoveryConfidence.MEDIUM, "The failure is actionable, but a bounded replan is required before another action.", "one bounded replan decision", ("no blind retry",))
        if context.failed_action_signatures and context.operation in context.failed_action_signatures:
            return self._result(context, classification, RecoveryAction.STOP, RecoveryStatus.FAILED, RecoveryConfidence.HIGH, "The identical failed action is already recorded; repetition is blocked.", "no retry", ("repeated identical failure",))
        if classification.category in {ErrorCategory.VERIFICATION_FAILURE, ErrorCategory.CONCURRENT_MODIFICATION}:
            action = RecoveryAction.USER_INTERVENTION_REQUIRED if classification.category is ErrorCategory.CONCURRENT_MODIFICATION else RecoveryAction.VERIFY
        elif classification.category in {ErrorCategory.FILE_NOT_FOUND, ErrorCategory.MATCH_NOT_FOUND, ErrorCategory.AMBIGUOUS_MATCH, ErrorCategory.TEST_FAILURE, ErrorCategory.TEST_ERROR, ErrorCategory.COMMAND_FAILURE, ErrorCategory.PROCESS_TIMEOUT, ErrorCategory.OUTPUT_LIMIT, ErrorCategory.APPLICATION_FAILURE}:
            action = RecoveryAction.INSPECT
        else:
            action = RecoveryAction.REPLAN
        return self._result(context, classification, action, RecoveryStatus.CONTINUE, RecoveryConfidence.MEDIUM, "Structured failure evidence supports a different bounded inspection or verification action.", "normal existing budget dimensions apply", ("do not repeat the failed action",))

    @staticmethod
    def _result(context: RecoveryContext, classification: ErrorClassification, action: RecoveryAction, status: RecoveryStatus, confidence: RecoveryConfidence, reason: str, impact: str, blocking: tuple[str, ...]) -> RecoveryResult:
        decision = RecoveryDecision(status, classification, action, reason, confidence, context.selected_tool or context.tool_result.tool_name, context.operation, status is RecoveryStatus.CONTINUE, False, False, context.verification_required or action is RecoveryAction.VERIFY, impact, warnings=("Recovery is bounded decision-making over structured evidence; it is not unrestricted self-correction.", "Do not blind retry the failed action."), blocking_conditions=blocking)
        record = RecoveryHistoryRecord(classification, context.operation, decision, None, reason, confidence, context.budget_snapshot, "DECIDED", decision.warnings)
        return RecoveryResult(decision, (record,))


def classify_error(result: ToolResult, *, operation: str | None = None) -> ErrorClassification:
    return ErrorClassifier().classify(result, operation=operation)


def decide_recovery(context: RecoveryContext) -> RecoveryResult:
    return RecoverabilityPolicy().decide(context)


def _bounded(value: str) -> str:
    return value if len(value) <= _MAX_TEXT else value[:_MAX_TEXT - 14] + "\n[truncated]"


def _bounded_items(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_bounded(v) for v in values if isinstance(v, str) and v)[:_MAX_ITEMS]


def _safe_result(result: ToolResult) -> dict[str, Any]:
    data = result.to_dict()
    if data.get("data") is not None:
        data["data"] = "[bounded]"
    return data


__all__ = ["ErrorCategory", "ErrorClassification", "ErrorClassifier", "RecoveryAction", "RecoveryConfidence", "RecoveryContext", "RecoveryDecision", "RecoveryHistoryRecord", "RecoveryResult", "RecoverySeverity", "RecoveryStatus", "RecoverabilityPolicy", "classify_error", "decide_recovery"]


@dataclass(frozen=True, slots=True)
class NormalizedError:
    category: ErrorCategory
    message: str
    tool_name: str | None = None
    command: str | None = None
    file_path: str | None = None
    exit_code: int | None = None
    stderr: str | None = None
    stdout: str | None = None
    recoverable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "message": _bounded(self.message),
            "tool_name": self.tool_name,
            "command": self.command,
            "file_path": self.file_path,
            "exit_code": self.exit_code,
            "stderr": _bounded(self.stderr) if self.stderr else None,
            "stdout": _bounded(self.stdout) if self.stdout else None,
            "recoverable": self.recoverable,
        }


def normalize_error(result: ToolResult, *, tool_name: str | None = None, command: str | None = None, file_path: str | None = None) -> NormalizedError:
    classifier = ErrorClassifier()
    classification = classifier.classify(result)
    stderr = getattr(result, "stderr", None)
    stdout = getattr(result, "stdout", None)
    exit_code = getattr(result, "exit_code", None)
    return NormalizedError(
        category=classification.category,
        message=result.message or classification.code,
        tool_name=tool_name or result.tool_name,
        command=command,
        file_path=file_path,
        exit_code=exit_code,
        stderr=stderr,
        stdout=stdout,
        recoverable=classification.recoverable,
    )


def compute_error_signature(result: ToolResult, *, tool_name: str | None = None, command: str | None = None) -> str:
    normalized = normalize_error(result, tool_name=tool_name, command=command)
    msg_cleaned = re.sub(r"\(timestamp \d+\)", "", normalized.message, flags=re.IGNORECASE)
    msg_cleaned = re.sub(r"\s+", " ", msg_cleaned).strip()
    return f"{normalized.category.value}:{normalized.tool_name}:{normalized.command}:{normalized.exit_code}:{msg_cleaned}"
