"""Pure, bounded task-completion verification over existing structured evidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from backend_ai.agent.execution_budget import ExecutionBudgetSnapshot
from backend_ai.agent.models import ToolResult
from backend_ai.agent.planner import ExecutionPlan, PlannerTaskType
from backend_ai.agent.recovery import RecoveryResult, RecoveryStatus
from backend_ai.agent.stop_conditions import VerificationEvidence, VerificationState

_MAX = 64
_MAX_TEXT = 1_024


class CompletionStatus(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class CriterionStatus(str, Enum):
    SATISFIED = "SATISFIED"
    UNSATISFIED = "UNSATISFIED"
    BLOCKED = "BLOCKED"
    UNVERIFIED = "UNVERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceStrength(str, Enum):
    DIRECT = "DIRECT"
    STRONG = "STRONG"
    INDIRECT = "INDIRECT"
    INSUFFICIENT = "INSUFFICIENT"
    NONE = "NONE"


class CompletionConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class CompletionDecision(str, Enum):
    COMPLETE = "COMPLETE"
    CONTINUE = "CONTINUE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class TaskCompletionCriterion:
    criterion_id: str
    description: str
    required: bool = True
    kind: str = "custom"
    target: str | None = None

    def __post_init__(self) -> None:
        if not self.criterion_id.strip() or not self.description.strip():
            raise ValueError("criterion_id and description must contain text")
        if not isinstance(self.required, bool):
            raise ValueError("required must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"criterion_id": self.criterion_id, "description": _bounded(self.description), "required": self.required, "kind": self.kind, "target": self.target}


@dataclass(frozen=True, slots=True)
class TaskCompletionEvidence:
    source: str
    detail: str
    strength: EvidenceStrength = EvidenceStrength.INDIRECT
    status: str = "OBSERVED"
    expected: str | None = None
    observed: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.detail.strip():
            raise ValueError("evidence source and detail must contain text")
        object.__setattr__(self, "detail", _bounded(self.detail))
        if self.expected is not None:
            object.__setattr__(self, "expected", _bounded(self.expected))
        if self.observed is not None:
            object.__setattr__(self, "observed", _bounded(self.observed))

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "detail": self.detail, "strength": self.strength.value, "status": self.status, "expected": self.expected, "observed": self.observed}


@dataclass(frozen=True, slots=True)
class TaskCompletionItem:
    criterion: TaskCompletionCriterion
    status: CriterionStatus
    evidence: tuple[TaskCompletionEvidence, ...] = ()
    reason: str = ""
    recoverable: bool = False
    user_intervention_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence[:_MAX]))
        object.__setattr__(self, "reason", _bounded(self.reason))

    def to_dict(self) -> dict[str, Any]:
        return {"criterion": self.criterion.to_dict(), "status": self.status.value, "evidence": [item.to_dict() for item in self.evidence], "reason": self.reason, "recoverable": self.recoverable, "user_intervention_required": self.user_intervention_required}


@dataclass(frozen=True, slots=True)
class TaskCompletionRequest:
    task: str
    plan: ExecutionPlan | None = None
    criteria: tuple[TaskCompletionCriterion, ...] = ()
    completed_step_ids: tuple[str, ...] = ()
    skipped_step_ids: tuple[str, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    verification: VerificationEvidence = field(default_factory=VerificationEvidence.not_required)
    recovery: RecoveryResult | None = None
    budget: ExecutionBudgetSnapshot | None = None
    evidence: tuple[TaskCompletionEvidence, ...] = ()
    unexpected_modifications: tuple[str, ...] = ()
    critical_unexpected_modifications: tuple[str, ...] = ()
    evidence_complete: bool = True
    final_response: str | None = None
    regression_required: bool = False
    regression_protection: Any | None = None
    final_verification_required: bool = False
    final_verification: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must contain text")
        if len(self.criteria) > _MAX or len(self.evidence) > _MAX or len(self.tool_results) > _MAX:
            raise ValueError("completion evidence exceeds bounded limit")
        for name in ("completed_step_ids", "skipped_step_ids", "unexpected_modifications", "critical_unexpected_modifications"):
            object.__setattr__(self, name, tuple(getattr(self, name)[:_MAX]))


@dataclass(frozen=True, slots=True)
class TaskCompletionResult:
    decision: CompletionDecision
    status: CompletionStatus
    confidence: CompletionConfidence
    items: tuple[TaskCompletionItem, ...]
    evidence: tuple[TaskCompletionEvidence, ...]
    remaining_criteria: tuple[str, ...]
    blocking_conditions: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("items", "evidence", "remaining_criteria", "blocking_conditions", "warnings"):
            object.__setattr__(self, name, tuple(getattr(self, name)[:_MAX]))

    @property
    def complete(self) -> bool:
        return self.decision is CompletionDecision.COMPLETE

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision.value, "status": self.status.value, "confidence": self.confidence.value, "complete": self.complete, "items": [item.to_dict() for item in self.items], "evidence": [item.to_dict() for item in self.evidence], "remaining_criteria": list(self.remaining_criteria), "blocking_conditions": list(self.blocking_conditions), "warnings": list(self.warnings)}


class TaskCompletionVerifier:
    """Deterministic evaluator; it performs no I/O, model call, or mutation."""

    def verify(self, request: TaskCompletionRequest) -> TaskCompletionResult:
        criteria = request.criteria or self._criteria_from_plan(request.plan, request.tool_results)
        items = [self._evaluate_criterion(criterion, request) for criterion in criteria]
        items.extend(self._global_items(request))
        evidence = tuple(request.evidence) + tuple(ev for item in items for ev in item.evidence)
        required = [item for item in items if item.criterion.required]
        unsatisfied = [item for item in required if item.status not in {CriterionStatus.SATISFIED, CriterionStatus.NOT_APPLICABLE}]
        blocking = [item.reason or item.criterion.description for item in unsatisfied if item.status is CriterionStatus.BLOCKED]
        remaining = [item.criterion.criterion_id for item in unsatisfied if item.status in {CriterionStatus.UNSATISFIED, CriterionStatus.UNVERIFIED}]
        if request.budget and request.budget.exhausted_dimensions:
            return self._result(CompletionDecision.BLOCKED, CompletionStatus.BLOCKED, CompletionConfidence.HIGH, items, evidence, remaining, ("execution budget is exhausted",), ("budget remains authoritative",))
        if request.recovery and request.recovery.decision.status in {RecoveryStatus.BLOCKED, RecoveryStatus.USER_INTERVENTION_REQUIRED}:
            return self._result(CompletionDecision.BLOCKED, CompletionStatus.BLOCKED, CompletionConfidence.HIGH, items, evidence, remaining, request.recovery.decision.blocking_conditions or ("recovery is blocked",), ())
        if request.recovery and request.recovery.decision.status is RecoveryStatus.FAILED:
            return self._result(CompletionDecision.FAILED, CompletionStatus.FAILED, CompletionConfidence.HIGH, items, evidence, remaining, ("recovery could not continue",), ())
        if not request.evidence_complete:
            return self._result(CompletionDecision.VERIFICATION_UNAVAILABLE, CompletionStatus.VERIFICATION_UNAVAILABLE, CompletionConfidence.LOW, items, evidence, remaining, ("completion evidence is incomplete or truncated",), ())
        if any(item.status is CriterionStatus.BLOCKED for item in required):
            return self._result(CompletionDecision.BLOCKED, CompletionStatus.BLOCKED, CompletionConfidence.HIGH, items, evidence, remaining, tuple(blocking) or ("required criterion is blocked",), ())
        if any(item.status is CriterionStatus.UNSATISFIED for item in required):
            return self._result(CompletionDecision.CONTINUE, CompletionStatus.INCOMPLETE, CompletionConfidence.MEDIUM, items, evidence, remaining, (), ("required completion evidence remains",))
        if any(item.status is CriterionStatus.UNVERIFIED for item in required):
            return self._result(CompletionDecision.VERIFICATION_UNAVAILABLE, CompletionStatus.VERIFICATION_UNAVAILABLE, CompletionConfidence.LOW, items, evidence, remaining, (), ("required verification evidence is unavailable",))
        confidence = CompletionConfidence.HIGH if all(item.evidence and item.evidence[0].strength in {EvidenceStrength.DIRECT, EvidenceStrength.STRONG} for item in required) else CompletionConfidence.MEDIUM
        return self._result(CompletionDecision.COMPLETE, CompletionStatus.COMPLETE, confidence, items, evidence, (), (), ())

    def _criteria_from_plan(self, plan: ExecutionPlan | None, results: Sequence[ToolResult]) -> tuple[TaskCompletionCriterion, ...]:
        if plan is None:
            return (TaskCompletionCriterion("evidence", "The requested task has sufficient direct evidence."),)
        criteria = [TaskCompletionCriterion(f"plan:{step.step_id}", f"Plan step completed: {step.title}", True, "plan_step", step.step_id) for step in plan.steps]
        verification_requested = bool(plan.verification_strategy) or any("test" in text.casefold() or "verif" in text.casefold() for text in plan.verification_strategy + plan.expected_changes)
        if verification_requested:
            criteria.append(TaskCompletionCriterion("verification", "Plan-required verification evidence is satisfied.", True, "verification"))
        elif any(name in {"write_file", "edit_file", "delete_file", "run_tests", "run_command", "run_command_with_policy", "run_application"} for name in (result.tool_name for result in results)):
            criteria.append(TaskCompletionCriterion("verification", "Mutation or execution result has required verification evidence.", True, "verification"))
        return tuple(criteria)

    def _evaluate_criterion(self, criterion: TaskCompletionCriterion, request: TaskCompletionRequest) -> TaskCompletionItem:
        if criterion.kind == "plan_step":
            if criterion.target in request.completed_step_ids:
                return TaskCompletionItem(criterion, CriterionStatus.SATISFIED, (TaskCompletionEvidence("plan", "Required plan step is recorded complete.", EvidenceStrength.DIRECT),), "plan step completed")
            if criterion.target in request.skipped_step_ids:
                return TaskCompletionItem(criterion, CriterionStatus.UNSATISFIED, (), "required plan step was skipped", True)
            return TaskCompletionItem(criterion, CriterionStatus.UNSATISFIED, (), "required plan step remains incomplete", True)
        if criterion.kind == "verification":
            state = request.verification.state
            if state is VerificationState.PASSED:
                return TaskCompletionItem(criterion, CriterionStatus.SATISFIED, (TaskCompletionEvidence(request.verification.source or "verification", request.verification.message or "verification passed", EvidenceStrength.DIRECT),), "verification passed")
            if state is VerificationState.UNAVAILABLE:
                return TaskCompletionItem(criterion, CriterionStatus.UNVERIFIED, (), "verification is unavailable", True)
            if state is VerificationState.FAILED:
                return TaskCompletionItem(criterion, CriterionStatus.UNSATISFIED, (), request.verification.message or "verification failed", True)
            return TaskCompletionItem(criterion, CriterionStatus.UNVERIFIED, (), "verification is required but has not passed", True)
        direct = tuple(item for item in request.evidence if item.strength in {EvidenceStrength.DIRECT, EvidenceStrength.STRONG})
        if direct:
            return TaskCompletionItem(criterion, CriterionStatus.SATISFIED, direct, "direct task evidence observed")
        return TaskCompletionItem(criterion, CriterionStatus.UNVERIFIED, (), "no direct evidence satisfies this criterion")

    def _global_items(self, request: TaskCompletionRequest) -> list[TaskCompletionItem]:
        items: list[TaskCompletionItem] = []
        failures = [result for result in request.tool_results if not result.success]
        if failures and not (request.recovery and request.recovery.decision.status is RecoveryStatus.CONTINUE):
            items.append(TaskCompletionItem(TaskCompletionCriterion("errors", "No unresolved required tool errors remain.", True, "errors"), CriterionStatus.BLOCKED, (), failures[-1].message or "a required tool failed", True))
        if request.critical_unexpected_modifications:
            items.append(TaskCompletionItem(TaskCompletionCriterion("unexpected", "No critical unexpected modifications remain.", True, "unexpected_changes"), CriterionStatus.BLOCKED, (), "critical unexpected modification observed", False, True))
        elif request.unexpected_modifications:
            items.append(TaskCompletionItem(TaskCompletionCriterion("unexpected", "Unexpected modifications are explicitly surfaced.", True, "unexpected_changes"), CriterionStatus.UNVERIFIED, tuple(TaskCompletionEvidence("git", item, EvidenceStrength.INDIRECT) for item in request.unexpected_modifications), "unexpected modification requires review", True))
        if request.plan and request.plan.completeness.name != "COMPLETE":
            items.append(TaskCompletionItem(TaskCompletionCriterion("plan_completeness", "The execution plan is complete enough to verify.", True, "plan"), CriterionStatus.UNVERIFIED, (), "plan is incomplete or requires clarification", False))
        if request.regression_required:
            if request.regression_protection is None:
                items.append(TaskCompletionItem(TaskCompletionCriterion("regression", "Required post-fix regression verification is complete.", True, "regression"), CriterionStatus.UNVERIFIED, (), "required regression verification is missing", True))
            else:
                status = getattr(getattr(request.regression_protection, "status", None), "value", str(getattr(request.regression_protection, "status", "UNKNOWN")))
                comparison = getattr(request.regression_protection, "comparison", None)
                message = getattr(comparison, "message", None) or f"regression protection returned {status}"
                evidence = (TaskCompletionEvidence("regression_protection", message, EvidenceStrength.DIRECT, status),)
                if status in {"REGRESSION_FREE", "PRE_EXISTING_FAILURES_ONLY"}:
                    items.append(TaskCompletionItem(TaskCompletionCriterion("regression", "Required post-fix regression verification is complete.", True, "regression"), CriterionStatus.SATISFIED, evidence, message))
                elif status in {"REGRESSION_DETECTED", "VERIFICATION_BLOCKED", "BUDGET_EXHAUSTED", "VERIFICATION_FAILED"}:
                    items.append(TaskCompletionItem(TaskCompletionCriterion("regression", "Required post-fix regression verification is complete.", True, "regression"), CriterionStatus.BLOCKED, evidence, message, True))
                else:
                    items.append(TaskCompletionItem(TaskCompletionCriterion("regression", "Required post-fix regression verification is complete.", True, "regression"), CriterionStatus.UNVERIFIED, evidence, message, True))
        if request.final_verification_required:
            if request.final_verification is None:
                items.append(TaskCompletionItem(TaskCompletionCriterion("final_verification", "Final Verification authoritatively confirms the task.", True, "final_verification"), CriterionStatus.UNVERIFIED, (), "final verification evidence is missing", True))
            else:
                status = getattr(getattr(request.final_verification, "status", None), "value", str(getattr(request.final_verification, "status", "UNKNOWN")))
                message = getattr(request.final_verification, "final_message", None) or f"final verification returned {status}"
                evidence = (TaskCompletionEvidence("final_verification", message, EvidenceStrength.DIRECT, status),)
                if status == "VERIFIED":
                    items.append(TaskCompletionItem(TaskCompletionCriterion("final_verification", "Final Verification authoritatively confirms the task.", True, "final_verification"), CriterionStatus.SATISFIED, evidence, message))
                elif status in {"BLOCKED", "BUDGET_EXHAUSTED", "FAILED"}:
                    items.append(TaskCompletionItem(TaskCompletionCriterion("final_verification", "Final Verification authoritatively confirms the task.", True, "final_verification"), CriterionStatus.BLOCKED, evidence, message, True))
                else:
                    items.append(TaskCompletionItem(TaskCompletionCriterion("final_verification", "Final Verification authoritatively confirms the task.", True, "final_verification"), CriterionStatus.UNVERIFIED, evidence, message, True))
        return items

    @staticmethod
    def _result(decision: CompletionDecision, status: CompletionStatus, confidence: CompletionConfidence, items: Sequence[TaskCompletionItem], evidence: Sequence[TaskCompletionEvidence], remaining: Sequence[str], blocking: Sequence[str], warnings: Sequence[str]) -> TaskCompletionResult:
        return TaskCompletionResult(decision, status, confidence, tuple(items), tuple(evidence), tuple(remaining), tuple(blocking), tuple(warnings))


def verify_task_completion(request: TaskCompletionRequest) -> TaskCompletionResult:
    return TaskCompletionVerifier().verify(request)


def _bounded(value: str) -> str:
    return value if len(value) <= _MAX_TEXT else value[:_MAX_TEXT - 14] + "\n[truncated]"


__all__ = ["CompletionConfidence", "CompletionDecision", "CompletionStatus", "CriterionStatus", "EvidenceStrength", "TaskCompletionCriterion", "TaskCompletionEvidence", "TaskCompletionItem", "TaskCompletionRequest", "TaskCompletionResult", "TaskCompletionVerifier", "verify_task_completion"]
