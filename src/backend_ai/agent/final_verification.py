from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from backend_ai.agent.completion import EvidenceStrength, TaskCompletionEvidence
from backend_ai.agent.execution_budget import ExecutionBudgetSnapshot
from backend_ai.agent.models import ToolResult
from backend_ai.agent.planner import ExecutionPlan, PlanCompleteness, PlanStepStatus, PlannerTaskType
from backend_ai.agent.recovery import RecoveryResult, RecoveryStatus
from backend_ai.agent.regression_protection import RegressionStatus
from backend_ai.agent.stop_conditions import VerificationEvidence, VerificationState

_MAX_ITEMS = 64
_MAX_TEXT = 1_024


class FinalVerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    INCOMPLETE = "INCOMPLETE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class FinalCriterionStatus(str, Enum):
    SATISFIED = "SATISFIED"
    UNSATISFIED = "UNSATISFIED"
    NOT_REQUIRED = "NOT_REQUIRED"
    UNKNOWN = "UNKNOWN"
    BLOCKED = "BLOCKED"
    INCOMPLETE = "INCOMPLETE"


class FinalEvidenceStrength(str, Enum):
    DIRECT = "DIRECT"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class FinalVerificationConfig:
    """Explicit overrides for criteria that the plan cannot infer safely."""

    require_tests: bool | None = None
    require_mutation_verification: bool | None = None
    require_regression_protection: bool = False
    require_fix_chain: bool = False
    require_completion_authority: bool = False
    require_stop_authority: bool = False
    accept_pre_existing_failures: bool = False

    def __post_init__(self) -> None:
        for name in ("require_regression_protection", "require_fix_chain", "require_completion_authority", "require_stop_authority", "accept_pre_existing_failures"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("require_tests", "require_mutation_verification"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean or None")


@dataclass(frozen=True, slots=True)
class FinalVerificationCriterion:
    criterion_id: str
    description: str
    required: bool
    status: FinalCriterionStatus
    evidence: tuple[TaskCompletionEvidence, ...] = ()
    reason: str = ""
    blocking: bool = False

    def __post_init__(self) -> None:
        if not self.criterion_id.strip() or not self.description.strip():
            raise ValueError("criterion_id and description must contain text")
        object.__setattr__(self, "evidence", tuple(self.evidence[:_MAX_ITEMS]))
        object.__setattr__(self, "reason", _bounded(self.reason))

    def to_dict(self) -> dict[str, Any]:
        return {"criterion_id": self.criterion_id, "description": _bounded(self.description), "required": self.required, "status": self.status.value, "evidence": [item.to_dict() for item in self.evidence], "reason": self.reason, "blocking": self.blocking}


@dataclass(frozen=True, slots=True)
class FinalVerificationRequest:
    """Structured evidence input; this layer performs no I/O, mutation, or execution."""

    task: str
    plan: ExecutionPlan | None = None
    completed_step_ids: tuple[str, ...] = ()
    skipped_step_ids: tuple[str, ...] = ()
    blocked_step_ids: tuple[str, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    verification: VerificationEvidence = field(default_factory=VerificationEvidence.not_required)
    mutation_verification: Any | None = None
    regression_protection: Any | None = None
    regression_required: bool | None = None
    self_correction: Any | None = None
    recovery: RecoveryResult | None = None
    budget: ExecutionBudgetSnapshot | None = None
    unexpected_modifications: tuple[str, ...] = ()
    critical_unexpected_modifications: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    safety_blocked: bool = False
    policy_denied: bool = False
    evidence_complete: bool = True
    final_action_claim: str | None = None
    task_completion: Any | None = None
    stop_evaluation: Any | None = None
    config: FinalVerificationConfig = field(default_factory=FinalVerificationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must contain text")
        if not isinstance(self.verification, VerificationEvidence):
            raise TypeError("verification must be VerificationEvidence")
        for name in ("completed_step_ids", "skipped_step_ids", "blocked_step_ids", "unexpected_modifications", "critical_unexpected_modifications", "missing_capabilities"):
            object.__setattr__(self, name, tuple(getattr(self, name)[:_MAX_ITEMS]))
        if len(self.tool_results) > _MAX_ITEMS:
            raise ValueError("tool_results exceed bounded limit")
        if not isinstance(self.evidence_complete, bool) or not isinstance(self.safety_blocked, bool) or not isinstance(self.policy_denied, bool):
            raise ValueError("evidence_complete, safety_blocked, and policy_denied must be boolean")
        if self.final_action_claim is not None:
            object.__setattr__(self, "final_action_claim", _bounded(self.final_action_claim))

    def to_dict(self) -> dict[str, Any]:
        return {"task": _bounded(self.task), "plan": self.plan.to_dict() if self.plan else None, "completed_step_ids": list(self.completed_step_ids), "skipped_step_ids": list(self.skipped_step_ids), "blocked_step_ids": list(self.blocked_step_ids), "tool_results": [_safe_tool_result(item) for item in self.tool_results], "verification": self.verification.to_dict(), "has_mutation_verification": self.mutation_verification is not None, "has_regression_protection": self.regression_protection is not None, "regression_required": self.regression_required, "has_self_correction": self.self_correction is not None, "recovery": self.recovery.to_dict() if self.recovery else None, "budget": self.budget.to_dict() if self.budget else None, "unexpected_modifications": list(self.unexpected_modifications), "critical_unexpected_modifications": list(self.critical_unexpected_modifications), "missing_capabilities": list(self.missing_capabilities), "safety_blocked": self.safety_blocked, "policy_denied": self.policy_denied, "evidence_complete": self.evidence_complete, "final_action_claim": _bounded(self.final_action_claim or "") if self.final_action_claim is not None else None, "has_task_completion": self.task_completion is not None, "has_stop_evaluation": self.stop_evaluation is not None, "config": {"require_tests": self.config.require_tests, "require_mutation_verification": self.config.require_mutation_verification, "require_regression_protection": self.config.require_regression_protection, "require_fix_chain": self.config.require_fix_chain, "require_completion_authority": self.config.require_completion_authority, "require_stop_authority": self.config.require_stop_authority, "accept_pre_existing_failures": self.config.accept_pre_existing_failures}}


@dataclass(frozen=True, slots=True)
class FinalVerificationResult:
    status: FinalVerificationStatus
    criteria: tuple[FinalVerificationCriterion, ...]
    evidence: tuple[TaskCompletionEvidence, ...]
    evidence_complete: bool
    evidence_sources: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    conflicting_evidence: tuple[str, ...]
    truncated_evidence: tuple[str, ...]
    verification_warnings: tuple[str, ...]
    confidence: str
    final_message: str

    def __post_init__(self) -> None:
        for name in ("criteria", "evidence", "evidence_sources", "missing_evidence", "conflicting_evidence", "truncated_evidence", "verification_warnings"):
            object.__setattr__(self, name, tuple(getattr(self, name)[:_MAX_ITEMS]))
        object.__setattr__(self, "final_message", _bounded(self.final_message))

    @property
    def verified(self) -> bool:
        return self.status is FinalVerificationStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "verified": self.verified, "criteria": [item.to_dict() for item in self.criteria], "evidence": [item.to_dict() for item in self.evidence], "evidence_complete": self.evidence_complete, "evidence_sources": list(self.evidence_sources), "missing_evidence": list(self.missing_evidence), "conflicting_evidence": list(self.conflicting_evidence), "truncated_evidence": list(self.truncated_evidence), "verification_warnings": list(self.verification_warnings), "confidence": self.confidence, "final_message": self.final_message}


class FinalVerification:
    """Authoritative evidence gate that never executes, mutates, or trusts model prose."""

    def verify(self, request: FinalVerificationRequest) -> FinalVerificationResult:
        if not isinstance(request, FinalVerificationRequest):
            raise TypeError("request must be FinalVerificationRequest")
        criteria: list[FinalVerificationCriterion] = []
        criteria.append(self._plan(request))
        criteria.append(self._mutation(request))
        criteria.append(self._tests(request))
        criteria.append(self._regression(request))
        criteria.append(self._recovery(request))
        criteria.append(self._budget(request))
        criteria.append(self._safety(request))
        criteria.append(self._execution(request))
        criteria.append(self._unexpected(request))
        criteria.append(self._evidence(request))
        criteria.append(self._observations(request))
        criteria.append(self._fix_chain(request))
        if request.config.require_completion_authority:
            criteria.append(self._authority("completion_authority", "TaskCompletionVerifier agrees with final verification.", request.task_completion))
        if request.config.require_stop_authority:
            criteria.append(self._authority("stop_authority", "StopConditionEvaluator permits terminal DONE.", request.stop_evaluation))
        evidence = tuple(item for criterion in criteria for item in criterion.evidence)
        sources = tuple(dict.fromkeys(item.source for item in evidence))
        missing = tuple(criterion.criterion_id for criterion in criteria if criterion.required and criterion.status in {FinalCriterionStatus.UNKNOWN, FinalCriterionStatus.INCOMPLETE})
        truncated = self._truncated_sources(request)
        conflicts = self._conflicts(request, criteria)
        warnings = tuple(dict.fromkeys(item.reason for item in criteria if item.reason and item.status in {FinalCriterionStatus.UNKNOWN, FinalCriterionStatus.INCOMPLETE}))
        complete = request.evidence_complete and not missing and not truncated
        status = self._status(criteria, request, complete, conflicts)
        return FinalVerificationResult(status, tuple(criteria), evidence, complete, sources, missing, conflicts, truncated, warnings, self._confidence(status, complete, conflicts), _message(status))

    @staticmethod
    def _plan(request: FinalVerificationRequest) -> FinalVerificationCriterion:
        if request.plan is None:
            return _criterion("plan", "Required plan steps are complete.", False, FinalCriterionStatus.NOT_REQUIRED, "no execution plan was supplied")
        plan = request.plan
        required_ids = {step.step_id for step in plan.steps}
        completed = set(request.completed_step_ids)
        missing = sorted(required_ids - completed - set(request.skipped_step_ids))
        blocked = sorted(set(request.blocked_step_ids) & required_ids)
        dependency_gap = [step.step_id for step in plan.steps if step.step_id in completed and any(dep not in completed for dep in step.dependencies)]
        clarification = [step.step_id for step in plan.steps if step.status is PlanStepStatus.NEEDS_CLARIFICATION]
        if plan.completeness is not PlanCompleteness.COMPLETE or clarification:
            return _criterion("plan", "All required plan steps and dependencies are complete.", True, FinalCriterionStatus.INCOMPLETE, "plan is partial or requires clarification")
        if blocked:
            return _criterion("plan", "All required plan steps and dependencies are complete.", True, FinalCriterionStatus.BLOCKED, "blocked plan steps remain", blocking=True)
        skipped = sorted(set(request.skipped_step_ids) & required_ids)
        if missing or dependency_gap or skipped:
            reason = "required plan steps remain or were skipped: " + ", ".join((missing + dependency_gap + skipped)[:8])
            return _criterion("plan", "All required plan steps and dependencies are complete.", True, FinalCriterionStatus.INCOMPLETE, reason)
        return _criterion("plan", "All required plan steps and dependencies are complete.", True, FinalCriterionStatus.SATISFIED, "execution plan is complete", evidence=("plan", "all required plan steps are recorded complete"))

    def _mutation(self, request: FinalVerificationRequest) -> FinalVerificationCriterion:
        required = request.config.require_mutation_verification if request.config.require_mutation_verification is not None else self._plan_requires_mutation(request.plan)
        if not required:
            return _criterion("mutation", "Expected mutation and post-state verification are authoritative.", False, FinalCriterionStatus.NOT_REQUIRED, "task does not require mutation verification")
        source = request.mutation_verification or self._find_structured_verification(request.tool_results)
        if source is None:
            return _criterion("mutation", "Expected mutation and post-state verification are authoritative.", True, FinalCriterionStatus.UNKNOWN, "mutation verification evidence is missing")
        success = _bool(source, "success")
        complete = _bool(source, "complete")
        unexpected = _sequence(source, "unexpected_changes")
        if success and complete and not unexpected:
            return _criterion("mutation", "Expected mutation and post-state verification are authoritative.", True, FinalCriterionStatus.SATISFIED, "structured mutation verification passed", evidence=("modification_verification", "expected post-state and bounded scope verification passed"))
        if not complete or _bool(source, "truncated"):
            return _criterion("mutation", "Expected mutation and post-state verification are authoritative.", True, FinalCriterionStatus.INCOMPLETE, "mutation verification is incomplete or truncated")
        return _criterion("mutation", "Expected mutation and post-state verification are authoritative.", True, FinalCriterionStatus.UNSATISFIED, "mutation verification did not pass")

    def _tests(self, request: FinalVerificationRequest) -> FinalVerificationCriterion:
        required = request.config.require_tests if request.config.require_tests is not None else self._plan_requires_tests(request.plan)
        if not required:
            return _criterion("tests", "Required targeted tests or checks have authoritative PASS evidence.", False, FinalCriterionStatus.NOT_REQUIRED, "plan does not require test execution")
        verification = request.verification
        source = (verification.source or "").casefold()
        if verification.state is VerificationState.PASSED and verification.complete and (source in {"parse_test_result", "run_tests", "tests", "test_result"} or "test" in source):
            return _criterion("tests", "Required targeted tests or checks have authoritative PASS evidence.", True, FinalCriterionStatus.SATISFIED, "structured test evidence reported PASS", evidence=(verification.source or "test_evidence", verification.message or "targeted tests passed"))
        if verification.state is VerificationState.FAILED:
            return _criterion("tests", "Required targeted tests or checks have authoritative PASS evidence.", True, FinalCriterionStatus.UNSATISFIED, verification.message or "required tests failed")
        if verification.state in {VerificationState.PENDING, VerificationState.INCOMPLETE} or not verification.complete:
            return _criterion("tests", "Required targeted tests or checks have authoritative PASS evidence.", True, FinalCriterionStatus.INCOMPLETE, verification.message or "test evidence is incomplete")
        return _criterion("tests", "Required targeted tests or checks have authoritative PASS evidence.", True, FinalCriterionStatus.UNKNOWN, "required test PASS evidence is missing")

    def _regression(self, request: FinalVerificationRequest) -> FinalVerificationCriterion:
        required = request.config.require_regression_protection if request.regression_required is None else request.regression_required
        if not required:
            return _criterion("regression", "Required regression protection is complete and regression-free.", False, FinalCriterionStatus.NOT_REQUIRED, "regression protection is not required by the supplied request")
        result = request.regression_protection
        if result is None:
            return _criterion("regression", "Required regression protection is complete and regression-free.", True, FinalCriterionStatus.INCOMPLETE, "regression evidence is missing")
        status = _status_value(result)
        if status == RegressionStatus.REGRESSION_FREE.value:
            return _criterion("regression", "Required regression protection is complete and regression-free.", True, FinalCriterionStatus.SATISFIED, "regression comparison reported REGRESSION_FREE", evidence=("regression_protection", "no newly introduced failures"))
        if status == RegressionStatus.PRE_EXISTING_FAILURES_ONLY.value and request.config.accept_pre_existing_failures:
            return _criterion("regression", "Required regression protection is complete and regression-free.", True, FinalCriterionStatus.SATISFIED, "only explicitly accepted pre-existing failures remain", evidence=("regression_protection", "pre-existing failures were explicitly accepted"))
        if status in {RegressionStatus.VERIFICATION_BLOCKED.value}:
            return _criterion("regression", "Required regression protection is complete and regression-free.", True, FinalCriterionStatus.BLOCKED, f"regression protection returned {status}", blocking=True)
        if status in {RegressionStatus.BUDGET_EXHAUSTED.value}:
            return _criterion("regression", "Required regression protection is complete and regression-free.", True, FinalCriterionStatus.BLOCKED, "regression protection exhausted the shared budget", blocking=True)
        if status in {RegressionStatus.VERIFICATION_FAILED.value, RegressionStatus.REGRESSION_DETECTED.value}:
            return _criterion("regression", "Required regression protection is complete and regression-free.", True, FinalCriterionStatus.UNSATISFIED, f"regression protection returned {status}")
        return _criterion("regression", "Required regression protection is complete and regression-free.", True, FinalCriterionStatus.INCOMPLETE, f"regression protection returned {status or 'UNKNOWN'}")

    @staticmethod
    def _recovery(request: FinalVerificationRequest) -> FinalVerificationCriterion:
        if request.recovery is None:
            return _criterion("recovery", "No unresolved recovery, retry, or user-intervention state remains.", False, FinalCriterionStatus.NOT_REQUIRED, "no recovery result was supplied")
        status = request.recovery.decision.status
        if status is RecoveryStatus.CONTINUE:
            if request.tool_results and request.tool_results[-1].success:
                return _criterion("recovery", "No unresolved recovery, retry, or user-intervention state remains.", True, FinalCriterionStatus.SATISFIED, "recovery continuation was followed by successful structured evidence", evidence=("recovery", "recovery action completed and the next tool observation succeeded"))
            return _criterion("recovery", "No unresolved recovery, retry, or user-intervention state remains.", True, FinalCriterionStatus.INCOMPLETE, "recovery requested continuation")
        if status in {RecoveryStatus.BLOCKED, RecoveryStatus.USER_INTERVENTION_REQUIRED}:
            return _criterion("recovery", "No unresolved recovery, retry, or user-intervention state remains.", True, FinalCriterionStatus.BLOCKED, "recovery remains blocked or requires user intervention", blocking=True)
        if status is RecoveryStatus.FAILED:
            return _criterion("recovery", "No unresolved recovery, retry, or user-intervention state remains.", True, FinalCriterionStatus.UNSATISFIED, "recovery failed")
        return _criterion("recovery", "No unresolved recovery, retry, or user-intervention state remains.", True, FinalCriterionStatus.SATISFIED, "recovery state is clean", evidence=("recovery", "no unresolved recovery state"))

    @staticmethod
    def _budget(request: FinalVerificationRequest) -> FinalVerificationCriterion:
        if request.budget is None:
            return _criterion("budget", "Shared execution budget was not exhausted before required verification.", False, FinalCriterionStatus.NOT_REQUIRED, "no budget snapshot was supplied")
        if request.budget.exhausted_dimensions:
            return _criterion("budget", "Shared execution budget was not exhausted before required verification.", True, FinalCriterionStatus.BLOCKED, "budget exhausted: " + ", ".join(request.budget.exhausted_dimensions[:8]), blocking=True)
        if not request.budget.usage_complete:
            return _criterion("budget", "Shared execution budget was not exhausted before required verification.", True, FinalCriterionStatus.INCOMPLETE, "budget usage evidence is incomplete")
        return _criterion("budget", "Shared execution budget was not exhausted before required verification.", True, FinalCriterionStatus.SATISFIED, "shared budget snapshot is available and not exhausted", evidence=("execution_budget", "no budget dimension is exhausted"))

    @staticmethod
    def _safety(request: FinalVerificationRequest) -> FinalVerificationCriterion:
        blocked = request.safety_blocked or request.policy_denied or bool(request.missing_capabilities) or any(_is_safety_error(item.error_code) for item in request.tool_results if not item.success)
        if blocked:
            return _criterion("safety", "No safety, policy, capability, or project-boundary block remains.", True, FinalCriterionStatus.BLOCKED, "safety or policy evidence blocks completion", blocking=True)
        return _criterion("safety", "No safety, policy, capability, or project-boundary block remains.", True, FinalCriterionStatus.SATISFIED, "no supplied safety or policy block", evidence=("safety_policy", "no safety or policy violation was observed"))

    @staticmethod
    def _execution(request: FinalVerificationRequest) -> FinalVerificationCriterion:
        failures = tuple(item for item in request.tool_results if not item.success)
        if not failures:
            return _criterion("execution", "No unresolved required tool execution failure remains.", True, FinalCriterionStatus.SATISFIED, "all supplied tool executions succeeded", evidence=("tool_results", "no unresolved tool execution failure"))
        if request.recovery is not None and request.recovery.decision.status is RecoveryStatus.CONTINUE:
            if request.tool_results and request.tool_results[-1].success:
                return _criterion("execution", "No unresolved required tool execution failure remains.", True, FinalCriterionStatus.SATISFIED, "recovery was followed by a successful structured tool observation", evidence=("tool_results", "latest post-recovery tool result succeeded"))
            return _criterion("execution", "No unresolved required tool execution failure remains.", True, FinalCriterionStatus.INCOMPLETE, "a failed tool result remains under recovery")
        if any("INTERNAL" in (item.error_code or "").upper() or "INFRASTRUCTURE" in (item.error_code or "").upper() for item in failures):
            return _criterion("execution", "No unresolved required tool execution failure remains.", True, FinalCriterionStatus.UNSATISFIED, "internal execution error: " + (failures[-1].message or "a required tool execution failed"))
        return _criterion("execution", "No unresolved required tool execution failure remains.", True, FinalCriterionStatus.UNSATISFIED, failures[-1].message or "a required tool execution failed")

    @staticmethod
    def _unexpected(request: FinalVerificationRequest) -> FinalVerificationCriterion:
        if request.critical_unexpected_modifications:
            return _criterion("unexpected_modifications", "No unauthorized or unexpected modifications remain.", True, FinalCriterionStatus.BLOCKED, "critical unexpected modifications: " + ", ".join(request.critical_unexpected_modifications[:8]), blocking=True)
        if request.unexpected_modifications:
            return _criterion("unexpected_modifications", "No unauthorized or unexpected modifications remain.", True, FinalCriterionStatus.UNSATISFIED, "unexpected modifications require review: " + ", ".join(request.unexpected_modifications[:8]))
        return _criterion("unexpected_modifications", "No unauthorized or unexpected modifications remain.", True, FinalCriterionStatus.SATISFIED, "no unexpected modifications were supplied", evidence=("scope", "expected change scope remained clean"))

    @staticmethod
    def _evidence(request: FinalVerificationRequest) -> FinalVerificationCriterion:
        if not request.evidence_complete:
            return _criterion("evidence", "All required evidence is complete, bounded, and non-conflicting.", True, FinalCriterionStatus.UNKNOWN, "completion evidence is incomplete or truncated")
        if request.verification.state is not VerificationState.NOT_REQUIRED and not request.verification.complete:
            return _criterion("evidence", "All required evidence is complete, bounded, and non-conflicting.", True, FinalCriterionStatus.INCOMPLETE, "verification evidence is marked incomplete")
        return _criterion("evidence", "All required evidence is complete, bounded, and non-conflicting.", True, FinalCriterionStatus.SATISFIED, "supplied evidence is marked complete", evidence=("evidence_record", "bounded evidence record is complete"))

    @staticmethod
    def _observations(request: FinalVerificationRequest) -> FinalVerificationCriterion:
        has_structured_observation = bool(request.tool_results or request.verification.state is not VerificationState.NOT_REQUIRED or request.mutation_verification is not None or request.regression_protection is not None or request.self_correction is not None or request.task_completion is not None)
        if has_structured_observation:
            return _criterion("observations", "At least one authoritative structured observation supports the final claim.", True, FinalCriterionStatus.SATISFIED, "structured execution evidence is present", evidence=("structured_evidence", "authoritative structured observation is present"))
        return _criterion("observations", "At least one authoritative structured observation supports the final claim.", True, FinalCriterionStatus.UNKNOWN, "only a final claim was supplied without structured observation")

    @staticmethod
    def _fix_chain(request: FinalVerificationRequest) -> FinalVerificationCriterion:
        if not request.config.require_fix_chain:
            return _criterion("fix_chain", "Required failure-analysis, RCA, fix, mutation, and retest chain is complete.", False, FinalCriterionStatus.NOT_REQUIRED, "fix-chain verification was not requested")
        correction = request.self_correction
        if correction is None:
            return _criterion("fix_chain", "Required failure-analysis, RCA, fix, mutation, and retest chain is complete.", True, FinalCriterionStatus.INCOMPLETE, "self-correction evidence is missing")
        status = _status_value(correction)
        final_test = getattr(correction, "final_parsed_result", None)
        final_fix = getattr(correction, "final_fix_result", None)
        failure = getattr(correction, "final_failure_analysis", None)
        rca = getattr(correction, "final_root_cause", None)
        if status in {"EXHAUSTED", "REPEATED_FAILURE", "NO_PROGRESS", "BLOCKED", "USER_INTERVENTION_REQUIRED", "BUDGET_EXHAUSTED"}:
            return _criterion("fix_chain", "Required failure-analysis, RCA, fix, mutation, and retest chain is complete.", True, FinalCriterionStatus.UNSATISFIED, f"self-correction ended in {status}")
        if failure is None or rca is None or final_fix is None:
            return _criterion("fix_chain", "Required failure-analysis, RCA, fix, mutation, and retest chain is complete.", True, FinalCriterionStatus.INCOMPLETE, "failure analysis, RCA, or verified fix evidence is missing")
        if not bool(getattr(final_fix, "verified", False)) or getattr(getattr(final_test, "overall_status", None), "value", None) != "PASS":
            return _criterion("fix_chain", "Required failure-analysis, RCA, fix, mutation, and retest chain is complete.", True, FinalCriterionStatus.UNSATISFIED, "fix or final retest was not verified")
        return _criterion("fix_chain", "Required failure-analysis, RCA, fix, mutation, and retest chain is complete.", True, FinalCriterionStatus.SATISFIED, "failure analysis, RCA, fix, and final retest evidence are present", evidence=("self_correction", "bounded failure-to-fix chain completed"))

    @staticmethod
    def _authority(criterion_id: str, description: str, value: Any) -> FinalVerificationCriterion:
        if value is None:
            return _criterion(criterion_id, description, True, FinalCriterionStatus.UNKNOWN, "authority result is missing")
        decision = getattr(getattr(value, "decision", None), "value", getattr(value, "decision", None))
        if criterion_id == "completion_authority":
            ok = bool(getattr(value, "complete", False)) or decision == "COMPLETE"
        else:
            ok = decision == "DONE"
        return _criterion(criterion_id, description, True, FinalCriterionStatus.SATISFIED if ok else FinalCriterionStatus.UNSATISFIED, "authoritative result agrees" if ok else "authoritative result does not allow terminal completion")

    @staticmethod
    def _plan_requires_mutation(plan: ExecutionPlan | None) -> bool:
        return plan is not None and plan.task_type is not PlannerTaskType.INVESTIGATION

    @staticmethod
    def _plan_requires_tests(plan: ExecutionPlan | None) -> bool:
        if plan is None:
            return False
        if plan.task_type in {PlannerTaskType.DOCUMENTATION_CHANGE, PlannerTaskType.INVESTIGATION}:
            return False
        return bool(plan.verification_strategy) or any("test" in item.casefold() or "check" in item.casefold() for item in plan.expected_changes)

    @staticmethod
    def _find_structured_verification(results: Sequence[ToolResult]) -> Any | None:
        for result in reversed(results):
            if not result.success:
                continue
            if result.tool_name == "verify_modification":
                return result.data
            data = result.data
            nested = getattr(data, "verification", None)
            if nested is not None:
                return nested
            if isinstance(data, Mapping) and data.get("verification") is not None:
                return data["verification"]
        return None

    @staticmethod
    def _truncated_sources(request: FinalVerificationRequest) -> tuple[str, ...]:
        sources: list[str] = []
        if any(result.truncated for result in request.tool_results):
            sources.append("tool_results")
        mutation = request.mutation_verification
        if mutation is not None and (_bool(mutation, "truncated") or not _bool(mutation, "complete")):
            sources.append("mutation_verification")
        if request.verification.state is not VerificationState.NOT_REQUIRED and not request.verification.complete:
            sources.append("verification")
        return tuple(dict.fromkeys(sources))

    @staticmethod
    def _conflicts(request: FinalVerificationRequest, criteria: Sequence[FinalVerificationCriterion]) -> tuple[str, ...]:
        conflicts: list[str] = []
        if request.final_action_claim and any(item.status in {FinalCriterionStatus.UNSATISFIED, FinalCriterionStatus.BLOCKED, FinalCriterionStatus.INCOMPLETE, FinalCriterionStatus.UNKNOWN} for item in criteria if item.required):
            conflicts.append("model FINAL claim conflicts with negative or missing authoritative evidence")
        if request.task_completion is not None and bool(getattr(request.task_completion, "complete", False)) and any(item.status in {FinalCriterionStatus.UNSATISFIED, FinalCriterionStatus.BLOCKED, FinalCriterionStatus.INCOMPLETE, FinalCriterionStatus.UNKNOWN} for item in criteria if item.required):
            conflicts.append("TaskCompletionVerifier completion claim conflicts with final verification evidence")
        return tuple(conflicts)

    @staticmethod
    def _status(criteria: Sequence[FinalVerificationCriterion], request: FinalVerificationRequest, complete: bool, conflicts: Sequence[str]) -> FinalVerificationStatus:
        if request.budget is not None and request.budget.exhausted_dimensions:
            return FinalVerificationStatus.BUDGET_EXHAUSTED
        if any(item.status is FinalCriterionStatus.BLOCKED for item in criteria if item.required):
            return FinalVerificationStatus.BLOCKED
        if any(item.status is FinalCriterionStatus.UNSATISFIED and item.criterion_id == "execution" and item.reason.startswith("internal execution error") for item in criteria if item.required):
            return FinalVerificationStatus.FAILED
        if any(item.status is FinalCriterionStatus.UNSATISFIED for item in criteria if item.required):
            return FinalVerificationStatus.NOT_VERIFIED
        if any(item.status is FinalCriterionStatus.INCOMPLETE for item in criteria if item.required):
            return FinalVerificationStatus.INCOMPLETE
        if any(item.status is FinalCriterionStatus.UNKNOWN for item in criteria if item.required) or not complete:
            return FinalVerificationStatus.INSUFFICIENT_EVIDENCE
        if conflicts:
            return FinalVerificationStatus.NOT_VERIFIED
        return FinalVerificationStatus.VERIFIED

    @staticmethod
    def _confidence(status: FinalVerificationStatus, complete: bool, conflicts: Sequence[str]) -> str:
        if status is FinalVerificationStatus.VERIFIED and complete and not conflicts:
            return "HIGH"
        if status in {FinalVerificationStatus.BLOCKED, FinalVerificationStatus.BUDGET_EXHAUSTED}:
            return "HIGH"
        if status in {FinalVerificationStatus.NOT_VERIFIED, FinalVerificationStatus.FAILED}:
            return "MEDIUM"
        return "LOW"


def verify_final_state(request: FinalVerificationRequest) -> FinalVerificationResult:
    return FinalVerification().verify(request)


def _criterion(criterion_id: str, description: str, required: bool, status: FinalCriterionStatus, reason: str, *, blocking: bool = False, evidence: tuple[str, str] | None = None) -> FinalVerificationCriterion:
    records = () if evidence is None else (TaskCompletionEvidence(evidence[0], evidence[1], EvidenceStrength.DIRECT),)
    return FinalVerificationCriterion(criterion_id, description, required, status, records, reason, blocking)


def _status_value(value: Any) -> str:
    raw = getattr(value, "status", value)
    return str(getattr(raw, "value", raw or ""))


def _bool(value: Any, name: str) -> bool:
    raw = getattr(value, name, None)
    if raw is None and isinstance(value, Mapping):
        raw = value.get(name)
    return bool(raw)


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    raw = getattr(value, name, None)
    if raw is None and isinstance(value, Mapping):
        raw = value.get(name)
    return tuple(raw or ())


def _is_safety_error(code: str | None) -> bool:
    normalized = (code or "").upper()
    return any(marker in normalized for marker in ("POLICY", "DENIED", "NOT_ALLOWED", "PATH_OUTSIDE_ROOT", "SAFETY", "PERMISSION"))


def _safe_tool_result(result: ToolResult) -> dict[str, Any]:
    return {"call_id": result.call_id, "tool_name": result.tool_name, "success": result.success, "error_code": result.error_code, "message": _bounded(result.message or ""), "truncated": result.truncated, "data": "[bounded]" if result.data is not None else None}


def _bounded(value: str, limit: int = _MAX_TEXT) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 14)] + "\n[truncated]"


def _message(status: FinalVerificationStatus) -> str:
    return {FinalVerificationStatus.VERIFIED: "Task completed and verified.", FinalVerificationStatus.NOT_VERIFIED: "Task is not fully verified.", FinalVerificationStatus.BLOCKED: "Task verification is blocked.", FinalVerificationStatus.INCOMPLETE: "Task is incomplete.", FinalVerificationStatus.FAILED: "Task verification failed.", FinalVerificationStatus.INSUFFICIENT_EVIDENCE: "Task cannot be verified with the available evidence.", FinalVerificationStatus.BUDGET_EXHAUSTED: "Task verification stopped because the execution budget is exhausted."}[status]


__all__ = ["FinalCriterionStatus", "FinalEvidenceStrength", "FinalVerification", "FinalVerificationConfig", "FinalVerificationCriterion", "FinalVerificationRequest", "FinalVerificationResult", "FinalVerificationStatus", "verify_final_state"]
