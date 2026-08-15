"""Explicit, evidence-driven one-attempt automatic fixing for Phase 7.4."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Sequence

from backend_ai.agent.execution_budget import (
    BudgetDecision,
    BudgetDimension,
    ExecutionBudget,
    ExecutionBudgetLedger,
    ExecutionBudgetSnapshot,
)
from backend_ai.agent.root_cause_analysis import (
    CausalStatus,
    RootCauseAnalysis,
    RootCauseAnalysisStatus,
    RootCauseConfidence,
)
from backend_ai.tools.modification_transaction import (
    ModificationOperation,
    ModificationTransaction,
    ModificationTransactionResult,
)
from backend_ai.tools.safe_editing import SafeEditPolicy


class FixStatus(str, Enum):
    PROPOSED = "PROPOSED"
    ACCEPTED = "ACCEPTED"
    MUTATION_SUCCEEDED = "MUTATION_SUCCEEDED"
    FIX_VERIFIED = "FIX_VERIFIED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    NO_SAFE_FIX = "NO_SAFE_FIX"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class FixDecisionType(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    BLOCK = "BLOCK"
    NO_SAFE_FIX = "NO_SAFE_FIX"


class FixConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class FixRiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FixFailureReason(str, Enum):
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNKNOWN_ROOT_CAUSE = "UNKNOWN_ROOT_CAUSE"
    MISSING_LOCATION = "MISSING_LOCATION"
    AMBIGUOUS_LOCATION = "AMBIGUOUS_LOCATION"
    MALFORMED_PLAN = "MALFORMED_PLAN"
    UNSUPPORTED_FIX_TYPE = "UNSUPPORTED_FIX_TYPE"
    MULTI_FILE_SCOPE = "MULTI_FILE_SCOPE"
    UNSAFE_PATH = "UNSAFE_PATH"
    SENSITIVE_PATH = "SENSITIVE_PATH"
    PROJECT_ROOT_ESCAPE = "PROJECT_ROOT_ESCAPE"
    POLICY_DENIAL = "POLICY_DENIAL"
    BUDGET_EXHAUSTION = "BUDGET_EXHAUSTION"
    TARGET_MISSING = "TARGET_MISSING"
    CONCURRENT_MODIFICATION = "CONCURRENT_MODIFICATION"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    TRANSACTION_FAILURE = "TRANSACTION_FAILURE"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    RECOVERY_UNAVAILABLE = "RECOVERY_UNAVAILABLE"


class FixChangeType(str, Enum):
    ASSERTION_LOGIC = "ASSERTION_LOGIC"
    SYNTAX = "SYNTAX"
    TYPE = "TYPE"
    IMPORT_MODULE = "IMPORT_MODULE"
    CONFIGURATION = "CONFIGURATION"
    FIXTURE = "FIXTURE"
    AUTHENTICATION_API = "AUTHENTICATION_API"
    DATABASE_CONNECTION = "DATABASE_CONNECTION"
    SMALL_IMPLEMENTATION = "SMALL_IMPLEMENTATION"


@dataclass(frozen=True, slots=True)
class AutomaticFixConfig:
    min_confidence: FixConfidence = FixConfidence.HIGH
    max_risk: FixRiskLevel = FixRiskLevel.MEDIUM
    max_files: int = 1
    max_content_bytes: int = 1_048_576
    max_evidence: int = 32
    allow_sensitive_paths: bool = False

    def __post_init__(self) -> None:
        if self.max_files != 1:
            raise ValueError("Phase 7.4 supports exactly one bounded target file")
        for name in ("max_content_bytes", "max_evidence"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_content_bytes > 4 * 1024 * 1024 or self.max_evidence > 256:
            raise ValueError("automatic-fix limit exceeds the safety ceiling")
        if not isinstance(self.allow_sensitive_paths, bool):
            raise ValueError("allow_sensitive_paths must be boolean")


@dataclass(frozen=True, slots=True)
class FixLocation:
    file_path: str | None
    line_number: int | None = None
    symbol: str | None = None
    component: str | None = None
    confidence: FixConfidence = FixConfidence.UNKNOWN
    exact: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"file_path": self.file_path, "line_number": self.line_number, "symbol": self.symbol, "component": self.component, "confidence": self.confidence.value, "exact": self.exact}


@dataclass(frozen=True, slots=True)
class FixEvidence:
    evidence_id: str
    source: str
    description: str
    strength: FixConfidence
    provenance: str
    related_failure_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, "source": self.source, "description": self.description, "strength": self.strength.value, "provenance": self.provenance, "related_failure_ids": list(self.related_failure_ids)}


@dataclass(frozen=True, slots=True)
class FixChangeSummary:
    change_type: FixChangeType
    relative_path: str
    old_size_bytes: int
    new_size_bytes: int
    changed: bool
    mutation_status: str
    verification_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"change_type": self.change_type.value, "relative_path": self.relative_path, "old_size_bytes": self.old_size_bytes, "new_size_bytes": self.new_size_bytes, "changed": self.changed, "mutation_status": self.mutation_status, "verification_status": self.verification_status}


@dataclass(frozen=True, slots=True)
class FixPlan:
    target_file: str
    location: FixLocation
    change_type: FixChangeType
    intended_change: str
    expected_post_state: str
    risk: FixRiskLevel
    confidence: FixConfidence
    affected_failure_ids: tuple[str, ...]
    hypothesis_id: str
    evidence: tuple[FixEvidence, ...]
    old_content: str
    new_content: str

    def to_dict(self) -> dict[str, Any]:
        return {"target_file": self.target_file, "location": self.location.to_dict(), "change_type": self.change_type.value, "intended_change": self.intended_change, "expected_post_state": self.expected_post_state, "risk": self.risk.value, "confidence": self.confidence.value, "affected_failure_ids": list(self.affected_failure_ids), "hypothesis_id": self.hypothesis_id, "evidence": [item.to_dict() for item in self.evidence], "has_old_content": bool(self.old_content), "has_new_content": bool(self.new_content)}


@dataclass(frozen=True, slots=True)
class FixDecision:
    decision: FixDecisionType
    status: FixStatus
    reason: FixFailureReason | None
    message: str
    plan: FixPlan | None
    evidence: tuple[FixEvidence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision.value, "status": self.status.value, "reason": self.reason.value if self.reason else None, "message": self.message, "plan": self.plan.to_dict() if self.plan else None, "evidence": [item.to_dict() for item in self.evidence]}


@dataclass(frozen=True, slots=True)
class AutomaticFixRequest:
    project_root: str
    root_cause_analysis: RootCauseAnalysis | None
    plan: FixPlan | None
    policy: SafeEditPolicy | None = None
    budget_ledger: ExecutionBudgetLedger | None = None
    config: AutomaticFixConfig = field(default_factory=AutomaticFixConfig)


@dataclass(frozen=True, slots=True)
class AutomaticFixResult:
    status: FixStatus
    decision: FixDecision
    attempted: bool
    verified: bool
    tests_rerun: bool
    retries: int
    change_summary: FixChangeSummary | None = None
    transaction: ModificationTransactionResult | None = None
    budget_decision: BudgetDecision | None = None
    execution_budget: ExecutionBudgetSnapshot | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "decision": self.decision.to_dict(), "attempted": self.attempted, "verified": self.verified, "tests_rerun": self.tests_rerun, "retries": self.retries, "change_summary": self.change_summary.to_dict() if self.change_summary else None, "transaction": self.transaction.to_dict() if self.transaction else None, "budget_decision": self.budget_decision.to_dict() if self.budget_decision else None, "execution_budget": self.execution_budget.to_dict() if self.execution_budget else None, "warnings": list(self.warnings), "errors": list(self.errors)}


class AutomaticFixPlanner:
    """Validate a structured proposal; it never turns prose into a mutation."""

    def validate(self, request: AutomaticFixRequest) -> FixDecision:
        plan = request.plan
        if not isinstance(plan, FixPlan):
            return self._reject(FixFailureReason.MALFORMED_PLAN, "A structured FixPlan is required.")
        error = _validate_plan(plan, request.config)
        if error is not None:
            return self._reject(error[0], error[1], plan)
        rca = request.root_cause_analysis
        if not isinstance(rca, RootCauseAnalysis):
            return self._reject(FixFailureReason.INSUFFICIENT_EVIDENCE, "RootCauseAnalysis evidence is required before mutation.", plan)
        if rca.status is not RootCauseAnalysisStatus.ANALYZED:
            return self._reject(FixFailureReason.INSUFFICIENT_EVIDENCE, "RootCauseAnalysis is not sufficiently conclusive for an automatic fix.", plan)
        hypotheses = {item.hypothesis_id: item for item in rca.hypotheses}
        hypothesis = hypotheses.get(plan.hypothesis_id)
        if hypothesis is None:
            return self._reject(FixFailureReason.UNKNOWN_ROOT_CAUSE, "FixPlan hypothesis_id is not present in RootCauseAnalysis.", plan)
        if hypothesis.causal_status in {CausalStatus.INSUFFICIENT_EVIDENCE, CausalStatus.REJECTED, CausalStatus.ALTERNATIVE}:
            return self._reject(FixFailureReason.INSUFFICIENT_EVIDENCE, "The selected hypothesis is not an actionable primary candidate.", plan)
        if not _confidence_allowed(hypothesis.confidence, request.config.min_confidence) or not _confidence_allowed(plan.confidence, request.config.min_confidence):
            return self._reject(FixFailureReason.LOW_CONFIDENCE, "Fix evidence does not satisfy the configured confidence threshold.", plan)
        if not request.policy or not request.policy.allow_edit:
            return self._reject(FixFailureReason.POLICY_DENIAL, "An explicit SafeEditPolicy allowing edit is required.", plan)
        return FixDecision(FixDecisionType.ACCEPT, FixStatus.ACCEPTED, None, "Structured fix plan passed evidence, scope, risk, policy, and RCA validation.", plan, plan.evidence)

    @staticmethod
    def _reject(reason: FixFailureReason, message: str, plan: FixPlan | None = None) -> FixDecision:
        return FixDecision(FixDecisionType.NO_SAFE_FIX if reason in {FixFailureReason.INSUFFICIENT_EVIDENCE, FixFailureReason.LOW_CONFIDENCE, FixFailureReason.UNKNOWN_ROOT_CAUSE, FixFailureReason.MISSING_LOCATION, FixFailureReason.AMBIGUOUS_LOCATION} else FixDecisionType.REJECT, FixStatus.NO_SAFE_FIX if reason in {FixFailureReason.INSUFFICIENT_EVIDENCE, FixFailureReason.LOW_CONFIDENCE, FixFailureReason.UNKNOWN_ROOT_CAUSE, FixFailureReason.MISSING_LOCATION, FixFailureReason.AMBIGUOUS_LOCATION} else FixStatus.REJECTED, reason, message, plan)


class AutomaticFixOrchestrator:
    """Apply at most one already-validated edit through ModificationTransaction."""

    def __init__(self, *, planner: AutomaticFixPlanner | None = None) -> None:
        self.planner = planner or AutomaticFixPlanner()

    def apply(self, request: AutomaticFixRequest) -> AutomaticFixResult:
        decision = self.planner.validate(request)
        if decision.decision is not FixDecisionType.ACCEPT or decision.plan is None:
            return AutomaticFixResult(decision.status, decision, False, False, False, 0, warnings=("No mutation was attempted because the fix plan was not accepted.",))
        ledger = request.budget_ledger or ExecutionBudgetLedger(ExecutionBudget.conservative_defaults())
        action_decision = ledger.check("automatic_fix", dimension=BudgetDimension.ACTION_STEPS)
        if not action_decision.allowed:
            return self._blocked(decision, FixFailureReason.BUDGET_EXHAUSTION, action_decision, ledger, "Action-step budget denied the fix.")
        mutation_decision = ledger.check("automatic_fix", dimension=BudgetDimension.MUTATIONS)
        if not mutation_decision.allowed:
            return self._blocked(decision, FixFailureReason.BUDGET_EXHAUSTION, mutation_decision, ledger, "Mutation budget denied the fix.")
        ledger.consume("automatic_fix", dimension=BudgetDimension.ACTION_STEPS)
        budget_decision = ledger.consume("automatic_fix", dimension=BudgetDimension.MUTATIONS)
        plan = decision.plan
        transaction = ModificationTransaction(request.project_root, ModificationOperation.edit(plan.target_file, plan.old_content, plan.new_content), policy=request.policy)
        transaction_result = transaction.execute()
        if transaction_result.status == "committed" and transaction_result.verification is not None and transaction_result.verification.success:
            summary = FixChangeSummary(plan.change_type, plan.target_file, plan.old_content.encode("utf-8").__len__(), plan.new_content.encode("utf-8").__len__(), plan.old_content != plan.new_content, transaction_result.status, "VERIFIED")
            final_decision = FixDecision(FixDecisionType.ACCEPT, FixStatus.FIX_VERIFIED, None, "One mutation attempt completed and its explicit post-state was verified; tests were not rerun.", plan, plan.evidence)
            return AutomaticFixResult(FixStatus.FIX_VERIFIED, final_decision, True, True, False, 0, summary, transaction_result, budget_decision, ledger.snapshot())
        status = FixStatus.RECOVERY_REQUIRED if transaction_result.status == "recovery_required" else FixStatus.FAILED
        reason = FixFailureReason.RECOVERY_REQUIRED if status is FixStatus.RECOVERY_REQUIRED else FixFailureReason.VERIFICATION_FAILURE if transaction_result.verification is not None else FixFailureReason.TRANSACTION_FAILURE
        final_decision = FixDecision(FixDecisionType.BLOCK, status, reason, "The single mutation attempt did not produce a verified fix; no retry or test rerun was performed.", plan, plan.evidence)
        return AutomaticFixResult(status, final_decision, True, False, False, 0, None, transaction_result, budget_decision, ledger.snapshot(), errors=transaction_result.errors)

    @staticmethod
    def _blocked(decision: FixDecision, reason: FixFailureReason, budget: BudgetDecision, ledger: ExecutionBudgetLedger, message: str) -> AutomaticFixResult:
        blocked = FixDecision(FixDecisionType.BLOCK, FixStatus.BLOCKED, reason, message, decision.plan)
        return AutomaticFixResult(FixStatus.BLOCKED, blocked, False, False, False, 0, budget_decision=budget, execution_budget=ledger.snapshot(), warnings=("Budget denial occurred before mutation start.",))


def apply_automatic_fix(request: AutomaticFixRequest) -> AutomaticFixResult:
    return AutomaticFixOrchestrator().apply(request)


def _validate_plan(plan: FixPlan, config: AutomaticFixConfig) -> tuple[FixFailureReason, str] | None:
    if not isinstance(plan.target_file, str) or not plan.target_file.strip(): return (FixFailureReason.MISSING_LOCATION, "FixPlan target_file is required.")
    if not _safe_relative_path(plan.target_file): return (FixFailureReason.UNSAFE_PATH, "FixPlan target_file must be a safe project-relative path.")
    if _sensitive_path(plan.target_file) and not config.allow_sensitive_paths: return (FixFailureReason.SENSITIVE_PATH, "Sensitive paths are never eligible for automatic fixes.")
    if plan.location.file_path != plan.target_file: return (FixFailureReason.AMBIGUOUS_LOCATION, "FixPlan location must identify the same exact target file.")
    if not plan.location.exact or plan.location.confidence is FixConfidence.UNKNOWN: return (FixFailureReason.MISSING_LOCATION, "An exact, confidence-bearing target location is required.")
    if not plan.intended_change.strip() or not plan.expected_post_state.strip(): return (FixFailureReason.MALFORMED_PLAN, "FixPlan intended_change and expected_post_state are required.")
    if not plan.hypothesis_id.strip() or not plan.affected_failure_ids: return (FixFailureReason.INSUFFICIENT_EVIDENCE, "FixPlan must link to a hypothesis and affected failures.")
    if not plan.evidence or len(plan.evidence) > config.max_evidence: return (FixFailureReason.INSUFFICIENT_EVIDENCE, "FixPlan requires bounded structured evidence.")
    if not isinstance(plan.old_content, str) or not isinstance(plan.new_content, str): return (FixFailureReason.MALFORMED_PLAN, "FixPlan content must be UTF-8 text strings.")
    if len(plan.old_content.encode("utf-8")) > config.max_content_bytes or len(plan.new_content.encode("utf-8")) > config.max_content_bytes: return (FixFailureReason.MALFORMED_PLAN, "FixPlan content exceeds the configured bounded size.")
    if _risk_rank(plan.risk) > _risk_rank(config.max_risk): return (FixFailureReason.UNSUPPORTED_FIX_TYPE, "FixPlan risk exceeds the configured automatic-fix risk limit.")
    return None


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return not value.startswith(("/", "\\")) and "\\" not in value and all(part not in {"", ".", ".."} for part in path.parts) and str(path) == value


def _sensitive_path(value: str) -> bool:
    lower = value.casefold()
    name = PurePosixPath(value).name.casefold()
    return name == ".env" or any(token in lower for token in ("secret", "credential", "private", "password")) or name.endswith((".pem", ".key", ".crt", ".p12", ".pfx"))


def _confidence_allowed(actual: FixConfidence | RootCauseConfidence, minimum: FixConfidence) -> bool:
    rank = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    return rank[actual.value] >= rank[minimum.value]


def _risk_rank(value: FixRiskLevel) -> int:
    return {FixRiskLevel.LOW: 0, FixRiskLevel.MEDIUM: 1, FixRiskLevel.HIGH: 2, FixRiskLevel.CRITICAL: 3}[value]


__all__ = ["AutomaticFixConfig", "AutomaticFixOrchestrator", "AutomaticFixPlanner", "AutomaticFixRequest", "AutomaticFixResult", "CausalStatus", "FixChangeSummary", "FixConfidence", "FixDecision", "FixDecisionType", "FixEvidence", "FixFailureReason", "FixLocation", "FixPlan", "FixRiskLevel", "FixStatus", "FixChangeType", "apply_automatic_fix"]
