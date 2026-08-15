"""Bounded post-fix regression comparison and execution for Phase 7.6."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import re
from typing import Any

from backend_ai.agent.automatic_testing import AutomaticTestOrchestrator, AutomaticTestRequest, AutomaticTestResult, AutomaticTestStatus
from backend_ai.agent.execution_budget import BudgetDecision, ExecutionBudget, ExecutionBudgetLedger, ExecutionBudgetSnapshot
from backend_ai.tools.test_result_parser import TestParseResult, TestParseStatus


class RegressionStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    READY = "READY"
    REGRESSION_FREE = "REGRESSION_FREE"
    REGRESSION_DETECTED = "REGRESSION_DETECTED"
    PRE_EXISTING_FAILURES_ONLY = "PRE_EXISTING_FAILURES_ONLY"
    VERIFICATION_INCOMPLETE = "VERIFICATION_INCOMPLETE"
    VERIFICATION_BLOCKED = "VERIFICATION_BLOCKED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class RegressionTestScope(str, Enum):
    AFFECTED_TEST = "AFFECTED_TEST"
    AFFECTED_MODULE = "AFFECTED_MODULE"
    RELATED_MODULE = "RELATED_MODULE"
    PROJECT_SUITE = "PROJECT_SUITE"


class RegressionFindingState(str, Enum):
    PRE_EXISTING = "PRE_EXISTING"
    RESOLVED = "RESOLVED"
    PERSISTENT = "PERSISTENT"
    NEW = "NEW"
    CHANGED = "CHANGED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RegressionProtectionConfig:
    enabled: bool = True
    require_complete_parser: bool = True
    max_failures: int = 64
    max_fingerprint_length: int = 512
    max_findings: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.require_complete_parser, bool):
            raise ValueError("regression protection flags must be boolean")
        for name in ("max_failures", "max_fingerprint_length", "max_findings"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_failures > 1024 or self.max_findings > 2048 or self.max_fingerprint_length > 4096:
            raise ValueError("regression protection limit exceeds safety ceiling")


@dataclass(frozen=True, slots=True)
class RegressionBaseline:
    execution_status: str
    parsed_status: str
    framework: str | None
    total: int | None
    passed: int | None
    failed: int | None
    errors: int | None
    skipped: int | None
    failure_fingerprints: tuple[str, ...]
    failure_identities: tuple[str, ...]
    parser_completeness: str
    parser_truncated: bool
    execution_started: bool
    execution_completed: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_fingerprints", tuple(self.failure_fingerprints[:128]))
        object.__setattr__(self, "failure_identities", tuple(self.failure_identities[:128]))
        object.__setattr__(self, "warnings", tuple(self.warnings[:16]))

    @classmethod
    def capture(cls, test_result: AutomaticTestResult, parsed_result: TestParseResult, *, max_failures: int = 64) -> "RegressionBaseline":
        if not isinstance(parsed_result, TestParseResult):
            raise TypeError("parsed_result must be TestParseResult")
        identities = _failure_identities(parsed_result, max_failures)
        fingerprints = _failure_fingerprints(parsed_result, max_failures)
        raw = test_result.test_run_result if isinstance(test_result, AutomaticTestResult) else None
        return cls(
            raw.status.value if raw is not None else "UNKNOWN",
            parsed_result.overall_status.value,
            parsed_result.framework,
            parsed_result.total,
            parsed_result.passed,
            parsed_result.failed,
            parsed_result.errors,
            parsed_result.skipped,
            fingerprints,
            identities,
            parsed_result.parse_completeness,
            parsed_result.truncated,
            bool(test_result and test_result.started),
            bool(test_result and test_result.execution.state.value == "COMPLETED"),
            parsed_result.warnings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"execution_status": self.execution_status, "parsed_status": self.parsed_status, "framework": self.framework, "total": self.total, "passed": self.passed, "failed": self.failed, "errors": self.errors, "skipped": self.skipped, "failure_fingerprints": list(self.failure_fingerprints), "failure_identities": list(self.failure_identities), "parser_completeness": self.parser_completeness, "parser_truncated": self.parser_truncated, "execution_started": self.execution_started, "execution_completed": self.execution_completed, "warnings": list(self.warnings)}


@dataclass(frozen=True, slots=True)
class RegressionTestExecution:
    test_result: AutomaticTestResult | None
    parsed_result: TestParseResult | None
    scope: RegressionTestScope
    started: bool
    completed: bool

    def to_dict(self) -> dict[str, Any]:
        return {"test_result": self.test_result.to_dict() if self.test_result else None, "parsed_result": self.parsed_result.to_dict() if self.parsed_result else None, "scope": self.scope.value, "started": self.started, "completed": self.completed}


@dataclass(frozen=True, slots=True)
class RegressionFinding:
    finding_id: str
    state: RegressionFindingState
    identity: str
    baseline_fingerprint: str | None
    post_fix_fingerprint: str | None
    message: str
    causal_inference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"finding_id": self.finding_id, "state": self.state.value, "identity": self.identity, "baseline_fingerprint": self.baseline_fingerprint, "post_fix_fingerprint": self.post_fix_fingerprint, "message": self.message, "causal_inference": self.causal_inference}


@dataclass(frozen=True, slots=True)
class RegressionComparison:
    status: RegressionStatus
    resolved_failures: tuple[str, ...]
    persistent_failures: tuple[str, ...]
    pre_existing_failures: tuple[str, ...]
    new_failures: tuple[str, ...]
    changed_failures: tuple[str, ...]
    unknown_failures: tuple[str, ...]
    findings: tuple[RegressionFinding, ...]
    evidence_complete: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "resolved_failures": list(self.resolved_failures), "persistent_failures": list(self.persistent_failures), "pre_existing_failures": list(self.pre_existing_failures), "new_failures": list(self.new_failures), "changed_failures": list(self.changed_failures), "unknown_failures": list(self.unknown_failures), "findings": [item.to_dict() for item in self.findings], "evidence_complete": self.evidence_complete, "message": self.message}


@dataclass(frozen=True, slots=True)
class RegressionProtectionRequest:
    baseline: RegressionBaseline | None
    regression_test_request: AutomaticTestRequest | None
    scope: RegressionTestScope = RegressionTestScope.PROJECT_SUITE
    config: RegressionProtectionConfig = field(default_factory=RegressionProtectionConfig)
    budget_ledger: ExecutionBudgetLedger | None = None


@dataclass(frozen=True, slots=True)
class RegressionProtectionResult:
    status: RegressionStatus
    scope: RegressionTestScope
    baseline: RegressionBaseline | None
    execution: RegressionTestExecution | None
    comparison: RegressionComparison | None
    budget_decision: BudgetDecision | None
    execution_budget: ExecutionBudgetSnapshot | None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "scope": self.scope.value, "baseline": self.baseline.to_dict() if self.baseline else None, "execution": self.execution.to_dict() if self.execution else None, "comparison": self.comparison.to_dict() if self.comparison else None, "budget_decision": self.budget_decision.to_dict() if self.budget_decision else None, "execution_budget": self.execution_budget.to_dict() if self.execution_budget else None, "warnings": list(self.warnings), "errors": list(self.errors)}


class RegressionProtection:
    """Execute one bounded regression scope and compare it to explicit baseline evidence."""

    def __init__(self, *, test_orchestrator: AutomaticTestOrchestrator | None = None) -> None:
        self.test_orchestrator = test_orchestrator or AutomaticTestOrchestrator()

    def run(self, request: RegressionProtectionRequest) -> RegressionProtectionResult:
        if not isinstance(request, RegressionProtectionRequest):
            raise TypeError("request must be RegressionProtectionRequest")
        ledger = request.budget_ledger
        if not request.config.enabled:
            return RegressionProtectionResult(RegressionStatus.NOT_REQUIRED, request.scope, request.baseline, None, None, None, ledger.snapshot() if ledger else None)
        if request.baseline is None:
            return RegressionProtectionResult(RegressionStatus.INSUFFICIENT_EVIDENCE, request.scope, None, None, None, None, ledger.snapshot() if ledger else None, warnings=("No authoritative baseline was supplied; regression-free cannot be claimed.",))
        if request.regression_test_request is None:
            return RegressionProtectionResult(RegressionStatus.VERIFICATION_INCOMPLETE, request.scope, request.baseline, None, None, None, ledger.snapshot() if ledger else None, warnings=("No evidence-backed regression test scope was supplied.",))
        if ledger is None:
            return RegressionProtectionResult(RegressionStatus.VERIFICATION_BLOCKED, request.scope, request.baseline, None, None, None, None, warnings=("Regression execution requires a shared execution budget ledger.",))
        test_request = replace(request.regression_test_request, budget_ledger=ledger)
        execution_result = self.test_orchestrator.run(test_request)
        execution = RegressionTestExecution(execution_result, None, request.scope, execution_result.started, execution_result.started)
        if execution_result.decision.status is AutomaticTestStatus.BUDGET_EXHAUSTED:
            return RegressionProtectionResult(RegressionStatus.BUDGET_EXHAUSTED, request.scope, request.baseline, execution, None, execution_result.decision.budget_decision, ledger.snapshot())
        if execution_result.decision.status in {AutomaticTestStatus.BLOCKED, AutomaticTestStatus.INVALID, AutomaticTestStatus.UNAVAILABLE} or execution_result.test_run_result is None:
            status = RegressionStatus.VERIFICATION_BLOCKED if execution_result.decision.status in {AutomaticTestStatus.BLOCKED, AutomaticTestStatus.INVALID} else RegressionStatus.VERIFICATION_INCOMPLETE
            return RegressionProtectionResult(status, request.scope, request.baseline, execution, None, execution_result.decision.budget_decision, ledger.snapshot(), warnings=(execution_result.decision.reason,))
        try:
            from backend_ai.tools.test_result_parser import TestResultParser
            parsed = TestResultParser().parse(execution_result.test_run_result)
        except Exception as exc:
            return RegressionProtectionResult(RegressionStatus.VERIFICATION_FAILED, request.scope, request.baseline, execution, None, execution_result.decision.budget_decision, ledger.snapshot(), errors=(str(exc),))
        execution = RegressionTestExecution(execution_result, parsed, request.scope, execution_result.started, True)
        comparison = compare_regression(request.baseline, parsed, config=request.config)
        return RegressionProtectionResult(comparison.status, request.scope, request.baseline, execution, comparison, execution_result.decision.budget_decision, ledger.snapshot(), warnings=parsed.warnings)


def compare_regression(baseline: RegressionBaseline | None, post_fix: TestParseResult | None, *, config: RegressionProtectionConfig | None = None) -> RegressionComparison:
    active = config or RegressionProtectionConfig()
    if baseline is None or not isinstance(post_fix, TestParseResult):
        return RegressionComparison(RegressionStatus.INSUFFICIENT_EVIDENCE, (), (), (), (), (), (), (), False, "Baseline and post-fix parsed evidence are both required.")
    if active.require_complete_parser and (baseline.parser_truncated or baseline.parser_completeness != "complete" or post_fix.truncated or post_fix.parse_completeness != "complete"):
        return RegressionComparison(RegressionStatus.VERIFICATION_INCOMPLETE, (), (), (), (), (), (), (), False, "Parser evidence is incomplete or truncated; regression-free cannot be claimed.")
    if baseline.parsed_status in {TestParseStatus.UNKNOWN.value, TestParseStatus.NO_TESTS.value} or post_fix.overall_status in {TestParseStatus.UNKNOWN, TestParseStatus.NO_TESTS}:
        return RegressionComparison(RegressionStatus.INSUFFICIENT_EVIDENCE, (), (), (), (), (), (), (), False, "Baseline or post-fix execution has no reliable failure identity evidence.")
    if post_fix.overall_status in {TestParseStatus.TIMEOUT, TestParseStatus.OUTPUT_LIMIT}:
        return RegressionComparison(RegressionStatus.VERIFICATION_INCOMPLETE, (), (), (), (), (), (), (), False, "Regression execution timed out or reached an output limit.")
    if post_fix.overall_status is TestParseStatus.EXECUTION_ERROR:
        return RegressionComparison(RegressionStatus.VERIFICATION_FAILED, (), (), (), (), (), (), (), False, "Regression execution failed before comparable semantic results were available.")
    baseline_map = dict(zip(baseline.failure_identities, baseline.failure_fingerprints))
    post_identities = _failure_identities(post_fix, active.max_failures)
    post_fingerprints = _failure_fingerprints(post_fix, active.max_failures)
    post_map = dict(zip(post_identities, post_fingerprints))
    resolved = tuple(sorted(set(baseline_map) - set(post_map)))
    persistent: list[str] = []
    changed: list[str] = []
    pre_existing: list[str] = []
    new: list[str] = []
    findings: list[RegressionFinding] = []
    for identity in sorted(set(baseline_map) | set(post_map)):
        old = baseline_map.get(identity)
        current = post_map.get(identity)
        if old is not None and current is None:
            findings.append(RegressionFinding(f"resolved-{len(findings)+1}", RegressionFindingState.RESOLVED, identity, old, None, "Failure was present in baseline and absent after the fix."))
        elif old is None and current is not None:
            new.append(identity)
            findings.append(RegressionFinding(f"new-{len(findings)+1}", RegressionFindingState.NEW, identity, None, current, "New failure detected after modification; causality is not asserted."))
        elif old == current:
            persistent.append(identity)
            pre_existing.append(identity)
            findings.append(RegressionFinding(f"persistent-{len(findings)+1}", RegressionFindingState.PERSISTENT, identity, old, current, "Failure identity and bounded fingerprint persisted from baseline."))
        else:
            changed.append(identity)
            findings.append(RegressionFinding(f"changed-{len(findings)+1}", RegressionFindingState.CHANGED, identity, old, current, "Existing failure identity remains but bounded evidence changed."))
    findings = findings[: active.max_findings]
    if new or changed:
        status = RegressionStatus.REGRESSION_DETECTED
        message = "New or materially changed failure evidence appeared after the fix."
    elif not post_identities:
        status = RegressionStatus.REGRESSION_FREE
        message = "Post-fix regression scope completed with no failure identities."
    elif persistent and not new and not changed:
        status = RegressionStatus.PRE_EXISTING_FAILURES_ONLY
        message = "Only baseline failure identities remain; no new failure was detected."
    else:
        status = RegressionStatus.REGRESSION_FREE
        message = "No new failure evidence was detected."
    return RegressionComparison(status, resolved, tuple(persistent), tuple(pre_existing), tuple(new), tuple(changed), (), tuple(findings), True, message)


def capture_regression_baseline(test_result: AutomaticTestResult, parsed_result: TestParseResult, *, max_failures: int = 64) -> RegressionBaseline:
    return RegressionBaseline.capture(test_result, parsed_result, max_failures=max_failures)


def run_regression_protection(request: RegressionProtectionRequest) -> RegressionProtectionResult:
    return RegressionProtection().run(request)


def _failure_identities(parsed: TestParseResult, limit: int) -> tuple[str, ...]:
    values: list[str] = []
    for item in parsed.failure_details[:limit]:
        values.append(_identity(item.test_name, item.file_path, item.line_number, item.failure_type, item.message))
    for item in parsed.error_details[: max(0, limit - len(values))]:
        values.append(_identity(item.test_name, item.file_path, item.line_number, item.error_type, item.message))
    if not values:
        values.extend(_normalize(name) for name in (*parsed.failed_tests[:limit], *parsed.error_tests[: max(0, limit - len(parsed.failed_tests))]))
    return tuple(dict.fromkeys(values))[:limit]


def _failure_fingerprints(parsed: TestParseResult, limit: int) -> tuple[str, ...]:
    values: list[str] = []
    for item in parsed.failure_details[:limit]:
        values.append(_hash("|".join((item.test_name or "", item.file_path or "", str(item.line_number or ""), item.failure_type or "", item.message or ""))))
    for item in parsed.error_details[: max(0, limit - len(values))]:
        values.append(_hash("|".join((item.test_name or "", item.file_path or "", str(item.line_number or ""), item.error_type or "", item.message or ""))))
    if not values:
        values.extend(_hash(value) for value in (*parsed.failed_tests[:limit], *parsed.error_tests[: max(0, limit - len(parsed.failed_tests))]))
    return tuple(values)[:limit]


def _identity(test_name: str | None, path: str | None, line: int | None, kind: str | None, message: str | None) -> str:
    if test_name: return _normalize(test_name)
    if path: return f"{_normalize(path)}:{line or '#'}:{_normalize(kind or 'error')}"
    return _normalize(kind or message or "unknown-failure")


def _hash(value: str) -> str:
    return hashlib.sha256(_normalize(value).encode("utf-8")).hexdigest()


def _normalize(value: str) -> str:
    text = str(value).casefold()
    text = re.sub(r"(?i)(password|token|secret|api[_ -]?key|credential|authorization)\s*[:=]\s*[^\s|]+", r"\1=[redacted]", text)
    text = re.sub(r"/tmp/[^\s|]+", "<tmp>", text)
    text = re.sub(r"\b0x[0-9a-f]+\b", "<address>", text)
    text = re.sub(r"\b\d{4,}\b", "#", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]


__all__ = ["RegressionBaseline", "RegressionComparison", "RegressionFinding", "RegressionFindingState", "RegressionProtection", "RegressionProtectionConfig", "RegressionProtectionRequest", "RegressionProtectionResult", "RegressionStatus", "RegressionTestExecution", "RegressionTestScope", "capture_regression_baseline", "compare_regression", "run_regression_protection"]
