"""Bounded evidence-driven test/analyze/fix/retest orchestration for Phase 7.5."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import re
from typing import Any, Callable

from backend_ai.agent.automatic_fix import (
    AutomaticFixConfig,
    AutomaticFixRequest,
    AutomaticFixResult,
    FixPlan,
    FixStatus,
    apply_automatic_fix,
)
from backend_ai.agent.automatic_testing import (
    AutomaticTestOrchestrator,
    AutomaticTestRequest,
    AutomaticTestResult,
    AutomaticTestStatus,
)
from backend_ai.agent.execution_budget import ExecutionBudget, ExecutionBudgetLedger, ExecutionBudgetSnapshot
from backend_ai.agent.root_cause_analysis import RootCauseAnalysis, RootCauseAnalyzer, RootCauseAnalysisRequest, RootCauseAnalysisStatus
from backend_ai.agent.regression_protection import RegressionBaseline, RegressionProtection, RegressionProtectionConfig, RegressionProtectionRequest, RegressionProtectionResult, RegressionStatus, RegressionTestScope
from backend_ai.agent.test_failure_analysis import TestFailureAnalysis, TestFailureAnalysisRequest, TestFailureAnalyzer
from backend_ai.agent.stop_conditions import StopConditionRequest, StopEvaluation, StopConditionEvaluator
from backend_ai.tools.safe_editing import SafeEditPolicy
from backend_ai.tools.test_result_parser import TestParseResult, TestParseStatus, TestResultParser


class SelfCorrectionStatus(str, Enum):
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    EXHAUSTED = "EXHAUSTED"
    NO_ACTIONABLE_FIX = "NO_ACTIONABLE_FIX"
    REPEATED_FAILURE = "REPEATED_FAILURE"
    NO_PROGRESS = "NO_PROGRESS"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    USER_INTERVENTION_REQUIRED = "USER_INTERVENTION_REQUIRED"
    REGRESSION_FREE = "REGRESSION_FREE"
    PRE_EXISTING_FAILURES_ONLY = "PRE_EXISTING_FAILURES_ONLY"
    REGRESSION_DETECTED = "REGRESSION_DETECTED"
    REGRESSION_INCOMPLETE = "REGRESSION_INCOMPLETE"
    REGRESSION_BLOCKED = "REGRESSION_BLOCKED"


class SelfCorrectionStep(str, Enum):
    RUN_TESTS = "RUN_TESTS"
    PARSE_RESULT = "PARSE_RESULT"
    ANALYZE_FAILURE = "ANALYZE_FAILURE"
    ROOT_CAUSE_ANALYSIS = "ROOT_CAUSE_ANALYSIS"
    APPLY_FIX = "APPLY_FIX"
    RECORD_ATTEMPT = "RECORD_ATTEMPT"
    PASS = "PASS"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class SelfCorrectionConfig:
    enabled: bool = True
    max_attempts: int = 3
    max_history: int = 16
    max_fingerprint_length: int = 512
    require_regression_protection: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.require_regression_protection, bool):
            raise ValueError("enabled and require_regression_protection must be boolean")
        for name in ("max_attempts", "max_history", "max_fingerprint_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_attempts > 64 or self.max_history > 128 or self.max_fingerprint_length > 4_096:
            raise ValueError("self-correction bound exceeds safety ceiling")


@dataclass(frozen=True, slots=True)
class SelfCorrectionAttempt:
    attempt_number: int
    steps: tuple[SelfCorrectionStep, ...]
    test_status: str
    parsed_status: str | None
    failure_count: int
    primary_failure: str | None
    root_cause_status: str | None
    root_cause_classification: str | None
    fix_status: str | None
    mutation_verified: bool
    failure_signature: str | None
    action_signature: str | None
    next_action: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"attempt_number": self.attempt_number, "steps": [item.value for item in self.steps], "test_status": self.test_status, "parsed_status": self.parsed_status, "failure_count": self.failure_count, "primary_failure": self.primary_failure, "root_cause_status": self.root_cause_status, "root_cause_classification": self.root_cause_classification, "fix_status": self.fix_status, "mutation_verified": self.mutation_verified, "failure_signature": self.failure_signature, "action_signature": self.action_signature, "next_action": self.next_action, "warnings": list(self.warnings)}


@dataclass(frozen=True, slots=True)
class RetryDecision:
    status: SelfCorrectionStatus
    reason: str
    attempt_count: int
    max_attempts: int
    stop_evaluation: StopEvaluation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "reason": self.reason, "attempt_count": self.attempt_count, "max_attempts": self.max_attempts, "stop_evaluation": self.stop_evaluation.to_dict() if self.stop_evaluation else None}


@dataclass(frozen=True, slots=True)
class SelfCorrectionRequest:
    test_request: AutomaticTestRequest
    fix_plan_provider: Callable[[RootCauseAnalysis, TestFailureAnalysis, int], FixPlan | None] | None = None
    fix_policy: SafeEditPolicy | None = None
    fix_config: AutomaticFixConfig = field(default_factory=AutomaticFixConfig)
    config: SelfCorrectionConfig = field(default_factory=SelfCorrectionConfig)
    budget_ledger: ExecutionBudgetLedger | None = None
    regression_baseline: RegressionBaseline | None = None
    regression_test_request: AutomaticTestRequest | None = None
    regression_scope: RegressionTestScope = RegressionTestScope.PROJECT_SUITE
    regression_config: RegressionProtectionConfig = field(default_factory=RegressionProtectionConfig)


@dataclass(frozen=True, slots=True)
class SelfCorrectionResult:
    status: SelfCorrectionStatus
    decision: RetryDecision
    attempts: tuple[SelfCorrectionAttempt, ...]
    final_test_result: AutomaticTestResult | None = None
    final_parsed_result: TestParseResult | None = None
    final_failure_analysis: TestFailureAnalysis | None = None
    final_root_cause: RootCauseAnalysis | None = None
    final_fix_result: AutomaticFixResult | None = None
    regression_protection: RegressionProtectionResult | None = None
    execution_budget: ExecutionBudgetSnapshot | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "decision": self.decision.to_dict(), "attempts": [item.to_dict() for item in self.attempts], "final_test_result": self.final_test_result.to_dict() if self.final_test_result else None, "final_parsed_result": self.final_parsed_result.to_dict() if self.final_parsed_result else None, "final_failure_analysis": self.final_failure_analysis.to_dict() if self.final_failure_analysis else None, "final_root_cause": self.final_root_cause.to_dict() if self.final_root_cause else None, "final_fix_result": self.final_fix_result.to_dict() if self.final_fix_result else None, "regression_protection": self.regression_protection.to_dict() if self.regression_protection else None, "execution_budget": self.execution_budget.to_dict() if self.execution_budget else None, "warnings": list(self.warnings), "errors": list(self.errors)}


class BoundedSelfCorrectionLoop:
    """Run bounded test → observe → analyze → RCA → one fix → retest transitions."""

    def __init__(self, *, test_orchestrator: AutomaticTestOrchestrator | None = None, parser: TestResultParser | None = None, failure_analyzer: TestFailureAnalyzer | None = None, root_cause_analyzer: RootCauseAnalyzer | None = None, fix_applier: Callable[[AutomaticFixRequest], AutomaticFixResult] | None = None) -> None:
        self.test_orchestrator = test_orchestrator or AutomaticTestOrchestrator()
        self.parser = parser or TestResultParser()
        self.failure_analyzer = failure_analyzer or TestFailureAnalyzer()
        self.root_cause_analyzer = root_cause_analyzer or RootCauseAnalyzer()
        self.fix_applier = fix_applier or apply_automatic_fix
        self.stop_evaluator = StopConditionEvaluator()
        self.regression_protection = RegressionProtection(test_orchestrator=self.test_orchestrator)

    def run(self, request: SelfCorrectionRequest) -> SelfCorrectionResult:
        if not isinstance(request, SelfCorrectionRequest):
            raise TypeError("request must be SelfCorrectionRequest")
        if not request.config.enabled:
            return self._finish(SelfCorrectionStatus.BLOCKED, "self-correction is disabled by host configuration", (), request, None)
        ledger = request.budget_ledger or request.test_request.budget_ledger or ExecutionBudgetLedger(request.test_request.budget or ExecutionBudget.conservative_defaults())
        test_request = replace(request.test_request, budget_ledger=ledger)
        attempts: list[SelfCorrectionAttempt] = []
        seen_actions: set[str] = set()
        final_test = final_parsed = final_failure = final_rca = final_fix = None
        final_regression: RegressionProtectionResult | None = None
        for number in range(1, request.config.max_attempts + 1):
            test_result = self.test_orchestrator.run(test_request)
            final_test = test_result
            steps = [SelfCorrectionStep.RUN_TESTS]
            if test_result.decision.status is AutomaticTestStatus.BUDGET_EXHAUSTED:
                attempts.append(self._attempt(number, steps, test_result, None, None, None, None, None, "STOP_BUDGET"))
                return self._finish(SelfCorrectionStatus.BUDGET_EXHAUSTED, "test execution budget was exhausted before this attempt started", attempts, request, ledger, stop_evaluation=self.stop_evaluator.evaluate(StopConditionRequest(budget_decision=test_result.decision.budget_decision)), final_test=test_result)
            if test_result.decision.status in {AutomaticTestStatus.BLOCKED, AutomaticTestStatus.INVALID} or test_result.test_run_result is None:
                attempts.append(self._attempt(number, steps, test_result, None, None, None, None, None, "STOP_BLOCKED"))
                return self._finish(SelfCorrectionStatus.BLOCKED, "automatic test execution was blocked or unavailable", attempts, request, ledger, stop_evaluation=self.stop_evaluator.evaluate(StopConditionRequest(safety_blocked=True)), final_test=test_result)
            parsed = self.parser.parse(test_result.test_run_result)
            final_parsed = parsed
            steps.append(SelfCorrectionStep.PARSE_RESULT)
            if parsed.overall_status is TestParseStatus.PASS:
                steps.append(SelfCorrectionStep.PASS)
                attempts.append(self._attempt(number, steps, test_result, parsed, None, None, None, None, "COMPLETE"))
                if request.config.require_regression_protection:
                    final_regression = self.regression_protection.run(RegressionProtectionRequest(request.regression_baseline, request.regression_test_request, request.regression_scope, request.regression_config, ledger))
                    if final_regression.status is RegressionStatus.REGRESSION_FREE:
                        return self._finish(SelfCorrectionStatus.REGRESSION_FREE, "targeted tests and regression scope passed with no new failures", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=final_failure, final_rca=final_rca, final_fix=final_fix, regression_protection=final_regression)
                    if final_regression.status is RegressionStatus.PRE_EXISTING_FAILURES_ONLY:
                        return self._finish(SelfCorrectionStatus.PRE_EXISTING_FAILURES_ONLY, final_regression.comparison.message if final_regression.comparison else "only baseline failures remain", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=final_failure, final_rca=final_rca, final_fix=final_fix, regression_protection=final_regression)
                    if final_regression.status is RegressionStatus.REGRESSION_DETECTED:
                        return self._finish(SelfCorrectionStatus.REGRESSION_DETECTED, final_regression.comparison.message if final_regression.comparison else "new regression detected after modification", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=final_failure, final_rca=final_rca, final_fix=final_fix, regression_protection=final_regression)
                    if final_regression.status is RegressionStatus.BUDGET_EXHAUSTED:
                        return self._finish(SelfCorrectionStatus.BUDGET_EXHAUSTED, "regression verification exhausted the shared execution budget", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=final_failure, final_rca=final_rca, final_fix=final_fix, regression_protection=final_regression)
                    if final_regression.status is RegressionStatus.VERIFICATION_BLOCKED:
                        return self._finish(SelfCorrectionStatus.REGRESSION_BLOCKED, "regression verification was blocked by policy or capability", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=final_failure, final_rca=final_rca, final_fix=final_fix, regression_protection=final_regression)
                    return self._finish(SelfCorrectionStatus.REGRESSION_INCOMPLETE, "regression verification was incomplete; DONE is not allowed", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=final_failure, final_rca=final_rca, final_fix=final_fix, regression_protection=final_regression)
                return self._finish(SelfCorrectionStatus.PASSED, "tests passed; loop stopped before any fix or retry", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=final_failure, final_rca=final_rca, final_fix=final_fix)
            failure = self.failure_analyzer.analyze(TestFailureAnalysisRequest(test_result.test_run_result, parsed))
            final_failure = failure
            steps.append(SelfCorrectionStep.ANALYZE_FAILURE)
            rca = self.root_cause_analyzer.analyze(RootCauseAnalysisRequest(failure))
            final_rca = rca
            steps.append(SelfCorrectionStep.ROOT_CAUSE_ANALYSIS)
            if rca.status not in {RootCauseAnalysisStatus.ANALYZED, RootCauseAnalysisStatus.INCONCLUSIVE} or not rca.hypotheses or request.fix_plan_provider is None:
                attempts.append(self._attempt(number, steps, test_result, parsed, failure, rca, None, None, "STOP_NO_ACTIONABLE_FIX"))
                return self._finish(SelfCorrectionStatus.NO_ACTIONABLE_FIX, "no actionable structured fix plan was available", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=failure, final_rca=rca)
            plan = request.fix_plan_provider(rca, failure, number)
            if plan is None:
                attempts.append(self._attempt(number, steps, test_result, parsed, failure, rca, None, None, "STOP_NO_ACTIONABLE_FIX"))
                return self._finish(SelfCorrectionStatus.NO_ACTIONABLE_FIX, "fix planner produced no safe FixPlan", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=failure, final_rca=rca)
            failure_signature = _fingerprint(_failure_material(failure), request.config.max_fingerprint_length)
            action_signature = _fingerprint(_action_material(plan), request.config.max_fingerprint_length)
            pair = failure_signature + ":" + action_signature
            if pair in seen_actions:
                attempts.append(self._attempt(number, steps, test_result, parsed, failure, rca, None, pair, "STOP_REPEATED_FAILURE"))
                return self._finish(SelfCorrectionStatus.REPEATED_FAILURE, "the same structured failure and proposed action repeated without progress", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=failure, final_rca=rca)
            seen_actions.add(pair)
            if number >= request.config.max_attempts:
                attempts.append(self._attempt(number, steps, test_result, parsed, failure, rca, None, pair, "STOP_MAX_ATTEMPTS"))
                return self._finish(SelfCorrectionStatus.EXHAUSTED, "max_attempts reached; no fix was applied after the final failure", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=failure, final_rca=rca)
            fix_request = AutomaticFixRequest(str(test_request.project_root), rca, plan, request.fix_policy, ledger, request.fix_config)
            fix = self.fix_applier(fix_request)
            final_fix = fix
            steps.append(SelfCorrectionStep.APPLY_FIX)
            if fix.status is FixStatus.BLOCKED or fix.status is FixStatus.RECOVERY_REQUIRED:
                attempts.append(self._attempt(number, steps, test_result, parsed, failure, rca, fix, pair, "STOP_BLOCKED"))
                return self._finish(SelfCorrectionStatus.BLOCKED, "automatic fix was blocked by budget, policy, or recovery boundary", attempts, request, ledger, stop_evaluation=self.stop_evaluator.evaluate(StopConditionRequest(safety_blocked=True)), final_test=test_result, final_parsed=parsed, final_failure=failure, final_rca=rca, final_fix=fix)
            if fix.status is not FixStatus.FIX_VERIFIED:
                attempts.append(self._attempt(number, steps, test_result, parsed, failure, rca, fix, pair, "STOP_NO_PROGRESS"))
                return self._finish(SelfCorrectionStatus.NO_PROGRESS, "the single fix attempt did not produce a verified post-state", attempts, request, ledger, final_test=test_result, final_parsed=parsed, final_failure=failure, final_rca=rca, final_fix=fix)
            attempts.append(self._attempt(number, steps, test_result, parsed, failure, rca, fix, pair, "RETEST"))
        return self._finish(SelfCorrectionStatus.EXHAUSTED, "max_attempts reached", attempts, request, ledger, final_test=final_test, final_parsed=final_parsed, final_failure=final_failure, final_rca=final_rca, final_fix=final_fix)

    def _attempt(self, number: int, steps: list[SelfCorrectionStep], test_result: AutomaticTestResult, parsed: TestParseResult | None, failure: TestFailureAnalysis | None, rca: RootCauseAnalysis | None, fix: AutomaticFixResult | None, action_signature: str | None, next_action: str) -> SelfCorrectionAttempt:
        return SelfCorrectionAttempt(number, tuple((*steps, SelfCorrectionStep.RECORD_ATTEMPT)), test_result.decision.status.value, parsed.overall_status.value if parsed else None, len(failure.findings) if failure else 0, failure.primary_failure_id if failure else None, rca.status.value if rca else None, rca.hypotheses[0].classification.value if rca and rca.hypotheses else None, fix.status.value if fix else None, bool(fix and fix.verified), _fingerprint(_failure_material(failure), 512) if failure else None, action_signature, next_action)

    @staticmethod
    def _finish(status: SelfCorrectionStatus, reason: str, attempts: list[SelfCorrectionAttempt], request: SelfCorrectionRequest, ledger: ExecutionBudgetLedger | None, *, stop_evaluation: StopEvaluation | None = None, final_test=None, final_parsed=None, final_failure=None, final_rca=None, final_fix=None, regression_protection=None) -> SelfCorrectionResult:
        decision = RetryDecision(status, reason, len(attempts), request.config.max_attempts, stop_evaluation)
        return SelfCorrectionResult(status=status, decision=decision, attempts=tuple(attempts[-request.config.max_history:]), final_test_result=final_test, final_parsed_result=final_parsed, final_failure_analysis=final_failure, final_root_cause=final_rca, final_fix_result=final_fix, regression_protection=regression_protection, execution_budget=ledger.snapshot() if ledger else None)


def run_self_correction(request: SelfCorrectionRequest) -> SelfCorrectionResult:
    return BoundedSelfCorrectionLoop().run(request)


def _failure_material(analysis: TestFailureAnalysis | None) -> str:
    if analysis is None: return "none"
    return "|".join(f"{item.classification.value}|{item.test_name or ''}|{item.location.file_path or ''}|{item.location.line_number or ''}|{item.observed_failure}" for item in analysis.findings[:32])


def _action_material(plan: FixPlan) -> str:
    return "|".join((plan.target_file, plan.change_type.value, plan.hypothesis_id, plan.intended_change, plan.new_content))


def _fingerprint(value: str, limit: int) -> str:
    safe = re.sub(r"(?i)(password|token|secret|api[_ -]?key|credential|authorization)\s*[:=]\s*[^|\s]+", r"\1=[REDACTED]", value)
    safe = re.sub(r"\d+", "#", safe.casefold())[:limit]
    return hashlib.sha256(safe.encode("utf-8")).hexdigest()


__all__ = ["BoundedSelfCorrectionLoop", "RetryDecision", "SelfCorrectionAttempt", "SelfCorrectionConfig", "SelfCorrectionRequest", "SelfCorrectionResult", "SelfCorrectionStatus", "SelfCorrectionStep", "run_self_correction"]
