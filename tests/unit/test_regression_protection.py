from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backend_ai.agent import (
    AutomaticTestDecision,
    AutomaticTestExecution,
    AutomaticTestExecutionState,
    AutomaticTestResult,
    AutomaticTestStatus,
    RegressionBaseline,
    RegressionProtection,
    RegressionProtectionConfig,
    RegressionProtectionRequest,
    RegressionStatus,
    RegressionTestScope,
    capture_regression_baseline,
    compare_regression,
)
from backend_ai.agent.automatic_testing import AutomaticTestRequest
from backend_ai.agent.registry import ToolRegistry
from backend_ai.agent.execution_budget import BudgetDimension, ExecutionBudget, ExecutionBudgetLedger
from backend_ai.tools.test_result_parser import TestErrorRecord, TestFailureRecord, TestParseResult, TestParseStatus
from backend_ai.tools.test_runner import TestRunStatus


def parsed(status=TestParseStatus.FAIL, *, failures=(), errors=(), truncated=False, completeness="complete", passed=10, failed=1, error_count=0):
    total = passed + failed + error_count if passed is not None and failed is not None else None
    return TestParseResult(status, TestRunStatus.COMPLETED, 1 if status is not TestParseStatus.PASS else 0, passed, failed, error_count, 0, 0, 0, total, "bounded stdout", "", tuple(failures), tuple(errors), tuple(item.test_name for item in failures if item.test_name), tuple(item.test_name for item in errors if item.test_name), "pytest", "high", "pytest", (), truncated, completeness, 0.1)


def failure(name, message="assertion differs", path="tests/test_value.py", line=4):
    return TestFailureRecord(test_name=name, file_path=path, line_number=line, failure_type="AssertionError", message=message, framework="pytest", raw_excerpt=message)


def test_baseline_capture_is_bounded_and_serializable():
    result = AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED))
    baseline = capture_regression_baseline(result, parsed(failures=(failure("test_value"),)))
    assert baseline.execution_started is True
    assert baseline.execution_completed is True
    assert baseline.failure_identities == ("test_value",)
    assert baseline.to_dict()["parser_completeness"] == "complete"


def test_resolved_failure_is_regression_free():
    baseline = RegressionBaseline.capture(AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED)), parsed(failures=(failure("test_value"),)))
    comparison = compare_regression(baseline, parsed(TestParseStatus.PASS, failures=(), passed=11, failed=0))
    assert comparison.status is RegressionStatus.REGRESSION_FREE
    assert comparison.resolved_failures == ("test_value",)


def test_new_failure_is_regression_detected():
    baseline = RegressionBaseline.capture(AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED)), parsed(failures=(failure("test_value"),)))
    comparison = compare_regression(baseline, parsed(failures=(failure("test_other"),)))
    assert comparison.status is RegressionStatus.REGRESSION_DETECTED
    assert comparison.new_failures == ("test_other",)
    assert comparison.findings[0].causal_inference is False


def test_persistent_baseline_failure_is_not_new_regression():
    baseline = RegressionBaseline.capture(AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED)), parsed(failures=(failure("test_other"),)))
    comparison = compare_regression(baseline, parsed(failures=(failure("test_other"),)))
    assert comparison.status is RegressionStatus.PRE_EXISTING_FAILURES_ONLY
    assert comparison.persistent_failures == ("test_other",)


def test_changed_failure_is_detected_by_fingerprint():
    baseline = RegressionBaseline.capture(AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED)), parsed(failures=(failure("test_other", "old assertion"),)))
    comparison = compare_regression(baseline, parsed(failures=(failure("test_other", "new type error"),)))
    assert comparison.status is RegressionStatus.REGRESSION_DETECTED
    assert comparison.changed_failures == ("test_other",)


def test_incomplete_parser_prevents_regression_free():
    baseline = RegressionBaseline.capture(AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED)), parsed(failures=(failure("test_value"),)))
    comparison = compare_regression(baseline, parsed(TestParseStatus.PASS, failures=(), truncated=True, completeness="partial", passed=11, failed=0))
    assert comparison.status is RegressionStatus.VERIFICATION_INCOMPLETE
    assert comparison.evidence_complete is False


def test_missing_baseline_cannot_claim_success():
    result = RegressionProtection().run(RegressionProtectionRequest(None, None))
    assert result.status is RegressionStatus.INSUFFICIENT_EVIDENCE


def test_scope_is_explicit_and_budget_can_block_before_execution():
    ledger = ExecutionBudgetLedger(ExecutionBudget(max_test_executions=0, max_tool_calls=0))
    request = AutomaticTestRequest(task="regression", project_root=Path("."), registry=ToolRegistry.with_test_execution(), user_requested=True)
    baseline = RegressionBaseline("COMPLETED", "FAIL", "pytest", 1, 0, 1, 0, 0, ("fingerprint",), ("test_value",), "complete", False, True, True)
    protection = RegressionProtection()
    result = protection.run(RegressionProtectionRequest(baseline, request, RegressionTestScope.AFFECTED_TEST, budget_ledger=ledger))
    assert result.status is RegressionStatus.BUDGET_EXHAUSTED or result.status is RegressionStatus.VERIFICATION_BLOCKED
    assert result.execution is not None
    assert result.execution.started is False


def test_sensitive_values_are_not_present_in_fingerprint_material():
    first = RegressionBaseline.capture(AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED)), parsed(failures=(failure("test_value", "token=secret-value"),)))
    second = RegressionBaseline.capture(AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED)), parsed(failures=(failure("test_value", "token=other-value"),)))
    assert first.failure_fingerprints == second.failure_fingerprints
    assert "secret-value" not in repr(first.to_dict())
    assert "other-value" not in repr(second.to_dict())


def test_comparison_is_deterministic():
    baseline = RegressionBaseline.capture(AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED)), parsed(failures=(failure("b"), failure("a"))))
    post = parsed(failures=(failure("c"), failure("a")))
    assert compare_regression(baseline, post).to_dict() == compare_regression(baseline, post).to_dict()


def test_multiple_framework_scope_is_preserved_without_new_runner():
    assert RegressionTestScope.AFFECTED_MODULE.value == "AFFECTED_MODULE"
    assert RegressionProtectionConfig(max_failures=2).max_failures == 2


def test_policy_denial_is_blocked_before_execution():
    baseline = RegressionBaseline("COMPLETED", "FAIL", "pytest", 1, 0, 1, 0, 0, ("fingerprint",), ("test_value",), "complete", False, True, True)
    request = AutomaticTestRequest(task="regression", project_root=Path("."), registry=None, user_requested=True)
    result = RegressionProtection().run(RegressionProtectionRequest(baseline, request, RegressionTestScope.AFFECTED_TEST, budget_ledger=ExecutionBudgetLedger(ExecutionBudget())))
    assert result.status is RegressionStatus.VERIFICATION_BLOCKED
    assert result.execution is not None and result.execution.started is False


def test_timeout_and_output_limit_are_incomplete():
    baseline = RegressionBaseline.capture(AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED)), parsed(failures=(failure("test_value"),)))
    assert compare_regression(baseline, parsed(TestParseStatus.TIMEOUT, truncated=False)).status is RegressionStatus.VERIFICATION_INCOMPLETE
    assert compare_regression(baseline, parsed(TestParseStatus.OUTPUT_LIMIT, truncated=False)).status is RegressionStatus.VERIFICATION_INCOMPLETE


def test_unknown_or_no_tests_cannot_be_regression_free():
    baseline = RegressionBaseline.capture(AutomaticTestResult(AutomaticTestDecision(AutomaticTestStatus.RUN, "run"), AutomaticTestExecution(AutomaticTestExecutionState.COMPLETED), SimpleNamespace(status=TestRunStatus.COMPLETED)), parsed(failures=(failure("test_value"),)))
    assert compare_regression(baseline, parsed(TestParseStatus.UNKNOWN, failures=(), passed=None, failed=None)).status is RegressionStatus.INSUFFICIENT_EVIDENCE
    assert compare_regression(baseline, parsed(TestParseStatus.NO_TESTS, failures=(), passed=0, failed=0)).status is RegressionStatus.INSUFFICIENT_EVIDENCE
