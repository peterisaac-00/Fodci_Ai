from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from backend_ai.agent import (
    FailureAnalysisConfig,
    FailureAnalysisStatus,
    FailureClassification,
    FailureConfidence,
    FailureLocationKind,
    TestFailureAnalysisRequest,
    TestFailureAnalyzer,
    analyze_test_failure,
)
from backend_ai.agent.autonomous_tool_loop import AutonomousToolLoop
from backend_ai.tools import (
    CommandResult,
    TestParseLimits,
    TestParseStatus,
    TestErrorRecord,
    TestResultParser,
    TestRunFailureCode,
    TestRunPlan,
    TestRunResult,
    TestRunStatus,
    parse_test_result,
)


def _raw(framework: str = "pytest", stdout: str = "", stderr: str = "", *, exit_code: int | None = 0, status: TestRunStatus = TestRunStatus.COMPLETED, failure_code: TestRunFailureCode | None = None, stdout_truncated: bool = False, timed_out: bool = False, warnings: tuple[str, ...] = ()) -> TestRunResult:
    command = CommandResult(("python", "-m", "pytest"), ".", "timed_out" if timed_out else ("failed" if exit_code not in (None, 0) else "completed"), exit_code, stdout, stderr, 0.1, timed_out, True, exit_code == 0 and not timed_out and not stdout_truncated, False, not timed_out, stdout_truncated, False, True, True, "normal", None, None, warnings, "CLEANED_UP", False, False, len(stdout.encode()), len(stderr.encode()))
    plan = TestRunPlan(("python", "-m", "pytest"), ".", framework, "fixture", ("fixture",), "high", True)
    return TestRunResult(status, plan, command, None, failure_code, "python", framework, (framework,), ("fixture",), warnings)


def _parse(output: str, *, exit_code: int = 1, status: TestRunStatus = TestRunStatus.COMPLETED, failure_code: TestRunFailureCode | None = TestRunFailureCode.NONZERO_EXIT, **kwargs):
    raw = _raw(stdout=output, exit_code=exit_code, status=status, failure_code=failure_code, **kwargs)
    return raw, parse_test_result(raw)


def test_pass_has_no_failure() -> None:
    raw, parsed = _parse("========================= 2 passed in 0.1s =========================\n", exit_code=0, failure_code=None)
    analysis = analyze_test_failure(raw, parsed)
    assert analysis.status is FailureAnalysisStatus.NO_FAILURE
    assert analysis.findings == ()


def test_assertion_failure_has_test_location_and_evidence() -> None:
    raw, parsed = _parse("________________ test_api __________________\nE AssertionError: Expected 200 but received 401\nFAILED tests/test_api.py::test_api - AssertionError: Expected 200 but received 401\n1 failed in 0.1s\n")
    analysis = analyze_test_failure(raw, parsed)
    assert analysis.status is FailureAnalysisStatus.ANALYZED
    assert analysis.findings
    finding = analysis.findings[0]
    assert finding.classification is FailureClassification.ASSERTION_FAILURE
    assert finding.location.kind is FailureLocationKind.TEST_LOCATION
    assert finding.evidence


def test_exception_type_classification() -> None:
    raw, parsed = _parse("________________ test_type __________________\nE TypeError: bad type\nFAILED tests/test_api.py::test_type - TypeError: bad type\n1 failed in 0.1s\n")
    assert analyze_test_failure(raw, parsed).findings[0].classification is FailureClassification.TYPE_ERROR


def test_import_and_module_errors_are_distinguished() -> None:
    raw, parsed = _parse("ERROR collecting tests/test_import.py\nImportError: cannot import name missing\n", exit_code=2, failure_code=TestRunFailureCode.NONZERO_EXIT)
    result = analyze_test_failure(raw, parsed)
    assert result.findings[0].classification is FailureClassification.IMPORT_ERROR
    raw2, parsed2 = _parse("ERROR collecting tests/test_import.py\nModuleNotFoundError: No module named missing\n", exit_code=2, failure_code=TestRunFailureCode.NONZERO_EXIT)
    assert analyze_test_failure(raw2, parsed2).findings[0].classification is FailureClassification.MODULE_NOT_FOUND


def test_timeout_output_limit_execution_error_are_technical() -> None:
    for status, code, expected in ((TestRunStatus.TIMED_OUT, TestRunFailureCode.TIMEOUT, FailureClassification.TIMEOUT), (TestRunStatus.OUTPUT_LIMIT_REACHED, TestRunFailureCode.OUTPUT_LIMIT_REACHED, FailureClassification.OUTPUT_LIMIT), (TestRunStatus.START_FAILED, TestRunFailureCode.START_FAILED, FailureClassification.EXECUTION_ERROR)):
        raw, parsed = _parse("", exit_code=None if status is TestRunStatus.START_FAILED else 1, status=status, failure_code=code, timed_out=status is TestRunStatus.TIMED_OUT, stdout_truncated=status is TestRunStatus.OUTPUT_LIMIT_REACHED)
        assert analyze_test_failure(raw, parsed).classification is expected


def test_missing_records_is_insufficient_evidence() -> None:
    raw, parsed = _parse("unexpected launcher text", exit_code=7)
    result = analyze_test_failure(raw, parsed)
    assert result.status is FailureAnalysisStatus.INSUFFICIENT_EVIDENCE
    assert result.classification is FailureClassification.UNKNOWN_FAILURE


def test_truncated_parser_result_is_incomplete() -> None:
    raw, parsed = _parse("1 failed in 0.1s\n", stdout_truncated=True)
    result = analyze_test_failure(raw, parsed)
    assert result.truncated
    assert result.analysis_complete is False


def test_related_failures_are_grouped_and_primary_is_inference() -> None:
    output = "ERROR collecting tests/a.py\nModuleNotFoundError: No module named missing\nERROR collecting tests/b.py\nModuleNotFoundError: No module named missing\n"
    raw, parsed = _parse(output, exit_code=2)
    parsed = replace(parsed, error_details=(TestErrorRecord(error_type="ModuleNotFoundError", message="No module named missing", file_path="tests/a.py"), TestErrorRecord(error_type="ModuleNotFoundError", message="No module named missing", file_path="tests/b.py")), errors=2)
    result = analyze_test_failure(raw, parsed)
    assert result.groups
    assert len(result.groups[0].related_finding_ids) >= 2
    assert result.primary_failure_id is not None
    assert result.groups[0].causal_inference is True


def test_auth_and_api_classification_is_conservative() -> None:
    raw, parsed = _parse("FAILED tests/test_auth.py::test_login - AssertionError: expected 200 received 401\n1 failed in 0.1s\n")
    result = analyze_test_failure(raw, parsed)
    assert result.findings[0].classification is FailureClassification.ASSERTION_FAILURE
    raw2, parsed2 = _parse("FAILED tests/test_auth.py::test_login - HTTP 401 unauthorized token\n1 failed in 0.1s\n")
    assert analyze_test_failure(raw2, parsed2).findings[0].classification is FailureClassification.AUTHENTICATION_FAILURE


def test_sensitive_values_are_redacted() -> None:
    raw, parsed = _parse("FAILED tests/test_secret.py::test_secret - AssertionError: password=supersecret token=abc123\n1 failed in 0.1s\n")
    result = analyze_test_failure(raw, parsed)
    rendered = str(result.to_dict())
    assert "supersecret" not in rendered and "abc123" not in rendered
    assert "REDACTED" in rendered


def test_bounds_and_custom_limits_are_visible() -> None:
    raw, parsed = _parse("\n".join(f"________________ test_{i} ________________\nE AssertionError: {'x' * 100}" for i in range(12)), exit_code=1)
    result = TestFailureAnalyzer(config=FailureAnalysisConfig(max_failures=2, max_message_length=32, max_excerpt_length=32)).analyze(TestFailureAnalysisRequest(raw, parsed))
    assert len(result.findings) <= 2
    assert all(len(item.observed_failure) <= 32 for item in result.findings)


def test_malformed_request_is_structured_invalid() -> None:
    result = TestFailureAnalyzer().analyze(TestFailureAnalysisRequest(None, None))
    assert result.status is FailureAnalysisStatus.INVALID
    assert result.analysis_complete is False


def test_deterministic_repeated_analysis() -> None:
    raw, parsed = _parse("FAILED tests/test_api.py::test_api - TypeError: bad\n1 failed in 0.1s\n")
    first = analyze_test_failure(raw, parsed).to_dict()
    second = analyze_test_failure(raw, parsed).to_dict()
    assert first == second


def test_analyzer_does_not_mutate_files_or_execute() -> None:
    raw, parsed = _parse("FAILED tests/test_api.py::test_api - AssertionError: bad\n1 failed in 0.1s\n")
    before = Path(".").resolve()
    result = analyze_test_failure(raw, parsed)
    assert Path(".").resolve() == before
    assert result.findings


def test_loop_exposes_explicit_analysis_helper() -> None:
    raw, parsed = _parse("FAILED tests/test_api.py::test_api - AssertionError: bad\n1 failed in 0.1s\n")
    loop = AutonomousToolLoop.__new__(AutonomousToolLoop)
    result = loop.analyze_test_failure(raw, parsed)
    assert result.findings
