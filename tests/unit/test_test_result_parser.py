from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from backend_ai.agent.registry import ToolRegistry
from backend_ai.tools import (
    CommandResult,
    TestParseLimits,
    TestParseStatus,
    TestResultParser,
    TestResultParserTool,
    TestRunFailureCode,
    TestRunPlan,
    TestRunResult,
    TestRunStatus,
    ToolError,
    parse_test_result,
)


def _raw(
    framework: str | None,
    stdout: str = "",
    stderr: str = "",
    *,
    exit_code: int | None = 0,
    status: TestRunStatus = TestRunStatus.COMPLETED,
    failure_code: TestRunFailureCode | None = None,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    timed_out: bool = False,
    start_failed: bool = False,
    warnings: tuple[str, ...] = (),
) -> TestRunResult:
    command = CommandResult(
        argv=("python", "-m", "test"),
        working_directory=".",
        lifecycle="timed_out" if timed_out else ("failed" if start_failed or exit_code not in (None, 0) else "completed"),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.125,
        timed_out=timed_out,
        started=not start_failed,
        completed=not start_failed,
        succeeded=exit_code == 0 and not timed_out and not start_failed and not stdout_truncated and not stderr_truncated,
        start_failed=start_failed,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        stdout_utf8_valid=True,
        stderr_utf8_valid=True,
        termination="timeout_graceful" if timed_out else "normal",
        error_code=None if exit_code in (None, 0) and not timed_out and not start_failed else "PROCESS_ERROR",
        error_message=None,
        warnings=warnings,
        process_state="CLEANED_UP",
        lifecycle_history=("REQUESTED", "VALIDATING", "STARTING", "RUNNING", "COMPLETED", "CLEANED_UP"),
        termination_attempted=timed_out,
        killed=False,
        stdout_bytes=len(stdout.encode("utf-8")),
        stderr_bytes=len(stderr.encode("utf-8")),
    )
    plan = TestRunPlan(
        argv=("python", "-m", "test"),
        working_directory=".",
        framework=framework or "explicit",
        source="test fixture",
        evidence=("fixture evidence",),
        confidence="high",
        explicit=True,
    )
    return TestRunResult(
        status=status,
        plan=plan,
        command_result=command,
        decision=None,
        failure_code=failure_code,
        project_type="python",
        framework=framework,
        frameworks=(framework,) if framework else (),
        evidence=("fixture evidence",),
        warnings=warnings,
    )


def test_pytest_passing_summary_is_pass() -> None:
    result = parse_test_result(_raw("pytest", "============================== 3 passed in 0.12s ==============================\n"))
    assert result.overall_status is TestParseStatus.PASS
    assert result.passed == 3
    assert result.failed == 0
    assert result.total == 3
    assert result.detected_format == "pytest-summary"


def test_pytest_failures_extract_bounded_names_and_messages() -> None:
    output = """
============================= test session starts ==============================
____________________________ test_first ____________________________
E       AssertionError: first bad
____________________________ test_second ____________________________
E       TypeError: second bad
=========================== short test summary info ============================
FAILED tests/test_api.py::test_first - AssertionError: first bad
FAILED tests/test_api.py::test_second - TypeError: second bad
========================= 2 failed, 1 passed in 0.20s ==========================
"""
    result = parse_test_result(_raw("pytest", output, exit_code=1, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert result.overall_status is TestParseStatus.FAIL
    assert result.failed == 2
    assert result.passed == 1
    assert "test_first" in result.failed_tests
    assert len(result.failure_details) <= 64
    assert "first bad" in str(result.failure_details[0].to_dict())


def test_pytest_skipped_xfail_xpass_and_no_tests() -> None:
    output = "========================= 1 passed, 2 skipped, 1 xfailed, 1 xpassed in 0.08s =========================\n"
    result = parse_test_result(_raw("pytest", output))
    assert result.overall_status is TestParseStatus.PASS
    assert result.skipped == 2
    assert result.xfailed == 1
    assert result.xpassed == 1

    no_tests = parse_test_result(_raw("pytest", "============================ no tests ran in 0.01s ============================\n", exit_code=5, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert no_tests.overall_status is TestParseStatus.NO_TESTS
    assert no_tests.total == 0


def test_pytest_collection_error_is_error_not_fail() -> None:
    output = "ERROR collecting tests/test_import.py\nImportError: cannot import name 'missing'\n"
    result = parse_test_result(_raw("pytest", output, exit_code=2, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert result.overall_status is TestParseStatus.ERROR
    assert result.errors is not None and result.errors >= 1
    assert result.error_details[0].file_path == "tests/test_import.py"


def test_unittest_pass_fail_error_skipped_and_no_tests() -> None:
    passed = parse_test_result(_raw("unittest", "Ran 2 tests in 0.010s\n\nOK\n"))
    assert passed.overall_status is TestParseStatus.PASS
    assert passed.total == 2 and passed.passed == 2

    failed = parse_test_result(_raw("unittest", "FAIL: test_invalid (tests.test_api.ApiTests)\n\nRan 2 tests in 0.010s\n\nFAILED (failures=1)\n", exit_code=1, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert failed.overall_status is TestParseStatus.FAIL
    assert failed.failed == 1 and failed.failed_tests

    error = parse_test_result(_raw("unittest", "ERROR: test_setup (tests.test_api.ApiTests)\n\nRan 1 test in 0.010s\n\nFAILED (errors=1)\n", exit_code=1, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert error.overall_status is TestParseStatus.ERROR
    assert error.errors == 1 and error.error_tests

    skipped = parse_test_result(_raw("unittest", "Ran 1 test in 0.010s\n\nOK (skipped=1)\n"))
    assert skipped.overall_status is TestParseStatus.PASS
    assert skipped.skipped == 1

    empty = parse_test_result(_raw("unittest", "Ran 0 tests in 0.000s\n\nOK\n"))
    assert empty.overall_status is TestParseStatus.NO_TESTS


def test_jest_passing_failure_skipped_and_startup_error() -> None:
    passed = parse_test_result(_raw("Jest", "Test Suites: 1 passed, 1 total\nTests: 2 passed, 1 skipped, 3 total\nTime: 0.4 s\n"))
    assert passed.overall_status is TestParseStatus.PASS
    assert passed.total == 3 and passed.passed == 2 and passed.skipped == 1

    failed = parse_test_result(_raw("Jest", "FAIL tests/api.test.js\n  ✕ rejects bad input (5 ms)\nTests: 1 failed, 1 passed, 2 total\n", exit_code=1, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert failed.overall_status is TestParseStatus.FAIL
    assert failed.failed == 1 and failed.failed_tests == ("rejects bad input",)

    startup = parse_test_result(_raw("Jest", "Test suite failed to run\nCannot find module './missing'\n", exit_code=1, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert startup.overall_status is TestParseStatus.ERROR
    assert startup.error_details


def test_vitest_passing_failure_and_error() -> None:
    passed = parse_test_result(_raw("Vitest", "Test Files  1 passed (1)\nTests  2 passed | 1 skipped | 3 total\nDuration 200ms\n"))
    assert passed.overall_status is TestParseStatus.PASS
    assert passed.total == 3

    failed = parse_test_result(_raw("Vitest", "Test Files 1 failed (1)\nTests 1 failed | 1 passed | 2 total\n × rejects input\n", exit_code=1, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert failed.overall_status is TestParseStatus.FAIL
    assert failed.failed == 1

    error = parse_test_result(_raw("Vitest", "Unhandled Errors 1\n", exit_code=1, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert error.overall_status is TestParseStatus.ERROR


def test_technical_precedence_timeout_output_limit_and_execution_error() -> None:
    timeout = parse_test_result(_raw("pytest", "2 passed in 0.1s\n", status=TestRunStatus.TIMED_OUT, failure_code=TestRunFailureCode.TIMEOUT, timed_out=True))
    assert timeout.overall_status is TestParseStatus.TIMEOUT

    limited = parse_test_result(_raw("pytest", "3 passed in 0.1s\n", status=TestRunStatus.OUTPUT_LIMIT_REACHED, failure_code=TestRunFailureCode.OUTPUT_LIMIT_REACHED, stdout_truncated=True))
    assert limited.overall_status is TestParseStatus.OUTPUT_LIMIT
    assert limited.truncated is True and limited.parse_completeness == "partial"

    start_failed = parse_test_result(_raw("pytest", "", status=TestRunStatus.START_FAILED, failure_code=TestRunFailureCode.START_FAILED, start_failed=True, exit_code=None))
    assert start_failed.overall_status is TestParseStatus.EXECUTION_ERROR


def test_conflicting_evidence_does_not_blindly_trust_exit_code() -> None:
    exit_zero_failure = parse_test_result(_raw("pytest", "1 failed, 1 passed in 0.1s\n", exit_code=0))
    assert exit_zero_failure.overall_status is TestParseStatus.FAIL

    nonzero_unknown = parse_test_result(_raw("pytest", "unexpected launcher text\n", exit_code=7, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert nonzero_unknown.overall_status is TestParseStatus.UNKNOWN

    empty = parse_test_result(_raw("pytest", "", exit_code=0))
    assert empty.overall_status is TestParseStatus.UNKNOWN


def test_truncation_and_invalid_utf8_metadata_are_preserved() -> None:
    result = _raw("pytest", "1 passed in 0.1s\n", stdout_truncated=True, warnings=("stdout contained invalid UTF-8 and replacement decoding was used.",))
    parsed = parse_test_result(result)
    assert parsed.truncated is True
    assert parsed.parse_completeness == "partial"
    assert any("invalid UTF-8" in warning for warning in parsed.warnings)


def test_parser_limits_bound_input_failures_names_and_excerpts() -> None:
    output = "\n".join(f"________________ test_{index} ________________\nE AssertionError: {'x' * 100}" for index in range(20))
    limits = TestParseLimits(max_input_bytes=512, max_failures=2, max_errors=2, max_test_name_length=8, max_message_length=12, max_excerpt_length=20)
    parsed = TestResultParser(limits=limits).parse(_raw("pytest", output, exit_code=1, failure_code=TestRunFailureCode.NONZERO_EXIT))
    assert parsed.truncated is True
    assert parsed.parse_completeness == "partial"
    assert len(parsed.failure_details) <= 2
    assert all(len((item.test_name or "")) <= 8 for item in parsed.failure_details)
    assert all(len(item.raw_excerpt) <= 21 for item in parsed.failure_details)


def test_package_script_output_uses_strong_framework_evidence() -> None:
    output = "Test Suites: 1 passed, 1 total\nTests: 1 passed, 1 total\n"
    parsed = parse_test_result(_raw("npm test", output))
    assert parsed.overall_status is TestParseStatus.PASS
    assert parsed.framework == "npm test"
    assert parsed.detected_format == "package-script-output"


def test_determinism_read_only_and_no_subprocess_or_filesystem_access(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = _raw("pytest", "2 passed in 0.1s\n")
    first = parse_test_result(raw).to_dict()
    second = parse_test_result(raw).to_dict()
    assert first == second

    def forbidden(*args, **kwargs):
        raise AssertionError("parser must not execute or read files")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    parsed = parse_test_result(raw)
    assert parsed.overall_status is TestParseStatus.PASS
    assert not list(tmp_path.iterdir())


def test_parser_tool_and_registry_are_opt_in_and_validate_input() -> None:
    raw = _raw("pytest", "1 passed in 0.1s\n")
    tool = TestResultParserTool()
    parsed = tool.run({"test_result": raw})
    assert parsed.overall_status is TestParseStatus.PASS
    assert tool.metadata.name == "parse_test_result"
    with pytest.raises(ToolError):
        tool.run({})
    assert "parse_test_result" not in ToolRegistry.default().names()
    assert "parse_test_result" in ToolRegistry.with_test_result_parsing().names()
    assert "run_tests" not in ToolRegistry.default().names()


# Domain classes imported for assertions are not pytest test classes.
TestParseLimits.__test__ = False
TestParseStatus.__test__ = False
TestResultParser.__test__ = False
TestResultParserTool.__test__ = False
TestRunFailureCode.__test__ = False
TestRunPlan.__test__ = False
TestRunResult.__test__ = False
TestRunStatus.__test__ = False
