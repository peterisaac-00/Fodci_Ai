"""Manual Phase 5.6 parser smoke checks over bounded synthetic raw results."""

from __future__ import annotations

from pathlib import Path
import sys

from backend_ai.tools import (
    CommandResult,
    TestParseStatus,
    TestRunFailureCode,
    TestRunPlan,
    TestRunResult,
    TestRunStatus,
    parse_test_result,
)


def _raw(framework: str, stdout: str, *, exit_code: int = 0, status: TestRunStatus = TestRunStatus.COMPLETED, failure_code: TestRunFailureCode | None = None, truncated: bool = False) -> TestRunResult:
    command = CommandResult(
        argv=("python", "-m", "test"),
        working_directory=".",
        lifecycle="timed_out" if status is TestRunStatus.TIMED_OUT else "completed",
        exit_code=exit_code,
        stdout=stdout,
        stderr="",
        duration_seconds=0.1,
        timed_out=status is TestRunStatus.TIMED_OUT,
        started=True,
        completed=True,
        succeeded=exit_code == 0 and not truncated and status is not TestRunStatus.TIMED_OUT,
        start_failed=False,
        stdout_truncated=truncated,
        stderr_truncated=False,
        stdout_utf8_valid=True,
        stderr_utf8_valid=True,
        termination="timeout_graceful" if status is TestRunStatus.TIMED_OUT else "normal",
        error_code=None,
        error_message=None,
        warnings=(),
        process_state="CLEANED_UP",
        lifecycle_history=("REQUESTED", "VALIDATING", "STARTING", "RUNNING", "COMPLETED", "CLEANED_UP"),
        termination_attempted=status is TestRunStatus.TIMED_OUT,
        killed=False,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=0,
    )
    plan = TestRunPlan(("python", "-m", "test"), ".", framework, "smoke", ("local fixture",), "high", True)
    return TestRunResult(status, plan, command, None, failure_code, "python", framework, (framework,), ("local fixture",), ())


def main() -> None:
    cases = (
        (_raw("pytest", "3 passed in 0.1s\n"), TestParseStatus.PASS),
        (_raw("pytest", "1 failed, 1 passed in 0.1s\n", exit_code=1, failure_code=TestRunFailureCode.NONZERO_EXIT), TestParseStatus.FAIL),
        (_raw("unittest", "Ran 2 tests in 0.1s\n\nOK\n"), TestParseStatus.PASS),
        (_raw("Jest", "Tests: 2 passed, 2 total\n"), TestParseStatus.PASS),
        (_raw("Vitest", "Tests 2 passed | 2 total\n"), TestParseStatus.PASS),
        (_raw("pytest", "", status=TestRunStatus.TIMED_OUT, failure_code=TestRunFailureCode.TIMEOUT), TestParseStatus.TIMEOUT),
        (_raw("pytest", "3 passed\n", status=TestRunStatus.OUTPUT_LIMIT_REACHED, failure_code=TestRunFailureCode.OUTPUT_LIMIT_REACHED, truncated=True), TestParseStatus.OUTPUT_LIMIT),
    )
    for raw, expected in cases:
        parsed = parse_test_result(raw)
        assert parsed.overall_status is expected, (parsed.to_dict(), expected)
        assert parsed.to_dict() == parse_test_result(raw).to_dict()

    actual_root = Path(__file__).resolve().parents[1]
    actual = TestRunResult(
        status=TestRunStatus.COMPLETED,
        plan=None,
        command_result=None,
        decision=None,
        failure_code=None,
        project_type="python",
        framework=None,
        frameworks=(),
        evidence=(str(actual_root),),
        warnings=(),
    )
    assert parse_test_result(actual).overall_status is TestParseStatus.UNKNOWN
    print("Phase 5.6 test-result-parser smoke passed")


if __name__ == "__main__":
    main()
