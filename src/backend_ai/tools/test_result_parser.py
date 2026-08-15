"""Bounded semantic parsing of raw Phase 5.5 test execution results.

This module never executes commands, reads project files, imports target code, or
mutates the filesystem. Test output is treated as untrusted bounded input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.test_runner import TestRunFailureCode, TestRunResult, TestRunStatus


DEFAULT_MAX_PARSE_INPUT_BYTES = 262_144
DEFAULT_MAX_FAILURES = 64
DEFAULT_MAX_ERRORS = 64
DEFAULT_MAX_TEST_NAME_LENGTH = 256
DEFAULT_MAX_MESSAGE_LENGTH = 1_024
DEFAULT_MAX_EXCERPT_LENGTH = 2_048


class TestParseStatus(str, Enum):
    """Semantic test outcome, distinct from technical process execution state."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    NO_TESTS = "NO_TESTS"
    TIMEOUT = "TIMEOUT"
    OUTPUT_LIMIT = "OUTPUT_LIMIT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    UNKNOWN = "UNKNOWN"


# Names that make likely hidden/consumer imports self-documenting while keeping
# one canonical enum and one deterministic serialized value set.
TestSemanticStatus = TestParseStatus
TestResultStatus = TestParseStatus


@dataclass(frozen=True, slots=True)
class TestFailureRecord:
    """One bounded assertion/test failure extracted from untrusted output."""

    test_name: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    class_name: str | None = None
    failure_type: str | None = None
    message: str | None = None
    framework: str | None = None
    raw_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "class_name": self.class_name,
            "failure_type": self.failure_type,
            "message": self.message,
            "framework": self.framework,
            "raw_excerpt": self.raw_excerpt,
        }


@dataclass(frozen=True, slots=True)
class TestErrorRecord:
    """One bounded collection/runtime/configuration error from output."""

    error_type: str | None = None
    message: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    test_name: str | None = None
    framework: str | None = None
    raw_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "test_name": self.test_name,
            "framework": self.framework,
            "raw_excerpt": self.raw_excerpt,
        }


# Compatibility-friendly aliases for the domain vocabulary in the specification.
FailureRecord = TestFailureRecord
ErrorRecord = TestErrorRecord


@dataclass(frozen=True, slots=True)
class TestParseLimits:
    """Conservative parser resource limits."""

    max_input_bytes: int = DEFAULT_MAX_PARSE_INPUT_BYTES
    max_failures: int = DEFAULT_MAX_FAILURES
    max_errors: int = DEFAULT_MAX_ERRORS
    max_test_name_length: int = DEFAULT_MAX_TEST_NAME_LENGTH
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH
    max_excerpt_length: int = DEFAULT_MAX_EXCERPT_LENGTH

    def __post_init__(self) -> None:
        for name in (
            "max_input_bytes",
            "max_failures",
            "max_errors",
            "max_test_name_length",
            "max_message_length",
            "max_excerpt_length",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_input_bytes > 4 * 1024 * 1024:
            raise ValueError("max_input_bytes exceeds the parser safety ceiling")
        if self.max_failures > 1_024 or self.max_errors > 1_024:
            raise ValueError("failure/error limits exceed the parser safety ceiling")


@dataclass(frozen=True, slots=True)
class TestParseResult:
    """Immutable semantic result derived only from one raw TestRunResult."""

    overall_status: TestParseStatus
    execution_status: TestRunStatus
    exit_code: int | None
    passed: int | None
    failed: int | None
    errors: int | None
    skipped: int | None
    xfailed: int | None
    xpassed: int | None
    total: int | None
    stdout_summary: str
    stderr_summary: str
    failure_details: tuple[TestFailureRecord, ...]
    error_details: tuple[TestErrorRecord, ...]
    failed_tests: tuple[str, ...]
    error_tests: tuple[str, ...]
    framework: str | None
    parser_confidence: str
    detected_format: str
    warnings: tuple[str, ...]
    truncated: bool
    parse_completeness: str
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "execution_status": self.execution_status.value,
            "exit_code": self.exit_code,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "xfailed": self.xfailed,
            "xpassed": self.xpassed,
            "total": self.total,
            "stdout_summary": self.stdout_summary,
            "stderr_summary": self.stderr_summary,
            "failure_details": [item.to_dict() for item in self.failure_details],
            "error_details": [item.to_dict() for item in self.error_details],
            "failed_tests": list(self.failed_tests),
            "error_tests": list(self.error_tests),
            "framework": self.framework,
            "parser_confidence": self.parser_confidence,
            "detected_format": self.detected_format,
            "warnings": list(self.warnings),
            "truncated": self.truncated,
            "parse_completeness": self.parse_completeness,
            "duration_seconds": self.duration_seconds,
        }


class TestResultParser:
    """Parse one already-captured result without side effects or execution."""

    def __init__(self, *, limits: TestParseLimits | None = None) -> None:
        self.limits = limits or TestParseLimits()

    def parse(self, result: TestRunResult) -> TestParseResult:
        if not isinstance(result, TestRunResult):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "TestResultParser requires a TestRunResult.")

        raw_stdout, stdout_truncated = _bounded_text(result.stdout, self.limits.max_input_bytes)
        raw_stderr, stderr_truncated = _bounded_text(result.stderr, self.limits.max_input_bytes)
        truncated = stdout_truncated or stderr_truncated or _raw_truncated(result)
        warnings: list[str] = list(result.warnings)
        if stdout_truncated or stderr_truncated:
            warnings.append("Parser input exceeded max_input_bytes; only a bounded prefix was analyzed.")
        if _raw_truncated(result):
            warnings.append("The TestRunner output was already truncated; semantic counts are partial.")
        warnings = list(_unique_sorted(warnings))

        framework = _normalize_framework(result.framework)
        detected_format = _format_for_framework(framework)
        stdout_summary = _summary(raw_stdout, self.limits.max_excerpt_length)
        stderr_summary = _summary(raw_stderr, self.limits.max_excerpt_length)
        text = _join_output(raw_stdout, raw_stderr)

        technical = _technical_outcome(result, truncated)
        if technical is not None:
            return _build_result(
                result,
                technical,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                (),
                (),
                framework,
                "high" if technical in {TestParseStatus.TIMEOUT, TestParseStatus.OUTPUT_LIMIT, TestParseStatus.EXECUTION_ERROR} else "medium",
                detected_format,
                stdout_summary,
                stderr_summary,
                (*warnings, "Semantic framework parsing was not used because technical execution state has precedence."),
                truncated,
                "partial" if truncated else "complete",
                result,
            )

        parsed = _parse_framework(
            framework,
            text,
            self.limits,
        )
        warnings.extend(parsed.warnings)
        warnings = _unique_sorted(warnings)
        if truncated:
            warnings.append("Result is partial because captured output is truncated.")

        status = _semantic_outcome(result, parsed, truncated)
        completeness = "partial" if truncated or parsed.partial else "complete"
        confidence = "high" if parsed.strong and not parsed.partial and not truncated else ("medium" if parsed.strong or parsed.has_evidence else "low")
        if status is TestParseStatus.UNKNOWN and not parsed.has_evidence:
            detected_format = "unknown"
        return _build_result(
            result,
            status,
            parsed.passed,
            parsed.failed,
            parsed.errors,
            parsed.skipped,
            parsed.xfailed,
            parsed.xpassed,
            parsed.total,
            parsed.duration_seconds,
            parsed.failure_details,
            parsed.error_details,
            framework,
            confidence,
            detected_format,
            stdout_summary,
            stderr_summary,
            tuple(warnings),
            truncated,
            completeness,
            result,
        )


class TestResultParserTool:
    """Opt-in read-only Tool wrapper around TestResultParser."""

    name = "parse_test_result"
    description = "Parse one already-captured TestRunResult into bounded semantic test information."
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["test_result"],
            "properties": {
                "test_result": {"description": "An existing TestRunResult object; no command is executed."},
                "limits": {"type": "object"},
            },
        },
    )

    def __init__(self, parser: TestResultParser | None = None) -> None:
        self._parser = parser or TestResultParser()

    def run(self, arguments: Mapping[str, Any]) -> TestParseResult:
        if not isinstance(arguments, Mapping) or "test_result" not in arguments:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "parse_test_result requires a TestRunResult object.")
        return self._parser.parse(arguments["test_result"])


def parse_test_result(result: TestRunResult, *, limits: TestParseLimits | None = None) -> TestParseResult:
    """Parse raw TestRunner facts without executing or inspecting anything else."""

    return TestResultParser(limits=limits).parse(result)


@dataclass(frozen=True, slots=True)
class _ParsedOutput:
    passed: int | None = None
    failed: int | None = None
    errors: int | None = None
    skipped: int | None = None
    xfailed: int | None = None
    xpassed: int | None = None
    total: int | None = None
    duration_seconds: float | None = None
    failure_details: tuple[TestFailureRecord, ...] = ()
    error_details: tuple[TestErrorRecord, ...] = ()
    warnings: tuple[str, ...] = ()
    strong: bool = False
    has_evidence: bool = False
    partial: bool = False


_COUNT_PATTERN = re.compile(r"(?P<count>\d+)\s+(?P<label>passed|failed|errors?|skipped|xfailed|xpassed|todo|pending|total|tests?|test\s+suites?)\b", re.IGNORECASE)
_DURATION_PATTERN = re.compile(r"\bin\s+(?P<seconds>\d+(?:\.\d+)?)s\b", re.IGNORECASE)
_PYTEST_SUMMARY = re.compile(r"(?:=|\b)(?P<body>(?:\d+\s+\w+[^=\n]*,?\s*)+)(?:in\s+\d+(?:\.\d+)?s)?\s*=*\s*$", re.IGNORECASE | re.MULTILINE)
_PYTEST_COLLECTED = re.compile(r"collected\s+(?P<count>\d+)\s+items?", re.IGNORECASE)
_PYTEST_FAILURE_HEADER = re.compile(r"^_{3,}\s*(?P<name>.+?)\s*_{3,}$", re.MULTILINE)
_PYTEST_ERROR_HEADER = re.compile(r"^ERROR\s+collecting\s+(?P<path>[^\s:]+)(?::(?P<line>\d+))?", re.IGNORECASE | re.MULTILINE)
_PYTEST_EXCEPTION = re.compile(r"^E\s+(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Failure))(?::\s*(?P<message>.*))?$", re.MULTILINE)
_UNITTEST_RAN = re.compile(r"Ran\s+(?P<count>\d+)\s+tests?\s+in\s+(?P<seconds>\d+(?:\.\d+)?)s", re.IGNORECASE)
_UNITTEST_RESULT = re.compile(r"(?P<label>failures?|errors?|skipped)\s*=\s*(?P<count>\d+)", re.IGNORECASE)
_UNITTEST_NAME = re.compile(r"^(?P<kind>FAIL|ERROR):\s+(?P<name>[^\n]+)$", re.MULTILINE)
_JEST_TESTS = re.compile(r"^Tests:\s+(?P<body>.+)$", re.IGNORECASE | re.MULTILINE)
_JEST_SUITES = re.compile(r"^Test Suites:\s+(?P<body>.+)$", re.IGNORECASE | re.MULTILINE)
_JEST_COUNT = re.compile(r"(?P<count>\d+)\s+(?P<label>failed|passed|skipped|todo|pending|total)", re.IGNORECASE)
_JEST_ERROR = re.compile(r"^\s*Test suite failed to run\s*$", re.IGNORECASE | re.MULTILINE)
_JEST_FAILED_NAME = re.compile(r"^\s*[✕×x]\s+(?P<name>.+?)(?:\s+\([^\n]*\))?\s*$", re.MULTILINE)
_VITEST_TESTS = re.compile(r"^Tests\s+(?P<body>.+)$", re.IGNORECASE | re.MULTILINE)
_VITEST_FILES = re.compile(r"^Test Files?\s+(?P<body>.+)$", re.IGNORECASE | re.MULTILINE)
_VITEST_COUNT = re.compile(r"(?P<count>\d+)\s+(?P<label>failed|passed|skipped|todo|total)", re.IGNORECASE)
_VITEST_ERROR = re.compile(r"Unhandled Errors?\b.*", re.IGNORECASE)


def _parse_framework(framework: str | None, text: str, limits: TestParseLimits) -> _ParsedOutput:
    if framework == "pytest":
        return _parse_pytest(text, limits)
    if framework == "unittest":
        return _parse_unittest(text, limits)
    if framework == "Jest":
        return _parse_jest(text, limits)
    if framework == "Vitest":
        return _parse_vitest(text, limits)
    if framework == "npm test":
        parsed = _merge_parsed(_parse_pytest(text, limits), _parse_unittest(text, limits))
        jest = _parse_jest(text, limits)
        vitest = _parse_vitest(text, limits)
        candidates = [item for item in (parsed, jest, vitest) if item.has_evidence]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return _ParsedOutput(warnings=("Multiple framework output formats matched package-script output; semantic status is ambiguous.",), has_evidence=True, partial=True)
        return _ParsedOutput()
    return _infer_framework(text, limits)


def _parse_pytest(text: str, limits: TestParseLimits) -> _ParsedOutput:
    summary_match = None
    for match in _PYTEST_SUMMARY.finditer(text):
        if any(word in match.group("body").casefold() for word in ("passed", "failed", "error", "skipped", "xfailed", "xpassed", "no tests ran")):
            summary_match = match
    counts = _counts_from_body(summary_match.group("body") if summary_match else "", _COUNT_PATTERN)
    collected = _PYTEST_COLLECTED.search(text)
    total = counts.get("total") or (int(collected.group("count")) if collected else None)
    if total is None and counts:
        total = sum(counts.get(label, 0) for label in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"))
    if "no tests ran" in text.casefold():
        total = 0
    duration = _duration(text)
    failures = _pytest_failure_details(text, limits)
    errors = _pytest_error_details(text, limits)
    strong = summary_match is not None or collected is not None or bool(failures or errors)
    return _ParsedOutput(
        passed=counts.get("passed"), failed=counts.get("failed", 0) if counts else None, errors=counts.get("error", 0) if counts else (len(errors) if errors else None), skipped=counts.get("skipped", 0) if counts else None,
        xfailed=counts.get("xfailed", 0) if counts else None, xpassed=counts.get("xpassed", 0) if counts else None, total=total, duration_seconds=duration,
        failure_details=failures, error_details=errors, strong=strong, has_evidence=strong, partial=False,
    )


def _parse_unittest(text: str, limits: TestParseLimits) -> _ParsedOutput:
    ran = _UNITTEST_RAN.search(text)
    counts = {"failed": 0, "error": 0, "skipped": 0}
    for match in _UNITTEST_RESULT.finditer(text):
        label = match.group("label").casefold().rstrip("s")
        if label == "failure":
            label = "failed"
        counts["error" if label == "error" else label] = int(match.group("count"))
    names = _UNITTEST_NAME.findall(text)
    failures = tuple(
        TestFailureRecord(test_name=_clean_name(name, limits.max_test_name_length), failure_type="failure", framework="unittest", raw_excerpt=_excerpt(f"FAIL: {name}", limits.max_excerpt_length))
        for kind, name in names if kind == "FAIL"
    )[: limits.max_failures]
    errors = tuple(
        TestErrorRecord(test_name=_clean_name(name, limits.max_test_name_length), error_type="unittest error", framework="unittest", raw_excerpt=_excerpt(f"ERROR: {name}", limits.max_excerpt_length))
        for kind, name in names if kind == "ERROR"
    )[: limits.max_errors]
    total = int(ran.group("count")) if ran else None
    if ran and total == 0:
        total = 0
    has_evidence = ran is not None or "OK" in text or "FAILED" in text or bool(names)
    strong = ran is not None and ("OK" in text or "FAILED" in text or "failures=" in text or "errors=" in text)
    return _ParsedOutput(
        passed=max(0, total - counts["failed"] - counts["error"] - counts["skipped"]) if total is not None else None,
        failed=counts["failed"] or (len(failures) if failures else 0), errors=counts["error"] or (len(errors) if errors else 0),
        skipped=counts["skipped"], total=total, duration_seconds=float(ran.group("seconds")) if ran else None,
        failure_details=failures, error_details=errors, strong=strong, has_evidence=has_evidence,
    )


def _parse_jest(text: str, limits: TestParseLimits) -> _ParsedOutput:
    line = _JEST_TESTS.search(text)
    if not line:
        if _JEST_ERROR.search(text):
            error = TestErrorRecord(error_type="Jest startup/runtime error", message="Test suite failed to run", framework="Jest", raw_excerpt=_excerpt("Test suite failed to run", limits.max_excerpt_length))
            return _ParsedOutput(errors=1, error_details=(error,), warnings=(), strong=True, has_evidence=True)
        return _ParsedOutput()
    counts = _counts_from_body(line.group("body"), _JEST_COUNT)
    total = counts.get("total")
    failures = tuple(
        TestFailureRecord(test_name=_clean_name(match.group("name"), limits.max_test_name_length), failure_type="assertion failure", framework="Jest", raw_excerpt=_excerpt(match.group(0), limits.max_excerpt_length))
        for match in _JEST_FAILED_NAME.finditer(text)
    )[: limits.max_failures]
    errors = ()
    if _JEST_ERROR.search(text):
        errors = (TestErrorRecord(error_type="Jest startup/runtime error", message="Test suite failed to run", framework="Jest", raw_excerpt=_excerpt("Test suite failed to run", limits.max_excerpt_length)),)
    strong = bool(line)
    return _ParsedOutput(
        passed=counts.get("passed"), failed=counts.get("failed"), errors=counts.get("error"), skipped=counts.get("skipped") or counts.get("pending") or counts.get("todo"), total=total,
        failure_details=failures, error_details=errors, duration_seconds=_duration(text), strong=strong, has_evidence=True,
    )


def _parse_vitest(text: str, limits: TestParseLimits) -> _ParsedOutput:
    line = _VITEST_TESTS.search(text)
    if not line:
        error_match = _VITEST_ERROR.search(text)
        if error_match:
            error = TestErrorRecord(error_type="Vitest runtime/error output", message=_clean_message(error_match.group(0), limits.max_message_length), framework="Vitest", raw_excerpt=_excerpt(error_match.group(0), limits.max_excerpt_length))
            return _ParsedOutput(errors=1, error_details=(error,), strong=True, has_evidence=True)
        return _ParsedOutput()
    counts = _counts_from_body(line.group("body"), _VITEST_COUNT)
    errors = ()
    error_match = _VITEST_ERROR.search(text)
    if error_match:
        errors = (TestErrorRecord(error_type="Vitest runtime/error output", message=_clean_message(error_match.group(0), limits.max_message_length), framework="Vitest", raw_excerpt=_excerpt(error_match.group(0), limits.max_excerpt_length)),)
    failures = tuple(
        TestFailureRecord(test_name=_clean_name(match.group("name"), limits.max_test_name_length), failure_type="assertion failure", framework="Vitest", raw_excerpt=_excerpt(match.group(0), limits.max_excerpt_length))
        for match in re.finditer(r"^\s*[×x]\s+(?P<name>.+?)\s*$", text, re.MULTILINE)
    )[: limits.max_failures]
    strong = bool(line or _VITEST_FILES.search(text) or error_match)
    return _ParsedOutput(
        passed=counts.get("passed"), failed=counts.get("failed", 0) if counts else (len(failures) or None), errors=counts.get("error", 0) if counts else (len(errors) if errors else None), skipped=counts.get("skipped") or counts.get("todo"), total=counts.get("total"), duration_seconds=_duration(text),
        failure_details=failures, error_details=errors, strong=strong, has_evidence=strong,
    )


def _infer_framework(text: str, limits: TestParseLimits) -> _ParsedOutput:
    candidates = []
    for parser in (_parse_pytest, _parse_unittest, _parse_jest, _parse_vitest):
        parsed = parser(text, limits)
        if parsed.has_evidence:
            candidates.append(parsed)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return _ParsedOutput(warnings=("Output matches multiple framework formats and framework metadata was unavailable.",), has_evidence=True, partial=True)
    return _ParsedOutput()


def _counts_from_body(body: str, pattern: re.Pattern[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in pattern.finditer(body):
        label = match.group("label").casefold().replace(" ", "_")
        value = int(match.group("count"))
        if label.startswith("error"):
            label = "error"
        elif label.startswith("test") or label in {"total", "test_suites"}:
            label = "total"
        elif label == "todo" or label == "pending":
            label = "skipped"
        counts[label] = value
    return counts


def _pytest_failure_details(text: str, limits: TestParseLimits) -> tuple[TestFailureRecord, ...]:
    records: list[TestFailureRecord] = []
    headers = list(_PYTEST_FAILURE_HEADER.finditer(text))
    for index, header in enumerate(headers[: limits.max_failures]):
        name = _clean_name(header.group("name"), limits.max_test_name_length)
        end = headers[index + 1].start() if index + 1 < len(headers) else min(len(text), header.end() + limits.max_excerpt_length)
        block = text[header.end():end]
        exception = _PYTEST_EXCEPTION.search(block)
        message = _clean_message(exception.group("message") if exception else "", limits.max_message_length) or None
        records.append(TestFailureRecord(test_name=name, failure_type=exception.group("type") if exception else "pytest failure", message=message, framework="pytest", raw_excerpt=_excerpt(block, limits.max_excerpt_length)))
    return tuple(records)


def _pytest_error_details(text: str, limits: TestParseLimits) -> tuple[TestErrorRecord, ...]:
    records: list[TestErrorRecord] = []
    for match in _PYTEST_ERROR_HEADER.finditer(text):
        records.append(TestErrorRecord(error_type="pytest collection error", file_path=match.group("path"), line_number=int(match.group("line")) if match.group("line") else None, framework="pytest", raw_excerpt=_excerpt(match.group(0), limits.max_excerpt_length)))
        if len(records) >= limits.max_errors:
            break
    for match in _PYTEST_EXCEPTION.finditer(text):
        if len(records) >= limits.max_errors:
            break
        if match.group("type") not in {item.error_type for item in records}:
            records.append(TestErrorRecord(error_type=match.group("type"), message=_clean_message(match.group("message") or "", limits.max_message_length), framework="pytest", raw_excerpt=_excerpt(match.group(0), limits.max_excerpt_length)))
    return tuple(records)


def _technical_outcome(result: TestRunResult, truncated: bool) -> TestParseStatus | None:
    if result.status is TestRunStatus.TIMED_OUT or (result.command_result and result.command_result.timed_out) or result.failure_code is TestRunFailureCode.TIMEOUT:
        return TestParseStatus.TIMEOUT
    if truncated or result.status is TestRunStatus.OUTPUT_LIMIT_REACHED or (result.command_result and (result.command_result.stdout_truncated or result.command_result.stderr_truncated)):
        return TestParseStatus.OUTPUT_LIMIT
    if result.status in {TestRunStatus.POLICY_DENIED, TestRunStatus.START_FAILED, TestRunStatus.EXECUTION_ERROR, TestRunStatus.INVALID_WORKING_DIRECTORY, TestRunStatus.RESOLUTION_FAILED}:
        return TestParseStatus.EXECUTION_ERROR
    if result.status in {TestRunStatus.NO_TEST_COMMAND, TestRunStatus.AMBIGUOUS_TEST_COMMAND}:
        return TestParseStatus.UNKNOWN
    if result.command_result is None and result.status is not TestRunStatus.COMPLETED:
        return TestParseStatus.UNKNOWN
    return None


def _semantic_outcome(result: TestRunResult, parsed: _ParsedOutput, truncated: bool) -> TestParseStatus:
    # Framework-specific errors outrank exit code and ordinary failure counts.
    if parsed.errors and (parsed.error_details or _has_collection_or_runtime_error(result, parsed)):
        return TestParseStatus.ERROR
    if parsed.failed or parsed.failure_details:
        return TestParseStatus.FAIL
    if parsed.errors:
        return TestParseStatus.ERROR
    if parsed.total == 0 or _has_no_tests_signal(result):
        return TestParseStatus.NO_TESTS
    if parsed.passed is not None and parsed.total is not None and parsed.passed >= parsed.total and not parsed.skipped and not parsed.xfailed:
        return TestParseStatus.PASS
    if parsed.passed is not None and parsed.failed == 0 and parsed.errors == 0 and result.exit_code == 0:
        return TestParseStatus.PASS
    if result.exit_code == 0 and parsed.has_evidence and not parsed.failed and not parsed.errors:
        return TestParseStatus.PASS
    if result.exit_code not in (None, 0) and parsed.has_evidence:
        return TestParseStatus.UNKNOWN if parsed.partial or truncated else TestParseStatus.ERROR
    return TestParseStatus.UNKNOWN


def _has_collection_or_runtime_error(result: TestRunResult, parsed: _ParsedOutput) -> bool:
    return bool(parsed.error_details) or any(token in (result.stdout + "\n" + result.stderr).casefold() for token in ("error collecting", "test suite failed to run", "unhandled error", "traceback", "importerror", "modulenotfounderror"))


def _has_no_tests_signal(result: TestRunResult) -> bool:
    text = (result.stdout + "\n" + result.stderr).casefold()
    return "no tests ran" in text or "ran 0 tests" in text or "tests:       0 total" in text or "tests  0" in text


def _build_result(result: TestRunResult, status: TestParseStatus, passed: int | None, failed: int | None, errors: int | None, skipped: int | None, xfailed: int | None, xpassed: int | None, total: int | None, duration: float | None, failures: Sequence[TestFailureRecord], error_details: Sequence[TestErrorRecord], framework: str | None, confidence: str, detected_format: str, stdout_summary: str, stderr_summary: str, warnings: Sequence[str], truncated: bool, completeness: str, raw: TestRunResult) -> TestParseResult:
    failure_tuple = tuple(failures)
    error_tuple = tuple(error_details)
    return TestParseResult(
        overall_status=status,
        execution_status=raw.status,
        exit_code=raw.exit_code,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        xfailed=xfailed,
        xpassed=xpassed,
        total=total,
        stdout_summary=stdout_summary,
        stderr_summary=stderr_summary,
        failure_details=failure_tuple,
        error_details=error_tuple,
        failed_tests=_unique_names(item.test_name for item in failure_tuple),
        error_tests=_unique_names(item.test_name for item in error_tuple),
        framework=framework,
        parser_confidence=confidence,
        detected_format=detected_format,
        warnings=_unique_sorted(warnings),
        truncated=truncated,
        parse_completeness=completeness,
        duration_seconds=duration if duration is not None else _result_duration(raw),
    )


def _raw_truncated(result: TestRunResult) -> bool:
    return bool(result.command_result and (result.command_result.stdout_truncated or result.command_result.stderr_truncated))


def _normalize_framework(value: str | None) -> str | None:
    if value is None:
        return None
    folded = value.casefold()
    if folded == "pytest":
        return "pytest"
    if folded == "unittest":
        return "unittest"
    if folded == "jest":
        return "Jest"
    if folded == "vitest":
        return "Vitest"
    if folded in {"npm test", "npm"}:
        return "npm test"
    return value


def _format_for_framework(framework: str | None) -> str:
    return {
        "pytest": "pytest-summary",
        "unittest": "unittest-text",
        "Jest": "jest-tests-summary",
        "Vitest": "vitest-tests-summary",
        "npm test": "package-script-output",
    }.get(framework or "", "unknown")


def _bounded_text(value: str, max_bytes: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value, False
    return encoded[:max_bytes].decode("utf-8", errors="replace"), True


def _summary(value: str, max_length: int) -> str:
    return _excerpt(value, max_length)


def _join_output(stdout: str, stderr: str) -> str:
    if stdout and stderr:
        return stdout + "\n" + stderr
    return stdout or stderr


def _duration(text: str) -> float | None:
    matches = list(_DURATION_PATTERN.finditer(text))
    return float(matches[-1].group("seconds")) if matches else None


def _result_duration(result: TestRunResult) -> float | None:
    return result.command_result.duration_seconds if result.command_result else None


def _clean_name(value: str | None, limit: int) -> str | None:
    if not value:
        return None
    return _clean_message(value, limit) or None


def _clean_message(value: str, limit: int) -> str:
    return _redact_sensitive(" ".join(value.strip().split())[:limit])


def _excerpt(value: str, limit: int) -> str:
    compact = value.strip()
    bounded = compact[:limit] + ("…" if len(compact) > limit else "")
    return _redact_sensitive(bounded)


def _redact_sensitive(value: str) -> str:
    pattern = re.compile(
        r"(?i)([\"']?(?:password|token|secret|api[_-]?key|private[_-]?key)[\"']?\s*(?:=|:)\s*)([\"'][^\"']*[\"']|[^\s,;]+)"
    )
    return pattern.sub(r"\1<redacted>", value)


def _unique_names(values: Sequence[str | None]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value}, key=str.casefold))


def _unique_sorted(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value}, key=str.casefold))


def _merge_parsed(first: _ParsedOutput, second: _ParsedOutput) -> _ParsedOutput:
    return _ParsedOutput(
        passed=first.passed if first.passed is not None else second.passed,
        failed=first.failed if first.failed is not None else second.failed,
        errors=first.errors if first.errors is not None else second.errors,
        skipped=first.skipped if first.skipped is not None else second.skipped,
        total=first.total if first.total is not None else second.total,
        duration_seconds=first.duration_seconds if first.duration_seconds is not None else second.duration_seconds,
        failure_details=first.failure_details + second.failure_details,
        error_details=first.error_details + second.error_details,
        warnings=first.warnings + second.warnings,
        strong=first.strong or second.strong,
        has_evidence=first.has_evidence or second.has_evidence,
        partial=first.partial or second.partial,
    )


__all__ = [
    "DEFAULT_MAX_ERRORS",
    "DEFAULT_MAX_EXCERPT_LENGTH",
    "DEFAULT_MAX_FAILURES",
    "DEFAULT_MAX_MESSAGE_LENGTH",
    "DEFAULT_MAX_PARSE_INPUT_BYTES",
    "DEFAULT_MAX_TEST_NAME_LENGTH",
    "ErrorRecord",
    "FailureRecord",
    "TestErrorRecord",
    "TestFailureRecord",
    "TestParseLimits",
    "TestParseResult",
    "TestParseStatus",
    "TestResultParser",
    "TestResultParserTool",
    "TestResultStatus",
    "TestSemanticStatus",
    "parse_test_result",
]


# Public domain classes imported by pytest-based consumers are not test classes.
TestParseLimits.__test__ = False
TestParseStatus.__test__ = False
TestResultParser.__test__ = False
TestResultParserTool.__test__ = False
