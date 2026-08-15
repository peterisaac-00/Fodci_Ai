"""Pure, bounded analysis of already parsed test failures for Phase 7.2."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Sequence

from backend_ai.tools.test_result_parser import (
    TestErrorRecord,
    TestFailureRecord,
    TestParseResult,
    TestParseStatus,
)
from backend_ai.tools.test_runner import TestRunResult, TestRunStatus


class FailureAnalysisStatus(str, Enum):
    ANALYZED = "ANALYZED"
    NO_FAILURE = "NO_FAILURE"
    INCOMPLETE = "INCOMPLETE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class FailureClassification(str, Enum):
    ASSERTION_FAILURE = "ASSERTION_FAILURE"
    EXCEPTION = "EXCEPTION"
    IMPORT_ERROR = "IMPORT_ERROR"
    MODULE_NOT_FOUND = "MODULE_NOT_FOUND"
    TYPE_ERROR = "TYPE_ERROR"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    ROUTING_API_FAILURE = "ROUTING_API_FAILURE"
    FIXTURE_FAILURE = "FIXTURE_FAILURE"
    TEST_DISCOVERY_FAILURE = "TEST_DISCOVERY_FAILURE"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    TIMEOUT = "TIMEOUT"
    OUTPUT_LIMIT = "OUTPUT_LIMIT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class FailureSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FailureConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class FailureLocationKind(str, Enum):
    TEST_LOCATION = "TEST_LOCATION"
    SUSPECTED_IMPLEMENTATION_LOCATION = "SUSPECTED_IMPLEMENTATION_LOCATION"
    FRAMEWORK_LOCATION = "FRAMEWORK_LOCATION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class FailureAnalysisConfig:
    max_input_bytes: int = 262_144
    max_failures: int = 64
    max_groups: int = 32
    max_traceback_length: int = 4_096
    max_excerpt_length: int = 2_048
    max_related_failures: int = 16
    max_chain_length: int = 8
    max_path_length: int = 512
    max_message_length: int = 1_024

    def __post_init__(self) -> None:
        for name in ("max_input_bytes", "max_failures", "max_groups", "max_traceback_length", "max_excerpt_length", "max_related_failures", "max_chain_length", "max_path_length", "max_message_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_input_bytes > 4 * 1024 * 1024 or self.max_failures > 1_024 or self.max_groups > 256:
            raise ValueError("failure analysis limit exceeds safety ceiling")


@dataclass(frozen=True, slots=True)
class FailureLocation:
    kind: FailureLocationKind
    file_path: str | None = None
    line_number: int | None = None
    symbol: str | None = None
    framework_component: str | None = None
    confidence: FailureConfidence = FailureConfidence.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "file_path": self.file_path, "line_number": self.line_number, "symbol": self.symbol, "framework_component": self.framework_component, "confidence": self.confidence.value}


@dataclass(frozen=True, slots=True)
class FailureEvidence:
    source: str
    detail: str
    value: str | None = None
    strength: FailureConfidence = FailureConfidence.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "detail": self.detail, "value": self.value, "strength": self.strength.value}


@dataclass(frozen=True, slots=True)
class FailureFinding:
    finding_id: str
    classification: FailureClassification
    severity: FailureSeverity
    confidence: FailureConfidence
    observed_failure: str
    location: FailureLocation
    evidence: tuple[FailureEvidence, ...] = ()
    diagnostic_chain: tuple[str, ...] = ()
    test_name: str | None = None
    exception_type: str | None = None
    assertion_message: str | None = None
    is_primary: bool = False
    is_derived: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"finding_id": self.finding_id, "classification": self.classification.value, "severity": self.severity.value, "confidence": self.confidence.value, "observed_failure": self.observed_failure, "location": self.location.to_dict(), "evidence": [item.to_dict() for item in self.evidence], "diagnostic_chain": list(self.diagnostic_chain), "test_name": self.test_name, "exception_type": self.exception_type, "assertion_message": self.assertion_message, "is_primary": self.is_primary, "is_derived": self.is_derived}


@dataclass(frozen=True, slots=True)
class FailureGroup:
    group_id: str
    classification: FailureClassification
    representative_finding_id: str
    related_finding_ids: tuple[str, ...]
    shared_evidence: tuple[FailureEvidence, ...]
    confidence: FailureConfidence
    suspected_common_location: FailureLocation | None = None
    primary_failure_id: str | None = None
    derived_failure_ids: tuple[str, ...] = ()
    causal_inference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"group_id": self.group_id, "classification": self.classification.value, "representative_finding_id": self.representative_finding_id, "related_finding_ids": list(self.related_finding_ids), "shared_evidence": [item.to_dict() for item in self.shared_evidence], "confidence": self.confidence.value, "suspected_common_location": self.suspected_common_location.to_dict() if self.suspected_common_location else None, "primary_failure_id": self.primary_failure_id, "derived_failure_ids": list(self.derived_failure_ids), "causal_inference": self.causal_inference}


@dataclass(frozen=True, slots=True)
class TestFailureAnalysisRequest:
    test_result: TestRunResult | None
    parsed_result: TestParseResult | None
    config: FailureAnalysisConfig = field(default_factory=FailureAnalysisConfig)


@dataclass(frozen=True, slots=True)
class TestFailureAnalysis:
    status: FailureAnalysisStatus
    classification: FailureClassification | None
    findings: tuple[FailureFinding, ...] = ()
    groups: tuple[FailureGroup, ...] = ()
    primary_failure_id: str | None = None
    warnings: tuple[str, ...] = ()
    truncated: bool = False
    analysis_complete: bool = True
    parser_confidence: str = "unknown"
    parser_completeness: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "classification": self.classification.value if self.classification else None, "findings": [item.to_dict() for item in self.findings], "groups": [item.to_dict() for item in self.groups], "primary_failure_id": self.primary_failure_id, "warnings": list(self.warnings), "truncated": self.truncated, "analysis_complete": self.analysis_complete, "parser_confidence": self.parser_confidence, "parser_completeness": self.parser_completeness}


class TestFailureAnalyzer:
    """Analyze parser evidence only; never executes, reads, mutates, or calls a model."""

    def __init__(self, *, config: FailureAnalysisConfig | None = None) -> None:
        self.config = config or FailureAnalysisConfig()

    def analyze(self, request: TestFailureAnalysisRequest) -> TestFailureAnalysis:
        if not isinstance(request, TestFailureAnalysisRequest) or not isinstance(request.parsed_result, TestParseResult):
            return TestFailureAnalysis(FailureAnalysisStatus.INVALID, None, warnings=("TestFailureAnalyzer requires a valid TestParseResult.",), analysis_complete=False)
        parsed = request.parsed_result
        warnings = list(parsed.warnings)
        truncated = bool(parsed.truncated)
        if request.test_result is not None and not isinstance(request.test_result, TestRunResult):
            warnings.append("TestRunResult metadata was malformed and was not used.")
        if parsed.overall_status is TestParseStatus.PASS:
            return TestFailureAnalysis(FailureAnalysisStatus.NO_FAILURE, None, warnings=tuple(_unique(warnings)), truncated=truncated, analysis_complete=not truncated, parser_confidence=parsed.parser_confidence, parser_completeness=parsed.parse_completeness)
        if parsed.overall_status is TestParseStatus.NO_TESTS:
            return TestFailureAnalysis(FailureAnalysisStatus.INSUFFICIENT_EVIDENCE, FailureClassification.TEST_DISCOVERY_FAILURE, warnings=tuple(_unique((*warnings, "No test failures were available because no tests were discovered."))), truncated=truncated, analysis_complete=False, parser_confidence=parsed.parser_confidence, parser_completeness=parsed.parse_completeness)
        if parsed.overall_status in {TestParseStatus.TIMEOUT, TestParseStatus.OUTPUT_LIMIT, TestParseStatus.EXECUTION_ERROR}:
            classification = {TestParseStatus.TIMEOUT: FailureClassification.TIMEOUT, TestParseStatus.OUTPUT_LIMIT: FailureClassification.OUTPUT_LIMIT, TestParseStatus.EXECUTION_ERROR: FailureClassification.EXECUTION_ERROR}[parsed.overall_status]
            finding = self._technical_finding(classification, parsed)
            return TestFailureAnalysis(FailureAnalysisStatus.ANALYZED, classification, (finding,), (FailureGroup("group-1", classification, finding.finding_id, (finding.finding_id,), finding.evidence, finding.confidence),), finding.finding_id, tuple(_unique(warnings)), truncated, not truncated, parsed.parser_confidence, parsed.parse_completeness)
        records: list[tuple[str, TestFailureRecord | TestErrorRecord]] = [("failure", item) for item in parsed.failure_details[: self.config.max_failures]]
        records.extend(("error", item) for item in parsed.error_details[: max(0, self.config.max_failures - len(records))])
        if not records and parsed.failed_tests:
            records.extend(("failure", TestFailureRecord(test_name=name, message=parsed.stdout_summary, framework=parsed.framework, raw_excerpt=parsed.stdout_summary)) for name in parsed.failed_tests[: self.config.max_failures])
        if not records and parsed.error_tests:
            records.extend(("error", TestErrorRecord(test_name=name, message=parsed.stdout_summary, framework=parsed.framework, raw_excerpt=parsed.stdout_summary)) for name in parsed.error_tests[: self.config.max_failures])
        if not records and parsed.overall_status in {TestParseStatus.FAIL, TestParseStatus.ERROR} and (parsed.stdout_summary or parsed.stderr_summary):
            records.append(("failure" if parsed.overall_status is TestParseStatus.FAIL else "error", TestFailureRecord(message=parsed.stdout_summary or parsed.stderr_summary, framework=parsed.framework, raw_excerpt=parsed.stdout_summary or parsed.stderr_summary)))
        if not records:
            return TestFailureAnalysis(FailureAnalysisStatus.INSUFFICIENT_EVIDENCE, FailureClassification.UNKNOWN_FAILURE, warnings=tuple(_unique((*warnings, "Parser reported a non-pass outcome without structured failure records."))), truncated=truncated, analysis_complete=False, parser_confidence=parsed.parser_confidence, parser_completeness=parsed.parse_completeness)
        findings = tuple(self._finding(index + 1, kind, record, parsed) for index, (kind, record) in enumerate(records))
        groups = self._groups(findings)
        primary_id = self._primary(groups, findings)
        findings = tuple(_replace_primary(item, item.finding_id == primary_id, bool(primary_id and item.finding_id != primary_id and any(item.finding_id in group.derived_failure_ids for group in groups))) for item in findings)
        classification = findings[0].classification if len({item.classification for item in findings}) == 1 else None
        status = FailureAnalysisStatus.INCOMPLETE if truncated or parsed.parse_completeness != "complete" else FailureAnalysisStatus.ANALYZED
        return TestFailureAnalysis(status, classification, findings, groups, primary_id, tuple(_unique(warnings)), truncated, not truncated and parsed.parse_completeness == "complete", parsed.parser_confidence, parsed.parse_completeness)

    def _technical_finding(self, classification: FailureClassification, parsed: TestParseResult) -> FailureFinding:
        detail = _redact(_clip(parsed.stderr_summary or parsed.stdout_summary or classification.value, self.config.max_message_length))
        evidence = (FailureEvidence("test_result_parser", "technical execution state", classification.value, FailureConfidence.HIGH), FailureEvidence("execution", "exit code", str(parsed.exit_code) if parsed.exit_code is not None else None, FailureConfidence.HIGH))
        return FailureFinding("finding-1", classification, FailureSeverity.HIGH, FailureConfidence.HIGH, detail, FailureLocation(FailureLocationKind.UNKNOWN), evidence, (classification.value, detail))

    def _finding(self, index: int, kind: str, record: TestFailureRecord | TestErrorRecord, parsed: TestParseResult) -> FailureFinding:
        message = _record_message(record)
        safe_message = _redact(message)
        classification = _classify(kind, record, message, parsed)
        exception = _redact(getattr(record, "failure_type", None) or getattr(record, "error_type", None) or "") or None
        safe_test_name = _redact(getattr(record, "test_name", None) or "") or None
        location = FailureLocation(FailureLocationKind.TEST_LOCATION if getattr(record, "file_path", None) or getattr(record, "test_name", None) else FailureLocationKind.UNKNOWN, _clip_path(getattr(record, "file_path", None), self.config.max_path_length), getattr(record, "line_number", None), _redact(getattr(record, "class_name", None) or getattr(record, "test_name", None) or "") or None, parsed.framework, _confidence_for_record(record, parsed))
        evidence: list[FailureEvidence] = [FailureEvidence("test_result_parser", "structured parser record", _clip(safe_message, self.config.max_message_length), _confidence_for_record(record, parsed))]
        if safe_test_name: evidence.append(FailureEvidence("test_name", "failing test identifier", safe_test_name, FailureConfidence.HIGH))
        if getattr(record, "file_path", None): evidence.append(FailureEvidence("location", "parser-provided file location", _clip_path(record.file_path, self.config.max_path_length), FailureConfidence.HIGH))
        if getattr(record, "raw_excerpt", ""): evidence.append(FailureEvidence("output_excerpt", "bounded parser excerpt", _redact(_clip(record.raw_excerpt, self.config.max_excerpt_length)), FailureConfidence.MEDIUM))
        chain = tuple(_clip(item, self.config.max_message_length) for item in (safe_test_name, exception, safe_message) if item)[: self.config.max_chain_length]
        return FailureFinding(f"finding-{index}", classification, _severity(classification), _confidence_for_record(record, parsed), _clip(safe_message or classification.value, self.config.max_message_length), location, tuple(evidence), chain, safe_test_name, exception, _clip(safe_message, self.config.max_message_length) or None)

    def _groups(self, findings: Sequence[FailureFinding]) -> tuple[FailureGroup, ...]:
        buckets: dict[tuple[FailureClassification, str], list[FailureFinding]] = {}
        for item in findings:
            key = (item.classification, _group_key(item.observed_failure))
            buckets.setdefault(key, []).append(item)
        groups: list[FailureGroup] = []
        for index, ((classification, _), members) in enumerate(sorted(buckets.items(), key=lambda pair: (pair[0][0].value, pair[0][1]))):
            members = members[: self.config.max_related_failures]
            representative = members[0]
            related = tuple(item.finding_id for item in members)
            derived = tuple(item.finding_id for item in members[1:])
            groups.append(FailureGroup(f"group-{index + 1}", classification, representative.finding_id, related, representative.evidence[:4], _group_confidence(members), representative.location if len({item.location.file_path for item in members}) == 1 else None, representative.finding_id if len(members) >= 2 else None, derived, len(members) >= 2))
            if len(groups) >= self.config.max_groups: break
        return tuple(groups)

    def _primary(self, groups: Sequence[FailureGroup], findings: Sequence[FailureFinding]) -> str | None:
        if not groups: return None
        candidates = [group for group in groups if len(group.related_finding_ids) >= 2 and group.classification in {FailureClassification.IMPORT_ERROR, FailureClassification.MODULE_NOT_FOUND, FailureClassification.DEPENDENCY_ERROR, FailureClassification.CONFIGURATION_ERROR, FailureClassification.DATABASE_ERROR, FailureClassification.CONNECTION_ERROR, FailureClassification.ENVIRONMENT_FAILURE}]
        if not candidates: return None
        return sorted(candidates, key=lambda group: (-len(group.related_finding_ids), group.group_id))[0].representative_finding_id


def analyze_test_failure(test_result: TestRunResult | None, parsed_result: TestParseResult | None, *, config: FailureAnalysisConfig | None = None) -> TestFailureAnalysis:
    return TestFailureAnalyzer(config=config).analyze(TestFailureAnalysisRequest(test_result, parsed_result, config or FailureAnalysisConfig()))


def _classify(kind: str, record: TestFailureRecord | TestErrorRecord, message: str, parsed: TestParseResult) -> FailureClassification:
    text = " ".join(str(item or "") for item in (getattr(record, "failure_type", None), getattr(record, "error_type", None), message, getattr(record, "raw_excerpt", ""), parsed.stdout_summary, parsed.stderr_summary)).casefold()
    if "modulenotfounderror" in text or "no module named" in text or "cannot find module" in text: return FailureClassification.MODULE_NOT_FOUND
    if "importerror" in text or "import error" in text or "cannot import" in text: return FailureClassification.IMPORT_ERROR
    if kind == "failure" and any(token in text for token in ("assertionerror", "assert ", "expected", "actual")): return FailureClassification.ASSERTION_FAILURE
    patterns = ((FailureClassification.TYPE_ERROR, ("typeerror",)), (FailureClassification.SYNTAX_ERROR, ("syntaxerror", "syntax error")), (FailureClassification.FIXTURE_FAILURE, ("fixture", "setup_teardown")), (FailureClassification.AUTHENTICATION_FAILURE, ("401", "403", "unauthorized", "forbidden", "authentication", "jwt", "token")), (FailureClassification.ROUTING_API_FAILURE, ("404", "405", "500", "http", "route", "endpoint", "status code")), (FailureClassification.DATABASE_ERROR, ("database", "sql", "postgres", "mysql", "sqlite")), (FailureClassification.CONNECTION_ERROR, ("connection", "connectionrefused", "timed out", "connect")), (FailureClassification.DEPENDENCY_ERROR, ("dependency", "required package", "distribution not found")), (FailureClassification.CONFIGURATION_ERROR, ("configuration", "config", "settings")), (FailureClassification.TEST_DISCOVERY_FAILURE, ("no tests", "test discovery", "collection")))
    for classification, tokens in patterns:
        if any(token in text for token in tokens): return classification
    return FailureClassification.EXCEPTION if getattr(record, "failure_type", None) or getattr(record, "error_type", None) else FailureClassification.UNKNOWN_FAILURE


def _record_message(record: TestFailureRecord | TestErrorRecord) -> str:
    return str(getattr(record, "message", None) or getattr(record, "raw_excerpt", "") or getattr(record, "failure_type", None) or getattr(record, "error_type", None) or "Observed test failure")


def _confidence_for_record(record: TestFailureRecord | TestErrorRecord, parsed: TestParseResult) -> FailureConfidence:
    if getattr(record, "test_name", None) and getattr(record, "file_path", None) and getattr(record, "line_number", None) and parsed.parser_confidence.casefold() == "high": return FailureConfidence.HIGH
    if getattr(record, "test_name", None) or getattr(record, "failure_type", None) or getattr(record, "error_type", None): return FailureConfidence.MEDIUM
    return FailureConfidence.LOW


def _severity(classification: FailureClassification) -> FailureSeverity:
    if classification in {FailureClassification.TIMEOUT, FailureClassification.OUTPUT_LIMIT, FailureClassification.EXECUTION_ERROR, FailureClassification.ENVIRONMENT_FAILURE}: return FailureSeverity.HIGH
    if classification in {FailureClassification.DATABASE_ERROR, FailureClassification.CONNECTION_ERROR, FailureClassification.AUTHENTICATION_FAILURE, FailureClassification.CONFIGURATION_ERROR}: return FailureSeverity.MEDIUM
    return FailureSeverity.MEDIUM


def _group_confidence(items: Sequence[FailureFinding]) -> FailureConfidence:
    if len(items) >= 2 and all(item.confidence is FailureConfidence.HIGH for item in items): return FailureConfidence.HIGH
    if items and any(item.confidence in {FailureConfidence.HIGH, FailureConfidence.MEDIUM} for item in items): return FailureConfidence.MEDIUM
    return FailureConfidence.LOW


def _group_key(message: str) -> str:
    return re.sub(r"\d+", "#", _redact(message).casefold())[:256]


def _replace_primary(item: FailureFinding, primary: bool, derived: bool) -> FailureFinding:
    return FailureFinding(item.finding_id, item.classification, item.severity, item.confidence, item.observed_failure, item.location, item.evidence, item.diagnostic_chain, item.test_name, item.exception_type, item.assertion_message, primary, derived)


def _redact(value: str) -> str:
    text = str(value).replace("\x00", "")
    patterns = (r"(?i)(password|passwd|token|api[_ -]?key|secret|authorization|private[_ -]?key|credential)(\s*[:=]\s*)([^\s,;]+)", r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", r"(?i)-----BEGIN [^-]+ PRIVATE KEY-----.*?-----END [^-]+ PRIVATE KEY-----")
    text = re.sub(patterns[0], r"\1\2[REDACTED]", text)
    text = re.sub(patterns[1], "Bearer [REDACTED]", text)
    return re.sub(patterns[2], "[REDACTED PRIVATE KEY]", text, flags=re.DOTALL)


def _clip(value: str, limit: int) -> str:
    value = str(value)
    return value if len(value) <= limit else value[: max(0, limit - 14)] + "\n[truncated]"


def _clip_path(value: str | None, limit: int) -> str | None:
    return _clip(value, limit) if value else None


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if value))


__all__ = ["FailureAnalysisConfig", "FailureAnalysisStatus", "FailureClassification", "FailureConfidence", "FailureEvidence", "FailureFinding", "FailureGroup", "FailureLocation", "FailureLocationKind", "FailureSeverity", "TestFailureAnalysis", "TestFailureAnalysisRequest", "TestFailureAnalyzer", "analyze_test_failure"]
