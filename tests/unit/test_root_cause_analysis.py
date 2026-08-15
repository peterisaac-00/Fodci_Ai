from __future__ import annotations

from pathlib import Path

from backend_ai.agent import (
    CausalStatus,
    FailureAnalysisStatus,
    FailureClassification,
    FailureConfidence,
    FailureEvidence,
    FailureFinding,
    FailureGroup,
    FailureLocation,
    FailureLocationKind,
    RootCauseAnalysisRequest,
    RootCauseAnalysisStatus,
    RootCauseAnalyzer,
    RootCauseConfidence,
    RootCauseLocationKind,
    TestFailureAnalysis,
    RootCauseAnalysisConfig,
    analyze_root_cause,
)
from backend_ai.agent.autonomous_tool_loop import AutonomousToolLoop


def _finding(fid: str, classification: FailureClassification, message: str, *, test_name: str | None = "test_case", path: str | None = "tests/test_api.py", exception: str | None = "ModuleNotFoundError") -> FailureFinding:
    evidence = (FailureEvidence("parser", "structured failure", message, FailureConfidence.HIGH), FailureEvidence("exception", "exception type", exception or "", FailureConfidence.HIGH))
    location = FailureLocation(FailureLocationKind.TEST_LOCATION, path, 42, test_name, "pytest", FailureConfidence.HIGH)
    return FailureFinding(fid, classification,  # type: ignore[arg-type]
        __import__("backend_ai.agent").agent.FailureSeverity.MEDIUM,
        FailureConfidence.HIGH, message, location, evidence, (test_name or "failure", exception or "", message), test_name, exception, message)


def _analysis(*findings: FailureFinding, primary: str | None = None, status: FailureAnalysisStatus = FailureAnalysisStatus.ANALYZED) -> TestFailureAnalysis:
    groups = tuple(FailureGroup(f"g{index}", finding.classification, finding.finding_id, (finding.finding_id,), finding.evidence, FailureConfidence.HIGH) for index, finding in enumerate(findings, 1))
    return TestFailureAnalysis(status, findings[0].classification if findings and len({item.classification for item in findings}) == 1 else None, findings, groups, primary, (), False, True, "high", "complete")


def test_missing_dependency_produces_hypothesis_and_dependency_location() -> None:
    result = analyze_root_cause(_analysis(_finding("f1", FailureClassification.MODULE_NOT_FOUND, "No module named missing")))
    assert result.status in {RootCauseAnalysisStatus.ANALYZED, RootCauseAnalysisStatus.INCONCLUSIVE}
    assert result.hypotheses
    hypothesis = result.hypotheses[0]
    assert hypothesis.location.kind is RootCauseLocationKind.DEPENDENCY
    assert hypothesis.confirmed is False
    assert hypothesis.causal_status is CausalStatus.SECONDARY_CANDIDATE


def test_observed_failure_is_separate_from_hypothesis() -> None:
    result = analyze_root_cause(_analysis(_finding("f1", FailureClassification.ASSERTION_FAILURE, "expected 200 received 401", exception="AssertionError")))
    assert "expected 200" in result.observed_failure
    assert result.hypotheses[0].statement != result.observed_failure
    assert result.hypotheses[0].confirmed is False


def test_authentication_hypothesis_keeps_alternatives() -> None:
    result = analyze_root_cause(_analysis(_finding("f1", FailureClassification.AUTHENTICATION_FAILURE, "HTTP 401 unauthorized")))
    assert result.hypotheses[0].location.kind is RootCauseLocationKind.IMPLEMENTATION
    assert len(result.alternatives) >= 2
    assert all(item.confidence is RootCauseConfidence.LOW for item in result.alternatives)


def test_database_and_connection_locations_are_distinct() -> None:
    database = analyze_root_cause(_analysis(_finding("f1", FailureClassification.DATABASE_ERROR, "database connection failed")))
    connection = analyze_root_cause(_analysis(_finding("f2", FailureClassification.CONNECTION_ERROR, "connection refused")))
    assert database.hypotheses[0].location.kind is RootCauseLocationKind.DATABASE
    assert connection.hypotheses[0].location.kind is RootCauseLocationKind.EXTERNAL_SERVICE


def test_primary_and_derived_failures_are_inference_only() -> None:
    first = _finding("f1", FailureClassification.MODULE_NOT_FOUND, "No module named package", exception="ModuleNotFoundError")
    second = _finding("f2", FailureClassification.MODULE_NOT_FOUND, "No module named package", exception="ModuleNotFoundError")
    analysis = _analysis(first, second, primary="f1")
    result = analyze_root_cause(analysis)
    assert result.hypotheses
    assert any(item.causal_status is CausalStatus.PRIMARY_CANDIDATE for item in result.hypotheses)
    assert all(item.confirmed is False for item in result.hypotheses)


def test_causal_relations_are_bounded_and_inferred() -> None:
    result = analyze_root_cause(_analysis(_finding("f1", FailureClassification.AUTHENTICATION_FAILURE, "401 unauthorized")), config=RootCauseAnalysisConfig(max_causal_depth=2))
    assert result.causal_relations
    assert all(item.inferred for item in result.causal_relations)
    assert all(len(item.evidence_ids) <= 3 for item in result.causal_relations)


def test_causal_depth_truncation_is_explicit() -> None:
    result = analyze_root_cause(_analysis(_finding("f1", FailureClassification.AUTHENTICATION_FAILURE, "401 unauthorized")), config=RootCauseAnalysisConfig(max_causal_depth=1))
    assert result.causal_chain_truncated
    assert result.hypotheses[0].causal_chain_truncated


def test_unknown_classification_is_inconclusive() -> None:
    result = analyze_root_cause(_analysis(_finding("f1", FailureClassification.UNKNOWN_FAILURE, "generic failure", exception=None)))
    assert result.status is RootCauseAnalysisStatus.INCONCLUSIVE
    assert result.hypotheses[0].location.kind is RootCauseLocationKind.UNKNOWN
    assert result.hypotheses[0].confidence in {RootCauseConfidence.LOW, RootCauseConfidence.UNKNOWN}


def test_missing_failure_analysis_is_invalid() -> None:
    result = RootCauseAnalyzer().analyze(RootCauseAnalysisRequest(None))
    assert result.status is RootCauseAnalysisStatus.INVALID
    assert result.analysis_complete is False


def test_empty_findings_are_insufficient() -> None:
    result = analyze_root_cause(TestFailureAnalysis(FailureAnalysisStatus.INSUFFICIENT_EVIDENCE, FailureClassification.UNKNOWN_FAILURE))
    assert result.status is RootCauseAnalysisStatus.INSUFFICIENT_EVIDENCE
    assert result.evidence_complete is False


def test_no_failure_produces_no_failure() -> None:
    result = analyze_root_cause(TestFailureAnalysis(FailureAnalysisStatus.NO_FAILURE, None))
    assert result.status is RootCauseAnalysisStatus.NO_FAILURE
    assert result.hypotheses == ()


def test_sensitive_evidence_is_not_reintroduced_as_raw_diagnosis() -> None:
    result = analyze_root_cause(_analysis(_finding("f1", FailureClassification.CONFIGURATION_ERROR, "password=[REDACTED] token=[REDACTED]", exception="ConfigError")))
    rendered = str(result.to_dict())
    assert "supersecret" not in rendered
    assert result.hypotheses[0].confirmed is False


def test_contradicting_evidence_is_preserved_separately() -> None:
    failure = _finding("f1", FailureClassification.AUTHENTICATION_FAILURE, "401 unauthorized")
    contradiction = __import__("backend_ai.agent").agent.RootCauseEvidence("contradiction-1", __import__("backend_ai.agent").agent.RootCauseEvidenceType.PROJECT_CONTEXT, "ProjectContext", "another auth test succeeds", RootCauseConfidence.MEDIUM, "structured context", ("f1",), True)
    result = analyze_root_cause(_analysis(failure), evidence=(contradiction,))
    assert result.hypotheses[0].contradicting_evidence
    assert result.hypotheses[0].supporting_evidence


def test_alternatives_are_explicit_and_not_rejected_without_evidence() -> None:
    result = analyze_root_cause(_analysis(_finding("f1", FailureClassification.ASSERTION_FAILURE, "wrong value", exception="AssertionError")))
    assert result.alternatives
    assert all(item.why_possible for item in result.alternatives)


def test_deterministic_repeated_analysis() -> None:
    analysis = _analysis(_finding("f1", FailureClassification.DEPENDENCY_ERROR, "dependency missing", exception="DependencyError"))
    assert analyze_root_cause(analysis).to_dict() == analyze_root_cause(analysis).to_dict()


def test_loop_exposes_explicit_root_cause_helper() -> None:
    loop = AutonomousToolLoop.__new__(AutonomousToolLoop)
    result = loop.analyze_root_cause(_analysis(_finding("f1", FailureClassification.FIXTURE_FAILURE, "fixture failed", exception="FixtureError")))
    assert result.hypotheses
