"""Bounded evidence-backed root-cause hypotheses for Phase 7.3."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from backend_ai.agent.test_failure_analysis import (
    FailureClassification,
    FailureConfidence,
    FailureEvidence,
    FailureFinding,
    FailureGroup,
    FailureLocation,
    FailureLocationKind,
    TestFailureAnalysis,
    FailureAnalysisStatus,
)
from backend_ai.tools.project_context import ProjectContext


class RootCauseAnalysisStatus(str, Enum):
    ANALYZED = "ANALYZED"
    NO_FAILURE = "NO_FAILURE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INCONCLUSIVE = "INCONCLUSIVE"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class RootCauseConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class CausalStatus(str, Enum):
    PRIMARY_CANDIDATE = "PRIMARY_CANDIDATE"
    SECONDARY_CANDIDATE = "SECONDARY_CANDIDATE"
    CONTRIBUTING_FACTOR = "CONTRIBUTING_FACTOR"
    ALTERNATIVE = "ALTERNATIVE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REJECTED = "REJECTED"


class RootCauseLocationKind(str, Enum):
    TEST = "TEST"
    IMPLEMENTATION = "IMPLEMENTATION"
    CONFIGURATION = "CONFIGURATION"
    DEPENDENCY = "DEPENDENCY"
    FIXTURE = "FIXTURE"
    ENVIRONMENT = "ENVIRONMENT"
    DATABASE = "DATABASE"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    UNKNOWN = "UNKNOWN"


class RootCauseEvidenceType(str, Enum):
    TEST_FAILURE = "TEST_FAILURE"
    TRACEBACK = "TRACEBACK"
    ASSERTION = "ASSERTION"
    EXCEPTION = "EXCEPTION"
    ERROR_MESSAGE = "ERROR_MESSAGE"
    FAILURE_PATTERN = "FAILURE_PATTERN"
    SHARED_FAILURE = "SHARED_FAILURE"
    PROJECT_CONTEXT = "PROJECT_CONTEXT"
    LOCATION = "LOCATION"
    EXECUTION_STATE = "EXECUTION_STATE"
    PARSER_EVIDENCE = "PARSER_EVIDENCE"
    CONFIGURATION_EVIDENCE = "CONFIGURATION_EVIDENCE"
    DEPENDENCY_EVIDENCE = "DEPENDENCY_EVIDENCE"
    CONTRADICTING_EVIDENCE = "CONTRADICTING_EVIDENCE"


@dataclass(frozen=True, slots=True)
class RootCauseAnalysisConfig:
    max_input_bytes: int = 262_144
    max_hypotheses: int = 16
    max_alternatives: int = 16
    max_evidence_items: int = 64
    max_causal_depth: int = 4
    max_chain_length: int = 8
    max_message_length: int = 1_024
    max_context_items: int = 64

    def __post_init__(self) -> None:
        for name in ("max_input_bytes", "max_hypotheses", "max_alternatives", "max_evidence_items", "max_causal_depth", "max_chain_length", "max_message_length", "max_context_items"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_input_bytes > 4 * 1024 * 1024 or self.max_hypotheses > 256 or self.max_alternatives > 256:
            raise ValueError("root-cause analysis limit exceeds safety ceiling")


@dataclass(frozen=True, slots=True)
class RootCauseLocation:
    kind: RootCauseLocationKind
    file_path: str | None = None
    line_number: int | None = None
    symbol: str | None = None
    component: str | None = None
    confidence: RootCauseConfidence = RootCauseConfidence.UNKNOWN
    inferred: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "file_path": self.file_path, "line_number": self.line_number, "symbol": self.symbol, "component": self.component, "confidence": self.confidence.value, "inferred": self.inferred}


@dataclass(frozen=True, slots=True)
class RootCauseEvidence:
    evidence_id: str
    evidence_type: RootCauseEvidenceType
    source: str
    description: str
    strength: RootCauseConfidence
    provenance: str
    related_finding_ids: tuple[str, ...] = ()
    contradicts: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, "evidence_type": self.evidence_type.value, "source": self.source, "description": self.description, "strength": self.strength.value, "provenance": self.provenance, "related_finding_ids": list(self.related_finding_ids), "contradicts": self.contradicts}


@dataclass(frozen=True, slots=True)
class AlternativeCause:
    alternative_id: str
    statement: str
    location: RootCauseLocation
    evidence: tuple[RootCauseEvidence, ...]
    confidence: RootCauseConfidence
    why_possible: str

    def to_dict(self) -> dict[str, Any]:
        return {"alternative_id": self.alternative_id, "statement": self.statement, "location": self.location.to_dict(), "evidence": [item.to_dict() for item in self.evidence], "confidence": self.confidence.value, "why_possible": self.why_possible}


@dataclass(frozen=True, slots=True)
class CausalRelation:
    relation_id: str
    from_node: str
    to_node: str
    relation: str
    evidence_ids: tuple[str, ...]
    confidence: RootCauseConfidence
    inferred: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"relation_id": self.relation_id, "from_node": self.from_node, "to_node": self.to_node, "relation": self.relation, "evidence_ids": list(self.evidence_ids), "confidence": self.confidence.value, "inferred": self.inferred}


@dataclass(frozen=True, slots=True)
class RootCauseHypothesis:
    hypothesis_id: str
    statement: str
    classification: FailureClassification
    location: RootCauseLocation
    mechanism: tuple[str, ...]
    supporting_evidence: tuple[RootCauseEvidence, ...]
    contradicting_evidence: tuple[RootCauseEvidence, ...]
    affected_failure_ids: tuple[str, ...]
    confidence: RootCauseConfidence
    evidence_strength: RootCauseConfidence
    causal_status: CausalStatus
    confirmed: bool = False
    causal_chain_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id, "statement": self.statement, "classification": self.classification.value, "location": self.location.to_dict(), "mechanism": list(self.mechanism), "supporting_evidence": [item.to_dict() for item in self.supporting_evidence], "contradicting_evidence": [item.to_dict() for item in self.contradicting_evidence], "affected_failure_ids": list(self.affected_failure_ids), "confidence": self.confidence.value, "evidence_strength": self.evidence_strength.value, "causal_status": self.causal_status.value, "confirmed": self.confirmed, "causal_chain_truncated": self.causal_chain_truncated}


@dataclass(frozen=True, slots=True)
class RootCauseAnalysisRequest:
    failure_analysis: TestFailureAnalysis | None
    project_context: ProjectContext | None = None
    evidence: tuple[RootCauseEvidence, ...] = ()
    config: RootCauseAnalysisConfig = field(default_factory=RootCauseAnalysisConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence[: self.config.max_evidence_items]))


@dataclass(frozen=True, slots=True)
class RootCauseAnalysis:
    status: RootCauseAnalysisStatus
    observed_failure: str
    hypotheses: tuple[RootCauseHypothesis, ...] = ()
    alternatives: tuple[AlternativeCause, ...] = ()
    causal_relations: tuple[CausalRelation, ...] = ()
    supporting_evidence: tuple[RootCauseEvidence, ...] = ()
    contradicting_evidence: tuple[RootCauseEvidence, ...] = ()
    unknowns: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    truncated: bool = False
    analysis_complete: bool = True
    evidence_complete: bool = True
    causal_chain_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "observed_failure": self.observed_failure, "hypotheses": [item.to_dict() for item in self.hypotheses], "alternatives": [item.to_dict() for item in self.alternatives], "causal_relations": [item.to_dict() for item in self.causal_relations], "supporting_evidence": [item.to_dict() for item in self.supporting_evidence], "contradicting_evidence": [item.to_dict() for item in self.contradicting_evidence], "unknowns": list(self.unknowns), "warnings": list(self.warnings), "truncated": self.truncated, "analysis_complete": self.analysis_complete, "evidence_complete": self.evidence_complete, "causal_chain_truncated": self.causal_chain_truncated}


class RootCauseAnalyzer:
    """Pure bounded causal hypothesis builder; it never inspects or changes a project."""

    def __init__(self, *, config: RootCauseAnalysisConfig | None = None) -> None:
        self.config = config or RootCauseAnalysisConfig()

    def analyze(self, request: RootCauseAnalysisRequest) -> RootCauseAnalysis:
        if not isinstance(request, RootCauseAnalysisRequest) or not isinstance(request.failure_analysis, TestFailureAnalysis):
            return RootCauseAnalysis(RootCauseAnalysisStatus.INVALID, "", warnings=("RootCauseAnalyzer requires a valid TestFailureAnalysis.",), analysis_complete=False, evidence_complete=False)
        failure = request.failure_analysis
        if failure.status is FailureAnalysisStatus.NO_FAILURE:
            return RootCauseAnalysis(RootCauseAnalysisStatus.NO_FAILURE, "No observed test failure.")
        if failure.status in {FailureAnalysisStatus.INVALID, FailureAnalysisStatus.UNAVAILABLE}:
            return RootCauseAnalysis(RootCauseAnalysisStatus.UNAVAILABLE, "", warnings=("Failure analysis evidence is unavailable.",), analysis_complete=False, evidence_complete=False)
        if not failure.findings:
            return RootCauseAnalysis(RootCauseAnalysisStatus.INSUFFICIENT_EVIDENCE, "Observed failure is not structured enough for a causal hypothesis.", unknowns=("failure finding details",), analysis_complete=False, evidence_complete=False)
        evidence = list(request.evidence[: self.config.max_evidence_items])
        hypotheses: list[RootCauseHypothesis] = []
        alternatives: list[AlternativeCause] = []
        relations: list[CausalRelation] = []
        for finding in failure.findings[: self.config.max_hypotheses]:
            support = self._finding_evidence(finding)
            support.extend(evidence[: max(0, self.config.max_evidence_items - len(support))])
            hypothesis = self._hypothesis_for(finding, support, request.project_context)
            hypotheses.append(hypothesis)
            alternatives.extend(self._alternatives_for(finding, hypothesis, request.project_context))
            relations.extend(self._relations_for(hypothesis, support))
        hypotheses = self._dedupe_hypotheses(hypotheses)
        primary_id = failure.primary_failure_id
        if primary_id:
            hypotheses = [self._with_status(item, CausalStatus.PRIMARY_CANDIDATE if primary_id in item.affected_failure_ids else CausalStatus.SECONDARY_CANDIDATE) for item in hypotheses]
        alternatives = alternatives[: self.config.max_alternatives]
        truncated = len(failure.findings) > self.config.max_hypotheses or len(request.evidence) > self.config.max_evidence_items
        status = RootCauseAnalysisStatus.INCONCLUSIVE if any(item.confidence in {RootCauseConfidence.LOW, RootCauseConfidence.UNKNOWN} or item.classification is FailureClassification.UNKNOWN_FAILURE for item in hypotheses) else RootCauseAnalysisStatus.ANALYZED
        unknowns = ("implementation evidence", "runtime causal confirmation", "contradicting evidence") if not request.project_context else ("runtime causal confirmation",)
        return RootCauseAnalysis(status, _observed_summary(failure), tuple(hypotheses), tuple(alternatives), tuple(relations[: self.config.max_evidence_items]), tuple(item for hypothesis in hypotheses for item in hypothesis.supporting_evidence[:4]), tuple(item for hypothesis in hypotheses for item in hypothesis.contradicting_evidence[:4]), unknowns, tuple(failure.warnings), truncated, not truncated and failure.analysis_complete, bool(evidence or any(item.supporting_evidence for item in hypotheses)), any(item.causal_chain_truncated for item in hypotheses))

    def _finding_evidence(self, finding: FailureFinding) -> list[RootCauseEvidence]:
        result: list[RootCauseEvidence] = []
        for index, item in enumerate(finding.evidence[: self.config.max_evidence_items]):
            result.append(RootCauseEvidence(f"evidence-{finding.finding_id}-{index + 1}", RootCauseEvidenceType.PARSER_EVIDENCE, item.source, _clip(item.detail, self.config.max_message_length), _convert_confidence(item.strength), item.source, (finding.finding_id,)))
        result.append(RootCauseEvidence(f"evidence-{finding.finding_id}-failure", RootCauseEvidenceType.TEST_FAILURE, "failure_analysis", _clip(finding.observed_failure, self.config.max_message_length), _convert_confidence(finding.confidence), "TestFailureAnalysis.findings", (finding.finding_id,)))
        if finding.exception_type:
            result.append(RootCauseEvidence(f"evidence-{finding.finding_id}-exception", RootCauseEvidenceType.EXCEPTION, "failure_analysis", _clip(finding.exception_type, self.config.max_message_length), RootCauseConfidence.HIGH, "FailureFinding.exception_type", (finding.finding_id,)))
        return result[: self.config.max_evidence_items]

    def _hypothesis_for(self, finding: FailureFinding, support: Sequence[RootCauseEvidence], context: ProjectContext | None) -> RootCauseHypothesis:
        classification = finding.classification
        statement, location_kind, mechanism, alternatives = _candidate_for(classification)
        context_support = _context_evidence(context, classification) if context else ()
        all_evidence = tuple((*support, *context_support))[: self.config.max_evidence_items]
        supporting = tuple(item for item in all_evidence if not item.contradicts)
        contradicting = tuple(item for item in all_evidence if item.contradicts)
        confidence = _hypothesis_confidence(finding, supporting, context)
        location = RootCauseLocation(location_kind, finding.location.file_path if location_kind is RootCauseLocationKind.TEST else None, finding.location.line_number if location_kind is RootCauseLocationKind.TEST else None, finding.location.symbol if location_kind is RootCauseLocationKind.TEST else None, finding.location.framework_component, confidence, True)
        chain = tuple(mechanism[: self.config.max_causal_depth])
        return RootCauseHypothesis(f"hypothesis-{finding.finding_id}", statement, classification, location, chain, supporting, contradicting, (finding.finding_id,), confidence, confidence, CausalStatus.SECONDARY_CANDIDATE, False, len(mechanism) > self.config.max_causal_depth)

    def _alternatives_for(self, finding: FailureFinding, hypothesis: RootCauseHypothesis, context: ProjectContext | None) -> list[AlternativeCause]:
        statements = _alternatives_for(hypothesis.classification)
        return [AlternativeCause(f"alternative-{finding.finding_id}-{index}", statement, RootCauseLocation(RootCauseLocationKind.UNKNOWN, confidence=RootCauseConfidence.LOW), (), RootCauseConfidence.LOW, "The observed evidence does not exclude this possibility.") for index, statement in enumerate(statements[: self.config.max_alternatives], 1)]

    def _relations_for(self, hypothesis: RootCauseHypothesis, support: Sequence[RootCauseEvidence]) -> list[CausalRelation]:
        nodes = hypothesis.mechanism[: self.config.max_causal_depth]
        return [CausalRelation(f"relation-{hypothesis.hypothesis_id}-{index}", nodes[index], nodes[index + 1], "may_contribute_to", tuple(item.evidence_id for item in support[:3]), hypothesis.confidence, True) for index in range(max(0, len(nodes) - 1))]

    @staticmethod
    def _with_status(hypothesis: RootCauseHypothesis, status: CausalStatus) -> RootCauseHypothesis:
        return RootCauseHypothesis(hypothesis.hypothesis_id, hypothesis.statement, hypothesis.classification, hypothesis.location, hypothesis.mechanism, hypothesis.supporting_evidence, hypothesis.contradicting_evidence, hypothesis.affected_failure_ids, hypothesis.confidence, hypothesis.evidence_strength, status, False, hypothesis.causal_chain_truncated)

    @staticmethod
    def _dedupe_hypotheses(items: Sequence[RootCauseHypothesis]) -> list[RootCauseHypothesis]:
        seen: set[tuple[FailureClassification, str]] = set()
        result: list[RootCauseHypothesis] = []
        for item in items:
            key = (item.classification, item.statement)
            if key not in seen:
                result.append(item)
                seen.add(key)
        return result


def analyze_root_cause(failure_analysis: TestFailureAnalysis | None, project_context: ProjectContext | None = None, evidence: Sequence[RootCauseEvidence] = (), *, config: RootCauseAnalysisConfig | None = None) -> RootCauseAnalysis:
    active = config or RootCauseAnalysisConfig()
    return RootCauseAnalyzer(config=active).analyze(RootCauseAnalysisRequest(failure_analysis, project_context, tuple(evidence), active))


def _candidate_for(classification: FailureClassification) -> tuple[str, RootCauseLocationKind, tuple[str, ...], tuple[str, ...]]:
    mapping = {
        FailureClassification.MODULE_NOT_FOUND: ("A required module or dependency is unavailable to the test runtime.", RootCauseLocationKind.DEPENDENCY, ("module import fails", "application or test setup cannot load dependency", "observed test failure"), ("dependency metadata is incomplete", "environment mismatch")),
        FailureClassification.IMPORT_ERROR: ("Module initialization or import contract is incompatible with the test path.", RootCauseLocationKind.IMPLEMENTATION, ("module import fails", "test setup cannot initialize module", "observed failure"), ("missing dependency", "fixture or configuration failure")),
        FailureClassification.AUTHENTICATION_FAILURE: ("Authentication state is rejected before the expected behavior is observed.", RootCauseLocationKind.IMPLEMENTATION, ("authentication is evaluated", "request is rejected", "assertion observes authentication failure"), ("invalid fixture", "missing configuration", "token generation failure", "environment mismatch")),
        FailureClassification.DATABASE_ERROR: ("Database access or initialization is unavailable on the failing path.", RootCauseLocationKind.DATABASE, ("database access is attempted", "database operation fails", "test observes failure"), ("configuration failure", "fixture failure", "environment or external service")),
        FailureClassification.CONNECTION_ERROR: ("A required connection or external service is unavailable on the failing path.", RootCauseLocationKind.EXTERNAL_SERVICE, ("connection is attempted", "connection fails", "test observes downstream failure"), ("environment mismatch", "configuration failure", "fixture failure")),
        FailureClassification.CONFIGURATION_ERROR: ("Required runtime configuration is absent or inconsistent with the test path.", RootCauseLocationKind.CONFIGURATION, ("configuration is loaded", "required setting is unavailable", "test or startup fails"), ("dependency failure", "fixture failure", "environment mismatch")),
        FailureClassification.DEPENDENCY_ERROR: ("A required dependency is unavailable or incompatible.", RootCauseLocationKind.DEPENDENCY, ("dependency is resolved", "resolution fails", "test path fails"), ("configuration failure", "environment mismatch")),
        FailureClassification.FIXTURE_FAILURE: ("Test fixture setup or teardown prevents the intended test from reaching its assertion.", RootCauseLocationKind.FIXTURE, ("fixture setup runs", "fixture fails", "test cannot execute normally"), ("implementation failure", "configuration failure")),
        FailureClassification.ROUTING_API_FAILURE: ("The request reaches an unexpected routing or API behavior boundary.", RootCauseLocationKind.IMPLEMENTATION, ("request is routed", "response is unexpected", "assertion observes API failure"), ("authentication failure", "fixture failure", "external service")),
        FailureClassification.ASSERTION_FAILURE: ("Observed behavior differs from the test expectation on the reported path.", RootCauseLocationKind.IMPLEMENTATION, ("code path executes", "return value differs", "assertion fails"), ("incorrect expectation", "fixture or configuration failure", "environment mismatch")),
        FailureClassification.TYPE_ERROR: ("An incompatible value or interface is observed on the failing execution path.", RootCauseLocationKind.IMPLEMENTATION, ("function is called", "value/interface is incompatible", "exception is raised"), ("fixture failure", "dependency mismatch")),
        FailureClassification.SYNTAX_ERROR: ("The runtime cannot parse the reported source or configuration input.", RootCauseLocationKind.IMPLEMENTATION, ("source is parsed", "parsing fails", "test setup cannot continue"), ("configuration or generated-source issue",)),
        FailureClassification.TIMEOUT: ("The bounded execution did not complete within the allowed time.", RootCauseLocationKind.ENVIRONMENT, ("operation starts", "completion exceeds bound", "runner stops execution"), ("slow implementation", "external service delay", "deadlock or fixture issue")),
    }
    return mapping.get(classification, ("The observed failure has no sufficiently supported root-cause candidate.", RootCauseLocationKind.UNKNOWN, ("observed failure",), ("implementation", "configuration", "dependency", "environment")))


def _alternatives_for(classification: FailureClassification) -> tuple[str, ...]:
    return _candidate_for(classification)[3]


def _context_evidence(context: ProjectContext | None, classification: FailureClassification) -> tuple[RootCauseEvidence, ...]:
    if context is None: return ()
    values = [context.project_type, context.stack_summary]
    if classification in {FailureClassification.MODULE_NOT_FOUND, FailureClassification.DEPENDENCY_ERROR}:
        values.extend(context.dependency_files)
    if classification is FailureClassification.CONFIGURATION_ERROR:
        values.extend(context.config_files)
    if classification is FailureClassification.DATABASE_ERROR:
        values.extend(item.name for item in context.databases)
    return tuple(RootCauseEvidence(f"context-{index}", RootCauseEvidenceType.PROJECT_CONTEXT, "ProjectContext", _clip(str(value), 512), RootCauseConfidence.MEDIUM, "ProjectContext structured metadata") for index, value in enumerate(values[:8]) if value)


def _hypothesis_confidence(finding: FailureFinding, evidence: Sequence[RootCauseEvidence], context: ProjectContext | None) -> RootCauseConfidence:
    if finding.classification is FailureClassification.UNKNOWN_FAILURE: return RootCauseConfidence.UNKNOWN if not evidence else RootCauseConfidence.LOW
    if finding.confidence is FailureConfidence.HIGH and finding.exception_type and len(evidence) >= 2: return RootCauseConfidence.HIGH
    if finding.confidence in {FailureConfidence.HIGH, FailureConfidence.MEDIUM} and evidence: return RootCauseConfidence.MEDIUM
    return RootCauseConfidence.LOW if evidence else RootCauseConfidence.UNKNOWN


def _convert_confidence(value: FailureConfidence) -> RootCauseConfidence:
    return RootCauseConfidence[value.value] if value.value in RootCauseConfidence.__members__ else RootCauseConfidence.UNKNOWN


def _observed_summary(analysis: TestFailureAnalysis) -> str:
    if analysis.findings:
        return "; ".join(_clip(item.observed_failure, 256) for item in analysis.findings[:4])
    return "Observed test failure is unavailable."


def _clip(value: str, limit: int) -> str:
    value = str(value)
    return value if len(value) <= limit else value[: max(0, limit - 14)] + "\n[truncated]"


__all__ = ["AlternativeCause", "CausalRelation", "CausalStatus", "RootCauseAnalysis", "RootCauseAnalysisConfig", "RootCauseAnalysisRequest", "RootCauseAnalysisStatus", "RootCauseAnalyzer", "RootCauseConfidence", "RootCauseEvidence", "RootCauseEvidenceType", "RootCauseHypothesis", "RootCauseLocation", "RootCauseLocationKind", "analyze_root_cause"]
