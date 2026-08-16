"""Phase 8.4 deterministic comparison of completed evaluation results.

This module consumes existing Phase 8.3 ``EvaluationResult`` objects only. It
never executes benchmarks, tests, commands, or network operations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import math
from types import MappingProxyType
from typing import Any

from backend_ai.evaluation.scoring import EvaluationResult, EvaluationStatus, TaskScore


class ComparisonStatus(str, Enum):
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    EQUIVALENT = "EQUIVALENT"
    IMPROVED_WITH_REGRESSIONS = "IMPROVED_WITH_REGRESSIONS"
    REGRESSION_FREE = "REGRESSION_FREE"
    INCONCLUSIVE = "INCONCLUSIVE"
    INCOMPARABLE = "INCOMPARABLE"


class RegressionSeverity(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ComparisonClassification(str, Enum):
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    UNCHANGED = "UNCHANGED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class EvaluationVersion:
    version_id: str
    agent_version: str
    evaluation_version: str
    scoring_policy_version: str
    benchmark_definition_version: str
    commit_sha: str | None = None
    metadata: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        for name in ("version_id", "agent_version", "evaluation_version", "scoring_policy_version", "benchmark_definition_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must contain text")
        if self.commit_sha is not None and (not isinstance(self.commit_sha, str) or not self.commit_sha.strip()):
            raise ValueError("commit_sha must contain text when supplied")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType({str(k): str(v) for k, v in sorted(self.metadata.items(), key=lambda item: str(item[0]))}))

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class EvaluationSnapshot:
    version: EvaluationVersion
    result: EvaluationResult

    def __post_init__(self) -> None:
        if not isinstance(self.version, EvaluationVersion) or not isinstance(self.result, EvaluationResult):
            raise ValueError("snapshot requires an EvaluationVersion and EvaluationResult")

    @property
    def benchmark_id(self) -> str:
        return self.result.benchmark_score.benchmark_id

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    epsilon: float = 0.01
    max_evidence_ids: int = 32
    require_complete_evidence: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)) or not math.isfinite(self.epsilon) or not 0.0 < self.epsilon <= 0.5:
            raise ValueError("epsilon must be finite, positive, and at most 0.5")
        if isinstance(self.max_evidence_ids, bool) or not isinstance(self.max_evidence_ids, int) or not 0 < self.max_evidence_ids <= 256:
            raise ValueError("max_evidence_ids must be between 1 and 256")
        if not isinstance(self.require_complete_evidence, bool):
            raise ValueError("require_complete_evidence must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class DimensionComparison:
    name: str
    baseline_score: float | None
    candidate_score: float | None
    delta: float | None
    classification: ComparisonClassification
    evidence_status: ComparisonStatus
    evidence_ids: tuple[str, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        for name in ("baseline_score", "candidate_score", "delta"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite when supplied")
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class TaskComparison:
    task_id: str
    baseline_score: float | None
    candidate_score: float | None
    delta: float | None
    baseline_status: str | None
    candidate_status: str | None
    classification: ComparisonClassification
    severity: RegressionSeverity
    dimensions: tuple[DimensionComparison, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class AggregateComparison:
    baseline_score: float | None
    candidate_score: float | None
    delta: float | None
    classification: ComparisonClassification
    task_success_rate: DimensionComparison | None = None
    test_success_rate: DimensionComparison | None = None
    code_quality_score: DimensionComparison | None = None
    efficiency_score: DimensionComparison | None = None
    completed_task_count_delta: int | None = None
    failed_task_count_delta: int | None = None
    blocked_task_count_delta: int | None = None
    incomplete_task_count_delta: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class RegressionFinding:
    finding_id: str
    task_id: str | None
    dimension: str | None
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    classification: ComparisonClassification
    severity: RegressionSeverity
    evidence_ids: tuple[str, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class RegressionSummary:
    finding_count: int
    regression_count: int
    improvement_count: int
    inconclusive_count: int
    highest_severity: RegressionSeverity
    findings: tuple[RegressionFinding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class EvaluationComparisonRequest:
    baseline: EvaluationSnapshot
    candidate: EvaluationSnapshot
    config: ComparisonConfig = ComparisonConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, EvaluationSnapshot) or not isinstance(self.candidate, EvaluationSnapshot):
            raise ValueError("baseline and candidate must be EvaluationSnapshot objects")


@dataclass(frozen=True, slots=True)
class EvaluationComparisonResult:
    baseline: EvaluationSnapshot
    candidate: EvaluationSnapshot
    status: ComparisonStatus
    severity: RegressionSeverity
    aggregate: AggregateComparison | None
    task_comparisons: tuple[TaskComparison, ...]
    findings: RegressionSummary
    incompatibility_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    epsilon: float = 0.01

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_comparisons", tuple(sorted(self.task_comparisons, key=lambda item: item.task_id)))
        object.__setattr__(self, "incompatibility_reasons", tuple(sorted(set(self.incompatibility_reasons))))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


ComparisonResult = EvaluationComparisonResult


class EvaluationRegressionComparator:
    """Compare two explicitly supplied completed evaluation snapshots."""

    def compare(self, request: EvaluationComparisonRequest) -> EvaluationComparisonResult:
        reasons = self._compatibility_reasons(request.baseline, request.candidate)
        if reasons:
            summary = RegressionSummary(0, 0, 0, 1, RegressionSeverity.CRITICAL, ())
            return EvaluationComparisonResult(request.baseline, request.candidate, ComparisonStatus.INCOMPARABLE, RegressionSeverity.CRITICAL, None, (), summary, reasons, (), request.config.epsilon)
        baseline_scores = {item.task_id: item for item in request.baseline.result.benchmark_score.task_scores}
        candidate_scores = {item.task_id: item for item in request.candidate.result.benchmark_score.task_scores}
        if set(baseline_scores) != set(candidate_scores):
            reasons = ("task ID sets differ",)
            summary = RegressionSummary(0, 0, 0, 1, RegressionSeverity.CRITICAL, ())
            return EvaluationComparisonResult(request.baseline, request.candidate, ComparisonStatus.INCOMPARABLE, RegressionSeverity.CRITICAL, None, (), summary, reasons, (), request.config.epsilon)
        comparisons = tuple(self._task(item, candidate_scores[item.task_id], request.config) for item in sorted(baseline_scores.values(), key=lambda value: value.task_id))
        aggregate = self._aggregate(request.baseline.result, request.candidate.result, request.config)
        findings = self._findings(comparisons, aggregate, request.config)
        status, severity = self._overall(comparisons, aggregate, findings, request)
        warnings = () if request.baseline.result.benchmark_score.evidence_completeness == 1.0 and request.candidate.result.benchmark_score.evidence_completeness == 1.0 else ("one or both evaluation results have incomplete evidence",)
        if request.config.require_complete_evidence and warnings:
            status = ComparisonStatus.INCONCLUSIVE
            severity = RegressionSeverity.CRITICAL
        return EvaluationComparisonResult(request.baseline, request.candidate, status, severity, aggregate, comparisons, findings, (), warnings, request.config.epsilon)

    def _compatibility_reasons(self, baseline: EvaluationSnapshot, candidate: EvaluationSnapshot) -> tuple[str, ...]:
        reasons: list[str] = []
        if baseline.benchmark_id != candidate.benchmark_id: reasons.append("benchmark identity differs")
        if baseline.version.evaluation_version != candidate.version.evaluation_version: reasons.append("evaluation version differs")
        if baseline.version.scoring_policy_version != candidate.version.scoring_policy_version: reasons.append("scoring policy version differs")
        if baseline.version.benchmark_definition_version != candidate.version.benchmark_definition_version: reasons.append("benchmark definition version differs")
        baseline_dimensions = tuple(item.name for item in baseline.result.benchmark_score.dimension_scores)
        candidate_dimensions = tuple(item.name for item in candidate.result.benchmark_score.dimension_scores)
        if baseline_dimensions != candidate_dimensions: reasons.append("scoring dimensions differ")
        return tuple(sorted(reasons))

    def _task(self, baseline: TaskScore, candidate: TaskScore, config: ComparisonConfig) -> TaskComparison:
        delta = candidate.final_score - baseline.final_score
        status_class = _classify_status(baseline.status, candidate.status)
        score_class = _classify_delta(delta, config.epsilon)
        classification = status_class or score_class
        severity = RegressionSeverity.HIGH if status_class == ComparisonClassification.REGRESSED and _is_passing(baseline.status) else RegressionSeverity.MEDIUM if classification is ComparisonClassification.REGRESSED else RegressionSeverity.NONE
        dims = tuple(self._dimension(baseline, candidate, name, config) for name in ("task_success", "tests", "code_quality", "efficiency"))
        evidence = tuple(sorted({evidence_id for dimension in baseline.dimensions for evidence_id in dimension.evidence_ids}))[:config.max_evidence_ids]
        return TaskComparison(baseline.task_id, baseline.final_score, candidate.final_score, delta, _status_text(baseline.status), _status_text(candidate.status), classification, severity, dims, evidence, "Status transitions take precedence over score deltas." if status_class else "Classification uses the configured epsilon threshold.")

    def _dimension(self, baseline: TaskScore, candidate: TaskScore, name: str, config: ComparisonConfig) -> DimensionComparison:
        left = next((item for item in baseline.dimensions if item.name == name), None)
        right = next((item for item in candidate.dimensions if item.name == name), None)
        if left is None or right is None: return DimensionComparison(name, None if left is None else left.score, None if right is None else right.score, None, ComparisonClassification.INCONCLUSIVE, ComparisonStatus.INCONCLUSIVE, (), "Dimension evidence is missing.")
        delta = right.score - left.score
        classification = _classify_delta(delta, config.epsilon)
        status = ComparisonStatus.INCONCLUSIVE if left.status in (EvaluationStatus.INCOMPLETE, EvaluationStatus.UNAVAILABLE) or right.status in (EvaluationStatus.INCOMPLETE, EvaluationStatus.UNAVAILABLE) else ComparisonStatus.IMPROVED if classification is ComparisonClassification.IMPROVED else ComparisonStatus.REGRESSED if classification is ComparisonClassification.REGRESSED else ComparisonStatus.EQUIVALENT
        return DimensionComparison(name, left.score, right.score, delta, classification, status, tuple(sorted(set(left.evidence_ids + right.evidence_ids)))[:config.max_evidence_ids], "Dimension delta compared with epsilon.")

    def _aggregate(self, baseline: EvaluationResult, candidate: EvaluationResult, config: ComparisonConfig) -> AggregateComparison:
        left, right = baseline.benchmark_score, candidate.benchmark_score
        dims = {name: self._aggregate_dimension(left.dimension_scores, right.dimension_scores, name, config) for name in ("task_success", "tests", "code_quality", "efficiency")}
        return AggregateComparison(left.aggregate_score, right.aggregate_score, right.aggregate_score - left.aggregate_score, _classify_delta(right.aggregate_score - left.aggregate_score, config.epsilon), dims["task_success"], dims["tests"], dims["code_quality"], dims["efficiency"], right.completed_count - left.completed_count, right.failed_count - left.failed_count, right.blocked_count - left.blocked_count, right.incomplete_count - left.incomplete_count)

    def _aggregate_dimension(self, baseline: Sequence[Any], candidate: Sequence[Any], name: str, config: ComparisonConfig) -> DimensionComparison:
        left, right = next((item for item in baseline if item.name == name), None), next((item for item in candidate if item.name == name), None)
        if left is None or right is None: return DimensionComparison(name, None, None, None, ComparisonClassification.INCONCLUSIVE, ComparisonStatus.INCONCLUSIVE, (), "Aggregate dimension is missing.")
        delta = right.score - left.score
        classification = _classify_delta(delta, config.epsilon)
        return DimensionComparison(name, left.score, right.score, delta, classification, ComparisonStatus.IMPROVED if classification is ComparisonClassification.IMPROVED else ComparisonStatus.REGRESSED if classification is ComparisonClassification.REGRESSED else ComparisonStatus.EQUIVALENT, tuple(sorted(set(left.evidence_ids + right.evidence_ids)))[:config.max_evidence_ids], "Aggregate dimension delta compared with epsilon.")

    def _findings(self, tasks: Sequence[TaskComparison], aggregate: AggregateComparison, config: ComparisonConfig) -> RegressionSummary:
        findings: list[RegressionFinding] = []
        for task in tasks:
            if task.classification is ComparisonClassification.REGRESSED:
                findings.append(RegressionFinding(f"TASK-{task.task_id}", task.task_id, None, task.baseline_score, task.candidate_score, task.delta, task.classification, task.severity, task.evidence_ids, task.explanation))
            for dimension in task.dimensions:
                if dimension.classification is ComparisonClassification.REGRESSED:
                    findings.append(RegressionFinding(f"TASK-{task.task_id}-{dimension.name}", task.task_id, dimension.name, dimension.baseline_score, dimension.candidate_score, dimension.delta, dimension.classification, RegressionSeverity.MEDIUM, dimension.evidence_ids, dimension.explanation))
        task_regression_count = sum(item.classification is ComparisonClassification.REGRESSED and item.task_id is not None and item.dimension is None for item in findings)
        severity = RegressionSeverity.CRITICAL if task_regression_count >= 2 else max((item.severity for item in findings), key=_severity_rank, default=RegressionSeverity.NONE)
        return RegressionSummary(len(findings), sum(item.classification is ComparisonClassification.REGRESSED for item in findings), sum(item.classification is ComparisonClassification.IMPROVED for item in findings), 0, severity, tuple(findings))

    def _overall(self, tasks: Sequence[TaskComparison], aggregate: AggregateComparison, findings: RegressionSummary, request: EvaluationComparisonRequest) -> tuple[ComparisonStatus, RegressionSeverity]:
        regressions = [item for item in tasks if item.classification is ComparisonClassification.REGRESSED]
        improvements = [item for item in tasks if item.classification is ComparisonClassification.IMPROVED]
        if any(item.severity is RegressionSeverity.HIGH for item in regressions): return ComparisonStatus.REGRESSED, RegressionSeverity.HIGH
        if not regressions and improvements: return ComparisonStatus.IMPROVED, RegressionSeverity.NONE
        if not regressions and aggregate.classification is ComparisonClassification.UNCHANGED:
            return (ComparisonStatus.IMPROVED, RegressionSeverity.NONE) if improvements else (ComparisonStatus.REGRESSION_FREE, RegressionSeverity.NONE)
        if regressions and aggregate.classification is ComparisonClassification.IMPROVED: return ComparisonStatus.IMPROVED_WITH_REGRESSIONS, findings.highest_severity
        if regressions: return ComparisonStatus.REGRESSED, findings.highest_severity
        if aggregate.classification is ComparisonClassification.IMPROVED: return ComparisonStatus.IMPROVED, RegressionSeverity.NONE
        return ComparisonStatus.EQUIVALENT, RegressionSeverity.NONE


def compare_evaluations(baseline: EvaluationSnapshot | EvaluationResult, candidate: EvaluationSnapshot | EvaluationResult, config: ComparisonConfig | None = None, *, baseline_version: EvaluationVersion | None = None, candidate_version: EvaluationVersion | None = None) -> EvaluationComparisonResult:
    if isinstance(baseline, EvaluationResult):
        baseline = EvaluationSnapshot(baseline_version or _version_from_result(baseline, "baseline"), baseline)
    if isinstance(candidate, EvaluationResult):
        candidate = EvaluationSnapshot(candidate_version or _version_from_result(candidate, "candidate"), candidate)
    return EvaluationRegressionComparator().compare(EvaluationComparisonRequest(baseline, candidate, config or ComparisonConfig()))


def _version_from_result(result: EvaluationResult, label: str) -> EvaluationVersion:
    return EvaluationVersion(label, label, result.evaluation_version, result.scoring_policy_version, str(result.benchmark_result.benchmark_version))


def _classify_delta(delta: float | None, epsilon: float) -> ComparisonClassification:
    if delta is None: return ComparisonClassification.INCONCLUSIVE
    if delta > epsilon: return ComparisonClassification.IMPROVED
    if delta < -epsilon: return ComparisonClassification.REGRESSED
    return ComparisonClassification.UNCHANGED


def _classify_status(baseline: EvaluationStatus, candidate: EvaluationStatus) -> ComparisonClassification | None:
    if baseline == candidate: return None
    if _is_passing(baseline) and not _is_passing(candidate): return ComparisonClassification.REGRESSED
    if not _is_passing(baseline) and _is_passing(candidate): return ComparisonClassification.IMPROVED
    if candidate in (EvaluationStatus.INCOMPLETE, EvaluationStatus.UNAVAILABLE) or baseline in (EvaluationStatus.INCOMPLETE, EvaluationStatus.UNAVAILABLE): return ComparisonClassification.INCONCLUSIVE
    return None


def _is_passing(status: EvaluationStatus) -> bool:
    return status in (EvaluationStatus.VERIFIED, EvaluationStatus.PASS)


def _status_text(status: EvaluationStatus | None) -> str | None:
    return status.value if isinstance(status, EvaluationStatus) else None


def _severity_rank(value: RegressionSeverity) -> int:
    return {RegressionSeverity.NONE: 0, RegressionSeverity.LOW: 1, RegressionSeverity.MEDIUM: 2, RegressionSeverity.HIGH: 3, RegressionSeverity.CRITICAL: 4}[value]


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum): return value.value
    if isinstance(value, Mapping): return {str(k): _serialize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)): return [_serialize(item) for item in value]
    if hasattr(value, "__dataclass_fields__"): return {name: _serialize(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, "to_dict") and not isinstance(value, (str, bytes)): return value.to_dict()
    return value


__all__ = ["AggregateComparison", "ComparisonClassification", "ComparisonConfig", "ComparisonResult", "ComparisonStatus", "DimensionComparison", "EvaluationComparisonRequest", "EvaluationComparisonResult", "EvaluationRegressionComparator", "EvaluationSnapshot", "EvaluationVersion", "RegressionFinding", "RegressionSeverity", "RegressionSummary", "TaskComparison", "compare_evaluations"]
