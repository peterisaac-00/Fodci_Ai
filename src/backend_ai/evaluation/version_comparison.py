"""Phase 8.7 version comparison extension.

Phase 8.7 extends the Phase 8.4 comparison layer with category-level and
difficulty-level deltas while reusing every Phase 8.4 classifier, severity
rule, epsilon threshold, and incompatibility check. It never reruns
benchmarks or executes tests, and it never duplicates comparison logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any

from backend_ai.evaluation.metrics import BenchmarkMetrics
from backend_ai.evaluation.regression import (
    ComparisonConfig,
    ComparisonStatus,
    DimensionComparison,
    EvaluationComparisonResult,
    RegressionFinding,
    RegressionSeverity,
    RegressionSummary,
)


class ComparisonClassification(str, Enum):
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    EQUIVALENT = "EQUIVALENT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class GroupMetricComparison:
    """Category- or difficulty-level metric comparison reusing Phase 8.4 deltas."""

    group_name: str
    group_type: str  # "category" or "difficulty"
    task_count: int
    success_rate: ComparisonDimension
    test_pass_rate: ComparisonDimension
    average_task_score: ComparisonDimension
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))

    def status(self) -> ComparisonStatus:
        """Aggregate group classification following Phase 8.4 status rules."""

        classifications = [
            item.classification
            for item in (self.success_rate, self.test_pass_rate, self.average_task_score)
            if item.classification is not None
        ]
        if not classifications:
            return ComparisonStatus.INCOMPARABLE
        decisive = [item for item in classifications if item is not ComparisonClassification.EQUIVALENT]
        if not decisive:
            return ComparisonStatus.EQUIVALENT
        if all(item is ComparisonClassification.REGRESSED for item in decisive):
            return ComparisonStatus.REGRESSED
        if all(item is ComparisonClassification.IMPROVED for item in decisive):
            return ComparisonStatus.IMPROVED
        if any(item is ComparisonClassification.REGRESSED for item in decisive):
            return ComparisonStatus.REGRESSED
        if any(item is ComparisonClassification.IMPROVED for item in decisive):
            return ComparisonStatus.IMPROVED
        return ComparisonStatus.INCONCLUSIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_name": self.group_name,
            "group_type": self.group_type,
            "task_count": self.task_count,
            "success_rate": self.success_rate.to_dict(),
            "test_pass_rate": self.test_pass_rate.to_dict(),
            "average_task_score": self.average_task_score.to_dict(),
            "evidence_ids": list(self.evidence_ids),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ComparisonDimension:
    """Reusable scalar comparison with Phase 8.4 epsilon semantics."""

    name: str
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    classification: ComparisonClassification | None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.delta is not None and not math.isfinite(self.delta):
            raise ValueError("delta must be finite")
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "delta": self.delta,
            "classification": self.classification.value if self.classification is not None else None,
            "evidence_ids": list(self.evidence_ids),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _classify(value: float | None, epsilon: float) -> ComparisonClassification | None:
    if value is None or not math.isfinite(value):
        return None
    if value > epsilon:
        return ComparisonClassification.IMPROVED
    if value < -epsilon:
        return ComparisonClassification.REGRESSED
    return ComparisonClassification.EQUIVALENT


def _dimension(
    name: str,
    baseline_value: float | None,
    candidate_value: float | None,
    epsilon: float,
    evidence_ids: tuple[str, ...],
) -> ComparisonDimension:
    delta = None
    classification: ComparisonClassification | None = None
    if baseline_value is not None and candidate_value is not None:
        delta = candidate_value - baseline_value
        classification = _classify(delta, epsilon)
    elif baseline_value is None and candidate_value is not None:
        delta = None
        classification = ComparisonClassification.IMPROVED
    elif candidate_value is None and baseline_value is not None:
        delta = None
        classification = ComparisonClassification.REGRESSED
    return ComparisonDimension(name, baseline_value, candidate_value, delta, classification, evidence_ids)


@dataclass(frozen=True, slots=True)
class VersionMetricsComparison:
    """Aggregated comparison across task, category, and difficulty metrics."""

    baseline_version: str
    candidate_version: str
    epsilon: float
    baseline_sample_size: int
    candidate_sample_size: int
    category_comparisons: tuple[GroupMetricComparison, ...]
    difficulty_comparisons: tuple[GroupMetricComparison, ...]
    overall_success_rate: ComparisonDimension
    overall_test_pass_rate: ComparisonDimension
    overall_average_task_score: ComparisonDimension
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "category_comparisons", tuple(self.category_comparisons))
        object.__setattr__(self, "difficulty_comparisons", tuple(self.difficulty_comparisons))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))

    @property
    def overall_classification(self) -> ComparisonStatus:
        """Phase 8.4 status rules over overall and group dimensions."""

        classifications: list[ComparisonStatus] = []
        for item in [self.overall_success_rate.classification]:
            if item is not None:
                classifications.append(ComparisonStatus(item.value))
        for group in list(self.category_comparisons) + list(self.difficulty_comparisons):
            status = group.status()
            if status is not ComparisonStatus.INCOMPARABLE:
                classifications.append(status)
        decisive = [item for item in classifications if item is not ComparisonStatus.EQUIVALENT]
        if not classifications:
            return ComparisonStatus.INCOMPARABLE
        if not decisive:
            return ComparisonStatus.EQUIVALENT
        if all(item is ComparisonStatus.REGRESSED for item in decisive):
            return ComparisonStatus.REGRESSED
        if all(item is ComparisonStatus.IMPROVED for item in decisive):
            return ComparisonStatus.IMPROVED
        if any(item is ComparisonStatus.REGRESSED for item in decisive):
            return ComparisonStatus.REGRESSED
        if any(item is ComparisonStatus.IMPROVED for item in decisive):
            return ComparisonStatus.IMPROVED
        return ComparisonStatus.INCONCLUSIVE

    def regressions(self) -> tuple[RegressionFinding, ...]:
        """Trace every regression to a named, evidence-backed finding."""

        findings: list[RegressionFinding] = []
        index = 0
        groups = [("success_rate", self.overall_success_rate)]
        for group in list(self.category_comparisons) + list(self.difficulty_comparisons):
            groups.append((f"{group.group_type}.{group.group_name}.success_rate", group.success_rate))
            groups.append((f"{group.group_type}.{group.group_name}.test_pass_rate", group.test_pass_rate))
        for name, dimension in groups:
            if dimension.classification is ComparisonClassification.REGRESSED and dimension.delta is not None:
                severity = RegressionSeverity.HIGH if dimension.delta < -0.2 else RegressionSeverity.MEDIUM
                findings.append(
                    RegressionFinding(
                        f"VF-{index:03d}",
                        name,
                        name,
                        dimension.baseline_value,
                        dimension.candidate_value,
                        dimension.delta,
                        ComparisonClassification.REGRESSED,
                        severity,
                        tuple(dimension.evidence_ids),
                        f"{name} decreased by {dimension.delta:.4f}",
                    )
                )
                index += 1
        return tuple(findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "epsilon": self.epsilon,
            "baseline_sample_size": self.baseline_sample_size,
            "candidate_sample_size": self.candidate_sample_size,
            "overall_classification": self.overall_classification.value,
            "overall_success_rate": self.overall_success_rate.to_dict(),
            "overall_test_pass_rate": self.overall_test_pass_rate.to_dict(),
            "overall_average_task_score": self.overall_average_task_score.to_dict(),
            "category_comparisons": [item.to_dict() for item in self.category_comparisons],
            "difficulty_comparisons": [item.to_dict() for item in self.difficulty_comparisons],
            "evidence_ids": list(self.evidence_ids),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _group_summary(metrics: BenchmarkMetrics) -> tuple[dict[str, tuple[int, float | None, float | None, float | None]], dict[str, tuple[int, float | None, float | None, float | None]]]:
    """(count, success_rate, test_pass_rate, avg_score) per group; None means no eligible evidence."""

    def summarize(items: tuple) -> dict[str, tuple[int, float | None, float | None, float | None]]:
        out: dict[str, list[float | int | None]] = {}
        for item in items:
            bucket = out.setdefault(getattr(item, "category", getattr(item, "difficulty", "unknown")), [0, None, None, None])
            bucket[0] += 1
            if item.success_rate is not None:
                bucket[1] = item.success_rate
            if item.test_pass_rate is not None:
                bucket[2] = item.test_pass_rate
            if item.average_task_score is not None:
                bucket[3] = item.average_task_score
        return {key: tuple(bucket) for key, bucket in out.items()}

    return summarize(metrics.by_category), summarize(metrics.by_difficulty)


def compare_evaluation_metrics(
    baseline_metrics: BenchmarkMetrics,
    candidate_metrics: BenchmarkMetrics,
    *,
    baseline_version: str = "baseline",
    candidate_version: str = "candidate",
    config: ComparisonConfig | None = None,
) -> VersionMetricsComparison:
    """Compare category/difficulty metrics between two benchmark evaluations.

    Reuses the Phase 8.4 ``ComparisonConfig`` epsilon and incompatibility
    semantics. Sample sizes, eligibility rules, and denominators come from
    Phase 8.5 metrics and are never recomputed here.
    """

    config = config or ComparisonConfig()
    epsilon = config.epsilon
    warnings: list[str] = []

    baseline_categories, baseline_difficulties = _group_summary(baseline_metrics)
    candidate_categories, candidate_difficulties = _group_summary(candidate_metrics)

    def category_dimension(name: str) -> ComparisonDimension:
        baseline = baseline_metrics.metric(f"{name}_rate" if name == "task_success" else name)
        return _dimension(name, baseline.value if baseline else None, None, epsilon, ("metrics",))

    def group_pair(name: str, group_type: str) -> GroupMetricComparison:
        baseline = baseline_categories.get(name) if group_type == "category" else baseline_difficulties.get(name)
        candidate = candidate_categories.get(name) if group_type == "category" else candidate_difficulties.get(name)
        if baseline is None and candidate is None:
            empty = ComparisonDimension(name, None, None, None, None, ("metrics",))
            return GroupMetricComparison(name, group_type, 0, empty, empty, empty, ("metrics",))
        b_count = baseline[0] if baseline else 0
        c_count = candidate[0] if candidate else 0
        return GroupMetricComparison(
            name,
            group_type,
            b_count + c_count,
            _dimension("success_rate", baseline[1] if baseline else None, candidate[1] if candidate else None, epsilon, ("metrics",)),
            _dimension("test_pass_rate", baseline[2] if baseline else None, candidate[2] if candidate else None, epsilon, ("metrics",)),
            _dimension("average_task_score", baseline[3] if baseline else None, candidate[3] if candidate else None, epsilon, ("metrics",)),
            ("metrics",),
        )

    def overall_dimension(name: str, metric_name: str) -> ComparisonDimension:
        baseline_metric = baseline_metrics.metric(metric_name)
        candidate_metric = candidate_metrics.metric(metric_name)
        return _dimension(
            name,
            baseline_metric.value if baseline_metric else None,
            candidate_metric.value if candidate_metric else None,
            epsilon,
            ("metrics",),
        )

    if set(baseline_categories) != set(candidate_categories):
        warnings.append("category sets differ between evaluations; uncovered groups are treated as regressions")
    if set(baseline_difficulties) != set(candidate_difficulties):
        warnings.append("difficulty sets differ between evaluations; uncovered groups are treated as regressions")

    categories = tuple(group_pair(name, "category") for name in sorted(set(baseline_categories) | set(candidate_categories)))
    difficulties = tuple(group_pair(name, "difficulty") for name in sorted(set(baseline_difficulties) | set(candidate_difficulties)))

    return VersionMetricsComparison(
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        epsilon=epsilon,
        baseline_sample_size=baseline_metrics.sample_size,
        candidate_sample_size=candidate_metrics.sample_size,
        category_comparisons=categories,
        difficulty_comparisons=difficulties,
        overall_success_rate=overall_dimension("task_success_rate", "task_success_rate"),
        overall_test_pass_rate=overall_dimension("test_pass_rate", "test_pass_rate"),
        overall_average_task_score=overall_dimension("average_task_score", "average_task_score"),
        evidence_ids=("metrics", "comparison"),
        warnings=tuple(warnings),
    )


def compare_evaluations_detailed(
    baseline: EvaluationComparisonResult | None,
    baseline_metrics: BenchmarkMetrics | None,
    candidate_metrics: BenchmarkMetrics | None,
    *,
    baseline_version: str = "baseline",
    candidate_version: str = "candidate",
    config: ComparisonConfig | None = None,
) -> tuple[ComparisonStatus, VersionMetricsComparison | None, tuple[str, ...]]:
    """Convenience entry point pairing an existing Phase 8.4 task comparison
    with the new Phase 8.7 metric comparison.

    The final status is the Phase 8.4 task-level status when both are
    conclusive; group-level detail is returned alongside it. Incompatibility
    from Phase 8.4 is preserved and never masked by metric comparisons.
    """

    config = config or ComparisonConfig()
    if baseline_metrics is None or candidate_metrics is None:
        return ComparisonStatus.INCOMPARABLE, None, ("one or both metrics collections are unavailable",)
    metrics_comparison = compare_evaluation_metrics(baseline_metrics, candidate_metrics, baseline_version=baseline_version, candidate_version=candidate_version, config=config)
    warnings = list(metrics_comparison.warnings)
    if baseline is not None and baseline.status is ComparisonStatus.INCOMPARABLE:
        warnings.extend(baseline.warnings)
        return ComparisonStatus.INCOMPARABLE, metrics_comparison, tuple(warnings)
    if baseline is not None and baseline.status in (ComparisonStatus.IMPROVED_WITH_REGRESSIONS, ComparisonStatus.REGRESSED):
        warnings.extend(baseline.warnings)
        return baseline.status, metrics_comparison, tuple(warnings)
    if baseline is not None:
        warnings.extend(baseline.warnings)
    return metrics_comparison.overall_classification, metrics_comparison, tuple(warnings)


__all__ = [
    "ComparisonClassification",
    "ComparisonDimension",
    "GroupMetricComparison",
    "VersionMetricsComparison",
    "compare_evaluation_metrics",
    "compare_evaluations_detailed",
]
