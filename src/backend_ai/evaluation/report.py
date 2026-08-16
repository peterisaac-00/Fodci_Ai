"""Phase 8.6 deterministic evaluation report generation.

The report layer consumes Phase 8.3 ``EvaluationResult``, Phase 8.5
``BenchmarkMetrics``, Phase 8.4 comparison results, Phase 8.8 regression
results, and Phase 8.9 validation results. It never executes benchmarks,
tests, or comparisons; it formats and bounds evidence that already exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from backend_ai.evaluation.regression import (
    EvaluationComparisonResult,
    EvaluationSnapshot,
    RegressionSummary,
)
from backend_ai.evaluation.metrics import BenchmarkMetrics
from backend_ai.evaluation.scoring import BenchmarkScore, EvaluationResult, EvaluationStatus
from backend_ai.evaluation.benchmark_runner import BenchmarkResult


class ReportLimit(str, Enum):
    """Bounding domains applied to report content."""

    TASK_FINDINGS = "task_findings"
    EVIDENCE_REFERENCES = "evidence_references"
    WARNINGS = "warnings"
    FAILURE_EXCERPTS = "failure_excerpts"
    ARTIFACT_REFERENCES = "artifact_references"
    COMPARISON_DETAILS = "comparison_details"


@dataclass(frozen=True, slots=True)
class ReportConfig:
    """Bounding limits for report content."""

    report_version: str = "8.6"
    max_task_findings: int = 64
    max_evidence_references: int = 128
    max_warnings: int = 64
    max_failure_excerpts: int = 32
    max_artifact_references: int = 64
    max_comparison_details: int = 64

    def __post_init__(self) -> None:
        for name in (
            "max_task_findings",
            "max_evidence_references",
            "max_warnings",
            "max_failure_excerpts",
            "max_artifact_references",
            "max_comparison_details",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _bounded_sequence(values: tuple[Any, ...], limit: int) -> tuple[tuple[Any, ...], bool]:
    """Bound a tuple to a finite limit and report whether truncation occurred."""

    return tuple(values[:limit]), len(values) > limit


@dataclass(frozen=True, slots=True)
class TruncationInfo:
    """Explicit metadata when report content was bounded."""

    truncated: bool = False
    truncated_domains: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"truncated": self.truncated, "truncated_domains": list(self.truncated_domains)}


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Complete deterministic, bounded evaluation report."""

    report_version: str
    evaluation_id: str
    agent_version: str
    model_identity: str
    benchmark_identity: str
    scoring_policy_version: str
    evaluation_version: str
    metrics: tuple[dict[str, Any], ...]
    task_breakdown: tuple[dict[str, Any], ...]
    category_breakdown: tuple[dict[str, Any], ...]
    difficulty_breakdown: tuple[dict[str, Any], ...]
    regression: dict[str, Any] | None
    validation: dict[str, Any] | None
    comparison: dict[str, Any] | None
    evidence_completeness: float
    evidence_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    truncation: TruncationInfo

    def __post_init__(self) -> None:
        if not math.isfinite(self.evidence_completeness) or not 0.0 <= self.evidence_completeness <= 1.0:
            raise ValueError("evidence_completeness must be in [0, 1]")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "task_breakdown", tuple(self.task_breakdown))
        object.__setattr__(self, "category_breakdown", tuple(sorted(self.category_breakdown, key=lambda item: str(item.get("category", "")))))
        object.__setattr__(self, "difficulty_breakdown", tuple(sorted(self.difficulty_breakdown, key=lambda item: str(item.get("difficulty", "")))))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))
        object.__setattr__(self, "limitations", tuple(sorted(set(self.limitations))))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "evaluation_id": self.evaluation_id,
            "agent_version": self.agent_version,
            "model_identity": self.model_identity,
            "benchmark_identity": self.benchmark_identity,
            "scoring_policy_version": self.scoring_policy_version,
            "evaluation_version": self.evaluation_version,
            "metrics": list(self.metrics),
            "task_breakdown": list(self.task_breakdown),
            "category_breakdown": list(self.category_breakdown),
            "difficulty_breakdown": list(self.difficulty_breakdown),
            "regression": self.regression,
            "validation": self.validation,
            "comparison": self.comparison,
            "evidence_completeness": self.evidence_completeness,
            "evidence_ids": list(self.evidence_ids),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "truncation": self.truncation.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def to_text(self) -> str:
        """Deterministic human-readable report."""

        def percent(value: float | None) -> str:
            if value is None:
                return "n/a"
            return f"{value * 100.0:.0f}%"

        lines: list[str] = []
        lines.append("==================================================")
        lines.append("FODCI EVALUATION REPORT")
        lines.append("==================================================")
        lines.append("")
        lines.append("Agent:")
        lines.append(self.agent_version)
        lines.append("")
        lines.append("Model:")
        lines.append(self.model_identity)
        lines.append("")
        lines.append("Benchmark:")
        lines.append(self.benchmark_identity)
        lines.append("")
        lines.append("Evaluation version:")
        lines.append(self.evaluation_version)
        lines.append("")
        lines.append("Scoring policy:")
        lines.append(self.scoring_policy_version)
        lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("EVIDENCE")
        lines.append("--------------------------------------------------")
        lines.append("")
        lines.append(f"Completeness: {percent(self.evidence_completeness)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        for warning in self.warnings[: self.truncation.to_dict().get("max_warnings", 64)]:
            lines.append(f"- {warning}")
        for item in self.metrics:
            value = item.get("value")
            if isinstance(value, float):
                lines.append(f"{item['name']}: {percent(value)}")
            elif value is None:
                lines.append(f"{item['name']}: {item.get('status', 'n/a')}")
            else:
                lines.append(f"{item['name']}: {value}")
        for item in self.task_breakdown:
            lines.append(f"task:{item.get('task_id')}: {item.get('status')}")
        lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("CATEGORIES")
        lines.append("--------------------------------------------------")
        lines.append("")
        for item in self.category_breakdown:
            name = item.get("category", "?")
            rate = percent(item.get("success_rate"))
            lines.append(f"{name:<20}{rate}")
        lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("DIFFICULTY")
        lines.append("--------------------------------------------------")
        lines.append("")
        for item in self.difficulty_breakdown:
            name = item.get("difficulty", "?")
            rate = percent(item.get("success_rate"))
            lines.append(f"{name:<20}{rate}")
        lines.append("")
        if self.comparison is not None:
            lines.append("--------------------------------------------------")
            lines.append("COMPARISON")
            lines.append("--------------------------------------------------")
            lines.append("")
            lines.append(f"Status: {self.comparison.get('status')}")
            lines.append(f"Findings: {self.comparison.get('findings', {})}")
            lines.append("")
        if self.regression is not None:
            lines.append("--------------------------------------------------")
            lines.append("REGRESSION")
            lines.append("--------------------------------------------------")
            lines.append("")
            lines.append(f"Result: {self.regression.get('result')}")
            lines.append(f"Findings: {self.regression.get('finding_count')}")
            lines.append("")
        if self.validation is not None:
            lines.append("--------------------------------------------------")
            lines.append("BENCHMARK VALIDATION")
            lines.append("--------------------------------------------------")
            lines.append("")
            lines.append(f"Status: {self.validation.get('status')}")
            lines.append("")
        if self.truncation.truncated:
            lines.append("--------------------------------------------------")
            lines.append("TRUNCATION")
            lines.append("--------------------------------------------------")
            lines.append("")
            lines.append(f"truncated = true (domains: {', '.join(self.truncation.truncated_domains)})")
            lines.append("")
        lines.append("==================================================")
        return "\n".join(lines) + "\n"


def _metric_summary(metrics: BenchmarkMetrics, score: BenchmarkScore | None) -> tuple[dict[str, Any], ...]:
    """Deterministic metric summary rows preserving eligibility metadata."""

    rows: list[dict[str, Any]] = []
    for item in metrics.metrics:
        rows.append(
            {
                "name": item.name.value if isinstance(item.name, Enum) else item.name,
                "value": item.value,
                "status": item.status,
                "numerator": item.numerator,
                "denominator": item.denominator,
                "sample_size": item.sample_size,
                "eligible_count": item.eligible_count,
                "excluded_count": item.excluded_count,
                "exclusion_reasons": list(item.exclusion_reasons),
                "evidence_ids": list(item.evidence_ids),
            }
        )
    if score is not None:
        rows.append(
            {
                "name": "overall_score",
                "value": score.aggregate_score,
                "status": score.confidence,
                "numerator": None,
                "denominator": len(score.task_scores),
                "sample_size": len(score.task_scores),
                "eligible_count": len(score.task_scores),
                "excluded_count": 0,
                "exclusion_reasons": [],
                "evidence_ids": ["benchmark_score"],
            }
        )
    return tuple(rows)


def _category_rows(metrics: BenchmarkMetrics) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "category": item.category,
            "task_count": item.task_count,
            "success_rate": item.success_rate,
            "test_pass_rate": item.test_pass_rate,
            "average_task_score": item.average_task_score,
            "evidence_ids": [task.task_id for task in item.tasks],
        }
        for item in metrics.by_category
    )


def _difficulty_rows(metrics: BenchmarkMetrics) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "difficulty": item.difficulty,
            "task_count": item.task_count,
            "success_rate": item.success_rate,
            "test_pass_rate": item.test_pass_rate,
            "average_task_score": item.average_task_score,
            "evidence_ids": [task.task_id for task in item.tasks],
        }
        for item in metrics.by_difficulty
    )


def _task_breakdown(metrics: BenchmarkMetrics, config: ReportConfig) -> tuple[tuple[dict[str, Any], ...], bool]:
    rows, truncated = _bounded_sequence(
        tuple(
            {
                "task_id": item.task_id,
                "category": item.category,
                "difficulty": item.difficulty,
                "status": item.status,
                "evaluation_status": item.evaluation_status,
                "success": item.success,
                "duration_seconds": item.duration_seconds,
                "evidence_ids": list(item.evidence_ids),
            }
            for item in metrics.task_metrics
        ),
        config.max_task_findings,
    )
    return rows, truncated


def _category_rows_bounded(metrics: BenchmarkMetrics, config: ReportConfig) -> tuple[tuple[dict[str, Any], ...], bool]:
    rows, truncated = _bounded_sequence(_category_rows(metrics), config.max_task_findings)
    return rows, truncated


def _difficulty_rows_bounded(metrics: BenchmarkMetrics, config: ReportConfig) -> tuple[tuple[dict[str, Any], ...], bool]:
    rows, truncated = _bounded_sequence(_difficulty_rows(metrics), config.max_task_findings)
    return rows, truncated


def _regression_summary(comparison: EvaluationComparisonResult | None, config: ReportConfig) -> dict[str, Any] | None:
    if comparison is None:
        return None
    findings, truncated = _bounded_sequence(comparison.findings.findings, config.max_comparison_details)
    summary = {
        "status": comparison.status,
        "severity": comparison.severity,
        "epsilon": comparison.epsilon,
        "finding_count": len(comparison.findings.findings),
        "regression_count": comparison.findings.regression_count,
        "improvement_count": comparison.findings.improvement_count,
        "inconclusive_count": comparison.findings.inconclusive_count,
        "highest_severity": comparison.findings.highest_severity,
        "findings": [item.to_dict() for item in findings],
        "warnings": list(comparison.warnings),
    }
    if truncated:
        summary["truncated"] = True
        summary["warnings"] = summary["warnings"] + ["comparison findings were bounded", "regression_truncated"]
    return summary


def _comparison_summary(comparison: EvaluationComparisonResult | None, config: ReportConfig) -> dict[str, Any] | None:
    if comparison is None:
        return None
    comparisons, truncated = _bounded_sequence(comparison.task_comparisons, config.max_comparison_details)
    summary: dict[str, Any] = {
        "status": comparison.status,
        "severity": comparison.severity,
        "epsilon": comparison.epsilon,
        "baseline_version": comparison.baseline.version.to_dict(),
        "candidate_version": comparison.candidate.version.to_dict(),
        "incompatibility_reasons": list(comparison.incompatibility_reasons),
        "task_comparisons": [item.to_dict() for item in comparisons],
        "warnings": list(comparison.warnings),
    }
    if comparison.aggregate is not None:
        summary["aggregate"] = comparison.aggregate.to_dict()
    if truncated:
        summary["truncated"] = True
    return summary


def _validation_summary(validation_result: Any | None, config: ReportConfig) -> dict[str, Any] | None:
    """Accept either an EvaluationTaskValidationResult or a BenchmarkValidatorResult."""

    if validation_result is None:
        return None
    if hasattr(validation_result, "to_dict") and hasattr(validation_result, "issues"):
        issues, truncated = _bounded_sequence(validation_result.issues, config.max_warnings)
        summary = {
            "status": validation_result.status,
            "health": getattr(validation_result, "health", None),
            "issue_count": len(validation_result.issues),
            "issues": [item.to_dict() for item in issues],
            "warnings": list(getattr(validation_result, "warnings", ())[: config.max_warnings]),
        }
        if truncated:
            summary["truncated"] = True
        return summary
    return validation_result.to_dict()


@dataclass(frozen=True, slots=True)
class ReportInputs:
    """Explicit bounded inputs for report generation."""

    evaluation_result: EvaluationResult
    benchmark_result: BenchmarkResult
    metrics: BenchmarkMetrics
    benchmark_score: BenchmarkScore
    comparison: EvaluationComparisonResult | None = None
    regression_result: Any | None = None
    validation_result: Any | None = None
    identity: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        identity = self.identity
        if identity is not None and not isinstance(identity, Mapping):
            raise ValueError("identity must be a mapping or None")
        if identity is not None:
            object.__setattr__(self, "identity", MappingProxyType(dict(sorted({str(k): str(v) for k, v in identity.items()}.items(), key=lambda item: str(item[0])))))


def generate_evaluation_report(inputs: ReportInputs, config: ReportConfig | None = None) -> EvaluationReport:
    """Generate one deterministic, bounded evaluation report.

    The report formats existing Phase 8.3-8.9 results only. It never executes
    benchmarks, tests, comparisons, or regressions.
    """

    config = config or ReportConfig()
    identity = dict(inputs.identity or {})
    agent_version = identity.get("agent_version", "fodci-agent")
    model_identity = identity.get("model_identity", inputs.evaluation_result.deterministic_metadata.get("model", "unknown"))
    evaluation_id = identity.get("evaluation_id", f"{inputs.benchmark_result.benchmark_id}-{inputs.evaluation_result.evaluation_version}-{inputs.evaluation_result.scoring_policy_version}")

    metric_rows = _metric_summary(inputs.metrics, inputs.benchmark_score)
    task_rows, task_truncated = _task_breakdown(inputs.metrics, config)
    category_rows, category_truncated = _category_rows_bounded(inputs.metrics, config)
    difficulty_rows, difficulty_truncated = _difficulty_rows_bounded(inputs.metrics, config)
    benchmark_warnings = tuple(getattr(inputs.benchmark_result, "warnings", ()) or ())
    warnings, warnings_truncated = _bounded_sequence(
        tuple(sorted(set(inputs.evaluation_result.warnings + inputs.metrics.warnings + benchmark_warnings))), config.max_warnings
    )

    domains: list[str] = []
    if task_truncated:
        domains.append(ReportLimit.TASK_FINDINGS)
    if category_truncated:
        domains.append(ReportLimit.TASK_FINDINGS)
    if difficulty_truncated:
        domains.append(ReportLimit.TASK_FINDINGS)
    if warnings_truncated:
        domains.append(ReportLimit.WARNINGS)

    evidence_ids: set[str] = set()
    for item in inputs.metrics.metrics:
        evidence_ids.update(item.evidence_ids)
    if inputs.comparison is not None:
        evidence_ids.add("comparison")
    if inputs.regression_result is not None:
        evidence_ids.add("regression_evaluation")

    limitations = (
        "evaluation measures the Agent and does not modify the Agent",
        "missing evidence is never treated as success",
        "results depend on the supplied benchmark definition and scoring policy",
    )

    return EvaluationReport(
        report_version=config.report_version,
        evaluation_id=evaluation_id,
        agent_version=agent_version,
        model_identity=model_identity,
        benchmark_identity=inputs.benchmark_result.benchmark_id,
        scoring_policy_version=inputs.evaluation_result.scoring_policy_version,
        evaluation_version=inputs.evaluation_result.evaluation_version,
        metrics=metric_rows,
        task_breakdown=task_rows,
        category_breakdown=category_rows,
        difficulty_breakdown=difficulty_rows,
        regression=_regression_summary(inputs.comparison, config),
        validation=_validation_summary(inputs.validation_result, config),
        comparison=_comparison_summary(inputs.comparison, config),
        evidence_completeness=inputs.benchmark_score.evidence_completeness,
        evidence_ids=tuple(evidence_ids),
        warnings=warnings,
        limitations=limitations,
        truncation=TruncationInfo(bool(domains), tuple(sorted(set(domains)))),
    )


__all__ = [
    "EvaluationReport",
    "ReportConfig",
    "ReportInputs",
    "ReportLimit",
    "TruncationInfo",
    "generate_evaluation_report",
]
