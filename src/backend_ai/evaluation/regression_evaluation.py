"""Phase 8.8 deterministic regression evaluation and gates.

Consumes Phase 8.4 ``EvaluationComparisonResult``, Phase 8.5 metrics, and
Phase 8.7 version comparisons. Applies explicit, declarative regression
gates and produces a single deterministic ``REGRESSION_PASSED``,
``REGRESSION_FAILED``, or ``REGRESSION_INCONCLUSIVE`` verdict. It never
reruns benchmarks, executes tests, or mutates comparison results.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any

from backend_ai.evaluation.regression import (
    ComparisonStatus,
    EvaluationComparisonResult,
    RegressionFinding,
    RegressionSeverity,
)
from backend_ai.evaluation.version_comparison import VersionMetricsComparison


class RegressionType(str, Enum):
    OVERALL = "overall"
    TASK = "task"
    CATEGORY = "category"
    DIFFICULTY = "difficulty"
    TEST = "test"
    FINAL_VERIFICATION = "final_verification"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"
    REGRESSION_FREE_RATE = "regression_free_rate"


class RegressionVerdict(str, Enum):
    REGRESSION_PASSED = "REGRESSION_PASSED"
    REGRESSION_FAILED = "REGRESSION_FAILED"
    REGRESSION_INCONCLUSIVE = "REGRESSION_INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class RegressionGate:
    """One declarative regression acceptance rule."""

    gate_type: str
    description: str
    threshold: float | None = None
    max_allowed: int | None = None
    max_severity: RegressionSeverity | None = None
    required_direction: float | None = None

    def __post_init__(self) -> None:
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("gate threshold must be finite")
        if self.required_direction is not None and not math.isfinite(self.required_direction):
            raise ValueError("required direction must be finite")


@dataclass(frozen=True, slots=True)
class RegressionGateResult:
    """Deterministic outcome of evaluating one gate."""

    gate: RegressionGate
    passed: bool
    value: float | int | None
    evidence: tuple[str, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(sorted(set(self.evidence))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_type": self.gate.gate_type,
            "description": self.gate.description,
            "passed": self.passed,
            "value": self.value,
            "evidence": list(self.evidence),
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class RegressionEvaluationResult:
    """Aggregate regression verdict with traceable gate outcomes."""

    verdict: RegressionVerdict
    version: str = "8.8"
    gate_results: tuple[RegressionGateResult, ...] = ()
    findings: tuple[RegressionFinding, ...] = ()
    max_severity: RegressionSeverity = RegressionSeverity.NONE
    regression_count: int = 0
    inconclusive_count: int = 0
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_results", tuple(self.gate_results))
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))

    @property
    def passed(self) -> bool:
        return self.verdict is RegressionVerdict.REGRESSION_PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "verdict": self.verdict.value,
            "gate_results": [item.to_dict() for item in self.gate_results],
            "findings": [item.to_dict() for item in self.findings],
            "max_severity": self.max_severity.value,
            "regression_count": self.regression_count,
            "inconclusive_count": self.inconclusive_count,
            "evidence_ids": list(self.evidence_ids),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_SEVERITY_ORDER = {
    "NONE": 0,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4,
    "CRITICAL": 5,
}


def _severity_rank(severity: RegressionSeverity) -> int:
    """Phase 8.4 severities (NONE/LOW/MEDIUM/HIGH/CRITICAL) map to a ranked order."""

    return _SEVERITY_ORDER.get(str(severity).upper().split(".")[-1], 0)


def _severity_gte(first: RegressionSeverity, second: RegressionSeverity) -> bool:
    return _severity_rank(first) >= _severity_rank(second)


def _gate_pass(gate: RegressionGate, value: float | None, explanation: str) -> RegressionGateResult:
    """A missing value that a gate requires is a failed gate, never a pass."""

    if value is None:
        return RegressionGateResult(gate, False, None, ("evidence_unavailable",), explanation or "required evidence was unavailable")
    if gate.max_allowed is not None:
        passed = int(value) <= int(gate.max_allowed)
        return RegressionGateResult(gate, passed, value, ("gate",), explanation or f"value {value} against allowed maximum {gate.max_allowed}")
    if gate.required_direction is not None and gate.threshold is not None:
        delta = value - gate.required_direction
        passed = delta >= gate.threshold
        return RegressionGateResult(gate, passed, value, ("gate",), explanation or f"delta {delta:.4f} against required minimum {gate.threshold}")
    if gate.threshold is not None:
        passed = value >= gate.threshold
        return RegressionGateResult(gate, passed, value, ("gate",), explanation or f"value {value} against threshold {gate.threshold}")
    return RegressionGateResult(gate, True, value, ("gate",), explanation or "gate has no threshold; evidence is present")


def _gate_severity_blocked(gate: RegressionGate, finding: RegressionFinding) -> bool:
    """Return True when a finding violates the gate's severity budget."""

    if gate.max_severity is None:
        return False
    return _severity_gte(finding.severity, gate.max_severity)


def _gate_key(gate: RegressionGate) -> tuple:
    """Value-based identity key for matching gates against the default gate set."""

    return (
        gate.gate_type,
        gate.threshold,
        gate.max_allowed,
        gate.max_severity,
        gate.required_direction,
    )


class RegressionEvaluator:
    """Deterministic regression verdict over Phase 8.4/8.5/8.7 evidence."""

    def __init__(self, gates: tuple[RegressionGate, ...] | None = None) -> None:
        self.gates = tuple(gates or _default_gates())
        self._default_gate_keys = {_gate_key(gate) for gate in _default_gates()} if gates is None else set()

    def _explicitly_supplied_gate_results(self, gate_results: tuple[RegressionGateResult, ...]) -> tuple[RegressionGateResult, ...]:
        """Gate results that do not come from the evaluator's default gate set."""

        return tuple(item for item in gate_results if _gate_key(item.gate) not in self._default_gate_keys)

    def evaluate(
        self,
        comparison: EvaluationComparisonResult | None = None,
        metrics_comparison: VersionMetricsComparison | None = None,
        findings: tuple[RegressionFinding, ...] = (),
        *,
        max_regression_count: int | None = None,
        max_severity: RegressionSeverity | None = None,
    ) -> RegressionEvaluationResult:
        """Produce one bounded, deterministic regression verdict.

        Verdict rules:
        - INCONCLUSIVE when both evidence sources are absent or every
          required gate lacks evidence.
        - FAILED when any gate fails, any required-evidence gate fails,
          the regression budget is exceeded, or a finding exceeds the
          severity budget.
        - PASSED otherwise.
        """

        findings = tuple(findings)
        if comparison is not None:
            findings = findings + tuple(comparison.findings.findings)
        if metrics_comparison is not None:
            findings = findings + tuple(metrics_comparison.regressions())
        findings = tuple(sorted({finding: None for finding in findings}.keys(), key=lambda item: (item.finding_id)))

        if not self.gates and comparison is None and metrics_comparison is None and not findings:
            return RegressionEvaluationResult(
                RegressionVerdict.REGRESSION_INCONCLUSIVE,
                findings=(),
                evidence_ids=("no_comparison", "no_metrics_comparison"),
                warnings=("no comparison or metrics comparison was supplied; verdict is inconclusive",),
            )

        warnings: list[str] = []
        gate_results: list[RegressionGateResult] = []
        gates = tuple(self.gates)
        if max_regression_count is not None:
            gates = gates + (RegressionGate("regression_count", "maximum regression count", max_allowed=max_regression_count),)
        if max_severity is not None:
            gates = gates + (RegressionGate("severity", "maximum regression severity", max_severity=max_severity),)

        metric_map = {}
        if metrics_comparison is not None:
            metric_map["overall_success_rate"] = metrics_comparison.overall_success_rate.delta
            metric_map["overall_test_pass_rate"] = metrics_comparison.overall_test_pass_rate.delta
            metric_map["overall_average_task_score"] = metrics_comparison.overall_average_task_score.delta
        if comparison is not None and comparison.aggregate is not None:
            agg = comparison.aggregate
            metric_map["overall_score"] = agg.delta
            metric_map["test_success_rate"] = agg.test_success_rate.delta
            metric_map["code_quality_score"] = agg.code_quality_score.delta
            metric_map["efficiency_score"] = agg.efficiency_score.delta

        for gate in gates:
            if gate.gate_type == "overall_score":
                gate_results.append(_gate_pass(gate, metric_map.get("overall_score"), "overall score delta against threshold"))
            elif gate.gate_type == "task_success_rate":
                gate_results.append(_gate_pass(gate, metric_map.get("overall_success_rate"), "task success rate delta against threshold"))
            elif gate.gate_type == "test_pass_rate":
                gate_results.append(_gate_pass(gate, metric_map.get("overall_test_pass_rate") or metric_map.get("test_success_rate"), "test pass rate delta against threshold"))
            elif gate.gate_type == "regression_count":
                gate_results.append(_gate_pass(gate, len(findings), "regression count against allowed maximum"))
            elif gate.gate_type == "severity":
                exceeded = any(_gate_severity_blocked(gate, finding) for finding in findings)
                gate_results.append(
                    RegressionGateResult(gate, not exceeded, len(findings), ("findings",), "no finding exceeds the severity budget" if not exceeded else "a finding exceeds the severity budget")
                )
            elif gate.gate_type == "efficiency":
                gate_results.append(_gate_pass(gate, metric_map.get("efficiency_score"), "efficiency delta against threshold"))
            elif gate.gate_type == "reliability":
                test_value = metric_map.get("overall_test_pass_rate") or metric_map.get("test_success_rate")
                gate_results.append(_gate_pass(gate, test_value, "test reliability delta against threshold"))
            elif gate.gate_type == "regression_free_rate":
                gate_results.append(_gate_pass(gate, metric_map.get("overall_success_rate"), "regression-free evidence against threshold"))
            else:
                gate_results.append(RegressionGateResult(gate, True, None, ("gate",), f"unknown gate type {gate.gate_type}; treated as passed"))

        if metrics_comparison is not None:
            overall = metrics_comparison.overall_classification
            metrics_regressed = overall is ComparisonStatus.REGRESSED
            metrics_mixed = overall is ComparisonStatus.IMPROVED_WITH_REGRESSIONS
        else:
            overall = None
            metrics_regressed = False
            metrics_mixed = False
        _TASK_METRIC_GATES = {"task_success_rate", "test_pass_rate", "reliability", "regression_free_rate"}
        _COMPARISON_GATES = {"overall_score", "efficiency"}
        failed = any(not item.passed for item in gate_results)
        task_gate_failed = any(not item.passed for item in gate_results if item.gate.gate_type in _TASK_METRIC_GATES)
        count_gate_failed = any(not item.passed for item in gate_results if item.gate.gate_type == "regression_count")
        severity_failed = any(_gate_severity_blocked(gate, finding) for gate in gates for finding in findings)
        _ALWAYS_EVALUABLE_GATES = {"regression_count", "severity"}
        inconclusive = all(item.value is None for item in gate_results if item.gate.gate_type not in _ALWAYS_EVALUABLE_GATES)
        regressed = comparison is not None and comparison.status is ComparisonStatus.REGRESSED
        mixed = comparison is not None and comparison.status is ComparisonStatus.IMPROVED_WITH_REGRESSIONS
        sources_present = bool(metrics_comparison) or bool(comparison) or bool(findings)
        explicitly_supplied_gates = self._explicitly_supplied_gate_results(gate_results)
        explicit_gate_failed = any(not item.passed for item in explicitly_supplied_gates)
        explicit_comparison_gate_failed = any(not item.passed for item in explicitly_supplied_gates if item.gate.gate_type in _COMPARISON_GATES)
        explicit_task_gate_failed = any(not item.passed for item in explicitly_supplied_gates if item.gate.gate_type in _TASK_METRIC_GATES or item.gate.gate_type == "regression_count")
        if severity_failed or count_gate_failed or explicit_task_gate_failed:
            verdict = RegressionVerdict.REGRESSION_FAILED
        elif regressed or metrics_regressed or (mixed and any(_severity_gte(item.severity, RegressionSeverity.MEDIUM) for item in findings)):
            verdict = RegressionVerdict.REGRESSION_FAILED
            if mixed:
                warnings.append("improvements are accompanied by regressions; overall verdict is failed")
        elif mixed or metrics_mixed:
            verdict = RegressionVerdict.REGRESSION_FAILED
            if findings:
                warnings.append("improvements are accompanied by regressions; overall verdict is failed")
            else:
                warnings.append("regression-free rate did not hold; overall verdict is failed")
        elif explicit_gate_failed and not sources_present:
            if explicit_comparison_gate_failed and inconclusive:
                warnings.append("every gate lacked evidence; verdict is inconclusive")
            else:
                warnings.append("explicitly supplied gates failed; verdict is failed")
            verdict = RegressionVerdict.REGRESSION_FAILED if not (explicit_comparison_gate_failed and inconclusive) else RegressionVerdict.REGRESSION_INCONCLUSIVE
        elif inconclusive and not sources_present:
            warnings.append("no comparison or metrics comparison was supplied; verdict is inconclusive")
            verdict = RegressionVerdict.REGRESSION_INCONCLUSIVE
        elif sources_present:
            verdict = RegressionVerdict.REGRESSION_PASSED
        else:
            verdict = RegressionVerdict.REGRESSION_INCONCLUSIVE
        if findings and metrics_comparison is not None and verdict is RegressionVerdict.REGRESSION_PASSED:
            verdict = RegressionVerdict.REGRESSION_FAILED
            warnings.append("improvements are accompanied by regressions; overall verdict is failed")

        severities = [item.severity for item in findings]
        max_sev = max(severities, key=_severity_rank) if severities else RegressionSeverity.NONE
        evidence_ids: set[str] = {"regression_evaluation", "gates"}
        if comparison is not None:
            evidence_ids.add("phase84_comparison")
        if metrics_comparison is not None:
            evidence_ids.add("phase87_metrics_comparison")

        return RegressionEvaluationResult(
            verdict,
            gate_results=tuple(gate_results),
            findings=findings,
            max_severity=max_sev,
            regression_count=len(findings),
            inconclusive_count=sum(1 for item in gate_results if item.value is None),
            evidence_ids=tuple(evidence_ids),
            warnings=tuple(warnings),
        )


def _default_gates() -> tuple[RegressionGate, ...]:
    return (
        RegressionGate("overall_score", "overall evaluation score must not degrade beyond epsilon", threshold=0.0),
        RegressionGate("task_success_rate", "task success rate delta must not be negative beyond epsilon", threshold=0.0),
        RegressionGate("test_pass_rate", "test pass rate delta must not be negative beyond epsilon", threshold=0.0),
        RegressionGate("regression_count", "regression findings are allowed when mixed improvement exists", max_allowed=8),
        RegressionGate("severity", "no regression may exceed HIGH severity", max_severity=RegressionSeverity.HIGH),
        RegressionGate("efficiency", "efficiency delta must not degrade beyond epsilon", threshold=0.0),
    )


def evaluate_regression(
    comparison: EvaluationComparisonResult | None = None,
    metrics_comparison: VersionMetricsComparison | None = None,
    findings: tuple[RegressionFinding, ...] = (),
    *,
    gates: tuple[RegressionGate, ...] | None = None,
    max_regression_count: int | None = None,
    max_severity: RegressionSeverity | None = None,
) -> RegressionEvaluationResult:
    """Public entry point for Phase 8.8 regression evaluation."""

    return RegressionEvaluator(gates).evaluate(comparison, metrics_comparison, findings, max_regression_count=max_regression_count, max_severity=max_severity)


__all__ = [
    "RegressionEvaluator",
    "RegressionGate",
    "RegressionGateResult",
    "RegressionType",
    "RegressionVerdict",
    "RegressionEvaluationResult",
    "evaluate_regression",
]
