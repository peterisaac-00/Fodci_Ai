"""Phase 8.3 deterministic, evidence-driven benchmark scoring.

This module consumes :class:`BenchmarkResult` evidence only.  It never executes
commands, tests, tools, models, or network requests, and it does not compare
benchmark versions (that belongs to Phase 8.4).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from backend_ai.evaluation.benchmark_runner import BenchmarkResult, BenchmarkTaskRun, BenchmarkTaskStatus
from backend_ai.evaluation.task_model import EvaluationTask, SuccessCriterion, SuccessCriterionType


class EvaluationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    INCOMPLETE = "INCOMPLETE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NO_TESTS = "NO_TESTS"
    UNAVAILABLE = "UNAVAILABLE"
    INCOMPLETE = "INCOMPLETE"
    TIMEOUT = "TIMEOUT"
    OUTPUT_LIMIT = "OUTPUT_LIMIT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    REGRESSION_DETECTED = "REGRESSION_DETECTED"


@dataclass(frozen=True, slots=True)
class EvaluationWeights:
    task_success: float = 0.50
    tests: float = 0.30
    code_quality: float = 0.10
    efficiency: float = 0.10

    def __post_init__(self) -> None:
        values = (self.task_success, self.tests, self.code_quality, self.efficiency)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError("weights must be finite real numbers")
        if any(value < 0.0 for value in values):
            raise ValueError("weights must be non-negative")
        if sum(values) != 1.0:
            raise ValueError("weights must sum exactly to 1.0")

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ScoringPolicy:
    evaluation_version: str = "8.3"
    scoring_policy_version: str = "1.0"
    duration_target_seconds: float = 60.0
    iteration_target: float = 5.0
    tool_call_target: float = 20.0
    test_run_target: float = 3.0

    def __post_init__(self) -> None:
        if not self.evaluation_version.strip() or not self.scoring_policy_version.strip():
            raise ValueError("scoring policy versions must contain text")
        for name in ("duration_target_seconds", "iteration_target", "tool_call_target", "test_run_target"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ScoreDimension:
    name: str
    score: float
    weight: float
    weighted_score: float
    status: EvaluationStatus
    evidence_ids: tuple[str, ...] = ()
    explanation: str = ""

    def __post_init__(self) -> None:
        _validate_score(self.score, "score")
        _validate_score(self.weight, "weight")
        _validate_score(self.weighted_score, "weighted_score")
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class CriterionEvaluation:
    criterion_id: str
    status: EvaluationStatus
    satisfied: bool | None
    score: float
    evidence: tuple[str, ...] = ()
    explanation: str = ""
    evidence_strength: str = ""

    def __post_init__(self) -> None:
        _validate_score(self.score, "score")
        object.__setattr__(self, "evidence", tuple(sorted(set(self.evidence))))

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class TaskScore:
    task_id: str
    task_success_score: float
    test_score: float
    code_quality_score: float
    efficiency_score: float
    final_score: float
    percentage: float
    status: EvaluationStatus
    dimensions: tuple[ScoreDimension, ...] = ()

    def __post_init__(self) -> None:
        for name in ("task_success_score", "test_score", "code_quality_score", "efficiency_score", "final_score"):
            _validate_score(getattr(self, name), name)
        if not math.isfinite(self.percentage) or self.percentage < 0 or self.percentage > 100:
            raise ValueError("percentage must be in [0, 100]")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    task: EvaluationTask
    task_run: BenchmarkTaskRun
    criterion_results: tuple[CriterionEvaluation, ...]
    score: TaskScore
    blocking_reasons: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    benchmark_id: str
    task_scores: tuple[TaskScore, ...]
    aggregate_score: float
    percentage: float
    dimension_scores: tuple[ScoreDimension, ...]
    completed_count: int
    failed_count: int
    incomplete_count: int
    blocked_count: int
    unavailable_count: int
    confidence: str
    evidence_completeness: float

    def __post_init__(self) -> None:
        _validate_score(self.aggregate_score, "aggregate_score")
        _validate_score(self.evidence_completeness, "evidence_completeness")
        if not 0 <= self.percentage <= 100:
            raise ValueError("percentage must be in [0, 100]")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    benchmark_result: BenchmarkResult
    benchmark_score: BenchmarkScore
    evaluation_version: str = "8.3"
    scoring_policy_version: str = "1.0"
    warnings: tuple[str, ...] = ()
    deterministic_metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))
        object.__setattr__(self, "deterministic_metadata", MappingProxyType(dict(self.deterministic_metadata)))

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class BenchmarkScorer:
    """Pure deterministic scorer for an existing Phase 8.2 benchmark result."""

    def __init__(self, *, weights: EvaluationWeights | None = None, policy: ScoringPolicy | None = None) -> None:
        self.weights = weights or EvaluationWeights()
        self.policy = policy or ScoringPolicy()

    def evaluate_task(self, task: EvaluationTask, task_run: BenchmarkTaskRun) -> TaskEvaluation:
        if task.task_id != task_run.task_id:
            raise ValueError("task and task_run IDs must match")
        criteria = tuple(self._criterion(task, task_run, criterion) for criterion in sorted(task.success_criteria, key=lambda item: item.criterion_id))
        missing = tuple(item.criterion_id for item in criteria if item.status in (EvaluationStatus.INSUFFICIENT_EVIDENCE, EvaluationStatus.UNAVAILABLE))
        blockers = tuple(item.explanation for item in criteria if item.status in (EvaluationStatus.FAILED, EvaluationStatus.BLOCKED))
        task_success, success_status, success_ids = self._task_success(task, task_run, criteria)
        tests, test_status, test_ids = self._tests(task_run)
        quality, quality_status, quality_ids = self._quality(task_run)
        efficiency, efficiency_status, efficiency_ids = self._efficiency(task_run, task_success)
        dimensions = (
            ScoreDimension("task_success", task_success, self.weights.task_success, task_success * self.weights.task_success, success_status, success_ids, "Declared criteria and authoritative completion evidence."),
            ScoreDimension("tests", tests, self.weights.tests, tests * self.weights.tests, test_status, test_ids, "Structured test evidence only; unavailable tests are not treated as passes."),
            ScoreDimension("code_quality", quality, self.weights.code_quality, quality * self.weights.code_quality, quality_status, quality_ids, "Bounded objective scope, safety, mutation, and regression signals."),
            ScoreDimension("efficiency", efficiency, self.weights.efficiency, efficiency * self.weights.efficiency, efficiency_status, efficiency_ids, "Measured execution data, gated by minimum correctness evidence."),
        )
        final = sum(item.weighted_score for item in dimensions)
        status = EvaluationStatus.VERIFIED if success_status is EvaluationStatus.VERIFIED and not missing and not blockers else (EvaluationStatus.FAILED if blockers or task_run.status is BenchmarkTaskStatus.FAILED else EvaluationStatus.INCOMPLETE if missing else EvaluationStatus.PARTIAL)
        score = TaskScore(task.task_id, task_success, tests, quality, efficiency, final, final * 100.0, status, dimensions)
        return TaskEvaluation(task, task_run, criteria, score, blockers, missing, tuple(task_run.warnings))

    def evaluate_benchmark(self, benchmark_result: BenchmarkResult, tasks: Sequence[EvaluationTask]) -> EvaluationResult:
        task_map = {task.task_id: task for task in tasks}
        evaluations = tuple(self.evaluate_task(task_map[run.task_id], run) for run in sorted(benchmark_result.task_runs, key=lambda item: item.task_id) if run.task_id in task_map)
        scores = tuple(item.score for item in evaluations)
        if scores:
            aggregate = sum(score.final_score for score in scores) / len(scores)
            dimensions = tuple(self._aggregate_dimension(scores, name) for name in ("task_success", "tests", "code_quality", "efficiency"))
            completeness = sum(1.0 for item in evaluations if not item.missing_evidence) / len(evaluations)
        else:
            aggregate, dimensions, completeness = 0.0, (), 0.0
        counts = {"completed": 0, "failed": 0, "incomplete": 0, "blocked": 0, "unavailable": 0}
        for score in scores:
            key = {EvaluationStatus.VERIFIED: "completed", EvaluationStatus.PASS: "completed", EvaluationStatus.FAILED: "failed", EvaluationStatus.BLOCKED: "blocked", EvaluationStatus.UNAVAILABLE: "unavailable"}.get(score.status, "incomplete")
            counts[key] += 1
        confidence = "HIGH" if scores and completeness == 1.0 else "LOW" if not scores else "MEDIUM"
        benchmark_score = BenchmarkScore(benchmark_result.benchmark_id, scores, aggregate, aggregate * 100.0, dimensions, counts["completed"], counts["failed"], counts["incomplete"], counts["blocked"], counts["unavailable"], confidence, completeness)
        metadata = {"task_order": [score.task_id for score in scores], "deterministic": True, "evaluation_version": self.policy.evaluation_version, "scoring_policy_version": self.policy.scoring_policy_version}
        return EvaluationResult(benchmark_result, benchmark_score, self.policy.evaluation_version, self.policy.scoring_policy_version, benchmark_result.warnings, metadata)

    def _criterion(self, task: EvaluationTask, run: BenchmarkTaskRun, criterion: SuccessCriterion) -> CriterionEvaluation:
        ev = run.evidence
        evidence: list[str] = []
        status = _map_status(run.status)
        satisfied: bool | None = None
        ctype = criterion.criterion_type.value if isinstance(criterion.criterion_type, SuccessCriterionType) else str(criterion.criterion_type)
        final = _status_value(ev.final_verification_evidence)
        test = _status_value(ev.test_result)
        if ctype in ("TEST_PASS", "REGRESSION_FREE"):
            if ev.test_result is None or not ev.tests_executed:
                return CriterionEvaluation(criterion.criterion_id, EvidenceStatus.UNAVAILABLE, None, 0.0, ("test_result",), "Required test evidence was not executed or is unavailable.", "NONE")
            evidence.append("test_result")
            if "REGRESSION" in ctype and (_has_regression(ev) or final in ("NOT_VERIFIED", "FAILED")):
                return CriterionEvaluation(criterion.criterion_id, EvaluationStatus.FAILED, False, 0.0, tuple(evidence + ["final_verification"]), "Regression or final verification failure detected.", "STRONG")
            passed = test in ("PASS", "PASSED", "REGRESSION_FREE") and not _has_regression(ev)
            return CriterionEvaluation(criterion.criterion_id, EvaluationStatus.VERIFIED if passed else EvaluationStatus.FAILED, passed, 1.0 if passed else 0.0, tuple(evidence), "Structured test evidence indicates pass." if passed else "Structured test evidence indicates failure.", "STRONG")
        if ctype == "FILE_CHANGE":
            evidence.append("changed_paths")
            expected = set(criterion.requirement_ids)  # no path inference from prose
            touched = bool(ev.changed_paths or ev.expected_paths_touched)
            satisfied = touched
            return CriterionEvaluation(criterion.criterion_id, EvaluationStatus.VERIFIED if touched else EvaluationStatus.INSUFFICIENT_EVIDENCE, touched if touched else None, 1.0 if touched else 0.0, tuple(evidence), "Observed bounded mutation evidence." if touched else "No mutation evidence available.", "MEDIUM" if touched else "NONE")
        if ctype in ("VERIFICATION", "COMPLETION"):
            evidence.append("final_verification_evidence" if ctype == "VERIFICATION" else "completion_evidence")
            source = ev.final_verification_evidence if ctype == "VERIFICATION" else ev.completion_evidence
            value = _status_value(source)
            if source is None or not value:
                return CriterionEvaluation(criterion.criterion_id, EvaluationStatus.INSUFFICIENT_EVIDENCE, None, 0.0, tuple(evidence), "Required authoritative evidence is missing.", "NONE")
            satisfied = value in ("VERIFIED", "COMPLETE", "PASSED")
            return CriterionEvaluation(criterion.criterion_id, EvaluationStatus.VERIFIED if satisfied else EvaluationStatus.FAILED, satisfied, 1.0 if satisfied else 0.0, tuple(evidence), f"Authoritative status is {value}.", "STRONG")
        if ctype == "NO_UNRELATED_CHANGE":
            evidence.append("unexpected_modifications")
            satisfied = not ev.unexpected_modifications and not ev.forbidden_changes_detected
            return CriterionEvaluation(criterion.criterion_id, EvaluationStatus.VERIFIED if satisfied else EvaluationStatus.FAILED, satisfied, 1.0 if satisfied else 0.0, tuple(evidence), "No unexpected or forbidden changes." if satisfied else "Unexpected or forbidden changes detected.", "STRONG")
        if ev.final_verification_evidence is None and ev.completion_evidence is None and not ev.evidence_complete:
            return CriterionEvaluation(criterion.criterion_id, EvaluationStatus.INSUFFICIENT_EVIDENCE, None, 0.0, ("evidence_complete",), "Evidence is incomplete; criterion cannot be evaluated.", "NONE")
        return CriterionEvaluation(criterion.criterion_id, status, None if status is EvaluationStatus.INCOMPLETE else status is EvaluationStatus.VERIFIED, 1.0 if status is EvaluationStatus.VERIFIED else 0.0, ("benchmark_status",), "Criterion evaluated from bounded benchmark status.", "MEDIUM")

    def _task_success(self, task: EvaluationTask, run: BenchmarkTaskRun, criteria: Sequence[CriterionEvaluation]) -> tuple[float, EvaluationStatus, tuple[str, ...]]:
        if run.status is BenchmarkTaskStatus.BLOCKED: return 0.0, EvaluationStatus.BLOCKED, ("task_status",)
        if run.status in (BenchmarkTaskStatus.UNAVAILABLE, BenchmarkTaskStatus.INFRASTRUCTURE_ERROR, BenchmarkTaskStatus.TIMED_OUT): return 0.0, EvaluationStatus.UNAVAILABLE, ("task_status",)
        if run.evidence.final_verification_evidence is None or run.evidence.completion_evidence is None:
            return 0.0, EvaluationStatus.INSUFFICIENT_EVIDENCE, ("final_verification_evidence", "completion_evidence")
        final = _status_value(run.evidence.final_verification_evidence)
        complete = _status_value(run.evidence.completion_evidence)
        required = [item for item in criteria if item.criterion_id and next((c for c in task.success_criteria if c.criterion_id == item.criterion_id), None) and next(c for c in task.success_criteria if c.criterion_id == item.criterion_id).required]
        if any(item.status in (EvaluationStatus.FAILED, EvaluationStatus.BLOCKED) for item in required) or run.evidence.forbidden_changes_detected or _has_regression(run.evidence) or final not in ("VERIFIED", "PASSED") or complete not in ("COMPLETE", "COMPLETED", "PASSED"):
            return 0.0 if required else 0.5, EvaluationStatus.FAILED, ("final_verification", "completion", "required_criteria")
        return 1.0, EvaluationStatus.VERIFIED, ("final_verification", "completion")

    def _tests(self, run: BenchmarkTaskRun) -> tuple[float, EvaluationStatus, tuple[str, ...]]:
        if not run.evidence.tests_requested: return 1.0, EvaluationStatus.PASS, ("tests_not_required",)
        if not run.evidence.tests_executed or run.evidence.test_result is None: return 0.0, EvidenceStatus.UNAVAILABLE, ("test_result",)
        status = _status_value(run.evidence.test_result)
        if _has_regression(run.evidence): return 0.0, EvidenceStatus.REGRESSION_DETECTED, ("regression", "test_result")
        if status in ("PASS", "PASSED"): return 1.0, EvaluationStatus.VERIFIED, ("test_result",)
        return 0.0, EvaluationStatus.FAILED, ("test_result",)

    def _quality(self, run: BenchmarkTaskRun) -> tuple[float, EvaluationStatus, tuple[str, ...]]:
        negatives = int(bool(run.evidence.forbidden_changes_detected)) + int(bool(run.evidence.unexpected_modifications)) + int(_has_regression(run.evidence)) + int(bool(run.evidence.policy_safety_blocks))
        score = max(0.0, 1.0 - 0.25 * negatives)
        return score, EvaluationStatus.FAILED if negatives else EvaluationStatus.PASS, ("scope", "safety", "regression")

    def _efficiency(self, run: BenchmarkTaskRun, correctness: float) -> tuple[float, EvaluationStatus, tuple[str, ...]]:
        if correctness <= 0 or not run.evidence.budget_state: return 0.0, EvaluationStatus.UNAVAILABLE, ("budget_state",)
        budget = run.evidence.budget_state
        duration = _number(budget, "duration_seconds", run.evidence.duration_seconds)
        iterations = _number(budget, "iterations", _number(budget, "iteration_count", 0.0))
        calls = _number(budget, "tool_calls", _number(budget, "tool_call_count", 0.0))
        tests = _number(budget, "test_runs", 0.0)
        ratios = (duration / self.policy.duration_target_seconds, iterations / self.policy.iteration_target, calls / self.policy.tool_call_target, tests / self.policy.test_run_target)
        return max(0.0, min(1.0, 1.0 - sum(min(1.0, ratio) for ratio in ratios) / len(ratios))), EvaluationStatus.PASS, ("budget_state",)

    @staticmethod
    def _aggregate_dimension(scores: Sequence[TaskScore], name: str) -> ScoreDimension:
        field_name = "test_score" if name == "tests" else f"{name}_score"
        score = sum(getattr(item, field_name) for item in scores) / len(scores)
        return ScoreDimension(name, score, 0.0, score, EvaluationStatus.PASS, tuple(item.task_id for item in scores), "Arithmetic mean across all evaluated tasks, including failed and incomplete tasks.")


EvaluationScorer = BenchmarkScorer


def evaluate_benchmark(benchmark_result: BenchmarkResult, tasks: Sequence[EvaluationTask], *, weights: EvaluationWeights | None = None, policy: ScoringPolicy | None = None) -> EvaluationResult:
    return BenchmarkScorer(weights=weights, policy=policy).evaluate_benchmark(benchmark_result, tasks)


def _validate_score(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _status_value(value: Mapping[str, Any] | None) -> str:
    if not value: return ""
    for key in ("status", "overall_status", "result", "outcome", "completion_status"):
        item = value.get(key)
        if item is not None: return getattr(item, "value", str(item)).upper()
    return ""


def _has_regression(evidence: Any) -> bool:
    text = " ".join((str(evidence.execution_status), str(evidence.termination_reason), " ".join(evidence.failure_information)))
    status = _status_value(evidence.final_verification_evidence)
    return "REGRESSION" in text.upper() or "REGRESSION" in status or bool(evidence.final_verification_evidence and evidence.final_verification_evidence.get("regression_status") in ("REGRESSION_DETECTED", "FAILED"))


def _map_status(status: BenchmarkTaskStatus) -> EvaluationStatus:
    return {BenchmarkTaskStatus.PASSED: EvaluationStatus.PASS, BenchmarkTaskStatus.FAILED: EvaluationStatus.FAILED, BenchmarkTaskStatus.BLOCKED: EvaluationStatus.BLOCKED, BenchmarkTaskStatus.UNAVAILABLE: EvaluationStatus.UNAVAILABLE, BenchmarkTaskStatus.TIMED_OUT: EvaluationStatus.UNAVAILABLE, BenchmarkTaskStatus.INCOMPLETE_EVIDENCE: EvaluationStatus.INCOMPLETE}.get(status, EvaluationStatus.INCOMPLETE)


def _number(value: Mapping[str, Any], key: str, default: float) -> float:
    item = value.get(key, default)
    return float(item) if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) and item >= 0 else default


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum): return value.value
    if isinstance(value, Mapping): return {str(k): _serialize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)): return [_serialize(item) for item in value]
    if hasattr(value, "__dataclass_fields__"): return {name: _serialize(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, "to_dict") and not isinstance(value, (str, bytes)): return value.to_dict()
    return value


__all__ = ["BenchmarkScorer", "BenchmarkScore", "CriterionEvaluation", "EvaluationResult", "EvaluationScorer", "EvaluationStatus", "EvaluationWeights", "EvidenceStatus", "ScoreDimension", "ScoringPolicy", "TaskEvaluation", "TaskScore", "evaluate_benchmark"]
