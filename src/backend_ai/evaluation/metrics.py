"""Phase 8.5 deterministic, evidence-driven evaluation metrics.

The metrics layer transforms existing Phase 8.2 ``BenchmarkResult`` evidence
and Phase 8.3 ``EvaluationResult`` / ``BenchmarkScore`` objects into measurable,
deterministic metrics. It never reruns benchmarks, executes tests, or inspects
the filesystem. Missing evidence is never treated as success.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import math
from statistics import median
from types import MappingProxyType
from typing import Any

from backend_ai.evaluation.benchmark_runner import (
    BenchmarkResult,
    BenchmarkRunSummary,
    BenchmarkTaskRun,
    BenchmarkTaskStatus,
)
from backend_ai.evaluation.scoring import (
    BenchmarkScore,
    EvaluationResult,
    EvaluationStatus,
    ScoreDimension,
    TaskScore,
)
from backend_ai.evaluation.task_model import EvaluationTask


class MetricStatus(str, Enum):
    """Outcome of a single metric computation."""

    AVAILABLE = "AVAILABLE"
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"
    INCOMPARABLE = "INCOMPARABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class MetricName(str, Enum):
    """Names of supported core metrics."""

    TASK_SUCCESS_RATE = "task_success_rate"
    TASK_COMPLETION_RATE = "task_completion_rate"
    TEST_PASS_RATE = "test_pass_rate"
    CRITERION_SATISFACTION_RATE = "criterion_satisfaction_rate"
    FINAL_VERIFICATION_RATE = "final_verification_rate"
    REGRESSION_FREE_RATE = "regression_free_rate"
    FAILURE_RATE = "failure_rate"
    BLOCKED_RATE = "blocked_rate"
    INCOMPLETE_EVIDENCE_RATE = "incomplete_evidence_rate"
    INFRASTRUCTURE_FAILURE_RATE = "infrastructure_failure_rate"
    AVERAGE_TASK_SCORE = "average_task_score"
    AVERAGE_DURATION = "average_duration_seconds"
    MEDIAN_DURATION = "median_duration_seconds"
    TIMEOUT_RATE = "timeout_rate"
    AVERAGE_ATTEMPTS_PER_TASK = "average_attempts_per_task"
    AVERAGE_TESTS_PER_TASK = "average_tests_per_task"
    AVERAGE_MUTATIONS_PER_TASK = "average_mutations_per_task"
    AVERAGE_TOOL_CALLS_PER_TASK = "average_tool_calls_per_task"
    AVERAGE_ACTION_STEPS_PER_TASK = "average_action_steps_per_task"
    AVERAGE_ATTEMPTS_PER_SUCCESSFUL_TASK = "average_attempts_per_successful_task"
    AVERAGE_ATTEMPTS_PER_FAILED_TASK = "average_attempts_per_failed_task"
    AVERAGE_EXECUTION_TIME_PER_TASK = "average_execution_time_per_task"
    DIMENSION_SCORE = "dimension_score"
    SUCCESS_RATE_BY_CATEGORY = "success_rate_by_category"
    SUCCESS_RATE_BY_DIFFICULTY = "success_rate_by_difficulty"


def _number(value: object) -> float | None:
    """Return a finite non-negative number or None when evidence is absent."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _budget_number(budget: Mapping[str, Any] | None, *keys: str) -> float | None:
    """Read the first available numeric evidence from a bounded budget state."""

    if not budget:
        return None
    for key in keys:
        candidate = budget.get(key)
        value = _number(candidate)
        if value is not None:
            return value
    return None


@dataclass(frozen=True, slots=True)
class TaskMetrics:
    """Per-task metric evidence consumed by aggregation."""

    task_id: str
    category: str
    difficulty: str
    status: BenchmarkTaskStatus
    evaluation_status: EvaluationStatus | None
    success: bool
    tests_evaluated: bool
    tests_passed: bool
    final_verification_required: bool
    final_verification_passed: bool
    regression_free: bool
    duration_seconds: float
    attempts: float | None
    test_executions: float | None
    mutations: float | None
    tool_calls: float | None
    action_steps: float | None
    criterion_count: int
    satisfied_criteria: int
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    score: float | None = None
    regression_evidence_available: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))


def _task_metrics_from_run(task: EvaluationTask, run: BenchmarkTaskRun, task_score: TaskScore | None = None) -> TaskMetrics:
    """Derive per-task metric evidence strictly from existing run evidence."""

    ev = run.evidence
    execution_status = (ev.execution_status or run.status.value).upper()
    budget = ev.budget_state
    criteria = tuple(task.success_criteria)
    satisfied = 0
    for criterion in criteria:
        criterion_type = getattr(criterion.criterion_type, "value", str(criterion.criterion_type))
        if criterion_type == "TEST_PASS":
            satisfied += int(run.evidence.tests_executed and _test_evidence_passed(run.evidence))
        elif criterion_type in ("VERIFICATION", "COMPLETION"):
            source = ev.final_verification_evidence if criterion_type == "VERIFICATION" else ev.completion_evidence
            value = _status_value(source)
            satisfied += int(value in ("VERIFIED", "COMPLETE", "PASSED", "COMPLETED", "PASS"))
        elif criterion_type in ("REGRESSION_FREE", "NO_UNRELATED_CHANGE"):
            satisfied += int(
                not ev.forbidden_changes_detected
                and not ev.unexpected_modifications
                and not _regression_in(ev)
            )
        elif criterion_type == "BEHAVIOR":
            completed = ev.execution_completed and ev.execution_started
            final = _status_value(ev.final_verification_evidence)
            satisfied += int(completed and final in ("VERIFIED", "COMPLETE", "PASSED") and not _regression_in(ev))
        else:
            satisfied += int(ev.execution_completed and not _regression_in(ev))
    test_evaluated = ev.tests_executed and ev.test_result is not None
    test_passed = test_evaluated and _test_evidence_passed(ev) and not _regression_in(ev)
    final = _status_value(ev.final_verification_evidence)
    final_verification_required = any(
        getattr(criterion.criterion_type, "value", str(criterion.criterion_type)) in ("VERIFICATION", "COMPLETION", "REGRESSION_FREE") and criterion.required
        for criterion in criteria
    ) if criteria else bool(ev.final_verification_evidence)
    final_verification_passed = final in ("VERIFIED", "COMPLETE", "PASSED") if final_verification_required else False
    eligible = run.status not in (BenchmarkTaskStatus.SKIPPED, BenchmarkTaskStatus.UNAVAILABLE)
    if not eligible:
        final_verification_required = False
    if run.status is BenchmarkTaskStatus.INCOMPLETE_EVIDENCE:
        final_verification_passed = False
    regression_evidence_available = bool(
        ev.final_verification_evidence and ("regression_status" in ev.final_verification_evidence or ev.failure_information)
    ) or bool(ev.forbidden_changes_detected or ev.unexpected_modifications)
    evaluation_status = task_score.status if task_score is not None else _evaluation_status(run, final)
    success_value = task_score.status in (EvaluationStatus.VERIFIED, EvaluationStatus.PASS) if task_score is not None else run.status is BenchmarkTaskStatus.PASSED
    return TaskMetrics(
        task_id=run.task_id,
        category=run.category,
        difficulty=run.difficulty,
        status=run.status,
        evaluation_status=evaluation_status,
        success=success_value,
        tests_evaluated=test_evaluated,
        tests_passed=test_passed,
        final_verification_required=final_verification_required,
        final_verification_passed=final_verification_passed,
        regression_free=not _regression_in(ev) and not ev.forbidden_changes_detected,
        duration_seconds=ev.duration_seconds,
        attempts=_budget_number(budget, "iterations", "iteration_count", "attempts"),
        test_executions=_budget_number(budget, "test_runs", "test_executions", "tests_executed"),
        mutations=_number(ev.mutation_count) or _budget_number(budget, "mutations", "mutation_count"),
        tool_calls=_budget_number(budget, "tool_calls", "tool_call_count", "actions"),
        action_steps=_budget_number(budget, "action_steps", "steps"),
        criterion_count=len(criteria),
        satisfied_criteria=satisfied,
        evidence_ids=("task_status", "evidence", "budget_state") if budget else ("task_status", "evidence"),
        warnings=tuple(run.warnings) + (() if regression_evidence_available else ("regression evidence is unavailable",)),
        score=task_score.final_score if task_score is not None else None,
        regression_evidence_available=regression_evidence_available,
    )


def _test_evidence_passed(ev: Any) -> bool:
    value = _status_value(ev.test_result)
    return value in ("PASS", "PASSED", "REGRESSION_FREE")


def _status_value(value: Mapping[str, Any] | None) -> str:
    if not value:
        return ""
    for key in ("status", "overall_status", "result", "outcome", "completion_status"):
        item = value.get(key)
        if item is not None:
            return getattr(item, "value", str(item)).upper()
    return ""


def _regression_in(ev: Any) -> bool:
    text = " ".join((str(ev.execution_status), str(ev.termination_reason), " ".join(ev.failure_information))).upper()
    status = _status_value(ev.final_verification_evidence)
    regression_status = str(ev.final_verification_evidence.get("regression_status", "")).upper() if ev.final_verification_evidence else ""
    return any(token in text for token in ("REGRESSION_DETECTED", "REGRESSION_FAILED")) or status in ("REGRESSION_DETECTED", "REGRESSION_FAILED") or regression_status in ("REGRESSION_DETECTED", "REGRESSION_FAILED")


def _evaluation_status(run: BenchmarkTaskRun, final_verification: str) -> EvaluationStatus | None:
    mapping = {
        BenchmarkTaskStatus.PASSED: EvaluationStatus.VERIFIED if final_verification in ("VERIFIED", "COMPLETE", "PASSED") else EvaluationStatus.PASS,
        BenchmarkTaskStatus.FAILED: EvaluationStatus.FAILED,
        BenchmarkTaskStatus.BLOCKED: EvaluationStatus.BLOCKED,
        BenchmarkTaskStatus.TIMED_OUT: EvaluationStatus.UNAVAILABLE,
        BenchmarkTaskStatus.SKIPPED: None,
        BenchmarkTaskStatus.UNAVAILABLE: EvaluationStatus.UNAVAILABLE,
        BenchmarkTaskStatus.INFRASTRUCTURE_ERROR: EvaluationStatus.UNAVAILABLE,
        BenchmarkTaskStatus.INCOMPLETE_EVIDENCE: EvaluationStatus.INCOMPLETE,
        BenchmarkTaskStatus.PENDING: None,
        BenchmarkTaskStatus.RUNNING: None,
    }
    return mapping.get(run.status, EvaluationStatus.INCOMPLETE)


@dataclass(frozen=True, slots=True)
class SingleMetric:
    """One computed metric bound to its evidence and eligibility rules."""

    name: str
    value: float | None
    numerator: int | None
    denominator: int
    status: MetricStatus
    sample_size: int
    eligible_count: int
    excluded_count: int
    exclusion_reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "exclusion_reasons", tuple(sorted(set(self.exclusion_reasons))))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        if self.denominator < 0:
            raise ValueError("denominator must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


def _eligible_runs(runs: Sequence[BenchmarkTaskRun]) -> tuple[BenchmarkTaskRun, ...]:
    """Eligible tasks exclude those skipped because the benchmark was stopped."""

    return tuple(run for run in runs if run.status is not BenchmarkTaskStatus.SKIPPED)


def _count(runs: Sequence[BenchmarkTaskRun], status: BenchmarkTaskStatus) -> int:
    return sum(1 for run in runs if run.status is status)


def collect_benchmark_metrics(benchmark_result: BenchmarkResult, tasks: Sequence[EvaluationTask] | None = None, *, benchmark_score: BenchmarkScore | None = None) -> "BenchmarkMetrics":
    """Compute all core metrics from a completed benchmark result.

    The optional ``tasks`` sequence supplies category/difficulty labels and
    success-criterion counts. When omitted, category/difficulty labels are
    read from ``BenchmarkTaskRun`` evidence and criterion counts fall back to
    status-based eligibility.
    """

    runs = benchmark_result.task_runs
    task_map: dict[str, EvaluationTask] = {}
    if tasks is not None:
        task_map = {task.task_id: task for task in tasks}
    score_map = {
        item.task_id: item
        for item in (benchmark_score.task_scores if benchmark_score is not None else ())
        if isinstance(item, TaskScore)
    }
    metrics = TaskMetricsCollection(benchmark_result, task_map, score_map)
    eligible = _eligible_runs(runs)
    excluded = tuple(run for run in runs if run.status is BenchmarkTaskStatus.SKIPPED)
    exclusion_reasons = ("skipped because benchmark terminated before task started",) if excluded else ()
    summary = benchmark_result.summary
    return BenchmarkMetrics(
        benchmark_id=benchmark_result.benchmark_id,
        benchmark_version=benchmark_result.benchmark_version,
        task_metrics=metrics.per_task,
        metrics=_core_metrics(
            summary,
            len(runs),
            len(eligible),
            excluded,
            exclusion_reasons,
            benchmark_result,
            metrics,
            task_map,
        ),
        by_category=metrics.by_category(),
        by_difficulty=metrics.by_difficulty(),
        sample_size=len(runs),
        eligible_count=len(eligible),
        excluded_count=len(excluded),
        warnings=metrics.warnings,
    )


def _core_metrics(
    summary: BenchmarkRunSummary,
    total: int,
    eligible: int,
    excluded: tuple[BenchmarkTaskRun, ...],
    exclusion_reasons: tuple[str, ...],
    result: BenchmarkResult,
    metrics: "TaskMetricsCollection",
    task_map: dict[str, EvaluationTask],
) -> tuple[SingleMetric, ...]:
    out: list[SingleMetric] = []

    def metric(
        name: str,
        value: float | None,
        numerator: int | None,
        denominator: int,
        status: MetricStatus,
        evidence: tuple[str, ...] = (),
    ) -> None:
        out.append(SingleMetric(name, value, numerator, denominator, status, len(result.task_runs), eligible, len(excluded), exclusion_reasons, evidence))

    def rate(numerator: int, denominator: int) -> tuple[float | None, MetricStatus]:
        if denominator <= 0:
            return None, MetricStatus.UNAVAILABLE
        return numerator / denominator, MetricStatus.AVAILABLE

    value, status = rate(summary.completed_tasks, eligible)
    metric(MetricName.TASK_SUCCESS_RATE, value, summary.completed_tasks, eligible, status, ("task_status",))
    value, status = rate(summary.completed_tasks, total)
    metric(MetricName.TASK_COMPLETION_RATE, value, summary.completed_tasks, total, status, ("task_status",))
    test_value, test_status, test_numerator, test_denominator = metrics.test_pass_rate_details()
    criterion_value, criterion_status, criterion_numerator, criterion_denominator = metrics.criterion_satisfaction_details()
    final_value, final_status, final_numerator, final_denominator = metrics.final_verification_details()
    regression_value, regression_status, regression_numerator, regression_denominator = metrics.regression_free_details()
    metric(MetricName.TEST_PASS_RATE, test_value, test_numerator, test_denominator, test_status)
    metric(MetricName.CRITERION_SATISFACTION_RATE, criterion_value, criterion_numerator, criterion_denominator, criterion_status)
    metric(MetricName.FINAL_VERIFICATION_RATE, final_value, final_numerator, final_denominator, final_status)
    metric(MetricName.REGRESSION_FREE_RATE, regression_value, regression_numerator, regression_denominator, regression_status)
    value, status = rate(summary.failed_tasks, eligible)
    metric(MetricName.FAILURE_RATE, value, summary.failed_tasks, eligible, status, ("task_status",))
    value, status = rate(summary.blocked_tasks, eligible)
    metric(MetricName.BLOCKED_RATE, value, summary.blocked_tasks, eligible, status, ("task_status",))
    value, status = rate(summary.evidence_incomplete_tasks, eligible)
    metric(MetricName.INCOMPLETE_EVIDENCE_RATE, value, summary.evidence_incomplete_tasks, eligible, status, ("evidence_complete",))
    value, status = rate(summary.infrastructure_failures, eligible)
    metric(MetricName.INFRASTRUCTURE_FAILURE_RATE, value, summary.infrastructure_failures, eligible, status, ("task_status",))
    score_value, score_status = metrics.average_task_score()
    duration_value, duration_status = metrics.average_duration()
    median_value, median_status = metrics.median_duration()
    metric(MetricName.AVERAGE_TASK_SCORE, score_value, None, max(eligible, 0), score_status)
    metric(MetricName.AVERAGE_DURATION, duration_value, None, max(eligible, 0), duration_status)
    metric(MetricName.MEDIAN_DURATION, median_value, None, max(eligible, 0), median_status)
    value, status = rate(summary.timed_out_tasks, eligible)
    metric(MetricName.TIMEOUT_RATE, value, summary.timed_out_tasks, eligible, status, ("task_status",))
    eligible_runs = tuple(
        run for run in result.task_runs if run.status is not BenchmarkTaskStatus.SKIPPED
    )
    for mean_name, mean_evidence, mean_label in (
        (MetricName.AVERAGE_ATTEMPTS_PER_TASK, "attempts", "budget_state"),
        (MetricName.AVERAGE_TESTS_PER_TASK, "test_executions", "budget_state"),
        (MetricName.AVERAGE_MUTATIONS_PER_TASK, "mutations", "budget_state"),
        (MetricName.AVERAGE_TOOL_CALLS_PER_TASK, "tool_calls", "budget_state"),
        (MetricName.AVERAGE_ACTION_STEPS_PER_TASK, "action_steps", "budget_state"),
        (MetricName.AVERAGE_EXECUTION_TIME_PER_TASK, "duration", "duration_seconds"),
    ):
        mean_value, mean_status = metrics.mean_metric(mean_evidence, eligible_runs)
        metric(mean_name, mean_value, None, max(eligible, 0), mean_status, (mean_label,))
    mean_value, mean_status = metrics.mean_metric("attempts", eligible_runs, only="success")
    metric(MetricName.AVERAGE_ATTEMPTS_PER_SUCCESSFUL_TASK, mean_value, None, max(eligible, 0), mean_status, ("budget_state",))
    mean_value, mean_status = metrics.mean_metric("attempts", eligible_runs, only="failure")
    metric(MetricName.AVERAGE_ATTEMPTS_PER_FAILED_TASK, mean_value, None, max(eligible, 0), mean_status, ("budget_state",))
    return tuple(out)


class TaskMetricsCollection:
    """Derives per-task metric evidence and aggregate breakdowns deterministically."""

    def __init__(self, result: BenchmarkResult, task_map: dict[str, EvaluationTask], score_map: Mapping[str, TaskScore] | None = None) -> None:
        scores = score_map or {}
        self.per_task: tuple[TaskMetrics, ...] = tuple(self._from_run(run, task_map.get(run.task_id), scores.get(run.task_id)) for run in sorted(result.task_runs, key=lambda run: run.task_id))
        self.warnings: tuple[str, ...] = tuple(sorted({warning for item in self.per_task for warning in item.warnings}))

    def _from_run(self, run: BenchmarkTaskRun, task: EvaluationTask | None, task_score: TaskScore | None = None) -> TaskMetrics:
        if task is None:
            from backend_ai.evaluation.task_model import EvaluationTask

            task = EvaluationTask(task_id=run.task_id, title=run.task_id, description=run.task_id, version=run.task_version or "1.0", category=run.category or "UNKNOWN", difficulty=run.difficulty or "MEDIUM")
        return _task_metrics_from_run(task, run, task_score)

    def _group(self, key: str) -> dict[str, list[TaskMetrics]]:
        groups: dict[str, list[TaskMetrics]] = defaultdict(list)
        for item in self.per_task:
            groups[getattr(item, key)].append(item)
        return dict(sorted(groups.items(), key=lambda entry: entry[0]))

    def by_category(self) -> tuple["CategoryMetrics", ...]:
        return tuple(CategoryMetrics(category=category, task_count=len(items), tasks=tuple(items)) for category, items in self._group("category").items())

    def by_difficulty(self) -> tuple["DifficultyMetrics", ...]:
        return tuple(DifficultyMetrics(difficulty=difficulty, task_count=len(items), tasks=tuple(items)) for difficulty, items in self._group("difficulty").items())

    def test_pass_rate(self) -> tuple[float | None, MetricStatus]:
        runs = [item for item in self.per_task if item.status not in (BenchmarkTaskStatus.SKIPPED, BenchmarkTaskStatus.UNAVAILABLE, BenchmarkTaskStatus.INFRASTRUCTURE_ERROR)]
        evaluated = [item for item in runs if item.tests_evaluated]
        if not evaluated:
            return None, MetricStatus.INSUFFICIENT_EVIDENCE
        passed = sum(1 for item in evaluated if item.tests_passed)
        return passed / len(evaluated), MetricStatus.AVAILABLE

    def test_pass_rate_details(self) -> tuple[float | None, MetricStatus, int | None, int]:
        runs = [item for item in self.per_task if item.status not in (BenchmarkTaskStatus.SKIPPED, BenchmarkTaskStatus.UNAVAILABLE, BenchmarkTaskStatus.INFRASTRUCTURE_ERROR)]
        evaluated = [item for item in runs if item.tests_evaluated]
        if not evaluated:
            return None, MetricStatus.INSUFFICIENT_EVIDENCE, None, 0
        passed = sum(1 for item in evaluated if item.tests_passed)
        return passed / len(evaluated), MetricStatus.AVAILABLE, passed, len(evaluated)

    def criterion_satisfaction_details(self) -> tuple[float | None, MetricStatus, int | None, int]:
        runs = [item for item in self.per_task if item.status is not BenchmarkTaskStatus.SKIPPED]
        denominator = sum(item.criterion_count for item in runs)
        if denominator == 0:
            return None, MetricStatus.INSUFFICIENT_EVIDENCE, None, 0
        numerator = sum(item.satisfied_criteria for item in runs)
        return numerator / denominator, MetricStatus.AVAILABLE, numerator, denominator

    def final_verification_details(self) -> tuple[float | None, MetricStatus, int | None, int]:
        required = [item for item in self.per_task if item.final_verification_required]
        if not required:
            return None, MetricStatus.NOT_APPLICABLE, None, 0
        numerator = sum(1 for item in required if item.final_verification_passed)
        return numerator / len(required), MetricStatus.AVAILABLE, numerator, len(required)

    def regression_free_details(self) -> tuple[float | None, MetricStatus, int | None, int]:
        runs = [item for item in self.per_task if item.status is not BenchmarkTaskStatus.SKIPPED]
        if not runs:
            return None, MetricStatus.UNAVAILABLE, None, 0
        if any(not item.regression_evidence_available for item in runs):
            return None, MetricStatus.INSUFFICIENT_EVIDENCE, None, len(runs)
        numerator = sum(1 for item in runs if item.regression_free)
        return numerator / len(runs), MetricStatus.AVAILABLE, numerator, len(runs)

    def criterion_satisfaction_rate(self) -> tuple[float | None, MetricStatus]:
        runs = [item for item in self.per_task if item.status not in (BenchmarkTaskStatus.SKIPPED,)]
        criteria_total = sum(item.criterion_count for item in runs)
        if criteria_total == 0:
            return None, MetricStatus.INSUFFICIENT_EVIDENCE
        satisfied = sum(item.satisfied_criteria for item in runs)
        return satisfied / criteria_total, MetricStatus.AVAILABLE

    def final_verification_rate(self) -> tuple[float | None, MetricStatus]:
        required = [item for item in self.per_task if item.final_verification_required]
        if not required:
            return None, MetricStatus.NOT_APPLICABLE
        passed = sum(1 for item in required if item.final_verification_passed)
        return passed / len(required), MetricStatus.AVAILABLE

    def regression_free_rate(self) -> tuple[float | None, MetricStatus]:
        runs = [item for item in self.per_task if item.status not in (BenchmarkTaskStatus.SKIPPED,)]
        if not runs:
            return None, MetricStatus.UNAVAILABLE
        if any(not item.regression_evidence_available for item in runs):
            return None, MetricStatus.INSUFFICIENT_EVIDENCE
        free = sum(1 for item in runs if item.regression_free)
        return free / len(runs), MetricStatus.AVAILABLE

    def average_task_score(self) -> tuple[float | None, MetricStatus]:
        scored = [item for item in self.per_task if item.score is not None]
        if not scored:
            return None, MetricStatus.INSUFFICIENT_EVIDENCE
        total = sum(item.score for item in scored if item.score is not None)
        return total / len(scored), MetricStatus.AVAILABLE

    def average_duration(self) -> tuple[float | None, MetricStatus]:
        durations = self._positive_durations()
        if not durations:
            return None, MetricStatus.INSUFFICIENT_EVIDENCE
        return sum(durations) / len(durations), MetricStatus.AVAILABLE

    def median_duration(self) -> tuple[float | None, MetricStatus]:
        durations = self._positive_durations()
        if len(durations) < 2:
            return None, MetricStatus.INSUFFICIENT_EVIDENCE
        return float(median(durations)), MetricStatus.AVAILABLE

    def _positive_durations(self) -> list[float]:
        return [item.duration_seconds for item in self.per_task if item.duration_seconds > 0]

    def mean_metric(self, name: str, eligible: tuple[BenchmarkTaskRun, ...], only: str | None = None) -> tuple[float | None, MetricStatus]:
        eligible_ids = {run.task_id for run in eligible}
        items = [item for item in self.per_task if item.task_id in eligible_ids]
        if only == "success":
            items = [item for item in items if item.success]
        elif only == "failure":
            items = [item for item in items if not item.success]
        values: list[float] = []
        for item in items:
            value = getattr(item, {
                "attempts": "attempts",
                "test_executions": "test_executions",
                "mutations": "mutations",
                "tool_calls": "tool_calls",
                "action_steps": "action_steps",
                "duration": "duration_seconds",
            }[name])
            if value is not None:
                values.append(float(value))
        if not values:
            return None, MetricStatus.INSUFFICIENT_EVIDENCE
        return sum(values) / len(values), MetricStatus.AVAILABLE


@dataclass(frozen=True, slots=True)
class CategoryMetrics:
    category: str
    task_count: int
    success_rate: float | None = None
    test_pass_rate: float | None = None
    average_task_score: float | None = None
    tasks: tuple[TaskMetrics, ...] = ()

    def __post_init__(self) -> None:
        rates = _category_rates(self.tasks)
        object.__setattr__(self, "success_rate", rates[0])
        object.__setattr__(self, "test_pass_rate", rates[1])
        object.__setattr__(self, "average_task_score", rates[2])

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True, slots=True)
class DifficultyMetrics:
    difficulty: str
    task_count: int
    success_rate: float | None = None
    test_pass_rate: float | None = None
    average_task_score: float | None = None
    tasks: tuple[TaskMetrics, ...] = ()

    def __post_init__(self) -> None:
        rates = _category_rates(self.tasks)
        object.__setattr__(self, "success_rate", rates[0])
        object.__setattr__(self, "test_pass_rate", rates[1])
        object.__setattr__(self, "average_task_score", rates[2])

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


def _category_rates(tasks: Sequence[TaskMetrics]) -> tuple[float | None, float | None, float | None]:
    eligible = [item for item in tasks if item.status is not BenchmarkTaskStatus.SKIPPED]
    success = sum(1 for item in eligible if item.success)
    success_rate = success / len(eligible) if eligible else None
    evaluated = [item for item in eligible if item.tests_evaluated]
    test_rate = sum(1 for item in evaluated if item.tests_passed) / len(evaluated) if evaluated else None
    scored = [item for item in eligible if item.score is not None]
    if scored:
        average = sum(item.score for item in scored if item.score is not None) / len(scored)
    else:
        average = None
    return success_rate, test_rate, average


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """Complete deterministic metrics result for one benchmark execution."""

    benchmark_id: str
    benchmark_version: str
    task_metrics: tuple[TaskMetrics, ...]
    metrics: tuple[SingleMetric, ...]
    by_category: tuple[CategoryMetrics, ...]
    by_difficulty: tuple[DifficultyMetrics, ...]
    sample_size: int
    eligible_count: int
    excluded_count: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_metrics", tuple(sorted(self.task_metrics, key=lambda item: item.task_id)))
        object.__setattr__(self, "metrics", tuple(sorted(self.metrics, key=lambda item: item.name)))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))

    def metric(self, name: str) -> SingleMetric | None:
        """Look up one metric by its canonical name."""

        for item in self.metrics:
            if item.name == name:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def dimension_scores(self) -> dict[str, float | None]:
        """Aggregate dimension scores traceable to existing task scores."""

        return {item.name: item.value for item in self.metrics if item.name.startswith("dimension_")}


def collect_metrics(result: BenchmarkResult, tasks: Sequence[EvaluationTask] | None = None, *, benchmark_score: BenchmarkScore | None = None) -> BenchmarkMetrics:
    """Public entry point for Phase 8.5 metric collection.

    Consumes a completed ``BenchmarkResult`` and optional task definitions.
    When a Phase 8.3 ``BenchmarkScore`` is supplied, its task scores are
    merged as additional traceable evidence without changing eligibility
    rules.
    """

    collected = collect_benchmark_metrics(result, tasks, benchmark_score=benchmark_score)
    if benchmark_score is None:
        return collected
    return _merge_score(collected, benchmark_score)


def _merge_score(collected: BenchmarkMetrics, score: BenchmarkScore) -> BenchmarkMetrics:
    """Attach Phase 8.3 score evidence as an additional dimension metric."""

    merged: list[SingleMetric] = list(collected.metrics)
    existing = {item.name for item in merged}
    eligible = len(score.task_scores)
    for item in score.dimension_scores:
        name = f"dimension_{getattr(item, 'name', item)}"
        if name in existing:
            continue
        dimension_score: float | None = None
        if eligible:
            raw = getattr(item, "score", None)
            if raw is None and getattr(item, "final_score", None) is not None:
                raw = item.final_score
            elif raw is None and hasattr(score, "aggregate_score"):
                raw = score.aggregate_score
            dimension_score = float(raw) if raw is not None else None
        evidence_ids: tuple[str, ...] = ()
        if hasattr(item, "evidence_ids"):
            evidence_ids = tuple(sorted(set(item.evidence_ids)))
        merged.append(
            SingleMetric(
                name,
                dimension_score,
                eligible,
                eligible,
                MetricStatus.AVAILABLE if eligible else MetricStatus.UNAVAILABLE,
                len(collected.task_metrics),
                eligible,
                0,
                (),
                evidence_ids,
            )
        )
    return BenchmarkMetrics(
        collected.benchmark_id,
        collected.benchmark_version,
        collected.task_metrics,
        tuple(merged),
        collected.by_category,
        collected.by_difficulty,
        collected.sample_size,
        collected.eligible_count,
        collected.excluded_count,
        collected.warnings,
    )


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {name: _serialize(getattr(value, name)) for name in value.__dataclass_fields__}
    return value


__all__ = [
    "BenchmarkMetrics",
    "CategoryMetrics",
    "DifficultyMetrics",
    "MetricName",
    "MetricStatus",
    "SingleMetric",
    "TaskMetrics",
    "TaskMetricsCollection",
    "collect_benchmark_metrics",
    "collect_metrics",
]
