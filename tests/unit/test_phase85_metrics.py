"""Phase 8.5 metric collection behavioral contract.

Verifies that metric collection is deterministic, evidence-driven,
bound to explicit eligibility denominators, and never treats missing
evidence as success.
"""
from __future__ import annotations

import json

import pytest

from backend_ai.evaluation import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkScore,
    BenchmarkTaskRun,
    BenchmarkTaskStatus,
    MetricName,
    MetricStatus,
    ScoringPolicy,
    TaskMetrics,
    EvaluationStatus,
    SuccessCriterion,
    SuccessCriterionType,
    TestDefinition,
    EvaluationTestType,
    EvaluationTask,
    EvaluationTaskCategory,
    EvaluationDifficulty,
    collect_metrics,
)
from backend_ai.evaluation.benchmark_runner import BenchmarkRunSummary


def _budget_state(iterations: int | None = None, tool_calls: int | None = None) -> dict:
    budget = {"duration_seconds": 10.0}
    if iterations is not None:
        budget["iterations"] = iterations
    if tool_calls is not None:
        budget["tool_calls"] = tool_calls
    return budget


def _task_run(
    task_id: str = "EVAL-001",
    status: BenchmarkTaskStatus = BenchmarkTaskStatus.PASSED,
    tests_executed: bool = True,
    test_passed: bool = True,
    duration: float = 10.0,
    budget: dict | None = None,
    forbidden: bool = False,
    unexpected: bool = False,
    final_verification: str = "VERIFIED",
) -> BenchmarkTaskRun:
    from backend_ai.evaluation.benchmark_runner import BenchmarkEvidence

    evidence = BenchmarkEvidence(
        execution_started=True,
        execution_completed=True,
        execution_status="PASSED" if status is BenchmarkTaskStatus.PASSED else status.value,
        duration_seconds=duration,
        termination_reason="COMPLETED",
        workspace_identity=f"ws-{task_id}",
        project_definition_identity=f"proj-{task_id}",
        task_identity=task_id,
        cleanup_status="cleaned",
        changed_paths=("src/a.py",),
        expected_paths_touched=("src/a.py",),
        unexpected_modifications=("src/unexpected.py",) if unexpected else (),
        forbidden_changes_detected=(".env",) if forbidden else (),
        mutation_count=1,
        mutation_verification=None,
        tests_requested=True,
        tests_executed=tests_executed,
        test_result={"status": "PASS" if test_passed else "FAIL"} if tests_executed else None,
        completion_evidence={"status": "COMPLETE"},
        final_verification_evidence={"status": final_verification} if final_verification else None,
        stop_condition_evidence=None,
        failure_information=(),
        recovery_state=None,
        budget_state=budget or _budget_state(3, 10),
        policy_safety_blocks=(),
        artifacts=(),
        bounded_logs=(),
        evidence_complete=True,
        warnings=(),
    )
    return BenchmarkTaskRun(
        task_id, "1.0", "API_ENDPOINT", "MEDIUM", status, 0.0, 1.0, duration, None, evidence, None, (), (), (),
    )


def _passed_result(runs: tuple) -> BenchmarkResult:
    return BenchmarkResult(
        "bench-001",
        "1.0.0",
        "COMPLETED",
        runs,
        BenchmarkRunSummary(len(runs), sum(1 for r in runs if r.status is BenchmarkTaskStatus.PASSED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.FAILED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.BLOCKED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.TIMED_OUT), sum(1 for r in runs if r.status is BenchmarkTaskStatus.SKIPPED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.UNAVAILABLE), sum(1 for r in runs if r.status is BenchmarkTaskStatus.INFRASTRUCTURE_ERROR), sum(1 for r in runs if r.status is BenchmarkTaskStatus.INCOMPLETE_EVIDENCE)),
        20.0,
        "COMPLETED",
        (),
        {"task_order": [run.task_id for run in runs], "project_root_supplied": True, "deterministic_mode": True, "scoring": "8.5"},
        False,
        False,
        None,
    )


@pytest.mark.parametrize("status", BenchmarkTaskStatus)
def test_metrics_module_accepts_all_task_statuses(status: BenchmarkTaskStatus) -> None:
    """Metrics must bound over every defined task status without crashing."""

    result = _passed_result((_task_run("EVAL-001", status, tests_executed=False),))
    metrics = collect_metrics(result)
    assert metrics.sample_size == 1
    assert metrics.eligible_count == (0 if status is BenchmarkTaskStatus.SKIPPED else 1)


def test_task_success_rate_excludes_skipped_tasks() -> None:
    runs = (
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
        _task_run("EVAL-003", BenchmarkTaskStatus.SKIPPED),
    )
    metrics = collect_metrics(_passed_result(runs))
    success = metrics.metric(MetricName.TASK_SUCCESS_RATE)
    assert success is not None
    assert success.denominator == 2
    assert success.eligible_count == 2
    assert success.excluded_count == 1
    assert success.value == 0.5
    completion = metrics.metric(MetricName.TASK_COMPLETION_RATE)
    assert completion is not None
    assert completion.denominator == 3


def test_missing_test_evidence_is_not_success() -> None:
    runs = (_task_run("EVAL-001", BenchmarkTaskStatus.PASSED, tests_executed=False),)
    metrics = collect_metrics(_passed_result(runs))
    test_rate = metrics.metric(MetricName.TEST_PASS_RATE)
    assert test_rate is not None
    assert test_rate.status is MetricStatus.INSUFFICIENT_EVIDENCE
    assert test_rate.value is None


def test_failed_test_evidence_fails_the_test_rate() -> None:
    runs = (_task_run("EVAL-001", BenchmarkTaskStatus.PASSED, test_passed=False),)
    metrics = collect_metrics(_passed_result(runs))
    test_rate = metrics.metric(MetricName.TEST_PASS_RATE)
    assert test_rate is not None
    assert test_rate.value == 0.0


def test_regresion_evidence_fails_regression_free_rate() -> None:
    runs = (_task_run("EVAL-001", BenchmarkTaskStatus.PASSED, forbidden=True),)
    metrics = collect_metrics(_passed_result(runs))
    regression = metrics.metric(MetricName.REGRESSION_FREE_RATE)
    assert regression is not None
    assert regression.value == 0.0


def test_forbidden_changes_fail_final_verification() -> None:
    runs = (_task_run("EVAL-001", BenchmarkTaskStatus.INCOMPLETE_EVIDENCE, final_verification="VERIFIED"),)
    metrics = collect_metrics(_passed_result(runs))
    verification = metrics.metric(MetricName.FINAL_VERIFICATION_RATE)
    assert verification is not None
    assert verification.value == 0.0


def test_timeout_rate_counts_timed_out_tasks() -> None:
    runs = (
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.TIMED_OUT),
    )
    metrics = collect_metrics(_passed_result(runs))
    timeout = metrics.metric(MetricName.TIMEOUT_RATE)
    assert timeout is not None
    assert timeout.value == 0.5


def test_duration_metrics_require_multiple_samples() -> None:
    metrics = collect_metrics(_passed_result((_task_run("EVAL-001"),)))
    median = metrics.metric(MetricName.MEDIAN_DURATION)
    assert median is not None
    assert median.status is MetricStatus.INSUFFICIENT_EVIDENCE
    assert median.value is None


def test_benchmark_score_merge_attaches_dimension_evidence() -> None:
    score = BenchmarkScore(
        "bench-001",
        ({"task_id": "EVAL-001", "final_score": 0.8} if False else _task_score("EVAL-001"),),
        0.8,
        80.0,
        ("task_success", "tests", "code_quality", "efficiency"),
        1,
        0,
        0,
        0,
        0,
        "HIGH",
        1.0,
    )
    runs = (_task_run("EVAL-001"),)
    metrics = collect_metrics(_passed_result(runs), benchmark_score=score)
    dimension = metrics.metric("dimension_task_success")
    assert dimension is not None
    assert dimension.value == 0.8


def _task_score(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "task_success_score": 0.8,
        "test_score": 0.8,
        "code_quality_score": 0.8,
        "efficiency_score": 0.8,
        "final_score": 0.8,
        "percentage": 80.0,
        "status": EvaluationStatus.PASS,
        "dimensions": (),
        "evaluation_status": EvaluationStatus.PASS,
    }


def test_metrics_are_deterministic_for_identical_inputs() -> None:
    runs = (_task_run("EVAL-001"), _task_run("EVAL-002", BenchmarkTaskStatus.FAILED))
    first = collect_metrics(_passed_result(runs))
    second = collect_metrics(_passed_result(runs))
    assert first.to_json() == second.to_json()


def test_metric_lookup_returns_none_for_unknown_name() -> None:
    metrics = collect_metrics(_passed_result((_task_run("EVAL-001"),)))
    assert metrics.metric("unknown_metric") is None


def test_category_breakdown_groups_by_task_category() -> None:
    runs = (
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    )
    metrics = collect_metrics(_passed_result(runs))
    categories = {item.category: item for item in metrics.by_category}
    assert "API_ENDPOINT" in categories


def test_difficulty_breakdown_groups_by_task_difficulty() -> None:
    runs = (_task_run("EVAL-001", BenchmarkTaskStatus.PASSED),)
    metrics = collect_metrics(_passed_result(runs))
    difficulties = {item.difficulty: item for item in metrics.by_difficulty}
    assert "MEDIUM" in difficulties


def test_task_metrics_immutable_and_canonical() -> None:
    runs = (_task_run("EVAL-001"),)
    metrics = collect_metrics(_passed_result(runs))
    task = metrics.task_metrics[0]
    with pytest.raises(AttributeError):
        task.success = not task.success  # frozen slots dataclass rejects mutation
    payload = json.loads(metrics.to_json())
    assert payload["task_metrics"][0]["task_id"] == "EVAL-001"
    assert payload["metrics"][0]["name"] < payload["metrics"][-1]["name"]
