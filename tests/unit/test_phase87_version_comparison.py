"""Phase 8.7 version comparison behavioral contract.

Verifies category/difficulty-level deltas reuse Phase 8.4 semantics,
never duplicate comparison logic, and remain deterministic.
"""
from __future__ import annotations

import json

import pytest

from backend_ai.evaluation import (
    BenchmarkResult,
    BenchmarkRunSummary,
    BenchmarkTaskRun,
    BenchmarkTaskStatus,
    ComparisonConfig,
    VersionMetricsComparison,
    compare_evaluation_metrics,
)


def _task_run(task_id: str, status: BenchmarkTaskStatus, category: str = "API_ENDPOINT", difficulty: str = "MEDIUM", tests_executed: bool = True, test_passed: bool = True) -> BenchmarkTaskRun:
    from backend_ai.evaluation.benchmark_runner import BenchmarkEvidence

    evidence = BenchmarkEvidence(
        execution_started=True,
        execution_completed=True,
        execution_status=status.value,
        duration_seconds=10.0,
        termination_reason="COMPLETED",
        workspace_identity=f"ws-{task_id}",
        project_definition_identity=f"proj-{task_id}",
        task_identity=task_id,
        cleanup_status="cleaned",
        changed_paths=(),
        expected_paths_touched=(),
        unexpected_modifications=(),
        forbidden_changes_detected=(),
        mutation_count=0,
        mutation_verification=None,
        tests_requested=True,
        tests_executed=tests_executed,
        test_result={"status": "PASS" if test_passed else "FAIL"} if tests_executed else None,
        completion_evidence={"status": "COMPLETE"},
        final_verification_evidence={"status": "VERIFIED"},
        stop_condition_evidence=None,
        failure_information=(),
        recovery_state=None,
        budget_state={"duration_seconds": 10.0, "iterations": 3, "tool_calls": 10},
        policy_safety_blocks=(),
        artifacts=(),
        bounded_logs=(),
        evidence_complete=True,
        warnings=(),
    )
    return BenchmarkTaskRun(task_id, "1.0", category, difficulty, status, 0.0, 1.0, 10.0, None, evidence, None, (), (), ())


def _result(runs: tuple) -> BenchmarkResult:
    return BenchmarkResult(
        "bench-001",
        "1.0.0",
        "COMPLETED",
        runs,
        BenchmarkRunSummary(len(runs), sum(1 for r in runs if r.status is BenchmarkTaskStatus.PASSED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.FAILED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.BLOCKED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.TIMED_OUT), sum(1 for r in runs if r.status is BenchmarkTaskStatus.SKIPPED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.UNAVAILABLE), sum(1 for r in runs if r.status is BenchmarkTaskStatus.INFRASTRUCTURE_ERROR), sum(1 for r in runs if r.status is BenchmarkTaskStatus.INCOMPLETE_EVIDENCE)),
        20.0,
        "COMPLETED",
        (),
        {"task_order": [run.task_id for run in runs], "project_root_supplied": True, "deterministic_mode": True, "scoring": "8.7"},
        False,
        False,
        None,
    )


def _collect(result: BenchmarkResult) -> "BenchmarkMetrics":
    from backend_ai.evaluation import collect_benchmark_metrics

    return collect_benchmark_metrics(result)


def test_category_delta_is_positive_when_version_b_improves() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.FAILED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    comparison = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    api = next(item for item in comparison.category_comparisons if item.group_name == "API_ENDPOINT")
    assert api.success_rate.delta == 1.0
    assert api.success_rate.classification.value == "IMPROVED"


def test_category_delta_is_negative_when_version_b_regresses() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.FAILED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    comparison = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    api = next(item for item in comparison.category_comparisons if item.group_name == "API_ENDPOINT")
    assert api.success_rate.delta == -1.0
    assert api.success_rate.classification.value == "REGRESSED"


def test_equivalent_categories_within_epsilon() -> None:
    baseline = _result((_task_run("EVAL-001", BenchmarkTaskStatus.PASSED),))
    candidate = _result((_task_run("EVAL-001", BenchmarkTaskStatus.PASSED),))
    comparison = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    api = next(item for item in comparison.category_comparisons if item.group_name == "API_ENDPOINT")
    assert api.success_rate.classification.value == "EQUIVALENT"


def test_difficulty_breakdown_compared_separately() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED, difficulty="EASY"),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED, difficulty="HARD"),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED, difficulty="EASY"),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED, difficulty="HARD"),
    ))
    comparison = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    hard = next(item for item in comparison.difficulty_comparisons if item.group_name == "HARD")
    assert hard.success_rate.classification.value == "IMPROVED"


def test_missing_group_treated_as_regression_warning() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED, category="API_ENDPOINT"),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED, category="AUTHENTICATION"),
    ))
    candidate = _result((_task_run("EVAL-001", BenchmarkTaskStatus.PASSED, category="API_ENDPOINT"),))
    comparison = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    assert any("category sets differ" in warning for warning in comparison.warnings)


def test_overall_classification_is_regressed_when_groups_regress() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.FAILED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    comparison = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    assert comparison.overall_classification.value == "REGRESSED"


def test_overall_classification_is_improved_when_groups_improve() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.FAILED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    comparison = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    assert comparison.overall_classification.value == "IMPROVED"


def test_inconclusive_when_evidence_is_absent() -> None:
    baseline = _result(())
    candidate = _result(())
    comparison = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    assert comparison.overall_classification.value == "INCOMPARABLE"


def test_epsilon_bounds_determine_classification() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    strict = compare_evaluation_metrics(_collect(baseline), _collect(candidate), config=ComparisonConfig(epsilon=0.001))
    loose = compare_evaluation_metrics(_collect(baseline), _collect(candidate), config=ComparisonConfig(epsilon=0.99))
    strict_value = next(item for item in strict.category_comparisons).success_rate.classification.value
    loose_value = next(item for item in loose.category_comparisons).success_rate.classification.value
    assert strict_value != loose_value


def test_invalid_epsilon_rejected() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
    ))
    with pytest.raises(ValueError):
        compare_evaluation_metrics(_collect(baseline), _collect(baseline), config=ComparisonConfig(epsilon=0.0))
    with pytest.raises(ValueError):
        compare_evaluation_metrics(_collect(baseline), _collect(baseline), config=ComparisonConfig(epsilon=1.0))


def test_comparison_is_deterministic() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    first = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    second = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    assert first.to_json() == second.to_json()


def test_regression_findings_trace_to_group_metric_deltas() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.FAILED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    comparison = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    regressions = comparison.regressions()
    assert regressions
    assert all(item.delta is not None and item.delta < 0 for item in regressions)
    assert all(item.severity.value in ("HIGH", "MEDIUM") for item in regressions)


def test_group_status_mixed_improvement_and_regression() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    comparison = compare_evaluation_metrics(_collect(baseline), _collect(candidate))
    api = next(item for item in comparison.category_comparisons)
    assert api.status().value == "IMPROVED"
