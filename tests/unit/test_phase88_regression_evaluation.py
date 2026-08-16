"""Phase 8.8 regression evaluation behavioral contract.

Verifies deterministic gates, severity budgets, and verdict rules without
rerunning benchmarks, executing tests, or mutating comparison results.
"""
from __future__ import annotations

import json

import pytest

from backend_ai.evaluation import (
    RegressionGate,
    RegressionGateResult,
    RegressionSeverity,
    RegressionVerdict,
    RegressionEvaluationResult,
    evaluate_regression,
)
from backend_ai.evaluation.version_comparison import compare_evaluation_metrics
from backend_ai.evaluation.regression import (
    ComparisonClassification,
    ComparisonStatus,
    ComparisonConfig,
    EvaluationComparisonResult,
    RegressionFinding,
)
from backend_ai.evaluation.benchmark_runner import (
    BenchmarkResult,
    BenchmarkRunSummary,
    BenchmarkTaskRun,
    BenchmarkTaskStatus,
)
from backend_ai.evaluation import collect_benchmark_metrics


def _task_run(task_id: str, status: BenchmarkTaskStatus, tests_executed: bool = True, test_passed: bool = True) -> BenchmarkTaskRun:
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
    return BenchmarkTaskRun(task_id, "1.0", "API_ENDPOINT", "MEDIUM", status, 0.0, 1.0, 10.0, None, evidence, None, (), (), ())


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
        {"task_order": [run.task_id for run in runs], "project_root_supplied": True, "deterministic_mode": True, "scoring": "8.8"},
        False,
        False,
        None,
    )


def _comparison(status: ComparisonStatus) -> EvaluationComparisonResult | None:
    return EvaluationComparisonResult(None, None, status, RegressionSeverity.NONE, None, (), RegressionSummary(0, 0, 0, 0, RegressionSeverity.NONE, ()), (), (), 0.01) if status is not ComparisonStatus.INCOMPARABLE else None


def test_passed_verdict_when_no_regressions() -> None:
    baseline = _result((_task_run("EVAL-001", BenchmarkTaskStatus.PASSED),))
    candidate = _result((_task_run("EVAL-001", BenchmarkTaskStatus.PASSED),))
    metrics = compare_evaluation_metrics(collect_benchmark_metrics(baseline), collect_benchmark_metrics(candidate))
    result = evaluate_regression(metrics_comparison=metrics)
    assert result.verdict is RegressionVerdict.REGRESSION_PASSED


def test_failed_verdict_on_metrics_regression() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.FAILED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    metrics = compare_evaluation_metrics(collect_benchmark_metrics(baseline), collect_benchmark_metrics(candidate))
    result = evaluate_regression(metrics_comparison=metrics)
    assert result.verdict is RegressionVerdict.REGRESSION_FAILED


def test_failed_verdict_when_severity_exceeds_budget() -> None:
    finding = RegressionFinding("F-001", "overall", "overall", 1.0, 0.5, -0.5, ComparisonClassification.REGRESSED, RegressionSeverity.CRITICAL, ("gate",), "critical regression")
    result = evaluate_regression(findings=(finding,), max_severity=RegressionSeverity.LOW)
    assert result.verdict is RegressionVerdict.REGRESSION_FAILED
    assert result.max_severity is RegressionSeverity.CRITICAL


def test_gate_missing_value_fails_never_passes() -> None:
    gate = RegressionGate("task_success_rate", "task success rate", threshold=0.0)
    result = evaluate_regression(metrics_comparison=None, gates=(gate,))
    assert result.verdict is RegressionVerdict.REGRESSION_FAILED
    gate_result = result.gate_results[0]
    assert gate_result.passed is False
    assert gate_result.value is None


def test_inconclusive_verdict_when_no_evidence_supplied() -> None:
    result = evaluate_regression()
    assert result.verdict is RegressionVerdict.REGRESSION_INCONCLUSIVE
    assert "no comparison or metrics comparison was supplied" in " ".join(result.warnings)


def test_regresion_count_gate_binds_findings() -> None:
    findings = tuple(RegressionFinding(f"F-{index:03d}", "overall", "overall", 1.0, 0.9, -0.1, ComparisonClassification.REGRESSED, RegressionSeverity.LOW, ("gate",), "small regression") for index in range(5))
    result = evaluate_regression(findings=findings, max_regression_count=3)
    assert result.verdict is RegressionVerdict.REGRESSION_FAILED
    assert result.regression_count == 5


def test_severity_ordering_respects_phase84_hierarchy() -> None:
    findings = (RegressionFinding("F-001", "overall", "overall", 1.0, 0.9, -0.1, ComparisonClassification.REGRESSED, RegressionSeverity.MEDIUM, ("gate",), "medium regression"),)
    passed = evaluate_regression(findings=findings, max_severity=RegressionSeverity.HIGH)
    assert passed.verdict is RegressionVerdict.REGRESSION_PASSED
    failed = evaluate_regression(findings=findings, max_severity=RegressionSeverity.LOW)
    assert failed.verdict is RegressionVerdict.REGRESSION_FAILED


def test_inconclusive_when_every_gate_lacks_evidence() -> None:
    result = evaluate_regression(gates=(RegressionGate("overall_score", "overall score", threshold=0.0),))
    assert result.verdict is RegressionVerdict.REGRESSION_INCONCLUSIVE
    assert all(not item.passed for item in result.gate_results)


def test_evaluator_reuses_default_gates() -> None:
    from backend_ai.evaluation.regression_evaluation import RegressionEvaluator

    evaluator = RegressionEvaluator()
    assert any(gate.gate_type == "overall_score" for gate in evaluator.gates)
    assert any(gate.gate_type == "task_success_rate" for gate in evaluator.gates)
    assert any(gate.gate_type == "test_pass_rate" for gate in evaluator.gates)
    assert any(gate.gate_type == "severity" for gate in evaluator.gates)


def test_result_is_deterministic() -> None:
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    metrics = compare_evaluation_metrics(collect_benchmark_metrics(baseline), collect_benchmark_metrics(candidate))
    first = evaluate_regression(metrics_comparison=metrics)
    second = evaluate_regression(metrics_comparison=metrics)
    assert first.to_json() == second.to_json()


def test_result_canonical_json_uses_sorted_keys() -> None:
    result = evaluate_regression()
    payload = json.loads(result.to_json())
    assert list(payload.keys()) == sorted(payload.keys())


def test_improved_with_regressions_still_exposed_through_findings() -> None:
    findings = tuple(RegressionFinding(f"F-{index:03d}", "overall", "overall", 1.0, 0.9, -0.1, ComparisonClassification.REGRESSED, RegressionSeverity.MEDIUM, ("gate",), "regression") for index in range(2))
    baseline = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.FAILED),
        _task_run("EVAL-002", BenchmarkTaskStatus.PASSED),
    ))
    candidate = _result((
        _task_run("EVAL-001", BenchmarkTaskStatus.PASSED),
        _task_run("EVAL-002", BenchmarkTaskStatus.FAILED),
    ))
    metrics = compare_evaluation_metrics(collect_benchmark_metrics(baseline), collect_benchmark_metrics(candidate))
    result = evaluate_regression(metrics_comparison=metrics, findings=findings, max_regression_count=8)
    assert result.regression_count == len(findings)
    assert "improvements are accompanied by regressions" in " ".join(result.warnings)


def test_gate_result_records_evidence_ids() -> None:
    gate = RegressionGate("task_success_rate", "task success rate", threshold=0.0)
    result = evaluate_regression(metrics_comparison=None, gates=(gate,))
    assert result.gate_results[0].evidence == ("evidence_unavailable",)


def test_passed_flag_matches_verdict() -> None:
    baseline = _result((_task_run("EVAL-001", BenchmarkTaskStatus.PASSED),))
    metrics = compare_evaluation_metrics(collect_benchmark_metrics(baseline), collect_benchmark_metrics(baseline))
    assert evaluate_regression(metrics_comparison=metrics).passed is True
    assert evaluate_regression().passed is False
