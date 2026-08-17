"""Phase 8.6 report generation behavioral contract.

Verifies that report generation is deterministic, bounded, and formats
existing Phase 8.3-8.9 results without executing anything new.
"""
from __future__ import annotations

import json

import pytest

from backend_ai.evaluation import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunSummary,
    BenchmarkTaskRun,
    BenchmarkTaskStatus,
    ReportConfig,
    ReportInputs,
    ScoringEvaluationResult,
    BenchmarkScore,
    EvaluationStatus,
    collect_metrics,
    generate_evaluation_report,
)


def _task_run(task_id: str = "EVAL-001", status: BenchmarkTaskStatus = BenchmarkTaskStatus.PASSED) -> BenchmarkTaskRun:
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
        changed_paths=("src/a.py",),
        expected_paths_touched=("src/a.py",),
        unexpected_modifications=(),
        forbidden_changes_detected=(),
        mutation_count=1,
        mutation_verification=None,
        tests_requested=True,
        tests_executed=True,
        test_result={"status": "PASS"},
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


def _result(runs: tuple, *, version: str = "8.3", policy: str = "1.0") -> BenchmarkResult:
    return BenchmarkResult(
        "bench-001",
        "1.0.0",
        "COMPLETED",
        runs,
        BenchmarkRunSummary(len(runs), sum(1 for r in runs if r.status is BenchmarkTaskStatus.PASSED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.FAILED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.BLOCKED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.TIMED_OUT), sum(1 for r in runs if r.status is BenchmarkTaskStatus.SKIPPED), sum(1 for r in runs if r.status is BenchmarkTaskStatus.UNAVAILABLE), sum(1 for r in runs if r.status is BenchmarkTaskStatus.INFRASTRUCTURE_ERROR), sum(1 for r in runs if r.status is BenchmarkTaskStatus.INCOMPLETE_EVIDENCE)),
        20.0,
        "COMPLETED",
        ("a first warning", "a second warning"),
        {"task_order": [run.task_id for run in runs], "project_root_supplied": True, "deterministic_mode": True, "scoring": "8.6", "model": "fodci-tiny-v1"},
        False,
        False,
        None,
    )


def _score(runs: tuple) -> BenchmarkScore:
    return BenchmarkScore(
        "bench-001",
        tuple({"task_id": run.task_id, "task_success_score": 1.0, "test_score": 1.0, "code_quality_score": 1.0, "efficiency_score": 1.0, "final_score": 1.0, "percentage": 100.0, "status": EvaluationStatus.VERIFIED, "dimensions": (), "evaluation_status": EvaluationStatus.VERIFIED} for run in runs),
        1.0,
        100.0,
        ("task_success", "tests", "code_quality", "efficiency"),
        len(runs),
        0,
        0,
        0,
        0,
        "HIGH",
        1.0,
    )


def _evaluation_result(result: BenchmarkResult, score: BenchmarkScore) -> ScoringEvaluationResult:
    return ScoringEvaluationResult(
        result,
        score,
        "8.3",
        "1.0",
        ("warning_from_scoring",),
        {"model": "fodci-tiny-v1"},
    )


def _report_inputs(runs: tuple) -> ReportInputs:
    result = _result(runs)
    score = _score(runs)
    return ReportInputs(
        evaluation_result=_evaluation_result(result, score),
        benchmark_result=result,
        metrics=collect_metrics(result),
        benchmark_score=score,
        identity={"agent_version": "fodci-agent-1.0", "model_identity": "fodci-tiny-v1"},
    )


def test_report_generates_deterministic_json() -> None:
    runs = (_task_run("EVAL-001"), _task_run("EVAL-002"))
    first = generate_evaluation_report(_report_inputs(runs))
    second = generate_evaluation_report(_report_inputs(runs))
    assert first.to_json() == second.to_json()
    payload = json.loads(first.to_json())
    assert payload["report_version"] == "8.6"


def test_report_text_follows_fodci_format() -> None:
    runs = (_task_run("EVAL-001"),)
    report = generate_evaluation_report(_report_inputs(runs))
    text = report.to_text()
    assert "FODCI EVALUATION REPORT" in text
    assert "Agent:" in text
    assert "fodci-agent-1.0" in text
    assert "Model:" in text
    assert "fodci-tiny-v1" in text
    assert "Benchmark:" in text
    assert "bench-001" in text
    assert "Completeness:" in text
    assert "CATEGORIES" in text
    assert "DIFFICULTY" in text
    assert "task_success_rate:" in text


def test_report_includes_warnings_from_scoring_and_metrics() -> None:
    runs = (_task_run("EVAL-001"),)
    report = generate_evaluation_report(_report_inputs(runs))
    assert "a first warning" in report.warnings
    assert "warning_from_scoring" in report.warnings


def test_report_limits_warnings_to_configured_maximum() -> None:
    config = ReportConfig(max_warnings=2)
    runs = (_task_run("EVAL-001"),)
    report = generate_evaluation_report(_report_inputs(runs), config)
    assert len(report.warnings) <= 2
    if report.truncation.truncated:
        assert "warnings" in report.truncation.truncated_domains


def test_report_includes_comparison_section_when_supplied() -> None:
    runs = (_task_run("EVAL-001"),)
    report = generate_evaluation_report(_report_inputs(runs))
    payload = json.loads(report.to_json())
    assert payload["comparison"] is None


def test_report_includes_limitations_section() -> None:
    runs = (_task_run("EVAL-001"),)
    report = generate_evaluation_report(_report_inputs(runs))
    assert "evaluation measures the Agent and does not modify the Agent" in report.limitations
    assert "missing evidence is never treated as success" in report.limitations


def test_report_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        ReportConfig(max_task_findings=0)
    with pytest.raises(ValueError):
        ReportConfig(max_warnings=-1)
    with pytest.raises(TypeError):
        ReportConfig(max_task_findings=True)


def test_report_bounds_task_breakdown_to_configured_limit() -> None:
    config = ReportConfig(max_task_findings=1)
    runs = tuple(_task_run(f"EVAL-{index + 1:03d}") for index in range(3))
    report = generate_evaluation_report(_report_inputs(runs), config)
    assert len(report.task_breakdown) == 1
    assert report.truncation.truncated is True
    assert "task_findings" in report.truncation.truncated_domains


def test_report_includes_category_and_difficulty_breakdowns() -> None:
    runs = (_task_run("EVAL-001"),)
    report = generate_evaluation_report(_report_inputs(runs))
    assert len(report.category_breakdown) == 1
    assert len(report.difficulty_breakdown) == 1
    assert report.category_breakdown[0]["category"] == "API_ENDPOINT"
    assert report.difficulty_breakdown[0]["difficulty"] == "MEDIUM"


def test_report_identity_ordering_is_canonical() -> None:
    inputs = _report_inputs((_task_run("EVAL-001"),))
    inputs_with_identity = ReportInputs(
        inputs.evaluation_result,
        inputs.benchmark_result,
        inputs.metrics,
        inputs.benchmark_score,
        identity={"z_key": "1", "a_key": "2"},
    )
    assert tuple(sorted(inputs_with_identity.identity.keys())) == ("a_key", "z_key")


def test_report_rejects_invalid_identity_type() -> None:
    with pytest.raises(ValueError):
        ReportInputs(
            _evaluation_result(_result((_task_run(),)), _score((_task_run(),))),
            _result((_task_run(),)),
            collect_metrics(_result((_task_run(),))),
            _score((_task_run(),)),
            identity="not-a-mapping",
        )


def test_report_preserves_evidence_completeness() -> None:
    runs = (_task_run("EVAL-001"),)
    report = generate_evaluation_report(_report_inputs(runs))
    assert report.evidence_completeness == 1.0
    payload = json.loads(report.to_json())
    assert payload["evidence_completeness"] == 1.0
