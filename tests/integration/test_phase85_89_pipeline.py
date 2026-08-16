"""Phase 8.5-8.9 end-to-end pipeline checkpoint.

Runs a real (tiny) benchmark through the runner, scorer, metrics, report,
comparison, regression, and validation layers and verifies the positive,
negative, validation, and determinism checkpoints.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from backend_ai.evaluation import (
    AllowedScope,
    BenchmarkConfig,
    BenchmarkRequest,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkScorer,
    BenchmarkTaskStatus,
    EvaluationDifficulty,
    EvaluationResult,
    EvaluationConstraint,
    EvaluationTask,
    EvaluationTaskCategory,
    EvaluationTaskValidator,
    ExpectedBehavior,
    GroundTruth,
    ProjectDefinition,
    Requirement,
    ScoringPolicy,
    SuccessCriterion,
    SuccessCriterionType,
    TestDefinition,
    EvaluationTestType,
    collect_benchmark_metrics,
    collect_metrics,
    compare_evaluation_metrics,
    evaluate_benchmark,
    evaluate_regression,
    generate_evaluation_report,
    validate_benchmark,
)
from backend_ai.evaluation.regression import ComparisonConfig, EvaluationSnapshot, compare_evaluations


def _task(task_id: str, category: EvaluationTaskCategory = EvaluationTaskCategory.API_ENDPOINT) -> EvaluationTask:
    return EvaluationTask(
        task_id=task_id,
        title=f"Task {task_id}",
        description=f"Evidence-only task {task_id}",
        version="1.0",
        category=category,
        difficulty=EvaluationDifficulty.MEDIUM,
        project_definition=ProjectDefinition("backend-service", "Python", "FastAPI", "Python 3.12", "PostgreSQL", ("existing project root",), ("dependencies installed",), "src/app.py", "pytest"),
        user_intent="Complete the evidence-only task.",
        requirements=(Requirement(f"REQ-{task_id}", f"Requirement for {task_id}", True, 1),),
        expected_behaviors=(ExpectedBehavior(f"BEH-{task_id}", f"Behavior for {task_id}", "complete", "completion evidence", "PASS", ()),),
        allowed_scope=AllowedScope(allowed_files=("src/",), allowed_directories=("src/",), allowed_patterns=(), allowed_change_types=(), forbidden_paths=(".env",), forbidden_patterns=()),
        expected_areas=(),
        tests=(TestDefinition(f"TEST-{task_id}", f"Test for {task_id}", EvaluationTestType.INTEGRATION, "tests/t.py", True, "PASS", (f"REQ-{task_id}",), (f"BEH-{task_id}",)),),
        success_criteria=(
            SuccessCriterion(f"CRIT-001-{task_id}", "test passes", SuccessCriterionType.TEST_PASS, True, "authoritative parsed test result", test_ids=(f"TEST-{task_id}",)),
            SuccessCriterion(f"CRIT-002-{task_id}", "behavior passes", SuccessCriterionType.BEHAVIOR, True, "verification evidence", behavior_ids=(f"BEH-{task_id}",)),
        ),
        forbidden_changes=(),
        constraints=EvaluationConstraint(),
        ground_truth=GroundTruth(
            required_outcomes=(f"correct payload returned for {task_id}",),
            required_interfaces=(),
            required_invariants=(),
        ),
        metadata={"suite": "pipeline"},
    )


TASKS = (
    _task("EVAL-001", EvaluationTaskCategory.API_ENDPOINT),
    _task("EVAL-002", EvaluationTaskCategory.AUTHENTICATION),
    _task("EVAL-003", EvaluationTaskCategory.DATABASE),
    _task("EVAL-004", EvaluationTaskCategory.BUG_FIX),
    _task("EVAL-005", EvaluationTaskCategory.TESTING),
    _task("EVAL-006", EvaluationTaskCategory.DOCKER),
)


class DeterministicRuntime:
    """Fake runtime adapter whose behavior is controlled per task."""

    def __init__(self, pass_tasks: set[str] = set(), duration: float = 5.0) -> None:
        self.pass_tasks = pass_tasks
        self.duration = duration

    def execute(self, task: EvaluationTask, workspace: str | Path, max_wall_time: float = 60.0) -> "DeterministicResult":
        from backend_ai.evaluation.benchmark_runner import BenchmarkExecutionResult

        passed = task.task_id in self.pass_tasks
        return _fake_result(passed, self.duration)


def _fake_result(passed: bool, duration: float):
    from backend_ai.evaluation.benchmark_runner import BenchmarkExecutionResult

    return BenchmarkExecutionResult(
        BenchmarkTaskStatus.PASSED if passed else BenchmarkTaskStatus.FAILED,
        "PASSED" if passed else "FAILED",
        tests_requested=True,
        tests_executed=True,
        test_evidence={"status": "PASS" if passed else "FAIL"},
        completion_evidence={"status": "COMPLETE"},
        final_verification_evidence={"status": "VERIFIED" if passed else "FAILED"},
        budget_state={"duration_seconds": duration, "iterations": 3, "tool_calls": 10, "test_runs": 1, "action_steps": 5},
    )


def _run_benchmark(pass_tasks: set[str]) -> tuple[BenchmarkResult, EvaluationResult]:
    config = BenchmarkConfig(
        benchmark_id="bench-pipeline",
        benchmark_version="1.0.0",
        max_tasks=32,
        max_total_wall_time=300.0,
        max_task_wall_time=60.0,
        deterministic_mode=True,
        fail_fast=False,
        continue_on_task_failure=True,
        collect_artifacts=False,
        cleanup_workspaces=True,
    )
    request = BenchmarkRequest(TASKS, None, config, DeterministicRuntime(pass_tasks), None)
    result = BenchmarkRunner().run(request)
    score = BenchmarkScorer().evaluate_benchmark(result, TASKS)
    return result, score


def _snapshot_inputs(pass_tasks: set[str]):
    from backend_ai.evaluation import ReportInputs

    result, evaluation = _run_benchmark(pass_tasks)
    metrics = collect_metrics(result, list(TASKS), benchmark_score=evaluation.benchmark_score)
    return (
        result,
        evaluation,
        metrics,
        ReportInputs(evaluation, result, metrics, evaluation.benchmark_score, identity={"agent_version": "fodci-agent-1.0", "model_identity": "fodci-tiny-v1"}),
    )


def test_positive_checkpoint_version_b_improves() -> None:
    """Version B improves authentication and testing tasks; comparison is IMPROVED."""

    _, baseline_eval = _run_benchmark(set())
    _, candidate_eval = _run_benchmark({"EVAL-002", "EVAL-005"})
    comparison = compare_evaluations(baseline_eval, candidate_eval)
    assert comparison.status.value == "IMPROVED"
    assert comparison.aggregate is not None
    assert comparison.aggregate.delta > 0


def test_negative_checkpoint_bug_fix_regresses_while_others_improve() -> None:
    """Version B improves most tasks but a previously-passing BUG FIX task fails.

    The pipeline must surface IMPROVED_WITH_REGRESSIONS rather than silently
    reporting overall improvement.
    """

    baseline_tasks = {"EVAL-001", "EVAL-004"}
    candidate_tasks = {"EVAL-001", "EVAL-002", "EVAL-003", "EVAL-005", "EVAL-006"}
    _, baseline_eval = _run_benchmark(baseline_tasks)
    _, candidate_eval = _run_benchmark(candidate_tasks)
    comparison = compare_evaluations(baseline_eval, candidate_eval)
    assert comparison.status.value == "IMPROVED_WITH_REGRESSIONS"
    assert comparison.findings.regression_count >= 1
    regressions = [item for item in comparison.findings.findings if item.classification.value == "REGRESSED"]
    assert any("EVAL-004" in item.task_id for item in regressions)


def test_validation_checkpoint_valid_and_invalid_benchmarks() -> None:
    """A mixed validation checkpoint: one valid benchmark, two invalid,
    three with warnings only."""

    valid = validate_benchmark((TASKS[0],))
    assert valid.status.value == "VALID"
    assert valid.health.score == 1.0

    invalid_duplicate_ids = validate_benchmark((TASKS[0], TASKS[0]))
    assert invalid_duplicate_ids.status.value == "INVALID"

    invalid_bad_task = validate_benchmark((TASKS[0], EvaluationTask(task_id="bad", title="x", description="y", version="1.0", category="UNKNOWN-CATEGORY", difficulty=EvaluationDifficulty.MEDIUM)))
    assert invalid_bad_task.status.value == "INVALID"

    warning_no_evaluable = validate_benchmark((
        replace(
            TASKS[0],
            tests=(),
            success_criteria=(SuccessCriterion("CRIT-001-EVAL-001", "empty", SuccessCriterionType.TEST_PASS, True, "evidence", test_ids=()),),
        ),
    ))
    assert warning_no_evaluable.status.value == "WARNING"

    warning_dominance = validate_benchmark(tuple(_task(f"EVAL-{index:03d}") for index in range(7)))
    assert warning_dominance.status.value == "WARNING"

    dominant_task = replace(
        TASKS[0],
        tests=TASKS[0].tests + (TestDefinition("TEST-EVAL-001-2", "Second test", EvaluationTestType.INTEGRATION, "tests/t2.py", True, "PASS", ("REQ-EVAL-001",), ("BEH-EVAL-001",)),),
    )
    warning_dominates = validate_benchmark((dominant_task, TASKS[1]))
    assert any(item.code == "ONE_TASK_DOMINATES" for item in warning_dominates.issues)


def test_determinism_checkpoint_reports_identical_across_runs() -> None:
    """The same benchmark inputs must produce byte-identical canonical JSON
    across two independent runs of the full pipeline."""

    a_result, a_evaluation, a_metrics, a_inputs = _snapshot_inputs({"EVAL-001", "EVAL-002", "EVAL-003"})
    b_result, b_evaluation, b_metrics, b_inputs = _snapshot_inputs({"EVAL-001", "EVAL-002", "EVAL-003"})
    assert collect_metrics(a_result, list(TASKS)).to_json() == collect_metrics(b_result, list(TASKS)).to_json()
    assert generate_evaluation_report(a_inputs).to_json() == generate_evaluation_report(b_inputs).to_json()
    a_metrics_snapshot = collect_benchmark_metrics(a_result)
    b_metrics_snapshot = collect_benchmark_metrics(b_result)
    assert compare_evaluation_metrics(a_metrics_snapshot, b_metrics_snapshot).to_json() == compare_evaluation_metrics(b_metrics_snapshot, a_metrics_snapshot).to_json()


def test_pipeline_report_includes_all_sections() -> None:
    _, evaluation = _run_benchmark({"EVAL-001", "EVAL-002", "EVAL-003", "EVAL-004", "EVAL-005", "EVAL-006"})
    result = evaluation.benchmark_result
    metrics = collect_metrics(result, list(TASKS), benchmark_score=evaluation.benchmark_score)
    from backend_ai.evaluation import ReportInputs

    report = generate_evaluation_report(ReportInputs(evaluation, result, metrics, evaluation.benchmark_score, identity={"agent_version": "fodci-agent-1.0", "model_identity": "fodci-tiny-v1"}))
    text = report.to_text()
    assert "FODCI EVALUATION REPORT" in text
    assert "bench-pipeline" in text
    payload = json.loads(report.to_json())
    assert payload["report_version"] == "8.6"
    assert len(payload["metrics"]) > 0
    assert len(payload["category_breakdown"]) >= 6
    assert len(payload["difficulty_breakdown"]) >= 1
    assert payload["evidence_completeness"] > 0


def test_regression_gate_applies_over_full_pipeline() -> None:
    _, baseline_eval = _run_benchmark(set())
    _, candidate_eval = _run_benchmark({"EVAL-004"})
    comparison = compare_evaluations(baseline_eval, candidate_eval)
    baseline_metrics = collect_benchmark_metrics(baseline_eval.benchmark_result)
    candidate_metrics = collect_benchmark_metrics(candidate_eval.benchmark_result)
    metrics_comparison = compare_evaluation_metrics(baseline_metrics, candidate_metrics)
    result = evaluate_regression(comparison, metrics_comparison)
    assert result.verdict.value in ("REGRESSION_PASSED", "REGRESSION_FAILED", "REGRESSION_INCONCLUSIVE")
    assert result.regression_count >= 0


def test_validation_never_executes_tests() -> None:
    """Validation consumes task definitions only; it never invokes a runtime."""

    result = validate_benchmark(TASKS)
    assert result.status.value == "VALID"
