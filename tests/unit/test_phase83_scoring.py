from backend_ai.evaluation import (
    BenchmarkEvidence,
    BenchmarkResult,
    BenchmarkRunSummary,
    BenchmarkScorer,
    BenchmarkTaskRun,
    BenchmarkTaskStatus,
    EvaluationTask,
    EvaluationTaskCategory,
    EvaluationDifficulty,
    EvaluationWeights,
    EvaluationStatus,
    EvaluationTaskValidationResult,
    ProjectDefinition,
    SuccessCriterion,
    SuccessCriterionType,
)
from backend_ai.evaluation.benchmark_runner import BenchmarkStatus, BenchmarkTerminationReason


def _task(task_id: str = "EVAL-SCORE") -> EvaluationTask:
    return EvaluationTask(
        task_id=task_id, title="score", description="score", version="1.0",
        category=EvaluationTaskCategory.BUG_FIX, difficulty=EvaluationDifficulty.EASY,
        project_definition=ProjectDefinition(project_type="backend", language="Python", runtime="Python", test_framework="pytest"),
        user_intent="score", success_criteria=(
            SuccessCriterion("C-TEST", "tests", SuccessCriterionType.TEST_PASS, True),
            SuccessCriterion("C-VERIFY", "verify", SuccessCriterionType.VERIFICATION, True),
        ),
    )


def _run(task_id: str, *, passed: bool = True, complete: bool = True, regression: bool = False) -> BenchmarkTaskRun:
    status = BenchmarkTaskStatus.PASSED if passed else BenchmarkTaskStatus.FAILED
    final = "VERIFIED" if passed and not regression else "NOT_VERIFIED"
    test = "PASS" if passed and not regression else "FAIL"
    evidence = BenchmarkEvidence(
        execution_started=True, execution_completed=True, execution_status="PASS" if passed else "FAIL",
        duration_seconds=1.0, task_identity=task_id, tests_requested=True, tests_executed=True,
        test_result={"status": test}, completion_evidence={"status": "COMPLETE" if complete else "INCOMPLETE"},
        final_verification_evidence={"status": final, "regression_status": "REGRESSION_DETECTED" if regression else "REGRESSION_FREE"},
        budget_state={"duration_seconds": 1.0, "iterations": 1, "tool_calls": 1, "test_runs": 1},
        failure_information=("REGRESSION_DETECTED",) if regression else (),
    )
    return BenchmarkTaskRun(task_id, "1.0", "BUG_FIX", "EASY", status, 0.0, 1.0, 1.0, None, evidence, EvaluationTaskValidationResult(True))


def _benchmark(runs):
    summary = BenchmarkRunSummary(len(runs), sum(r.status is BenchmarkTaskStatus.PASSED for r in runs), sum(r.status is BenchmarkTaskStatus.FAILED for r in runs), 0, 0, 0, 0, 0, 0)
    return BenchmarkResult("B-1", "1.0", BenchmarkStatus.COMPLETED, tuple(runs), summary, 1.0, BenchmarkTerminationReason.COMPLETED)


def test_weights_are_immutable_and_validated():
    assert EvaluationWeights().to_dict() == {"task_success": 0.5, "tests": 0.3, "code_quality": 0.1, "efficiency": 0.1}
    try:
        EvaluationWeights(task_success=0.6, tests=0.3, code_quality=0.1, efficiency=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid weights must be rejected")


def test_successful_evaluation_is_traceable_and_deterministic():
    task = _task()
    result = BenchmarkScorer().evaluate_benchmark(_benchmark((_run(task.task_id),)), (task,))
    score = result.benchmark_score.task_scores[0]
    assert score.status is EvaluationStatus.VERIFIED
    assert score.final_score > 0.9
    assert result.to_json() == result.to_json()
    assert all(d.evidence_ids for d in score.dimensions)


def test_regression_is_not_a_pass_and_stays_in_denominator():
    good, bad = _task("EVAL-A"), _task("EVAL-B")
    result = BenchmarkScorer().evaluate_benchmark(_benchmark((_run("EVAL-A"), _run("EVAL-B", passed=False, regression=True))), (good, bad))
    assert result.benchmark_score.failed_count == 1
    assert result.benchmark_score.aggregate_score < result.benchmark_score.task_scores[0].final_score
    assert result.benchmark_score.task_scores[1].status is EvaluationStatus.FAILED


def test_missing_required_evidence_is_explicitly_incomplete():
    task = _task()
    run = _run(task.task_id)
    evidence = BenchmarkEvidence(execution_started=True, execution_completed=True, tests_requested=True, tests_executed=False, evidence_complete=False)
    incomplete = BenchmarkTaskRun(task.task_id, "1.0", "BUG_FIX", "EASY", BenchmarkTaskStatus.INCOMPLETE_EVIDENCE, 0.0, 1.0, 1.0, None, evidence, EvaluationTaskValidationResult(True))
    evaluation = BenchmarkScorer().evaluate_task(task, incomplete)
    assert evaluation.missing_evidence
    assert evaluation.score.status is EvaluationStatus.INCOMPLETE
