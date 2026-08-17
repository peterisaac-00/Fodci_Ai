from dataclasses import replace

import pytest

from backend_ai.evaluation import (
    BenchmarkResult,
    BenchmarkRunSummary,
    BenchmarkScore,
    BenchmarkStatus,
    ComparisonClassification,
    ComparisonConfig,
    ComparisonStatus,
    DimensionComparison,
    ScoringEvaluationResult,
    EvaluationStatus,
    EvaluationVersion,
    RegressionSeverity,
    ScoreDimension,
    ScoringPolicy,
    TaskScore,
    compare_evaluations,
)
from backend_ai.evaluation.benchmark_runner import BenchmarkTerminationReason


def _version(name: str, *, definition: str = "1.0", policy: str = "1.0") -> EvaluationVersion:
    return EvaluationVersion(name, name, "8.3", policy, definition, commit_sha=name)


def _task_score(task_id: str, value: float, status: EvaluationStatus = EvaluationStatus.VERIFIED) -> TaskScore:
    dimensions = tuple(
        ScoreDimension(name, value, 0.25, value * 0.25, status, (f"{task_id}-{name}",), f"{name} evidence")
        for name in ("task_success", "tests", "code_quality", "efficiency")
    )
    return TaskScore(task_id, value, value, value, value, value, value * 100, status, dimensions)


def _result(values: dict[str, float], statuses: dict[str, EvaluationStatus] | None = None, *, benchmark: str = "BENCH-1", evidence: float = 1.0, definition: str = "1.0", policy: str = "1.0") -> ScoringEvaluationResult:
    statuses = statuses or {}
    scores = tuple(_task_score(task_id, value, statuses.get(task_id, EvaluationStatus.VERIFIED)) for task_id, value in sorted(values.items()))
    aggregate = sum(item.final_score for item in scores) / len(scores) if scores else 0.0
    dimensions = tuple(ScoreDimension(name, aggregate, 0.25, aggregate * 0.25, EvaluationStatus.VERIFIED, (name,), "aggregate") for name in ("task_success", "tests", "code_quality", "efficiency"))
    score = BenchmarkScore(benchmark, scores, aggregate, aggregate * 100, dimensions, sum(item.status is EvaluationStatus.VERIFIED for item in scores), sum(item.status is EvaluationStatus.FAILED for item in scores), sum(item.status is EvaluationStatus.INCOMPLETE for item in scores), sum(item.status is EvaluationStatus.BLOCKED for item in scores), sum(item.status is EvaluationStatus.UNAVAILABLE for item in scores), "HIGH" if evidence == 1.0 else "LOW", evidence)
    summary = BenchmarkRunSummary(len(scores), score.completed_count, score.failed_count, score.blocked_count, 0, 0, score.unavailable_count, 0, score.incomplete_count)
    run = BenchmarkResult(benchmark, "1.0", BenchmarkStatus.COMPLETED, (), summary, 0.1, BenchmarkTerminationReason.COMPLETED)
    return ScoringEvaluationResult(run, score, "8.3", policy, (), {"definition": definition})


def _compare(base, candidate, **kwargs):
    return compare_evaluations(base, candidate, ComparisonConfig(**kwargs), baseline_version=_version("v0.1", policy=base.scoring_policy_version), candidate_version=_version("v0.2", policy=candidate.scoring_policy_version))


def test_identical_versions_are_regression_free_and_deterministic():
    result = _compare(_result({"T1": 0.8}), _result({"T1": 0.8}))
    assert result.status is ComparisonStatus.REGRESSION_FREE
    assert result.severity is RegressionSeverity.NONE
    assert result.to_json() == result.to_json()


def test_improvement_and_regression_use_epsilon():
    assert _compare(_result({"T1": 0.5}), _result({"T1": 0.7})).status is ComparisonStatus.IMPROVED
    assert _compare(_result({"T1": 0.7}), _result({"T1": 0.5})).status is ComparisonStatus.REGRESSED
    assert _compare(_result({"T1": 0.5}), _result({"T1": 0.505})).status is ComparisonStatus.REGRESSION_FREE
    assert _compare(_result({"T1": 0.5}), _result({"T1": 0.52})).task_comparisons[0].classification is ComparisonClassification.IMPROVED


def test_pass_fail_and_fail_pass_transitions_override_score():
    failed = _result({"T1": 0.9}, {"T1": EvaluationStatus.FAILED})
    passed = _result({"T1": 0.1})
    regression = _compare(passed, failed)
    improvement = _compare(failed, passed)
    assert regression.status is ComparisonStatus.REGRESSED
    assert regression.task_comparisons[0].severity is RegressionSeverity.HIGH
    assert improvement.status is ComparisonStatus.IMPROVED


@pytest.mark.parametrize("status", [EvaluationStatus.BLOCKED, EvaluationStatus.INCOMPLETE, EvaluationStatus.UNAVAILABLE])
def test_passing_to_blocked_incomplete_or_unavailable_is_regression(status):
    result = _compare(_result({"T1": 0.2}), _result({"T1": 0.9}, {"T1": status}))
    assert result.status is ComparisonStatus.REGRESSED


def test_improvement_with_task_regression_is_not_plain_improvement():
    result = _compare(_result({"A": 0.5, "B": 0.9}), _result({"A": 0.9, "B": 0.7}))
    assert result.status is ComparisonStatus.IMPROVED_WITH_REGRESSIONS
    assert result.findings.regression_count >= 1
    assert any(item.task_id == "B" for item in result.findings.findings)


def test_incompatible_benchmark_policy_definition_and_task_set():
    assert _compare(_result({"T1": 0.5}, benchmark="A"), _result({"T1": 0.5}, benchmark="B")).status is ComparisonStatus.INCOMPARABLE
    assert _compare(_result({"T1": 0.5}), _result({"T1": 0.5}, policy="2.0")).status is ComparisonStatus.INCOMPARABLE
    assert _compare(_result({"T1": 0.5}), _result({"T2": 0.5})).status is ComparisonStatus.INCOMPARABLE


def test_incomplete_evidence_is_inconclusive_not_regression_free():
    result = _compare(_result({"T1": 0.5}, evidence=0.5), _result({"T1": 0.6}))
    assert result.status is ComparisonStatus.INCONCLUSIVE
    assert result.warnings


def test_missing_dimension_and_bounded_evidence_are_explicit():
    baseline = _result({"T1": 0.5})
    candidate = _result({"T1": 0.6})
    original = candidate.benchmark_score.task_scores[0]
    candidate_score = replace(original, dimensions=original.dimensions[:2])
    candidate_benchmark = replace(candidate.benchmark_score, task_scores=(candidate_score,))
    candidate = replace(candidate, benchmark_score=candidate_benchmark)
    result = _compare(baseline, candidate, max_evidence_ids=1)
    assert result.task_comparisons[0].dimensions[2].classification is ComparisonClassification.INCONCLUSIVE
    assert len(result.task_comparisons[0].evidence_ids) <= 1


def test_immutable_result_and_public_version_metadata():
    version = _version("v0.1")
    with pytest.raises(TypeError):
        version.metadata["x"] = "y"
    assert version.to_dict()["version_id"] == "v0.1"
    assert ScoringPolicy().scoring_policy_version == "1.0"


def test_reproducible_six_task_checkpoint_with_mixed_results():
    task_ids = {"API-ENDPOINT": 0.70, "AUTHENTICATION": 0.72, "DATABASE": 0.68, "BUG-FIX": 0.75, "TESTING": 0.66, "DOCKER": 0.71}
    candidate = {"API-ENDPOINT": 0.78, "AUTHENTICATION": 0.80, "DATABASE": 0.74, "BUG-FIX": 0.70, "TESTING": 0.76, "DOCKER": 0.79}
    first = _compare(_result(task_ids), _result(candidate))
    second = _compare(_result(task_ids), _result(candidate))
    assert first.status is ComparisonStatus.IMPROVED_WITH_REGRESSIONS
    assert first.to_json() == second.to_json()
    assert len(first.task_comparisons) == 6
    assert any(item.task_id == "BUG-FIX" and item.classification is ComparisonClassification.REGRESSED for item in first.task_comparisons)
