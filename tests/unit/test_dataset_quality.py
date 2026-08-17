from __future__ import annotations

from dataclasses import replace

import pytest

from backend_ai.agent.dataset_quality import (
    DatasetQualityEvaluator,
    DatasetQualityPolicy,
    QualityCheckStatus,
    QualityDecision,
)
from backend_ai.agent.dataset_schema import (
    DatasetEvaluation,
    DatasetOutcome,
    DatasetSolution,
    DatasetTrajectory,
    DatasetVerification,
)
from backend_ai.agent.experience_dataset import ExperienceDatasetExtractor
from backend_ai.agent.experience_records import ExperienceRecords

from tests.unit.test_dataset_schema import _candidate


STAMP = "2026-08-17T00:00:00Z"


def _strong_record():
    return __import__("backend_ai.agent.dataset_schema", fromlist=["DatasetRecord"]).DatasetRecord.from_candidate(_candidate())


def _failed_record():
    records = ExperienceRecords(clock=lambda: STAMP)
    session = records.start_experience("Fix the Redis connection timeout")
    session.start_attempt()
    session.record_attempt_result("tests failed")
    source = session.finalize(status="failed", outcome="failure", final_summary="tests failed")
    return __import__("backend_ai.agent.dataset_schema", fromlist=["DatasetRecord"]).DatasetRecord.from_candidate(ExperienceDatasetExtractor().extract(source))


def test_strong_verified_backend_record_is_accepted_with_explainable_score() -> None:
    record = _strong_record()
    assessment = DatasetQualityEvaluator().evaluate(record)
    assert assessment.decision is QualityDecision.ACCEPT
    assert assessment.score.final_score >= 0.75
    assert {item.check_id for item in assessment.checks} == {"security", "consistency", "task_quality", "relevance", "solution_completeness", "verification", "trajectory", "noise", "outcome"}
    assert assessment.reasons == ()
    assert assessment.provenance.experience_id == record.experience_id


def test_score_formula_is_explicit_and_deterministic() -> None:
    evaluator = DatasetQualityEvaluator()
    first = evaluator.evaluate(_strong_record())
    second = evaluator.evaluate(_strong_record())
    expected = round(0.20 * first.score.task_score + 0.20 * first.score.completeness_score + 0.25 * first.score.verification_score + 0.15 * first.score.trajectory_score + 0.10 * first.score.relevance_score + 0.10 * first.score.consistency_score, 6)
    assert first.score.final_score == expected
    assert first.to_dict() == second.to_dict()


def test_failed_outcome_is_rejected_without_deleting_or_mutating_record() -> None:
    record = _failed_record()
    before = record.to_dict()
    assessment = DatasetQualityEvaluator().evaluate(record)
    assert assessment.decision is QualityDecision.REJECT
    assert "failed_outcome_not_high_quality" in assessment.reasons
    assert record.to_dict() == before


def test_missing_verification_on_success_is_review_not_silent_acceptance() -> None:
    record = _strong_record()
    weak = replace(record, verification=DatasetVerification(False, None, None, None, None, None, None, {}))
    assessment = DatasetQualityEvaluator().evaluate(weak)
    assert assessment.decision is QualityDecision.REVIEW
    assert "verification_missing" in assessment.warnings


def test_contradictory_success_verification_is_hard_rejected() -> None:
    record = _strong_record()
    contradictory = replace(record, verification=DatasetVerification(True, 2, 1, 1, "FAILED", "one test failed", STAMP, {}))
    assessment = DatasetQualityEvaluator().evaluate(contradictory)
    assert assessment.decision is QualityDecision.REJECT
    assert "verification_failed_tests" in assessment.reasons or "success_verification_failed" in assessment.reasons


def test_task_quality_is_conservative_for_short_backend_and_relevance_uncertain() -> None:
    record = _strong_record()
    short_backend = replace(record, task="Fix Redis")
    assert DatasetQualityEvaluator().evaluate(short_backend).decision is QualityDecision.ACCEPT
    ambiguous = replace(record, task="Improve the system")
    ambiguous_assessment = DatasetQualityEvaluator().evaluate(ambiguous)
    assert ambiguous_assessment.decision is QualityDecision.REVIEW
    assert "relevance_uncertain" in ambiguous_assessment.warnings


def test_placeholder_and_irrelevant_tasks_are_not_accepted() -> None:
    record = _strong_record()
    placeholder = replace(record, task="hello")
    assert DatasetQualityEvaluator().evaluate(placeholder).decision is QualityDecision.REVIEW
    irrelevant = replace(record, task="Write a wedding speech")
    irrelevant_assessment = DatasetQualityEvaluator().evaluate(irrelevant)
    assert irrelevant_assessment.decision is QualityDecision.REVIEW
    assert "relevance_uncertain" in irrelevant_assessment.warnings


def test_missing_solution_and_placeholder_solution_produce_review_or_reject() -> None:
    record = _strong_record()
    missing = replace(record, solution=DatasetSolution(None, None, None))
    missing_assessment = DatasetQualityEvaluator().evaluate(missing)
    assert missing_assessment.decision is QualityDecision.REJECT
    assert "solution_missing" in missing_assessment.reasons
    placeholder = replace(record, solution=DatasetSolution("TODO", "TODO", "TODO"))
    placeholder_assessment = DatasetQualityEvaluator().evaluate(placeholder)
    assert placeholder_assessment.decision is QualityDecision.REVIEW
    assert "solution_placeholder" in placeholder_assessment.warnings


def test_recovery_errors_are_valuable_and_not_automatically_rejected() -> None:
    record = _strong_record()
    assessment = DatasetQualityEvaluator().evaluate(record)
    assert assessment.decision is QualityDecision.ACCEPT
    assert assessment.score.trajectory_score == 1.0
    assert assessment.checks[[item.check_id for item in assessment.checks].index("trajectory")].status is QualityCheckStatus.PASS


def test_repeated_events_are_reviewed_but_source_record_is_unchanged() -> None:
    record = _strong_record()
    original = record.to_dict()
    actions = []
    for index in range(4):
        action = dict(record.trajectory.actions[0])
        action["action_id"] = f"action-repeated-{index}"
        actions.append(action)
    noisy_trajectory = replace(record.trajectory, actions=tuple(actions))
    noisy = replace(record, trajectory=noisy_trajectory)
    assessment = DatasetQualityEvaluator().evaluate(noisy)
    assert assessment.decision is QualityDecision.REVIEW
    assert "trajectory_repetition_review" in assessment.warnings
    assert record.to_dict() == original


def test_exact_duplicate_is_rejected_only_in_batch_and_records_remain_available() -> None:
    record = _strong_record()
    result = DatasetQualityEvaluator().filter_many((record, record))
    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert result.review_count == 0
    assert result.rejected[0].duplicate_of == record.record_id
    assert "duplicate_of:" + record.record_id in result.rejected[0].reasons


def test_invalid_schema_and_secret_are_hard_rejected_with_safe_diagnostics() -> None:
    evaluator = DatasetQualityEvaluator()
    invalid = evaluator.evaluate({"record_id": "bad", "experience_id": "exp-bad"})
    assert invalid.decision is QualityDecision.REJECT
    assert invalid.score.final_score == 0.0
    secret = evaluator.evaluate({"record_id": "bad", "experience_id": "exp-bad", "task": "token=super-secret"})
    assert secret.decision is QualityDecision.REJECT
    assert "super-secret" not in str(secret.to_dict())


def test_custom_policy_is_inspectable_and_cancelled_is_review() -> None:
    policy = DatasetQualityPolicy(minimum_quality_score=0.99, cancelled_outcome_decision=QualityDecision.REVIEW)
    evaluator = DatasetQualityEvaluator(policy=policy)
    record = _strong_record()
    cancelled = replace(record, outcome=DatasetOutcome.CANCELLED, provenance=replace(record.provenance, original_outcome="cancelled"))
    assessment = evaluator.evaluate(cancelled)
    assert assessment.decision is QualityDecision.REVIEW
    assert "cancelled_outcome_review" in assessment.warnings
    assert policy.to_dict()["minimum_quality_score"] == 0.99
