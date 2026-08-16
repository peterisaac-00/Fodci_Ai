"""Phase 8.9 benchmark validation behavioral contract.

Verifies that benchmark validation reuses the Phase 8.1 task validator,
detects structural/reference/scoring/fairness issues, and never executes
tests or benchmarks.
"""
from __future__ import annotations

import json

import pytest

from backend_ai.evaluation import (
    AllowedScope,
    BenchmarkValidationResult,
    BenchmarkValidator,
    EvaluationTask,
    EvaluationTaskCategory,
    EvaluationDifficulty,
    ExpectedBehavior,
    GroundTruth,
    IssueLevel,
    ProjectDefinition,
    Requirement,
    ScoringPolicy,
    SuccessCriterion,
    SuccessCriterionType,
    TestDefinition,
    EvaluationTestType,
    ValidationStatus,
    validate_benchmark,
    validate_evaluation_task,
)


def _minimal_task(
    task_id: str = "EVAL-001",
    category: EvaluationTaskCategory = EvaluationTaskCategory.API_ENDPOINT,
    with_tests: bool = True,
    with_criteria: bool = True,
    with_requirements: bool = True,
    requirement_id: str = "REQ-001",
    behavior_id: str = "BEH-001",
    test_id: str = "TEST-001",
    criterion_id: str = "CRIT-001",
) -> EvaluationTask:
    behaviors = (ExpectedBehavior(behavior_id, "POST /api/login with valid credentials", "authenticate the user", "a JWT token is returned", "PASS", ()),)
    ground_truth = GroundTruth(expected_behavior_ids=(behavior_id,), required_outcomes=("valid login returns a token",), required_interfaces=("POST /api/login",), required_invariants=("invalid credentials never produce a token",), allowed_implementation_alternatives=())
    tests = (TestDefinition(test_id, "valid login test", EvaluationTestType.API, "tests/test_auth.py", True, "PASS", (requirement_id,), (behavior_id,)),) if with_tests else ()
    criteria = (SuccessCriterion(criterion_id, "required tests pass", SuccessCriterionType.TEST_PASS, True, "authoritative parsed test result", test_ids=(test_id,)),) if with_criteria else ()
    requirements = (Requirement(requirement_id, "Create POST /api/login endpoint", True, 1),) if with_requirements else ()
    return EvaluationTask(
        task_id=task_id,
        title="Add JWT Authentication",
        description="Implement a login endpoint.",
        version="1.0",
        category=category,
        difficulty=EvaluationDifficulty.MEDIUM,
        project_definition=ProjectDefinition("backend-service", "Python", "FastAPI", "Python 3.12", "PostgreSQL", ("existing project root",), ("dependencies installed",), "src/app.py", "pytest"),
        user_intent="Create a secure login endpoint.",
        requirements=requirements,
        expected_behaviors=behaviors,
        allowed_scope=AllowedScope(allowed_files=("src/auth.py",), allowed_directories=("src/",), allowed_patterns=(), allowed_change_types=(), forbidden_paths=(), forbidden_patterns=()),
        expected_areas=(),
        ground_truth=ground_truth,
        tests=tests,
        success_criteria=criteria,
        forbidden_changes=(),
        metadata={"suite": "backend"},
    )


def test_valid_benchmark_validates_successfully() -> None:
    result = validate_benchmark((_minimal_task("EVAL-001"),))
    assert result.status is ValidationStatus.VALID
    assert result.health.score == 1.0
    assert result.health.error_count == 0


def test_duplicate_task_ids_fail_validation() -> None:
    task = _minimal_task("EVAL-001")
    result = validate_benchmark((task, task))
    assert result.status is ValidationStatus.INVALID
    assert any(item.code == "DUPLICATE_TASK_ID" for item in result.issues)


def test_invalid_task_fails_the_benchmark() -> None:
    task = _minimal_task("EVAL-001")
    invalid = EvaluationTask(
        task_id="bad",
        title="x",
        description="y",
        version="1.0",
        category="UNKNOWN-CATEGORY",
        difficulty=EvaluationDifficulty.MEDIUM,
    )
    result = validate_benchmark((task, invalid))
    assert result.status is ValidationStatus.INVALID
    assert any(item.task_id == "bad" for item in result.issues)


def test_empty_benchmark_is_invalid() -> None:
    result = validate_benchmark(())
    assert result.status is ValidationStatus.INVALID
    assert any(item.code == "EMPTY_COLLECTION" for item in result.issues)


def test_unresolved_references_warn() -> None:
    task = _minimal_task("EVAL-001", with_tests=True, with_criteria=True)
    referenced = EvaluationTask(
        task_id="EVAL-002",
        title="Second task",
        description="Refer to the first task.",
        version="1.0",
        category=EvaluationTaskCategory.AUTHENTICATION,
        expected_behaviors=(ExpectedBehavior("BEH-002", "second behavior", "complete", "evidence", "PASS", ()),),
        success_criteria=(SuccessCriterion("CRIT-002", "test from other task passes", SuccessCriterionType.TEST_PASS, True, "evidence", test_ids=("TEST-999",)),),
    )
    result = validate_benchmark((task, referenced))
    assert any(item.code == "UNRESOLVED_REFERENCE" for item in result.issues)


def test_test_criterion_without_test_ids_warns() -> None:
    task = _minimal_task("EVAL-001", with_tests=True, with_criteria=False)
    task = EvaluationTask(
        task_id=task.task_id,
        title=task.title,
        description=task.description,
        version=task.version,
        category=task.category,
        difficulty=task.difficulty,
        tests=(TestDefinition("TEST-001", "test", EvaluationTestType.API, "tests/t.py", True, "PASS", (), ()),),
        success_criteria=(SuccessCriterion("CRIT-001", "tests pass", SuccessCriterionType.TEST_PASS, True, "evidence", test_ids=()),),
    )
    result = validate_benchmark((task,))
    assert any(item.code == "TEST_CRITERION_NO_TEST" for item in result.issues)


def test_task_without_evaluable_criterion_warns() -> None:
    task = EvaluationTask(
        task_id="EVAL-001",
        title="No evaluable criteria",
        description="Empty.",
        version="1.0",
        category=EvaluationTaskCategory.API_ENDPOINT,
        expected_behaviors=(ExpectedBehavior("BEH-001", "behavior", "complete", "evidence", "PASS", ()),),
        success_criteria=(SuccessCriterion("CRIT-001", "empty criterion", SuccessCriterionType.TEST_PASS, True, "evidence", test_ids=()),),
    )
    result = validate_benchmark((task,))
    assert any(item.code == "NO_EVALUABLE_CRITERION" for item in result.issues)


def test_single_category_dominance_warns() -> None:
    tasks = tuple(_minimal_task(f"EVAL-{index + 1:03d}") for index in range(6))
    result = validate_benchmark(tasks)
    assert any(item.code == "CATEGORY_DOMINANCE" for item in result.issues)


def test_scoring_weights_must_sum_to_one() -> None:
    task = _minimal_task("EVAL-001")
    result = validate_benchmark((task,), scoring_weights_sum=0.9)
    assert result.status is ValidationStatus.INVALID
    assert any(item.code == "WEIGHTS_DO_NOT_SUM_TO_ONE" for item in result.issues)
    assert result.scoring_policy_valid is False


def test_valid_scoring_weights_pass() -> None:
    task = _minimal_task("EVAL-001")
    result = validate_benchmark((task,), scoring_weights_sum=1.0)
    assert result.scoring_policy_valid is True


def test_scoring_policy_invalid_version_fails() -> None:
    task = _minimal_task("EVAL-001")
    policy = ScoringPolicy(scoring_policy_version="not-a-version")
    result = validate_benchmark((task,), scoring_policy=policy)
    assert result.scoring_policy_valid is False
    assert any(item.code == "INVALID_POLICY_VERSION" for item in result.issues)


def test_scoring_policy_invalid_target_fails() -> None:
    """The scoring policy itself refuses non-positive targets at construction."""

    with pytest.raises(ValueError):
        ScoringPolicy(scoring_policy_version="1.0", duration_target_seconds=-1.0)
    with pytest.raises(ValueError):
        ScoringPolicy(scoring_policy_version="1.0", iteration_target=0.0)
    with pytest.raises(ValueError):
        ScoringPolicy(scoring_policy_version="1.0", tool_call_target=float("inf"))


def test_duplicate_reference_ids_fail_across_tasks() -> None:
    first = _minimal_task("EVAL-001")
    second = EvaluationTask(
        task_id="EVAL-002",
        title="Second",
        description="Second.",
        version="1.0",
        category=EvaluationTaskCategory.AUTHENTICATION,
        difficulty=EvaluationDifficulty.MEDIUM,
        requirements=(Requirement("REQ-001", "Reuses requirement id", True, 1),),
        tests=(TestDefinition("TEST-001", "Reuses test id", EvaluationTestType.API, "tests/t.py", True, "PASS", ("REQ-001",), ()),),
    )
    result = validate_benchmark((first, second))
    assert any(item.code == "DUPLICATE_REFERENCE" for item in result.issues)


def test_health_score_penalizes_errors() -> None:
    tasks = tuple(_minimal_task(f"EVAL-{index + 1:03d}") for index in range(4))
    invalid = EvaluationTask(task_id="bad", title="x", description="y", version="1.0", category="UNKNOWN-CATEGORY", expected_behaviors=(ExpectedBehavior("B-1", "b", "complete", "evidence", "PASS", ()),))
    result = validate_benchmark(tasks + (invalid,))
    assert result.health.error_count > 0
    assert result.health.score < 1.0
    assert result.health.task_count == 5
    assert result.health.validated_task_count == 4


def test_one_task_dominates_warning() -> None:
    heavy = _minimal_task("EVAL-001")
    heavy = EvaluationTask(
        task_id=heavy.task_id,
        title=heavy.title,
        description=heavy.description,
        version=heavy.version,
        category=heavy.category,
        difficulty=heavy.difficulty,
        tests=tuple(TestDefinition(f"TEST-{index:03d}", f"test {index}", EvaluationTestType.API, "tests/t.py", True, "PASS", (), ()) for index in range(8)),
        success_criteria=(SuccessCriterion("CRIT-001", "tests pass", SuccessCriterionType.TEST_PASS, True, "evidence", test_ids=tuple(f"TEST-{index:03d}" for index in range(8))),),
    )
    light = _minimal_task("EVAL-002")
    light = EvaluationTask(
        task_id=light.task_id,
        title=light.title,
        description=light.description,
        version=light.version,
        category=light.category,
        difficulty=light.difficulty,
        tests=(TestDefinition("TEST-L001", "light test", EvaluationTestType.API, "tests/t.py", True, "PASS", (), ()),),
        success_criteria=(SuccessCriterion("CRIT-001", "tests pass", SuccessCriterionType.TEST_PASS, True, "evidence", test_ids=("TEST-L001",)),),
    )
    result = validate_benchmark((heavy, light))
    assert any(item.code == "ONE_TASK_DOMINATES" for item in result.issues)


def test_valid_benchmark_serializes_canonically() -> None:
    result = validate_benchmark((_minimal_task("EVAL-001"),))
    payload = json.loads(result.to_json())
    assert payload["status"] == "VALID"
    assert payload["health"]["score"] == 1.0
    assert list(payload.keys()) == sorted(payload.keys())


def test_issues_sorted_deterministically() -> None:
    task = _minimal_task("EVAL-001")
    invalid = EvaluationTask(task_id="bad", title="x", description="y", version="1.0", category="UNKNOWN-CATEGORY", expected_behaviors=(ExpectedBehavior("B-1", "b", "complete", "evidence", "PASS", ()),))
    first = validate_benchmark((task, invalid))
    second = validate_benchmark((task, invalid))
    assert first.to_json() == second.to_json()


def test_validation_reuses_phase81_task_validator() -> None:
    task = _minimal_task("EVAL-001")
    result = validate_benchmark((task,))
    standalone = validate_evaluation_task(task)
    assert result.task_validations[0]["valid"] is standalone.valid


def _minimal_task_unique(task_id: str) -> EvaluationTask:
    return _minimal_task(
        task_id,
        requirement_id=f"REQ-{task_id}",
        behavior_id=f"BEH-{task_id}",
        test_id=f"TEST-{task_id}",
        criterion_id=f"CRIT-{task_id}",
    )


def test_warning_status_when_only_warnings_exist() -> None:
    tasks = tuple(_minimal_task_unique(f"EVAL-{index + 1:03d}") for index in range(6))
    task = tasks[0]
    task = EvaluationTask(
        task_id=task.task_id,
        title=task.title,
        description=task.description,
        version=task.version,
        category=task.category,
        difficulty=task.difficulty,
        user_intent=task.user_intent,
        project_definition=task.project_definition,
        expected_behaviors=task.expected_behaviors,
        ground_truth=task.ground_truth,
        requirements=task.requirements,
        allowed_scope=task.allowed_scope,
        tests=task.tests,
        success_criteria=(SuccessCriterion(f"CRIT-{task.task_id}", "tests pass", SuccessCriterionType.TEST_PASS, True, "evidence", test_ids=()),),
    )
    result = validate_benchmark((task,) + tasks[1:])
    assert result.status is ValidationStatus.WARNING
    assert result.health.error_count == 0
    assert result.health.warning_count > 0


def test_issue_level_values_are_exhaustive() -> None:
    assert IssueLevel.ERROR.value == "ERROR"
    assert IssueLevel.WARNING.value == "WARNING"
    assert IssueLevel.INFO.value == "INFO"
