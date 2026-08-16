from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from backend_ai.evaluation import (
    AllowedScope,
    ChangeType,
    EvaluationConstraint,
    EvaluationDifficulty,
    EvaluationTask,
    EvaluationTaskCategory,
    EvaluationTaskValidator,
    EvaluationTestType,
    ExpectedArea,
    ExpectedAreaType,
    ExpectedBehavior,
    ForbiddenChange,
    ForbiddenChangeType,
    GroundTruth,
    ProjectDefinition,
    Requirement,
    SuccessCriterion,
    SuccessCriterionType,
    TestDefinition,
    create_evaluation_task,
    serialize_evaluation_task,
    validate_evaluation_task,
)


def make_task(category: EvaluationTaskCategory = EvaluationTaskCategory.API_ENDPOINT) -> EvaluationTask:
    return EvaluationTask(
        task_id="EVAL-001",
        title="Add JWT Authentication",
        description="Implement a login endpoint in the existing backend project.",
        version="1.0",
        category=category,
        difficulty=EvaluationDifficulty.MEDIUM,
        project_definition=ProjectDefinition(
            project_type="backend-service",
            language="Python",
            framework="FastAPI",
            runtime="Python 3.12",
            database="PostgreSQL",
            project_root_requirements=("existing project root",),
            setup_requirements=("dependencies are already installed",),
            entry_point="src/app.py",
            test_framework="pytest",
        ),
        user_intent="Create a secure login endpoint.",
        requirements=(
            Requirement("REQ-001", "Create POST /api/login endpoint", True, 1),
            Requirement("REQ-002", "Reject invalid credentials with HTTP 401", True, 2),
        ),
        expected_behaviors=(
            ExpectedBehavior(
                "BEH-001",
                "POST /api/login with valid credentials",
                "authenticate the user",
                "a JWT token is returned",
                "200",
                ("no unrelated file changes",),
            ),
            ExpectedBehavior(
                "BEH-002",
                "POST /api/login with invalid credentials",
                "reject authentication",
                "an error response is returned",
                "401",
            ),
        ),
        allowed_scope=AllowedScope(
            allowed_files=("src/auth.py", "tests/test_auth.py"),
            allowed_directories=("src/",),
            allowed_patterns=("tests/test_*.py",),
            allowed_change_types=(ChangeType.EDIT, ChangeType.CREATE),
            forbidden_paths=(".env", "secrets/"),
            forbidden_patterns=("*.key",),
        ),
        expected_areas=(
            ExpectedArea("authentication layer", ("src/auth.py",), ExpectedAreaType.REQUIRED_CHANGE),
            ExpectedArea("existing API tests", ("tests/test_auth.py",), ExpectedAreaType.OPTIONAL_CHANGE),
        ),
        tests=(
            TestDefinition("TEST-001", "valid login test", EvaluationTestType.API, "tests/test_auth.py", True, "PASS", ("REQ-001",), ("BEH-001",)),
            TestDefinition("TEST-002", "invalid login test", EvaluationTestType.INTEGRATION, "tests/test_auth.py", True, "PASS", ("REQ-002",), ("BEH-002",)),
        ),
        success_criteria=(
            SuccessCriterion("CRIT-001", "valid behavior passes", SuccessCriterionType.BEHAVIOR, True, "parsed API response", behavior_ids=("BEH-001",)),
            SuccessCriterion("CRIT-002", "required tests pass", SuccessCriterionType.TEST_PASS, True, "authoritative parsed test result", test_ids=("TEST-001", "TEST-002")),
        ),
        forbidden_changes=(
            ForbiddenChange("Do not modify secrets.", ForbiddenChangeType.SECRETS, (".env",), ("*.key",)),
            ForbiddenChange("Do not disable benchmark tests.", ForbiddenChangeType.DISABLE_TESTS, (), ("tests/benchmark_*.py",)),
        ),
        constraints=EvaluationConstraint(
            max_files_expected=3,
            max_scope="authentication and its tests",
            required_framework="FastAPI",
            required_language="Python",
            required_test_framework="pytest",
            prohibited_technologies=("shell scripts",),
            prohibited_modifications=("secrets",),
        ),
        ground_truth=GroundTruth(
            expected_behavior_ids=("BEH-001", "BEH-002"),
            required_outcomes=("valid credentials produce a valid JWT",),
            required_interfaces=("POST /api/login",),
            required_invariants=("invalid credentials never produce a token",),
            allowed_implementation_alternatives=("JWT library A", "JWT library B"),
        ),
        metadata={"suite": "backend", "owner": "evaluation"},
    )


def test_valid_api_task_validates_and_serializes() -> None:
    task = make_task()
    result = validate_evaluation_task(task)
    assert result.valid is True
    payload = json.loads(serialize_evaluation_task(task))
    assert payload["task_id"] == "EVAL-001"
    assert payload["category"] == "API_ENDPOINT"
    assert payload["tests"][0]["test_type"] == "API"
    assert payload["ground_truth"]["allowed_implementation_alternatives"] == ["JWT library A", "JWT library B"]


@pytest.mark.parametrize(
    "category",
    [
        EvaluationTaskCategory.AUTHENTICATION,
        EvaluationTaskCategory.DATABASE,
        EvaluationTaskCategory.BUG_FIX,
        EvaluationTaskCategory.TESTING,
        EvaluationTaskCategory.DOCKER,
        EvaluationTaskCategory.REFACTOR,
        EvaluationTaskCategory.CONFIGURATION,
        EvaluationTaskCategory.DOCUMENTATION,
        EvaluationTaskCategory.INVESTIGATION,
    ],
)
def test_supported_categories_are_declarative(category: EvaluationTaskCategory) -> None:
    task = make_task(category)
    assert EvaluationTaskValidator().validate(task).valid is True


def test_factory_and_public_validator_are_equivalent() -> None:
    task = make_task()
    assert validate_evaluation_task(task) == EvaluationTaskValidator().validate(task)


def test_invalid_task_id_is_reported() -> None:
    result = EvaluationTaskValidator().validate(replace(make_task(), task_id="bad"))
    assert result.valid is False
    assert any(issue.code == "INVALID_TASK_ID" for issue in result.errors)


def test_duplicate_requirement_test_and_criterion_ids_are_reported() -> None:
    base = make_task()
    task = replace(
        base,
        requirements=base.requirements + (Requirement("REQ-001", "duplicate", True, 1),),
        tests=base.tests + (TestDefinition("TEST-001", "duplicate", EvaluationTestType.UNIT, "x", True, "PASS"),),
        success_criteria=base.success_criteria + (SuccessCriterion("CRIT-001", "duplicate", SuccessCriterionType.BEHAVIOR, True, "evidence"),),
    )
    result = validate_evaluation_task(task)
    codes = {issue.code for issue in result.errors}
    assert {"DUPLICATE_REQUIREMENT_ID", "DUPLICATE_TEST_ID", "DUPLICATE_CRITERION_ID"} <= codes


def test_invalid_project_definition_is_reported() -> None:
    task = replace(make_task(), project_definition=ProjectDefinition(language="", test_framework=""))
    result = validate_evaluation_task(task)
    assert result.valid is False
    assert any(issue.code == "MALFORMED_PROJECT" for issue in result.errors)


def test_contradictory_allowed_and_forbidden_scope_is_reported() -> None:
    scope = AllowedScope(allowed_files=("src/auth.py",), forbidden_paths=("src/auth.py",))
    task = replace(make_task(), allowed_scope=scope)
    result = validate_evaluation_task(task)
    assert any(issue.code == "CONTRADICTORY_SCOPE" for issue in result.errors)


def test_invalid_parent_traversal_scope_is_reported() -> None:
    scope = AllowedScope(allowed_files=("../secrets.txt",))
    task = replace(make_task(), allowed_scope=scope)
    result = validate_evaluation_task(task)
    assert any(issue.code == "INVALID_PATH" for issue in result.errors)


def test_invalid_ground_truth_reference_is_reported() -> None:
    ground_truth = GroundTruth(expected_behavior_ids=("BEH-404",), required_outcomes=("outcome",))
    task = replace(make_task(), ground_truth=ground_truth)
    result = validate_evaluation_task(task)
    assert any(issue.code == "INVALID_REFERENCE" for issue in result.errors)


def test_invalid_constraints_are_reported() -> None:
    constraints = EvaluationConstraint(max_files_expected=0, required_framework="FastAPI", prohibited_technologies=("FastAPI",))
    task = replace(make_task(), constraints=constraints)
    result = validate_evaluation_task(task)
    codes = {issue.code for issue in result.errors}
    assert {"INVALID_CONSTRAINTS", "CONTRADICTORY_CONSTRAINTS"} <= codes


def test_empty_task_is_invalid_without_silent_repair() -> None:
    result = validate_evaluation_task(EvaluationTask())
    assert result.valid is False
    assert len(result.errors) >= 8


def test_serialization_is_deterministic_and_unicode_safe() -> None:
    task = replace(make_task(), title="إضافة مصادقة عربية")
    first = task.to_json()
    second = serialize_evaluation_task(task)
    assert first == second
    assert "إضافة" in first
    assert list(json.loads(first)["metadata"]) == ["owner", "suite"]


def test_models_are_immutable_and_collections_are_snapshotted() -> None:
    task = make_task()
    with pytest.raises(FrozenInstanceError):
        task.title = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        task.metadata["new"] = "value"  # type: ignore[index]
    assert isinstance(task.requirements, tuple)
    assert isinstance(task.allowed_scope.allowed_files, tuple)


def test_multiple_ground_truth_implementation_alternatives_are_preserved() -> None:
    task = make_task()
    assert task.ground_truth.allowed_implementation_alternatives == ("JWT library A", "JWT library B")
    assert "JWT library A" in task.to_json()


def test_forbidden_change_definitions_are_structural() -> None:
    task = make_task()
    assert task.forbidden_changes[0].change_type is ForbiddenChangeType.SECRETS
    assert task.forbidden_changes[1].patterns == ("tests/benchmark_*.py",)


def test_validation_is_deterministic() -> None:
    task = replace(make_task(), task_id="bad")
    first = validate_evaluation_task(task)
    second = validate_evaluation_task(task)
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_evaluation_task_has_no_runtime_execution_state() -> None:
    names = set(make_task().__dataclass_fields__)
    assert not {"result", "run", "process", "score", "metrics"} & names
