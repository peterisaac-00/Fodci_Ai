"""Declarative, immutable evaluation-task definitions for Phase 8.1.

This module defines benchmark task data only. It never executes tasks, tests,
commands, tools, network calls, or model evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, TypeVar


class EvaluationTaskCategory(str, Enum):
    API_ENDPOINT = "API_ENDPOINT"
    AUTHENTICATION = "AUTHENTICATION"
    DATABASE = "DATABASE"
    BUG_FIX = "BUG_FIX"
    TESTING = "TESTING"
    DOCKER = "DOCKER"
    REFACTOR = "REFACTOR"
    CONFIGURATION = "CONFIGURATION"
    DOCUMENTATION = "DOCUMENTATION"
    INVESTIGATION = "INVESTIGATION"


class EvaluationDifficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    EXPERT = "EXPERT"


class ChangeType(str, Enum):
    CREATE = "CREATE"
    EDIT = "EDIT"
    DELETE = "DELETE"
    RENAME = "RENAME"


class ExpectedAreaType(str, Enum):
    REQUIRED_CHANGE = "REQUIRED_CHANGE"
    OPTIONAL_CHANGE = "OPTIONAL_CHANGE"
    INSPECTION_ONLY = "INSPECTION_ONLY"


class EvaluationTestType(str, Enum):
    UNIT = "UNIT"
    INTEGRATION = "INTEGRATION"
    API = "API"
    REGRESSION = "REGRESSION"
    END_TO_END = "END_TO_END"


class SuccessCriterionType(str, Enum):
    BEHAVIOR = "BEHAVIOR"
    TEST_PASS = "TEST_PASS"
    FILE_CHANGE = "FILE_CHANGE"
    NO_UNRELATED_CHANGE = "NO_UNRELATED_CHANGE"
    REGRESSION_FREE = "REGRESSION_FREE"
    VERIFICATION = "VERIFICATION"
    COMPLETION = "COMPLETION"


class ForbiddenChangeType(str, Enum):
    UNRELATED_FILE = "UNRELATED_FILE"
    SECRETS = "SECRETS"
    ENVIRONMENT = "ENVIRONMENT"
    BENCHMARK_TEST = "BENCHMARK_TEST"
    DISABLE_TESTS = "DISABLE_TESTS"
    WEAKEN_SECURITY = "WEAKEN_SECURITY"
    DELETE_FUNCTIONALITY = "DELETE_FUNCTIONALITY"
    CUSTOM = "CUSTOM"


class ValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


def _tuple(value: object) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        return (value,)
    return tuple(value)  # type: ignore[arg-type]


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return value  # type: ignore[return-value]
    return MappingProxyType(dict(sorted(value.items(), key=lambda item: str(item[0]))))


@dataclass(frozen=True, slots=True)
class ProjectDefinition:
    """Declarative description of the project in which a task is evaluated."""

    project_type: str = ""
    language: str = ""
    framework: str = ""
    runtime: str = ""
    database: str = ""
    project_root_requirements: tuple[str, ...] = ()
    setup_requirements: tuple[str, ...] = ()
    entry_point: str = ""
    test_framework: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root_requirements", _tuple(self.project_root_requirements))
        object.__setattr__(self, "setup_requirements", _tuple(self.setup_requirements))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class Requirement:
    """One independently identifiable task requirement."""

    requirement_id: str = ""
    description: str = ""
    mandatory: bool = True
    priority: int = 1

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class ExpectedBehavior:
    """One declarative input/action/expected-result behavior case."""

    behavior_id: str = ""
    input: str = ""
    action: str = ""
    expected_output: str = ""
    expected_status: str = ""
    expected_side_effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_side_effects", _tuple(self.expected_side_effects))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class AllowedScope:
    """Declarative allowed and forbidden path/change boundaries."""

    allowed_files: tuple[str, ...] = ()
    allowed_directories: tuple[str, ...] = ()
    allowed_patterns: tuple[str, ...] = ()
    allowed_change_types: tuple[ChangeType | str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "allowed_files",
            "allowed_directories",
            "allowed_patterns",
            "allowed_change_types",
            "forbidden_paths",
            "forbidden_patterns",
        ):
            object.__setattr__(self, field_name, _tuple(getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class ExpectedArea:
    """A logical area or exact path set expected to be inspected or changed."""

    name: str = ""
    paths: tuple[str, ...] = ()
    area_type: ExpectedAreaType | str = ExpectedAreaType.REQUIRED_CHANGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", _tuple(self.paths))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class TestDefinition:
    """Declarative test evidence expected by a future benchmark runner."""

    __test__ = False
    test_id: str = ""
    description: str = ""
    test_type: EvaluationTestType | str = EvaluationTestType.UNIT
    target: str = ""
    required: bool = True
    expected_result: str = ""
    requirement_ids: tuple[str, ...] = ()
    behavior_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_ids", _tuple(self.requirement_ids))
        object.__setattr__(self, "behavior_ids", _tuple(self.behavior_ids))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class SuccessCriterion:
    """One later-evaluated correctness criterion, without pass/fail logic."""

    criterion_id: str = ""
    description: str = ""
    criterion_type: SuccessCriterionType | str = SuccessCriterionType.BEHAVIOR
    required: bool = True
    evidence_expectation: str = ""
    requirement_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()
    behavior_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "requirement_ids", _tuple(self.requirement_ids))
        object.__setattr__(self, "test_ids", _tuple(self.test_ids))
        object.__setattr__(self, "behavior_ids", _tuple(self.behavior_ids))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class ForbiddenChange:
    """Declarative change that a future evaluator must reject or report."""

    description: str = ""
    change_type: ForbiddenChangeType | str = ForbiddenChangeType.CUSTOM
    paths: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", _tuple(self.paths))
        object.__setattr__(self, "patterns", _tuple(self.patterns))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class EvaluationConstraint:
    """Declarative task constraints, distinct from runtime execution budgets."""

    max_files_expected: int | None = None
    max_scope: str = ""
    required_framework: str = ""
    required_language: str = ""
    required_test_framework: str = ""
    prohibited_technologies: tuple[str, ...] = ()
    prohibited_modifications: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prohibited_technologies", _tuple(self.prohibited_technologies))
        object.__setattr__(self, "prohibited_modifications", _tuple(self.prohibited_modifications))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Correctness contract that permits multiple valid implementations."""

    expected_behavior_ids: tuple[str, ...] = ()
    required_outcomes: tuple[str, ...] = ()
    required_interfaces: tuple[str, ...] = ()
    required_invariants: tuple[str, ...] = ()
    allowed_implementation_alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "expected_behavior_ids",
            "required_outcomes",
            "required_interfaces",
            "required_invariants",
            "allowed_implementation_alternatives",
        ):
            object.__setattr__(self, field_name, _tuple(getattr(self, field_name)))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    """Complete immutable benchmark definition; it contains no runtime state."""

    task_id: str = ""
    title: str = ""
    description: str = ""
    version: str = ""
    category: EvaluationTaskCategory | str = EvaluationTaskCategory.BUG_FIX
    difficulty: EvaluationDifficulty | str = EvaluationDifficulty.MEDIUM
    project_definition: ProjectDefinition = ProjectDefinition()
    user_intent: str = ""
    expected_behaviors: tuple[ExpectedBehavior, ...] = ()
    requirements: tuple[Requirement, ...] = ()
    allowed_scope: AllowedScope = AllowedScope()
    expected_areas: tuple[ExpectedArea, ...] = ()
    tests: tuple[TestDefinition, ...] = ()
    success_criteria: tuple[SuccessCriterion, ...] = ()
    forbidden_changes: tuple[ForbiddenChange, ...] = ()
    constraints: EvaluationConstraint = EvaluationConstraint()
    ground_truth: GroundTruth = GroundTruth()
    metadata: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        for field_name in (
            "expected_behaviors",
            "requirements",
            "expected_areas",
            "tests",
            "success_criteria",
            "forbidden_changes",
        ):
            object.__setattr__(self, field_name, _tuple(getattr(self, field_name)))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)

    def to_json(self) -> str:
        """Return canonical JSON suitable for benchmark files and version control."""

        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    path: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


@dataclass(frozen=True, slots=True)
class EvaluationTaskValidationResult:
    valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", _tuple(self.errors))
        object.__setattr__(self, "warnings", _tuple(self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return _serialize_dataclass(self)


class EvaluationTaskValidator:
    """Pure deterministic validator; it never executes or repairs a task."""

    _version_pattern = re.compile(r"^\d+\.\d+$")
    _task_id_pattern = re.compile(r"^EVAL-[A-Z0-9][A-Z0-9_-]*$")

    def validate(self, task: EvaluationTask) -> EvaluationTaskValidationResult:
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        if not isinstance(task, EvaluationTask):
            return EvaluationTaskValidationResult(
                False,
                (ValidationIssue("INVALID_TASK", "task must be an EvaluationTask", "task"),),
            )

        self._text(task.task_id, "task_id", "MISSING_TASK_ID", errors)
        if isinstance(task.task_id, str) and task.task_id and not self._task_id_pattern.fullmatch(task.task_id):
            self._error(errors, "INVALID_TASK_ID", "task_id must use a stable EVAL-* identifier", "task_id")
        self._text(task.title, "title", "EMPTY_TITLE", errors)
        self._text(task.description, "description", "EMPTY_DESCRIPTION", errors)
        self._text(task.version, "version", "INVALID_VERSION", errors)
        if isinstance(task.version, str) and task.version and not self._version_pattern.fullmatch(task.version):
            self._error(errors, "INVALID_VERSION", "version must have MAJOR.MINOR form", "version")
        self._enum(task.category, EvaluationTaskCategory, "category", errors)
        self._enum(task.difficulty, EvaluationDifficulty, "difficulty", errors)
        self._text(task.user_intent, "user_intent", "EMPTY_USER_INTENT", errors)

        if not isinstance(task.project_definition, ProjectDefinition):
            self._error(errors, "MALFORMED_PROJECT", "project_definition must be ProjectDefinition", "project_definition")
        else:
            for field_name in ("project_type", "language", "runtime", "test_framework"):
                self._text(getattr(task.project_definition, field_name), f"project_definition.{field_name}", "MALFORMED_PROJECT", errors)
            self._paths(task.project_definition.project_root_requirements, "project_definition.project_root_requirements", errors)
            self._paths(task.project_definition.setup_requirements, "project_definition.setup_requirements", errors)

        behavior_ids = self._validate_behaviors(task.expected_behaviors, errors)
        requirement_ids = self._validate_requirements(task.requirements, errors)
        test_ids = self._validate_tests(task.tests, requirement_ids, behavior_ids, errors)
        self._validate_areas(task.expected_areas, errors)
        self._validate_criteria(task.success_criteria, requirement_ids, test_ids, behavior_ids, errors)
        self._validate_scope(task.allowed_scope, errors)
        self._validate_forbidden(task.forbidden_changes, errors)
        self._validate_constraints(task.constraints, errors)
        self._validate_ground_truth(task.ground_truth, behavior_ids, errors)
        self._validate_metadata(task.metadata, errors)
        if not task.expected_behaviors:
            self._error(errors, "MISSING_BEHAVIORS", "at least one expected behavior is required", "expected_behaviors")
        if not task.requirements:
            warnings.append(ValidationIssue("NO_REQUIREMENTS", "task has no requirements", "requirements", ValidationSeverity.WARNING))
        if not task.success_criteria:
            self._error(errors, "MISSING_CRITERIA", "at least one success criterion is required", "success_criteria")
        if not task.ground_truth.required_outcomes and not task.ground_truth.required_interfaces and not task.ground_truth.required_invariants:
            self._error(errors, "INVALID_GROUND_TRUTH", "ground truth must contain an outcome, interface, or invariant", "ground_truth")
        return EvaluationTaskValidationResult(not errors, tuple(errors), tuple(warnings))

    def _validate_behaviors(self, values: tuple[ExpectedBehavior, ...], errors: list[ValidationIssue]) -> set[str]:
        ids: set[str] = set()
        for index, value in enumerate(values):
            path = f"expected_behaviors[{index}]"
            if not isinstance(value, ExpectedBehavior):
                self._error(errors, "INVALID_BEHAVIOR", "expected behavior has an invalid type", path)
                continue
            self._unique(value.behavior_id, ids, "DUPLICATE_BEHAVIOR_ID", path, errors)
            self._text(value.behavior_id, f"{path}.behavior_id", "INVALID_BEHAVIOR", errors)
            for name in ("input", "action", "expected_output"):
                self._text(getattr(value, name), f"{path}.{name}", "INVALID_BEHAVIOR", errors)
        return ids

    def _validate_requirements(self, values: tuple[Requirement, ...], errors: list[ValidationIssue]) -> set[str]:
        ids: set[str] = set()
        for index, value in enumerate(values):
            path = f"requirements[{index}]"
            if not isinstance(value, Requirement):
                self._error(errors, "INVALID_REQUIREMENT", "requirement has an invalid type", path)
                continue
            self._unique(value.requirement_id, ids, "DUPLICATE_REQUIREMENT_ID", path, errors)
            self._text(value.requirement_id, f"{path}.requirement_id", "INVALID_REQUIREMENT", errors)
            self._text(value.description, f"{path}.description", "INVALID_REQUIREMENT", errors)
            if not isinstance(value.mandatory, bool):
                self._error(errors, "INVALID_REQUIREMENT", "mandatory must be boolean", f"{path}.mandatory")
            if not isinstance(value.priority, int) or isinstance(value.priority, bool) or value.priority <= 0:
                self._error(errors, "INVALID_REQUIREMENT", "priority must be a positive integer", f"{path}.priority")
        return ids

    def _validate_tests(self, values: tuple[TestDefinition, ...], requirement_ids: set[str], behavior_ids: set[str], errors: list[ValidationIssue]) -> set[str]:
        ids: set[str] = set()
        for index, value in enumerate(values):
            path = f"tests[{index}]"
            if not isinstance(value, TestDefinition):
                self._error(errors, "INVALID_TEST", "test has an invalid type", path)
                continue
            self._unique(value.test_id, ids, "DUPLICATE_TEST_ID", path, errors)
            self._text(value.test_id, f"{path}.test_id", "INVALID_TEST", errors)
            self._text(value.description, f"{path}.description", "INVALID_TEST", errors)
            self._text(value.target, f"{path}.target", "INVALID_TEST", errors)
            self._text(value.expected_result, f"{path}.expected_result", "INVALID_TEST", errors)
            self._enum(value.test_type, EvaluationTestType, f"{path}.test_type", errors)
            self._references(value.requirement_ids, requirement_ids, f"{path}.requirement_ids", errors)
            self._references(value.behavior_ids, behavior_ids, f"{path}.behavior_ids", errors)
        return ids

    def _validate_areas(self, values: tuple[ExpectedArea, ...], errors: list[ValidationIssue]) -> None:
        for index, value in enumerate(values):
            path = f"expected_areas[{index}]"
            if not isinstance(value, ExpectedArea):
                self._error(errors, "INVALID_AREA", "expected area has an invalid type", path)
                continue
            self._text(value.name, f"{path}.name", "INVALID_AREA", errors)
            self._paths(value.paths, f"{path}.paths", errors)
            self._enum(value.area_type, ExpectedAreaType, f"{path}.area_type", errors)

    def _validate_criteria(self, values: tuple[SuccessCriterion, ...], requirement_ids: set[str], test_ids: set[str], behavior_ids: set[str], errors: list[ValidationIssue]) -> None:
        ids: set[str] = set()
        for index, value in enumerate(values):
            path = f"success_criteria[{index}]"
            if not isinstance(value, SuccessCriterion):
                self._error(errors, "INVALID_CRITERION", "criterion has an invalid type", path)
                continue
            self._unique(value.criterion_id, ids, "DUPLICATE_CRITERION_ID", path, errors)
            self._text(value.criterion_id, f"{path}.criterion_id", "INVALID_CRITERION", errors)
            self._text(value.description, f"{path}.description", "INVALID_CRITERION", errors)
            self._text(value.evidence_expectation, f"{path}.evidence_expectation", "INVALID_CRITERION", errors)
            self._enum(value.criterion_type, SuccessCriterionType, f"{path}.criterion_type", errors)
            self._references(value.requirement_ids, requirement_ids, f"{path}.requirement_ids", errors)
            self._references(value.test_ids, test_ids, f"{path}.test_ids", errors)
            self._references(value.behavior_ids, behavior_ids, f"{path}.behavior_ids", errors)

    def _validate_scope(self, value: AllowedScope, errors: list[ValidationIssue]) -> None:
        if not isinstance(value, AllowedScope):
            self._error(errors, "INVALID_SCOPE", "allowed_scope must be AllowedScope", "allowed_scope")
            return
        if not (value.allowed_files or value.allowed_directories or value.allowed_patterns):
            self._error(errors, "INVALID_SCOPE", "at least one allowed file, directory, or pattern is required", "allowed_scope")
        for name in ("allowed_files", "allowed_directories", "allowed_patterns", "forbidden_paths", "forbidden_patterns"):
            self._paths(getattr(value, name), f"allowed_scope.{name}", errors)
        for index, item in enumerate(value.allowed_change_types):
            if not isinstance(item, ChangeType) and item not in {member.value for member in ChangeType}:
                self._error(errors, "INVALID_SCOPE", "unknown change type", f"allowed_scope.allowed_change_types[{index}]")
        allowed = set(value.allowed_files) | set(value.allowed_directories) | set(value.allowed_patterns)
        forbidden = set(value.forbidden_paths) | set(value.forbidden_patterns)
        for path in sorted(allowed & forbidden):
            self._error(errors, "CONTRADICTORY_SCOPE", f"path/pattern is both allowed and forbidden: {path}", "allowed_scope")

    def _validate_forbidden(self, values: tuple[ForbiddenChange, ...], errors: list[ValidationIssue]) -> None:
        for index, value in enumerate(values):
            path = f"forbidden_changes[{index}]"
            if not isinstance(value, ForbiddenChange):
                self._error(errors, "INVALID_FORBIDDEN_CHANGE", "forbidden change has an invalid type", path)
                continue
            self._text(value.description, f"{path}.description", "INVALID_FORBIDDEN_CHANGE", errors)
            self._enum(value.change_type, ForbiddenChangeType, f"{path}.change_type", errors)
            if not value.paths and not value.patterns:
                self._error(errors, "INVALID_FORBIDDEN_CHANGE", "a forbidden change needs a path or pattern", path)
            self._paths(value.paths, f"{path}.paths", errors)
            self._paths(value.patterns, f"{path}.patterns", errors)

    def _validate_constraints(self, value: EvaluationConstraint, errors: list[ValidationIssue]) -> None:
        if not isinstance(value, EvaluationConstraint):
            self._error(errors, "INVALID_CONSTRAINTS", "constraints must be EvaluationConstraint", "constraints")
            return
        if value.max_files_expected is not None and (not isinstance(value.max_files_expected, int) or isinstance(value.max_files_expected, bool) or value.max_files_expected <= 0):
            self._error(errors, "INVALID_CONSTRAINTS", "max_files_expected must be a positive integer or None", "constraints.max_files_expected")
        for name in ("prohibited_technologies", "prohibited_modifications"):
            self._paths(getattr(value, name), f"constraints.{name}", errors)
        if value.required_framework and value.required_framework in value.prohibited_technologies:
            self._error(errors, "CONTRADICTORY_CONSTRAINTS", "required framework is prohibited", "constraints")
        if value.required_language and value.required_language in value.prohibited_technologies:
            self._error(errors, "CONTRADICTORY_CONSTRAINTS", "required language is prohibited", "constraints")

    def _validate_ground_truth(self, value: GroundTruth, behavior_ids: set[str], errors: list[ValidationIssue]) -> None:
        if not isinstance(value, GroundTruth):
            self._error(errors, "INVALID_GROUND_TRUTH", "ground_truth must be GroundTruth", "ground_truth")
            return
        self._references(value.expected_behavior_ids, behavior_ids, "ground_truth.expected_behavior_ids", errors)
        all_values = (
            value.required_outcomes,
            value.required_interfaces,
            value.required_invariants,
            value.allowed_implementation_alternatives,
        )
        for index, group in enumerate(all_values):
            self._paths(group, f"ground_truth[{index}]", errors)

    def _validate_metadata(self, value: Mapping[str, str], errors: list[ValidationIssue]) -> None:
        if not isinstance(value, Mapping):
            self._error(errors, "INVALID_METADATA", "metadata must be a mapping", "metadata")
            return
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(item, str):
                self._error(errors, "INVALID_METADATA", "metadata keys and values must be non-empty text", "metadata")

    @staticmethod
    def _text(value: object, path: str, code: str, errors: list[ValidationIssue]) -> None:
        if not isinstance(value, str) or not value.strip():
            errors.append(ValidationIssue(code, "value must contain non-whitespace text", path))

    @staticmethod
    def _enum(value: object, enum_type: type[Enum], path: str, errors: list[ValidationIssue]) -> None:
        valid = {member.value for member in enum_type}
        if not isinstance(value, enum_type) and value not in valid:
            errors.append(ValidationIssue("INVALID_ENUM", f"value must be one of {sorted(valid)}", path))

    @staticmethod
    def _unique(value: object, seen: set[str], code: str, path: str, errors: list[ValidationIssue]) -> None:
        if isinstance(value, str) and value in seen:
            errors.append(ValidationIssue(code, f"duplicate identifier: {value}", path))
        elif isinstance(value, str):
            seen.add(value)

    @staticmethod
    def _references(values: object, valid: set[str], path: str, errors: list[ValidationIssue]) -> None:
        if not isinstance(values, tuple):
            errors.append(ValidationIssue("INVALID_REFERENCE", "references must be an ordered tuple", path))
            return
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip() or value not in valid:
                errors.append(ValidationIssue("INVALID_REFERENCE", f"unknown reference: {value!r}", f"{path}[{index}]"))

    @staticmethod
    def _paths(values: object, path: str, errors: list[ValidationIssue]) -> None:
        if not isinstance(values, tuple):
            errors.append(ValidationIssue("INVALID_PATHS", "paths/patterns must be ordered tuples", path))
            return
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                errors.append(ValidationIssue("INVALID_PATH", "path or pattern must contain text", f"{path}[{index}]"))
                continue
            normalized = value.replace("\\", "/")
            if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized) or ".." in PurePosixPath(normalized).parts:
                errors.append(ValidationIssue("INVALID_PATH", "absolute or parent-traversal paths are not allowed", f"{path}[{index}]"))

    @staticmethod
    def _error(errors: list[ValidationIssue], code: str, message: str, path: str) -> None:
        errors.append(ValidationIssue(code, message, path))


def create_evaluation_task(**kwargs: Any) -> EvaluationTask:
    """Construct one declarative task without executing or validating it implicitly."""

    return EvaluationTask(**kwargs)


def validate_evaluation_task(task: EvaluationTask) -> EvaluationTaskValidationResult:
    """Validate one task definition and return structured immutable issues."""

    return EvaluationTaskValidator().validate(task)


def serialize_evaluation_task(task: EvaluationTask) -> str:
    """Serialize one task using canonical deterministic JSON."""

    if not isinstance(task, EvaluationTask):
        raise TypeError("task must be an EvaluationTask")
    return task.to_json()


def _tuple(value: object) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        return (value,)
    return tuple(value)  # type: ignore[arg-type]


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return value  # type: ignore[return-value]
    return MappingProxyType(dict(sorted(value.items(), key=lambda item: str(item[0]))))


def _serialize_dataclass(value: object) -> dict[str, Any]:
    from dataclasses import fields

    return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}


def _serialize(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


__all__ = [
    "AllowedScope",
    "ChangeType",
    "EvaluationConstraint",
    "EvaluationDifficulty",
    "EvaluationTask",
    "EvaluationTaskCategory",
    "EvaluationTaskValidationResult",
    "EvaluationTaskValidator",
    "EvaluationTestType",
    "ExpectedArea",
    "ExpectedAreaType",
    "ExpectedBehavior",
    "ForbiddenChange",
    "ForbiddenChangeType",
    "GroundTruth",
    "ProjectDefinition",
    "Requirement",
    "SuccessCriterion",
    "SuccessCriterionType",
    "TestDefinition",
    "ValidationIssue",
    "ValidationSeverity",
    "create_evaluation_task",
    "serialize_evaluation_task",
    "validate_evaluation_task",
]
