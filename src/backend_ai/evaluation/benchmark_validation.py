"""Phase 8.9 deterministic benchmark definition validation.

Validates that a collection of Phase 8.1 ``EvaluationTask`` objects and the
Phase 8.3 scoring policy form a fair, consistent, and executable benchmark
definition. It reuses the Phase 8.1 task validator for structural rules,
never executes tests or benchmarks, and reports every issue with a bounded
diagnostic health score.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any

from backend_ai.evaluation.scoring import BenchmarkScore, ScoringPolicy
from backend_ai.evaluation.task_model import (
    EvaluationTask,
    EvaluationTaskValidator,
    SuccessCriterionType,
)


class ValidationStatus(str, Enum):
    """Aggregate validation status for an entire benchmark definition."""

    VALID = "VALID"
    INVALID = "INVALID"
    WARNING = "WARNING"
    INCONCLUSIVE = "INCONCLUSIVE"


class IssueLevel(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One bounded diagnostic issue with deterministic identity."""

    issue_id: str
    level: IssueLevel
    task_id: str | None
    domain: str
    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.issue_id.startswith(("TASK-", "SCOPE-", "SCORING-", "FAIRNESS-", "HEALTH-")):
            raise ValueError(f"issue_id must carry a domain prefix: {self.issue_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "level": self.level.value,
            "task_id": self.task_id,
            "domain": self.domain,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkHealth:
    """Bounded aggregate diagnostic score for a benchmark definition."""

    score: float
    task_count: int
    validated_task_count: int
    issue_count: int
    error_count: int
    warning_count: int
    info_count: int
    max_score: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0.0 <= self.score <= self.max_score:
            raise ValueError("health score must be in [0, max_score]")
        if self.max_score <= 0:
            raise ValueError("max_score must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "task_count": self.task_count,
            "validated_task_count": self.validated_task_count,
            "issue_count": self.issue_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "max_score": self.max_score,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkValidationResult:
    """Complete deterministic validation result for a benchmark definition."""

    status: ValidationStatus
    health: BenchmarkHealth
    issues: tuple[ValidationIssue, ...]
    task_validations: tuple[dict[str, Any], ...]
    scoring_policy_valid: bool
    scoring_warnings: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ("task_validation", "scope_validation", "scoring_validation", "fairness_diagnostics")

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(sorted(self.issues, key=lambda item: item.issue_id)))
        object.__setattr__(self, "task_validations", tuple(self.task_validations))
        object.__setattr__(self, "scoring_warnings", tuple(sorted(set(self.scoring_warnings))))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "health": self.health.to_dict(),
            "issues": [item.to_dict() for item in self.issues],
            "task_validations": list(self.task_validations),
            "scoring_policy_valid": self.scoring_policy_valid,
            "scoring_warnings": list(self.scoring_warnings),
            "warnings": list(self.warnings),
            "evidence_ids": list(self.evidence_ids),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _issue(counter: list[int], prefix: str, level: IssueLevel, task_id: str | None, code: str, message: str) -> ValidationIssue:
    counter[0] += 1
    return ValidationIssue(f"{prefix}-{counter[0]:03d}", level, task_id, prefix.lower(), code, message)


class BenchmarkValidator:
    """Deterministic validation over a complete benchmark definition.

    Reuses the Phase 8.1 task validator for per-task structural rules and
    adds collection-level scope, reference, scoring, and fairness checks.
    No test execution or benchmark execution occurs.
    """

    def __init__(self, task_validator: EvaluationTaskValidator | None = None) -> None:
        self.task_validator = task_validator or EvaluationTaskValidator()

    def validate(
        self,
        tasks: tuple[EvaluationTask, ...] | list[EvaluationTask],
        scoring_policy: ScoringPolicy | None = None,
        scoring_weights_sum: float | None = None,
    ) -> BenchmarkValidationResult:
        issues: list[ValidationIssue] = []
        counter = [0]
        scoring_policy_valid = True
        scoring_warnings: list[str] = []

        tasks_tuple = tuple(tasks)

        if not tasks_tuple:
            return BenchmarkValidationResult(
                ValidationStatus.INVALID,
                BenchmarkHealth(0.0, 0, 0, 0, 0, 0, 0),
                (_issue(counter, "SCOPE", IssueLevel.ERROR, None, "EMPTY_COLLECTION", "a benchmark must define at least one task"),),
                (),
                True,
                (),
                ("benchmark definition is empty",),
            )

        task_validations = []
        validated_count = 0
        task_ids: list[str] = []
        requirement_ids: dict[str, str] = {}
        behavior_ids: dict[str, str] = {}
        test_ids: dict[str, str] = {}
        criterion_sources: dict[str, list[str]] = {}

        for index, task in enumerate(tasks_tuple):
            validation = self.task_validator.validate(task)
            if not validation.valid:
                validated_count += 0
            else:
                validated_count += 1
            task_validations.append(
                {
                    "task_id": task.task_id,
                    "valid": validation.valid,
                    "error_count": len(validation.errors),
                    "warning_count": len(validation.warnings),
                }
            )
            if not validation.valid:
                for error in validation.errors:
                    issues.append(_issue(counter, "TASK", IssueLevel.ERROR, task.task_id, error.code, f"{error.code}: {error.message}"))
            for warning in validation.warnings:
                warning_message = warning.message if isinstance(warning.message, str) else f"{warning.code}: {warning.message}"
                issues.append(_issue(counter, "TASK", IssueLevel.WARNING, task.task_id, "VALIDATION_WARNING", warning_message))

            task_ids.append(task.task_id)

            no_test_criteria = any(criterion.criterion_type is SuccessCriterionType.TEST_PASS and not criterion.test_ids for criterion in task.success_criteria)
            if no_test_criteria:
                issues.append(_issue(counter, "SCOPE", IssueLevel.WARNING, task.task_id, "TEST_CRITERION_NO_TEST", "TEST_PASS criterion declares no test IDs; the criterion has no evidence source"))

            declared_ids: set[str] = set()
            for requirement in task.requirements:
                if requirement.requirement_id in requirement_ids:
                    issues.append(_issue(counter, "SCOPE", IssueLevel.ERROR, task.task_id, "DUPLICATE_REFERENCE", f"requirement ID {requirement.requirement_id} is declared in multiple tasks"))
                else:
                    requirement_ids[requirement.requirement_id] = task.task_id
                declared_ids.add(requirement.requirement_id)
            for behavior in task.expected_behaviors:
                if behavior.behavior_id in behavior_ids:
                    issues.append(_issue(counter, "SCOPE", IssueLevel.ERROR, task.task_id, "DUPLICATE_REFERENCE", f"behavior ID {behavior.behavior_id} is declared in multiple tasks"))
                else:
                    behavior_ids[behavior.behavior_id] = task.task_id
                declared_ids.add(behavior.behavior_id)
            for test in task.tests:
                if test.test_id in test_ids:
                    issues.append(_issue(counter, "SCOPE", IssueLevel.ERROR, task.task_id, "DUPLICATE_REFERENCE", f"test ID {test.test_id} is declared in multiple tasks"))
                else:
                    test_ids[test.test_id] = task.task_id
                declared_ids.add(test.test_id)
                criterion_sources.setdefault(test.test_id, []).append(task.task_id)
            for criterion in task.success_criteria:
                missing_tests = [test_id for test_id in criterion.test_ids if test_id not in test_ids and test_id not in declared_ids]
                missing_behaviors = [behavior_id for behavior_id in criterion.behavior_ids if behavior_id not in behavior_ids and behavior_id not in declared_ids]
                if missing_tests:
                    issues.append(_issue(counter, "SCOPE", IssueLevel.WARNING, task.task_id, "UNRESOLVED_REFERENCE", f"criterion {criterion.criterion_id} references undeclared tests {missing_tests}"))
                if missing_behaviors:
                    issues.append(_issue(counter, "SCOPE", IssueLevel.WARNING, task.task_id, "UNRESOLVED_REFERENCE", f"criterion {criterion.criterion_id} references undeclared behaviors {missing_behaviors}"))

            no_meaningful = all(criterion.criterion_type is SuccessCriterionType.TEST_PASS and not criterion.test_ids for criterion in task.success_criteria) and not task.tests
            if no_meaningful:
                issues.append(_issue(counter, "FAIRNESS", IssueLevel.WARNING, task.task_id, "NO_EVALUABLE_CRITERION", "task declares no tests and no criterion with an evidence source; it cannot be evaluated"))

        if len(task_ids) != len(set(task_ids)):
            duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
            for duplicate in duplicates:
                issues.append(_issue(counter, "SCOPE", IssueLevel.ERROR, duplicate, "DUPLICATE_TASK_ID", f"task ID {duplicate} appears more than once in the benchmark"))

        category_counts: dict[str, int] = {}
        for task in tasks_tuple:
            category_counts[task.category] = category_counts.get(task.category, 0) + 1
        total = len(tasks_tuple)
        for category, count in category_counts.items():
            if total >= 6 and count == total:
                issues.append(_issue(counter, "FAIRNESS", IssueLevel.WARNING, None, "CATEGORY_DOMINANCE", f"all {total} tasks belong to category {category}; the benchmark cannot measure breadth"))

        test_counts = [len(task.tests) for task in tasks_tuple]
        if test_counts and max(test_counts) > 0 and total >= 2:
            dominant = [task for task in tasks_tuple if len(task.tests) == max(test_counts)]
            if len(dominant) == 1 and max(test_counts) >= 2 * (sum(test_counts) - max(test_counts)):
                issues.append(_issue(counter, "FAIRNESS", IssueLevel.WARNING, dominant[0].task_id, "ONE_TASK_DOMINATES", "one task dominates the test weight; benchmark scores are skewed"))

        required_tasks = [task for task in tasks_tuple if any(requirement.mandatory for requirement in task.requirements)]
        criteria_without_requirement = [
            task for task in tasks_tuple
            if any(criterion.required and criterion.criterion_type is SuccessCriterionType.BEHAVIOR for criterion in task.success_criteria) and not task.requirements
        ]
        if criteria_without_requirement:
            for task in criteria_without_requirement:
                issues.append(_issue(counter, "SCOPE", IssueLevel.WARNING, task.task_id, "CONSTRAINT_GAP", "task requires passing criteria without declaring any requirements"))

        if scoring_policy is not None:
            if not isinstance(scoring_policy, ScoringPolicy):
                raise TypeError("scoring_policy must be a ScoringPolicy or None")
            if scoring_policy.scoring_policy_version and not scoring_policy.scoring_policy_version.replace(".", "").isdigit():
                scoring_policy_valid = False
                issues.append(_issue(counter, "SCORING", IssueLevel.ERROR, None, "INVALID_POLICY_VERSION", f"scoring policy version {scoring_policy.scoring_policy_version} is not a valid numeric version"))
            if scoring_policy.duration_target_seconds is not None and scoring_policy.duration_target_seconds <= 0:
                scoring_policy_valid = False
                issues.append(_issue(counter, "SCORING", IssueLevel.ERROR, None, "INVALID_TARGET", "duration target must be positive"))
            if scoring_policy.iteration_target is not None and scoring_policy.iteration_target <= 0:
                scoring_policy_valid = False
                issues.append(_issue(counter, "SCORING", IssueLevel.ERROR, None, "INVALID_TARGET", "iteration target must be positive"))
        if scoring_weights_sum is not None:
            if not math.isfinite(scoring_weights_sum):
                scoring_policy_valid = False
                issues.append(_issue(counter, "SCORING", IssueLevel.ERROR, None, "INVALID_WEIGHT_SUM", "scoring weights must sum to a finite number"))
            elif abs(scoring_weights_sum - 1.0) > 1e-9:
                scoring_policy_valid = False
                issues.append(_issue(counter, "SCORING", IssueLevel.ERROR, None, "WEIGHTS_DO_NOT_SUM_TO_ONE", f"scoring weights sum to {scoring_weights_sum}, but they must sum to exactly 1.0"))

        error_count = sum(1 for item in issues if item.level is IssueLevel.ERROR)
        warning_count = sum(1 for item in issues if item.level is IssueLevel.WARNING)
        info_count = sum(1 for item in issues if item.level is IssueLevel.INFO)

        if total > 0:
            penalty = min(1.0, (error_count * 0.25) + (warning_count * 0.05))
            health_score = max(0.0, 1.0 - penalty)
            structure_bonus = min(0.05, total * 0.01)
            health_score = min(1.0, health_score + structure_bonus if error_count == 0 else health_score)
        else:
            health_score = 0.0

        health = BenchmarkHealth(
            float(f"{health_score:.4f}"),
            total,
            validated_count,
            len(issues),
            error_count,
            warning_count,
            info_count,
        )

        if error_count > 0:
            status = ValidationStatus.INVALID
        elif warning_count > 0:
            status = ValidationStatus.WARNING
        elif not issues:
            status = ValidationStatus.VALID
        else:
            status = ValidationStatus.INCONCLUSIVE

        warnings: list[str] = []
        if validated_count < total:
            warnings.append(f"{total - validated_count} task(s) failed structural validation")
        if warning_count > 0:
            warnings.append("benchmark definition contains warnings that may affect fairness or reliability")

        return BenchmarkValidationResult(status, health, tuple(issues), tuple(task_validations), scoring_policy_valid, tuple(scoring_warnings), tuple(warnings))


def validate_benchmark(
    tasks: tuple[EvaluationTask, ...] | list[EvaluationTask],
    scoring_policy: ScoringPolicy | None = None,
    scoring_weights_sum: float | None = None,
) -> BenchmarkValidationResult:
    """Public entry point for Phase 8.9 benchmark validation."""

    return BenchmarkValidator().validate(tasks, scoring_policy, scoring_weights_sum)


__all__ = [
    "BenchmarkHealth",
    "BenchmarkValidationResult",
    "BenchmarkValidator",
    "IssueLevel",
    "ValidationIssue",
    "ValidationStatus",
    "validate_benchmark",
]
