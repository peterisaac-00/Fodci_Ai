"""Deterministic Filtering & Quality Gates for canonical DatasetRecord values.

Phase 10.3 evaluates structural dataset quality only.  It never mutates source
records, persists filtered output, calls an LLM, performs semantic search, or
changes training/model state.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from backend_ai.agent.dataset_schema import (
    DatasetOutcome,
    DatasetRecord,
    DatasetRecordLimits,
    DatasetRecordProvenance,
    DatasetSchemaError,
    DatasetSchemaValidationResult,
)
from backend_ai.agent.experience_dataset import _canonical_json, _contains_prohibited_secret


class DatasetQualityError(ValueError):
    """Invalid quality policy or filtering input."""


class QualityDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


class QualityCheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class DatasetQualityPolicy:
    """Inspectable named thresholds and decisions for deterministic filtering."""

    minimum_quality_score: float = 0.75
    minimum_task_length: int = 4
    maximum_task_length: int = 1_024
    minimum_verification_strength: float = 0.50
    maximum_noise_ratio: float = 0.50
    maximum_repeated_event_count: int = 3
    minimum_successful_trajectory_events: int = 1
    accepted_outcomes: tuple[DatasetOutcome, ...] = (DatasetOutcome.SUCCESS,)
    failed_outcome_decision: QualityDecision = QualityDecision.REJECT
    cancelled_outcome_decision: QualityDecision = QualityDecision.REVIEW
    duplicate_decision: QualityDecision = QualityDecision.REJECT
    irrelevant_decision: QualityDecision = QualityDecision.REVIEW
    schema_limits: DatasetRecordLimits = field(default_factory=DatasetRecordLimits)
    domain_terms: tuple[str, ...] = (
        "api", "authentication", "authorization", "backend", "database", "debug", "docker", "django",
        "deployment", "express", "fastapi", "flask", "git", "http", "javascript", "migration", "mysql",
        "node", "orm", "performance", "postgres", "postgresql", "python", "redis", "rest", "scalability",
        "security", "service", "sql", "sqlite", "testing", "typescript", "validation", "webhook",
    )
    irrelevant_terms: tuple[str, ...] = ("poem", "recipe", "logo", "marketing slogan", "wedding speech", "song lyrics")
    placeholder_terms: tuple[str, ...] = ("test", "hello", "hi", "fix", "asdf", "...", "todo", "placeholder")

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_quality_score <= 1.0:
            raise DatasetQualityError("minimum_quality_score must be within [0, 1]")
        if not 0.0 <= self.minimum_verification_strength <= 1.0:
            raise DatasetQualityError("minimum_verification_strength must be within [0, 1]")
        if not 0.0 <= self.maximum_noise_ratio <= 1.0:
            raise DatasetQualityError("maximum_noise_ratio must be within [0, 1]")
        for name in ("minimum_task_length", "maximum_task_length", "maximum_repeated_event_count", "minimum_successful_trajectory_events"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise DatasetQualityError(f"{name} must be a positive integer")
        if self.minimum_task_length > self.maximum_task_length:
            raise DatasetQualityError("minimum_task_length cannot exceed maximum_task_length")
        if not isinstance(self.schema_limits, DatasetRecordLimits):
            raise DatasetQualityError("schema_limits must be DatasetRecordLimits")
        if not self.accepted_outcomes or any(not isinstance(item, DatasetOutcome) for item in self.accepted_outcomes):
            raise DatasetQualityError("accepted_outcomes must contain DatasetOutcome values")
        for name in ("domain_terms", "irrelevant_terms", "placeholder_terms"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, str) or not item.strip() for item in values):
                raise DatasetQualityError(f"{name} must be a tuple of non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_quality_score": self.minimum_quality_score,
            "minimum_task_length": self.minimum_task_length,
            "maximum_task_length": self.maximum_task_length,
            "minimum_verification_strength": self.minimum_verification_strength,
            "maximum_noise_ratio": self.maximum_noise_ratio,
            "maximum_repeated_event_count": self.maximum_repeated_event_count,
            "minimum_successful_trajectory_events": self.minimum_successful_trajectory_events,
            "accepted_outcomes": [item.value for item in self.accepted_outcomes],
            "failed_outcome_decision": self.failed_outcome_decision.value,
            "cancelled_outcome_decision": self.cancelled_outcome_decision.value,
            "duplicate_decision": self.duplicate_decision.value,
            "irrelevant_decision": self.irrelevant_decision.value,
            "domain_terms": list(self.domain_terms),
            "irrelevant_terms": list(self.irrelevant_terms),
            "placeholder_terms": list(self.placeholder_terms),
        }


@dataclass(frozen=True, slots=True)
class QualityCheck:
    """One named, bounded and explainable quality check."""

    check_id: str
    status: QualityCheckStatus
    score: float | None
    reason: str
    hard_gate: bool

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str) or not self.check_id.strip():
            raise DatasetQualityError("check_id must contain text")
        if not isinstance(self.status, QualityCheckStatus):
            raise DatasetQualityError("check status must be QualityCheckStatus")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise DatasetQualityError("check score must be within [0, 1]")
        if not isinstance(self.reason, str) or not self.reason.strip() or len(self.reason) > 512:
            raise DatasetQualityError("check reason must be bounded text")
        if not isinstance(self.hard_gate, bool):
            raise DatasetQualityError("hard_gate must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"check_id": self.check_id, "status": self.status.value, "score": self.score, "reason": self.reason, "hard_gate": self.hard_gate}


@dataclass(frozen=True, slots=True)
class QualityScore:
    """Explicit weighted score: task .20, solution .20, verification .25, trajectory .15, relevance .10, consistency .10."""

    completeness_score: float
    verification_score: float
    trajectory_score: float
    relevance_score: float
    consistency_score: float
    task_score: float
    final_score: float

    def __post_init__(self) -> None:
        for name in ("completeness_score", "verification_score", "trajectory_score", "relevance_score", "consistency_score", "task_score", "final_score"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise DatasetQualityError(f"{name} must be within [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return {"completeness_score": self.completeness_score, "verification_score": self.verification_score, "trajectory_score": self.trajectory_score, "relevance_score": self.relevance_score, "consistency_score": self.consistency_score, "task_score": self.task_score, "final_score": self.final_score}


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Decision and diagnostics for one canonical DatasetRecord."""

    record_id: str
    experience_id: str
    decision: QualityDecision
    score: QualityScore
    checks: tuple[QualityCheck, ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    provenance: DatasetRecordProvenance | None
    duplicate_of: str | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.record_id, "record_id"), (self.experience_id, "experience_id")):
            if not isinstance(value, str) or not value.strip():
                raise DatasetQualityError(f"{name} must contain text")
        if not isinstance(self.decision, QualityDecision):
            raise DatasetQualityError("decision must be QualityDecision")
        if not isinstance(self.score, QualityScore):
            raise DatasetQualityError("score must be QualityScore")
        if not isinstance(self.checks, tuple) or any(not isinstance(item, QualityCheck) for item in self.checks):
            raise DatasetQualityError("checks must be a tuple of QualityCheck")
        if any(not isinstance(item, str) or not item.strip() for item in self.reasons + self.warnings):
            raise DatasetQualityError("reasons and warnings must contain text")
        if self.duplicate_of is not None and (not isinstance(self.duplicate_of, str) or not self.duplicate_of.strip()):
            raise DatasetQualityError("duplicate_of must be text or None")
        if self.provenance is not None and not isinstance(self.provenance, DatasetRecordProvenance):
            raise DatasetQualityError("provenance must be DatasetRecordProvenance or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "experience_id": self.experience_id,
            "decision": self.decision.value,
            "score": self.score.to_dict(),
            "checks": [item.to_dict() for item in self.checks],
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "duplicate_of": self.duplicate_of,
        }


@dataclass(frozen=True, slots=True)
class DatasetFilteringResult:
    """Read-only batch output preserving accepted records and all decisions."""

    accepted: tuple[DatasetRecord, ...]
    rejected: tuple[QualityAssessment, ...]
    review: tuple[QualityAssessment, ...]
    assessments: tuple[QualityAssessment, ...]
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("accepted", "rejected", "review", "assessments"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise DatasetQualityError(f"{name} must be a tuple")
        if any(not isinstance(item, DatasetRecord) for item in self.accepted):
            raise DatasetQualityError("accepted must contain DatasetRecord")
        if any(not isinstance(item, QualityAssessment) for item in self.rejected + self.review + self.assessments):
            raise DatasetQualityError("decision collections must contain QualityAssessment")
        if any(not isinstance(item, str) or not item.strip() for item in self.diagnostics):
            raise DatasetQualityError("diagnostics must contain text")

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def review_count(self) -> int:
        return len(self.review)

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": [item.to_dict() for item in self.accepted], "rejected": [item.to_dict() for item in self.rejected], "review": [item.to_dict() for item in self.review], "assessments": [item.to_dict() for item in self.assessments], "diagnostics": list(self.diagnostics), "counts": {"accepted": self.accepted_count, "rejected": self.rejected_count, "review": self.review_count, "total": len(self.assessments)}}


class DatasetQualityEvaluator:
    """Deterministic quality evaluator for canonical DatasetRecord objects."""

    def __init__(self, *, policy: DatasetQualityPolicy | None = None) -> None:
        self.policy = policy or DatasetQualityPolicy()

    def evaluate(self, record: DatasetRecord | Mapping[str, Any]) -> QualityAssessment:
        try:
            canonical = record if isinstance(record, DatasetRecord) else DatasetRecord.from_dict(record, limits=self.policy.schema_limits)
            if not isinstance(canonical, DatasetRecord):
                raise DatasetQualityError("input is not DatasetRecord")
        except (DatasetSchemaError, DatasetQualityError, TypeError, ValueError) as exc:
            return self._hard_reject_invalid(exc, record)
        return self._evaluate_valid(canonical)

    def filter(self, record: DatasetRecord | Mapping[str, Any]) -> QualityAssessment:
        return self.evaluate(record)

    def filter_many(self, records: Sequence[DatasetRecord | Mapping[str, Any]]) -> DatasetFilteringResult:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise DatasetQualityError("records must be a bounded sequence")
        assessments: list[QualityAssessment] = []
        accepted: list[DatasetRecord] = []
        rejected: list[QualityAssessment] = []
        review: list[QualityAssessment] = []
        diagnostics: list[str] = []
        seen: dict[str, str] = {}
        for raw in records:
            assessment = self.evaluate(raw)
            if assessment.decision is not QualityDecision.REJECT and isinstance(raw, DatasetRecord):
                fingerprint = _duplicate_fingerprint(raw)
                previous = seen.get(fingerprint)
                if previous is not None:
                    assessment = self._with_duplicate(assessment, previous)
                else:
                    seen[fingerprint] = assessment.record_id
            assessments.append(assessment)
            if assessment.decision is QualityDecision.ACCEPT and isinstance(raw, DatasetRecord):
                accepted.append(raw)
            elif assessment.decision is QualityDecision.REJECT:
                rejected.append(assessment)
            else:
                review.append(assessment)
            diagnostics.extend(assessment.reasons)
            diagnostics.extend(assessment.warnings)
        return DatasetFilteringResult(tuple(accepted), tuple(rejected), tuple(review), tuple(assessments), tuple(dict.fromkeys(diagnostics)))

    def _evaluate_valid(self, record: DatasetRecord) -> QualityAssessment:
        checks: list[QualityCheck] = []
        reasons: list[str] = []
        warnings: list[str] = []
        hard_failure = False

        security_check = self._security_check(record)
        checks.append(security_check)
        if security_check.status is QualityCheckStatus.FAIL:
            hard_failure = True
            reasons.append(security_check.reason)

        consistency_check, consistency_score = self._consistency_check(record)
        checks.append(consistency_check)
        if consistency_check.status is QualityCheckStatus.FAIL:
            hard_failure = True
            reasons.append(consistency_check.reason)

        task_check, task_score, relevance_check, relevance_score = self._task_and_relevance(record)
        checks.extend((task_check, relevance_check))
        if task_check.status is QualityCheckStatus.FAIL:
            reasons.append(task_check.reason)
        elif task_check.status is QualityCheckStatus.WARN:
            warnings.append(task_check.reason)
        relevance_policy_reject = relevance_check.reason == "relevance_uncertain" and self.policy.irrelevant_decision is QualityDecision.REJECT
        if relevance_check.status is QualityCheckStatus.FAIL:
            reasons.append(relevance_check.reason)
        elif relevance_check.status is QualityCheckStatus.WARN:
            warnings.append(relevance_check.reason)

        solution_check, completeness_score = self._solution_check(record)
        checks.append(solution_check)
        if solution_check.status is QualityCheckStatus.FAIL:
            reasons.append(solution_check.reason)
        elif solution_check.status is QualityCheckStatus.WARN:
            warnings.append(solution_check.reason)

        verification_check, verification_score = self._verification_check(record)
        checks.append(verification_check)
        if verification_check.status is QualityCheckStatus.FAIL:
            reasons.append(verification_check.reason)
        elif verification_check.status is QualityCheckStatus.WARN:
            warnings.append(verification_check.reason)

        trajectory_check, trajectory_score = self._trajectory_check(record)
        checks.append(trajectory_check)
        if trajectory_check.status is QualityCheckStatus.FAIL:
            reasons.append(trajectory_check.reason)
        elif trajectory_check.status is QualityCheckStatus.WARN:
            warnings.append(trajectory_check.reason)

        noise_check = self._noise_check(record)
        checks.append(noise_check)
        if noise_check.status is QualityCheckStatus.FAIL:
            reasons.append(noise_check.reason)
        elif noise_check.status is QualityCheckStatus.WARN:
            warnings.append(noise_check.reason)

        outcome_check = self._outcome_check(record)
        checks.append(outcome_check)
        if outcome_check.status is QualityCheckStatus.FAIL:
            reasons.append(outcome_check.reason)
        elif outcome_check.status is QualityCheckStatus.WARN:
            warnings.append(outcome_check.reason)

        quality_score = QualityScore(
            completeness_score=completeness_score,
            verification_score=verification_score,
            trajectory_score=trajectory_score,
            relevance_score=relevance_score,
            consistency_score=consistency_score,
            task_score=task_score,
            final_score=round(0.20 * task_score + 0.20 * completeness_score + 0.25 * verification_score + 0.15 * trajectory_score + 0.10 * relevance_score + 0.10 * consistency_score, 6),
        )
        if hard_failure or any(item.status is QualityCheckStatus.FAIL and item.hard_gate for item in checks):
            decision = QualityDecision.REJECT
        elif outcome_check.status is QualityCheckStatus.FAIL:
            decision = self.policy.failed_outcome_decision if record.outcome is DatasetOutcome.FAILURE else QualityDecision.REJECT
        elif record.outcome is DatasetOutcome.CANCELLED and outcome_check.status is QualityCheckStatus.WARN:
            decision = self.policy.cancelled_outcome_decision
        elif relevance_policy_reject:
            decision = self.policy.irrelevant_decision
        elif any(item.status is QualityCheckStatus.FAIL for item in checks):
            decision = QualityDecision.REJECT
        elif quality_score.final_score < self.policy.minimum_quality_score or any(item.status is QualityCheckStatus.WARN for item in checks):
            decision = QualityDecision.REVIEW
        else:
            decision = QualityDecision.ACCEPT
        if decision is QualityDecision.REJECT and not reasons:
            reasons.append("quality policy hard gate failed")
        if decision is QualityDecision.REVIEW and not warnings:
            warnings.append("record requires review before dataset acceptance")
        return QualityAssessment(record.record_id, record.experience_id, decision, quality_score, tuple(checks), tuple(dict.fromkeys(reasons)), tuple(dict.fromkeys(warnings)), record.provenance)

    def _hard_reject_invalid(self, exc: Exception, raw: Any) -> QualityAssessment:
        record_id = str(raw.get("record_id", "invalid")) if isinstance(raw, Mapping) else "invalid"
        experience_id = str(raw.get("experience_id", "invalid")) if isinstance(raw, Mapping) else "invalid"
        reason = _safe_reason(exc)
        score = QualityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        check = QualityCheck("schema", QualityCheckStatus.FAIL, 0.0, "schema_invalid", True)
        return QualityAssessment(record_id, experience_id, QualityDecision.REJECT, score, (check,), (reason,), (), None)

    def _security_check(self, record: DatasetRecord) -> QualityCheck:
        if _contains_prohibited_secret(_canonical_json(record.to_dict())):
            return QualityCheck("security", QualityCheckStatus.FAIL, 0.0, "security_violation", True)
        return QualityCheck("security", QualityCheckStatus.PASS, 1.0, "security_pass", True)

    def _consistency_check(self, record: DatasetRecord) -> tuple[QualityCheck, float]:
        problems: list[str] = []
        attempts = {str(item.get("attempt_id")) for item in record.trajectory.attempts}
        error_ids = {str(item.get("error_id")) for item in record.trajectory.errors}
        for event in record.trajectory.actions + record.trajectory.observations + record.trajectory.errors + record.trajectory.corrections:
            if str(event.get("attempt_id")) not in attempts:
                problems.append("trajectory_attempt_reference_missing")
        for correction in record.trajectory.corrections:
            error_id = correction.get("error_id")
            if error_id is not None and str(error_id) not in error_ids:
                problems.append("correction_error_reference_missing")
        if record.outcome is DatasetOutcome.SUCCESS:
            if not any(value for value in (record.solution.solution, record.solution.final_result, record.solution.final_summary)):
                problems.append("success_solution_missing")
            if record.verification.present and (record.verification.tests_failed or (record.verification.test_status and record.verification.test_status.casefold() in {"fail", "failed", "error"})):
                problems.append("success_verification_failed")
        if problems:
            return QualityCheck("consistency", QualityCheckStatus.FAIL, 0.0, problems[0], True), 0.0
        return QualityCheck("consistency", QualityCheckStatus.PASS, 1.0, "consistency_pass", True), 1.0

    def _task_and_relevance(self, record: DatasetRecord) -> tuple[QualityCheck, float, QualityCheck, float]:
        text = record.task.casefold().strip()
        tokens = set(re.findall(r"[\w+#.-]+", text, flags=re.UNICODE))
        if not text:
            task = QualityCheck("task_quality", QualityCheckStatus.FAIL, 0.0, "task_missing", True)
            task_score = 0.0
        elif text in {item.casefold() for item in self.policy.placeholder_terms}:
            task = QualityCheck("task_quality", QualityCheckStatus.WARN, 0.0, "task_placeholder", False)
            task_score = 0.0
        elif len(text) < self.policy.minimum_task_length:
            task = QualityCheck("task_quality", QualityCheckStatus.WARN, 0.4, "task_short_review", False)
            task_score = 0.4
        else:
            task = QualityCheck("task_quality", QualityCheckStatus.PASS, 1.0, "task_meaningful", False)
            task_score = 1.0
        domain_hits = sum(1 for term in self.policy.domain_terms if term.casefold() in tokens or term.casefold() in text)
        irrelevant_hits = sum(1 for term in self.policy.irrelevant_terms if term.casefold() in text)
        if irrelevant_hits:
            relevance = QualityCheck("relevance", QualityCheckStatus.WARN, 0.25, "relevance_uncertain", False)
            relevance_score = 0.25
        elif domain_hits > 0:
            relevance = QualityCheck("relevance", QualityCheckStatus.PASS, 1.0, "backend_relevance_detected", False)
            relevance_score = 1.0
        else:
            relevance = QualityCheck("relevance", QualityCheckStatus.WARN, 0.5, "relevance_uncertain", False)
            relevance_score = 0.5
        return task, task_score, relevance, relevance_score

    def _solution_check(self, record: DatasetRecord) -> tuple[QualityCheck, float]:
        values = (record.solution.solution, record.solution.final_result, record.solution.final_summary)
        present = sum(bool(value and value.strip()) for value in values)
        placeholders = {"none", "unknown", "n/a", "todo", "not implemented", "placeholder"}
        if record.outcome is DatasetOutcome.SUCCESS and present == 0:
            return QualityCheck("solution_completeness", QualityCheckStatus.FAIL, 0.0, "solution_missing", True), 0.0
        if any(value and value.casefold().strip() in placeholders for value in values):
            return QualityCheck("solution_completeness", QualityCheckStatus.WARN, 0.35, "solution_placeholder", False), 0.35
        if record.outcome is DatasetOutcome.SUCCESS and present == 1:
            return QualityCheck("solution_completeness", QualityCheckStatus.WARN, 0.65, "solution_partially_recorded", False), 0.65
        if present == 0:
            return QualityCheck("solution_completeness", QualityCheckStatus.NOT_APPLICABLE, 0.5, "solution_not_expected_for_outcome", False), 0.5
        return QualityCheck("solution_completeness", QualityCheckStatus.PASS, 1.0, "solution_recorded", False), 1.0

    def _verification_check(self, record: DatasetRecord) -> tuple[QualityCheck, float]:
        verification = record.verification
        if not verification.present:
            if record.outcome is DatasetOutcome.SUCCESS:
                return QualityCheck("verification", QualityCheckStatus.WARN, 0.25, "verification_missing", False), 0.25
            return QualityCheck("verification", QualityCheckStatus.NOT_APPLICABLE, 0.5, "verification_not_present", False), 0.5
        if verification.tests_failed and verification.tests_failed > 0:
            return QualityCheck("verification", QualityCheckStatus.FAIL, 0.0, "verification_failed_tests", True), 0.0
        status = (verification.test_status or "").casefold()
        if status in {"fail", "failed", "error"}:
            return QualityCheck("verification", QualityCheckStatus.FAIL, 0.0, "verification_status_failed", True), 0.0
        if verification.tests_executed and verification.tests_executed > 0 and verification.tests_passed == verification.tests_executed and status in {"pass", "passed", "success", "ok"}:
            return QualityCheck("verification", QualityCheckStatus.PASS, 1.0, "verification_strong", False), 1.0
        if verification.summary and verification.test_status:
            return QualityCheck("verification", QualityCheckStatus.WARN, 0.6, "verification_partial", False), 0.6
        return QualityCheck("verification", QualityCheckStatus.WARN, 0.4, "verification_weak", False), 0.4

    def _trajectory_check(self, record: DatasetRecord) -> tuple[QualityCheck, float]:
        event_count = sum(len(getattr(record.trajectory, name)) for name in ("actions", "observations", "errors", "corrections"))
        if record.outcome is DatasetOutcome.SUCCESS and event_count < self.policy.minimum_successful_trajectory_events:
            return QualityCheck("trajectory", QualityCheckStatus.WARN, 0.35, "trajectory_sparse", False), 0.35
        if event_count == 0:
            return QualityCheck("trajectory", QualityCheckStatus.WARN, 0.4, "trajectory_empty", False), 0.4
        if record.trajectory.errors and record.trajectory.corrections:
            return QualityCheck("trajectory", QualityCheckStatus.PASS, 1.0, "trajectory_contains_recovery_evidence", False), 1.0
        return QualityCheck("trajectory", QualityCheckStatus.PASS, 0.85, "trajectory_structured", False), 0.85

    def _noise_check(self, record: DatasetRecord) -> QualityCheck:
        values = []
        for event in record.trajectory.actions + record.trajectory.observations + record.trajectory.errors + record.trajectory.corrections:
            for key in ("summary", "name", "category", "outcome"):
                if event.get(key):
                    values.append(str(event[key]).casefold().strip())
                    break
        if not values:
            return QualityCheck("noise", QualityCheckStatus.NOT_APPLICABLE, 0.75, "noise_not_observed", False)
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        repeated = max(counts.values())
        duplicate_ratio = (len(values) - len(counts)) / len(values)
        if repeated > self.policy.maximum_repeated_event_count or duplicate_ratio > self.policy.maximum_noise_ratio:
            return QualityCheck("noise", QualityCheckStatus.WARN, max(0.0, 1.0 - duplicate_ratio), "trajectory_repetition_review", False)
        return QualityCheck("noise", QualityCheckStatus.PASS, 1.0, "noise_within_bound", False)

    def _outcome_check(self, record: DatasetRecord) -> QualityCheck:
        if record.outcome is DatasetOutcome.FAILURE:
            return QualityCheck("outcome", QualityCheckStatus.FAIL, 0.0, "failed_outcome_not_high_quality", True)
        if record.outcome is DatasetOutcome.CANCELLED:
            return QualityCheck("outcome", QualityCheckStatus.WARN, 0.35, "cancelled_outcome_review", False)
        return QualityCheck("outcome", QualityCheckStatus.PASS, 1.0, "successful_outcome", False)

    def _with_duplicate(self, assessment: QualityAssessment, duplicate_of: str) -> QualityAssessment:
        checks = assessment.checks + (QualityCheck("duplicate", QualityCheckStatus.FAIL, 0.0, "duplicate_record", False),)
        return QualityAssessment(assessment.record_id, assessment.experience_id, self.policy.duplicate_decision, assessment.score, checks, assessment.reasons + (f"duplicate_of:{duplicate_of}",), assessment.warnings, assessment.provenance, duplicate_of)


def _duplicate_fingerprint(record: DatasetRecord) -> str:
    payload = record.to_dict()
    payload.pop("record_id", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _safe_reason(exc: Exception) -> str:
    text = str(exc).strip() or "invalid_dataset_record"
    text = re.sub(r"(?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie|database_url)\s*(?:=|:)\s*[^,\s}\]]+", "[REDACTED]", text, flags=re.IGNORECASE)
    return text[:512]


__all__ = [
    "DatasetFilteringResult",
    "DatasetQualityError",
    "DatasetQualityEvaluator",
    "DatasetQualityPolicy",
    "QualityAssessment",
    "QualityCheck",
    "QualityCheckStatus",
    "QualityDecision",
    "QualityScore",
]
