"""Bounded historical Experience Records for actual Agent executions.

Experience Records are passive historical data.  They are intentionally
separate from Short-Term Memory, Project Memory, Long-Term Memory, training,
and execution permissions.  Persistence is explicit through ExperienceRecordStore.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

from backend_ai.agent.short_term_memory import _redact_text, _redact_value


EXPERIENCE_RECORD_SCHEMA_VERSION = "9.4"
EXPERIENCE_RECORD_FORMAT = "fodci.experience_records"
_DEFAULT_DIRECTORY = ".fodci"
_DEFAULT_FILENAME = "experience_records.json"
_MAX_COLLECTION_ITEMS = 64


class ExperienceRecordError(RuntimeError):
    """Base error for controlled Experience Record operations."""


class ExperienceRecordValidationError(ExperienceRecordError, ValueError):
    """Invalid record data or a configured resource limit."""


class ExperienceRecordClosedError(ExperienceRecordError):
    """A write was attempted after an experience was finalized."""


class ExperienceRecordConflictError(ExperienceRecordError):
    """The persisted experience collection changed after it was loaded."""


class ExperienceRecordLoadStatus(str, Enum):
    LOADED = "LOADED"
    MEMORY_MISSING = "MEMORY_MISSING"
    MEMORY_CORRUPTED = "MEMORY_CORRUPTED"
    MEMORY_INVALID = "MEMORY_INVALID"
    MEMORY_UNAVAILABLE = "MEMORY_UNAVAILABLE"


class ExperienceLifecycleStatus(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperienceOutcomeStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"


class ExperienceAttemptStatus(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperienceErrorStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class ExperienceRecordLimits:
    max_experiences: int = 128
    max_attempts_per_experience: int = 8
    max_actions_per_attempt: int = 64
    max_observations_per_attempt: int = 64
    max_errors_per_attempt: int = 32
    max_corrections_per_attempt: int = 32
    max_content_length: int = 1_024
    max_metadata_size: int = 4_096
    max_serialized_record_bytes: int = 262_144
    max_total_storage_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        names = (
            "max_experiences", "max_attempts_per_experience", "max_actions_per_attempt",
            "max_observations_per_attempt", "max_errors_per_attempt", "max_corrections_per_attempt",
            "max_content_length", "max_metadata_size", "max_serialized_record_bytes", "max_total_storage_bytes",
        )
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ExperienceRecordValidationError(f"{name} must be a positive integer")
        ceilings = {
            "max_experiences": 1_024,
            "max_attempts_per_experience": 64,
            "max_actions_per_attempt": 512,
            "max_observations_per_attempt": 512,
            "max_errors_per_attempt": 256,
            "max_corrections_per_attempt": 256,
            "max_content_length": 65_536,
            "max_metadata_size": 65_536,
            "max_serialized_record_bytes": 8 * 1024 * 1024,
            "max_total_storage_bytes": 32 * 1024 * 1024,
        }
        for name, ceiling in ceilings.items():
            if getattr(self, name) > ceiling:
                raise ExperienceRecordValidationError(f"{name} exceeds the safety ceiling")

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in (
            "max_experiences", "max_attempts_per_experience", "max_actions_per_attempt",
            "max_observations_per_attempt", "max_errors_per_attempt", "max_corrections_per_attempt",
            "max_content_length", "max_metadata_size", "max_serialized_record_bytes", "max_total_storage_bytes",
        )}


@dataclass(frozen=True, slots=True)
class ExperienceProjectIdentity:
    project_id: str
    project_root: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, str) or not self.project_id.strip():
            raise ExperienceRecordValidationError("project_id must contain text")
        if self.project_root is not None and (not isinstance(self.project_root, str) or not self.project_root.strip()):
            raise ExperienceRecordValidationError("project_root must contain text when provided")

    def to_dict(self) -> dict[str, str | None]:
        return {"project_id": self.project_id, "project_root": self.project_root}


@dataclass(frozen=True, slots=True)
class ExperienceAction:
    action_id: str
    attempt_id: str
    name: str
    summary: str
    timestamp: str
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id(self.action_id, "action_id")
        _validate_id(self.attempt_id, "attempt_id")
        _validate_text(self.name, "action name")
        _validate_text(self.summary, "action summary")
        _validate_text(self.timestamp, "action timestamp")
        _validate_text(self.status, "action status")
        object.__setattr__(self, "metadata", _freeze_value(_redact_experience_value(dict(self.metadata))))

    def to_dict(self) -> dict[str, Any]:
        return {"action_id": self.action_id, "attempt_id": self.attempt_id, "name": self.name, "summary": self.summary, "timestamp": self.timestamp, "status": self.status, "metadata": _thaw_value(self.metadata)}


@dataclass(frozen=True, slots=True)
class ExperienceObservation:
    observation_id: str
    attempt_id: str
    summary: str
    source: str
    timestamp: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id(self.observation_id, "observation_id")
        _validate_id(self.attempt_id, "attempt_id")
        _validate_text(self.summary, "observation summary")
        _validate_text(self.source, "observation source")
        _validate_text(self.timestamp, "observation timestamp")
        object.__setattr__(self, "metadata", _freeze_value(_redact_experience_value(dict(self.metadata))))

    def to_dict(self) -> dict[str, Any]:
        return {"observation_id": self.observation_id, "attempt_id": self.attempt_id, "summary": self.summary, "source": self.source, "timestamp": self.timestamp, "metadata": _thaw_value(self.metadata)}


@dataclass(frozen=True, slots=True)
class ExperienceError:
    error_id: str
    attempt_id: str
    category: str
    summary: str
    source: str
    timestamp: str
    status: ExperienceErrorStatus = ExperienceErrorStatus.UNRESOLVED

    def __post_init__(self) -> None:
        _validate_id(self.error_id, "error_id")
        _validate_id(self.attempt_id, "attempt_id")
        for value, name in ((self.category, "error category"), (self.summary, "error summary"), (self.source, "error source"), (self.timestamp, "error timestamp")):
            _validate_text(value, name)
        if not isinstance(self.status, ExperienceErrorStatus):
            raise ExperienceRecordValidationError("error status must be ExperienceErrorStatus")

    def to_dict(self) -> dict[str, Any]:
        return {"error_id": self.error_id, "attempt_id": self.attempt_id, "category": self.category, "summary": self.summary, "source": self.source, "timestamp": self.timestamp, "status": self.status.value}


@dataclass(frozen=True, slots=True)
class ExperienceCorrection:
    correction_id: str
    attempt_id: str
    error_id: str | None
    summary: str
    outcome: str
    timestamp: str

    def __post_init__(self) -> None:
        _validate_id(self.correction_id, "correction_id")
        _validate_id(self.attempt_id, "attempt_id")
        if self.error_id is not None:
            _validate_id(self.error_id, "error_id")
        _validate_text(self.summary, "correction summary")
        _validate_text(self.outcome, "correction outcome")
        _validate_text(self.timestamp, "correction timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {"correction_id": self.correction_id, "attempt_id": self.attempt_id, "error_id": self.error_id, "summary": self.summary, "outcome": self.outcome, "timestamp": self.timestamp}


@dataclass(frozen=True, slots=True)
class ExperienceVerification:
    tests_executed: int
    tests_passed: int
    tests_failed: int
    test_status: str
    summary: str
    timestamp: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in ((self.tests_executed, "tests_executed"), (self.tests_passed, "tests_passed"), (self.tests_failed, "tests_failed")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ExperienceRecordValidationError(f"{name} must be a non-negative integer")
        if self.tests_passed + self.tests_failed > self.tests_executed:
            raise ExperienceRecordValidationError("test counts exceed tests_executed")
        for value, name in ((self.test_status, "test status"), (self.summary, "verification summary"), (self.timestamp, "verification timestamp")):
            _validate_text(value, name)
        object.__setattr__(self, "metadata", _freeze_value(_redact_experience_value(dict(self.metadata))))

    def to_dict(self) -> dict[str, Any]:
        return {"tests_executed": self.tests_executed, "tests_passed": self.tests_passed, "tests_failed": self.tests_failed, "test_status": self.test_status, "summary": self.summary, "timestamp": self.timestamp, "metadata": _thaw_value(self.metadata)}


@dataclass(frozen=True, slots=True)
class ExperienceEvaluation:
    score: float | None
    status: str
    summary: str
    criteria: tuple[Mapping[str, Any], ...]
    evaluator_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.score is not None and (not isinstance(self.score, (int, float)) or isinstance(self.score, bool)):
            raise ExperienceRecordValidationError("evaluation score must be numeric or None")
        _validate_text(self.status, "evaluation status")
        _validate_text(self.summary, "evaluation summary")
        object.__setattr__(self, "criteria", tuple(_freeze_value(_redact_experience_value(dict(item))) for item in self.criteria))
        object.__setattr__(self, "evaluator_metadata", _freeze_value(_redact_experience_value(dict(self.evaluator_metadata))))

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "status": self.status, "summary": self.summary, "criteria": [_thaw_value(item) for item in self.criteria], "evaluator_metadata": _thaw_value(self.evaluator_metadata)}


@dataclass(frozen=True, slots=True)
class ExperienceAttempt:
    attempt_id: str
    started_at: str
    completed_at: str | None
    status: ExperienceAttemptStatus
    actions: tuple[ExperienceAction, ...]
    observations: tuple[ExperienceObservation, ...]
    errors: tuple[ExperienceError, ...]
    corrections: tuple[ExperienceCorrection, ...]
    result: str | None

    def __post_init__(self) -> None:
        _validate_id(self.attempt_id, "attempt_id")
        _validate_text(self.started_at, "attempt started_at")
        if self.completed_at is not None:
            _validate_text(self.completed_at, "attempt completed_at")
        if not isinstance(self.status, ExperienceAttemptStatus):
            raise ExperienceRecordValidationError("attempt status must be ExperienceAttemptStatus")
        for values, expected, name in ((self.actions, ExperienceAction, "actions"), (self.observations, ExperienceObservation, "observations"), (self.errors, ExperienceError, "errors"), (self.corrections, ExperienceCorrection, "corrections")):
            if not isinstance(values, tuple) or any(not isinstance(item, expected) for item in values):
                raise ExperienceRecordValidationError(f"invalid {name}")
        if self.result is not None:
            _validate_text(self.result, "attempt result")

    def to_dict(self) -> dict[str, Any]:
        return {"attempt_id": self.attempt_id, "started_at": self.started_at, "completed_at": self.completed_at, "status": self.status.value, "actions": [item.to_dict() for item in self.actions], "observations": [item.to_dict() for item in self.observations], "errors": [item.to_dict() for item in self.errors], "corrections": [item.to_dict() for item in self.corrections], "result": self.result}


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    experience_id: str
    task: str
    project_identity: ExperienceProjectIdentity | None
    started_at: str
    completed_at: str | None
    status: ExperienceLifecycleStatus
    attempts: tuple[ExperienceAttempt, ...]
    final_solution: str | None
    final_summary: str | None
    verification: ExperienceVerification | None
    evaluation: ExperienceEvaluation | None
    outcome: ExperienceOutcomeStatus | None
    metadata: Mapping[str, Any]
    schema_version: str = EXPERIENCE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_id(self.experience_id, "experience_id")
        _validate_text(self.task, "task")
        _validate_text(self.started_at, "started_at")
        if self.completed_at is not None:
            _validate_text(self.completed_at, "completed_at")
        if not isinstance(self.status, ExperienceLifecycleStatus):
            raise ExperienceRecordValidationError("status must be ExperienceLifecycleStatus")
        if not isinstance(self.attempts, tuple) or any(not isinstance(item, ExperienceAttempt) for item in self.attempts):
            raise ExperienceRecordValidationError("attempts must be a tuple of ExperienceAttempt")
        for value, name in ((self.final_solution, "final_solution"), (self.final_summary, "final_summary")):
            if value is not None:
                _validate_text(value, name)
        if self.verification is not None and not isinstance(self.verification, ExperienceVerification):
            raise ExperienceRecordValidationError("verification must be ExperienceVerification")
        if self.evaluation is not None and not isinstance(self.evaluation, ExperienceEvaluation):
            raise ExperienceRecordValidationError("evaluation must be ExperienceEvaluation")
        if self.outcome is not None and not isinstance(self.outcome, ExperienceOutcomeStatus):
            raise ExperienceRecordValidationError("outcome must be ExperienceOutcomeStatus")
        if self.status in {ExperienceLifecycleStatus.COMPLETED, ExperienceLifecycleStatus.FAILED, ExperienceLifecycleStatus.CANCELLED}:
            if self.completed_at is None or self.outcome is None:
                raise ExperienceRecordValidationError("finalized experience requires completed_at and outcome")
        if self.schema_version != EXPERIENCE_RECORD_SCHEMA_VERSION:
            raise ExperienceRecordValidationError("unsupported Experience Record schema version")
        object.__setattr__(self, "metadata", _freeze_value(_redact_experience_value(dict(self.metadata))))

    @property
    def finalized(self) -> bool:
        return self.status in {ExperienceLifecycleStatus.COMPLETED, ExperienceLifecycleStatus.FAILED, ExperienceLifecycleStatus.CANCELLED}

    def to_dict(self) -> dict[str, Any]:
        return {"experience_id": self.experience_id, "task": self.task, "project_identity": self.project_identity.to_dict() if self.project_identity else None, "started_at": self.started_at, "completed_at": self.completed_at, "status": self.status.value, "attempts": [item.to_dict() for item in self.attempts], "final_solution": self.final_solution, "final_summary": self.final_summary, "verification": self.verification.to_dict() if self.verification else None, "evaluation": self.evaluation.to_dict() if self.evaluation else None, "outcome": self.outcome.value if self.outcome else None, "metadata": _thaw_value(self.metadata), "schema_version": self.schema_version}


@dataclass(frozen=True, slots=True)
class ExperienceRecordLoadResult:
    status: ExperienceRecordLoadStatus
    records: "ExperienceRecords" | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ExperienceRecordsSnapshot:
    records: tuple[ExperienceRecord, ...]
    status: ExperienceRecordLoadStatus
    sequence: int
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"format": EXPERIENCE_RECORD_FORMAT, "schema_version": EXPERIENCE_RECORD_SCHEMA_VERSION, "records": [item.to_dict() for item in self.records], "status": self.status.value, "sequence": self.sequence, "warnings": list(self.warnings)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ExperienceSession:
    """Explicit mutable owner that produces one immutable historical record."""

    def __init__(self, manager: "ExperienceRecords", record: ExperienceRecord, *, clock: Callable[[], str]) -> None:
        self._manager = manager
        self._record = record
        self._clock = clock
        self._attempts: list[dict[str, Any]] = []
        self._finalized = False

    @property
    def experience_id(self) -> str:
        return self._record.experience_id

    @property
    def record(self) -> ExperienceRecord:
        return self.snapshot()

    def start_attempt(self) -> str:
        self._ensure_open()
        if len(self._attempts) >= self._manager.limits.max_attempts_per_experience:
            raise ExperienceRecordValidationError("max_attempts_per_experience exceeded")
        attempt_id = f"{self._record.experience_id}-attempt-{len(self._attempts) + 1:04d}"
        self._attempts.append({"attempt_id": attempt_id, "started_at": self._clock(), "completed_at": None, "status": ExperienceAttemptStatus.RUNNING, "actions": [], "observations": [], "errors": [], "corrections": [], "result": None})
        self._record = replace(self._record, status=ExperienceLifecycleStatus.RUNNING)
        return attempt_id

    def record_action(self, name: str, summary: str, *, status: str = "observed", metadata: Mapping[str, Any] | None = None, attempt_id: str | None = None) -> ExperienceAction:
        attempt = self._attempt(attempt_id)
        self._ensure_capacity(attempt["actions"], self._manager.limits.max_actions_per_attempt, "actions")
        action = ExperienceAction(self._next_id("action"), attempt["attempt_id"], self._safe(summary), self._safe(summary), self._clock(), self._safe(status), _safe_metadata(metadata or {}, self._manager.limits))
        action = replace(action, name=self._safe(name))
        attempt["actions"].append(action)
        return action

    def record_observation(self, summary: str, *, source: str = "agent", metadata: Mapping[str, Any] | None = None, attempt_id: str | None = None) -> ExperienceObservation:
        attempt = self._attempt(attempt_id)
        self._ensure_capacity(attempt["observations"], self._manager.limits.max_observations_per_attempt, "observations")
        observation = ExperienceObservation(self._next_id("observation"), attempt["attempt_id"], self._safe(summary), self._safe(source), self._clock(), _safe_metadata(metadata or {}, self._manager.limits))
        attempt["observations"].append(observation)
        return observation

    def record_error(self, category: str, summary: str, *, source: str = "agent", status: ExperienceErrorStatus | str = ExperienceErrorStatus.UNRESOLVED, attempt_id: str | None = None) -> ExperienceError:
        attempt = self._attempt(attempt_id)
        self._ensure_capacity(attempt["errors"], self._manager.limits.max_errors_per_attempt, "errors")
        error = ExperienceError(self._next_id("error"), attempt["attempt_id"], self._safe(category), self._safe(summary), self._safe(source), self._clock(), _coerce_enum(status, ExperienceErrorStatus, "error status"))
        attempt["errors"].append(error)
        return error

    def record_correction(self, summary: str, outcome: str, *, error_id: str | None = None, attempt_id: str | None = None) -> ExperienceCorrection:
        attempt = self._attempt(attempt_id)
        self._ensure_capacity(attempt["corrections"], self._manager.limits.max_corrections_per_attempt, "corrections")
        correction = ExperienceCorrection(self._next_id("correction"), attempt["attempt_id"], error_id, self._safe(summary), self._safe(outcome), self._clock())
        attempt["corrections"].append(correction)
        return correction

    def record_attempt_result(self, result: str, *, status: ExperienceAttemptStatus | str = ExperienceAttemptStatus.COMPLETED, attempt_id: str | None = None) -> None:
        attempt = self._attempt(attempt_id)
        attempt["result"] = self._safe(result)
        attempt["status"] = _coerce_enum(status, ExperienceAttemptStatus, "attempt status")
        attempt["completed_at"] = self._clock()

    def record_verification(self, verification: ExperienceVerification) -> None:
        self._ensure_open()
        if not isinstance(verification, ExperienceVerification):
            raise ExperienceRecordValidationError("verification must be ExperienceVerification")
        self._record = replace(self._record, verification=verification)

    def record_evaluation(self, evaluation: ExperienceEvaluation) -> None:
        self._ensure_open()
        if not isinstance(evaluation, ExperienceEvaluation):
            raise ExperienceRecordValidationError("evaluation must be ExperienceEvaluation")
        self._record = replace(self._record, evaluation=evaluation)

    def finalize(self, *, status: ExperienceLifecycleStatus | str, outcome: ExperienceOutcomeStatus | str, final_solution: str | None = None, final_summary: str | None = None) -> ExperienceRecord:
        self._ensure_open()
        lifecycle = _coerce_enum(status, ExperienceLifecycleStatus, "lifecycle status")
        outcome_value = _coerce_enum(outcome, ExperienceOutcomeStatus, "outcome")
        if lifecycle not in {ExperienceLifecycleStatus.COMPLETED, ExperienceLifecycleStatus.FAILED, ExperienceLifecycleStatus.CANCELLED}:
            raise ExperienceRecordValidationError("finalize requires completed, failed, or cancelled status")
        if outcome_value is ExperienceOutcomeStatus.SUCCESS and self._record.verification is None:
            raise ExperienceRecordValidationError("success requires recorded verification evidence")
        for attempt in self._attempts:
            if attempt["completed_at"] is None:
                attempt["completed_at"] = self._clock()
                attempt["status"] = ExperienceAttemptStatus.COMPLETED if lifecycle is ExperienceLifecycleStatus.COMPLETED else ExperienceAttemptStatus.FAILED
        attempts = tuple(_attempt_snapshot(item) for item in self._attempts)
        finalized_record = replace(self._record, completed_at=self._clock(), status=lifecycle, attempts=attempts, outcome=outcome_value, final_solution=self._safe(final_solution) if final_solution is not None else None, final_summary=self._safe(final_summary) if final_summary is not None else None)
        self._manager._accept(finalized_record)
        self._record = finalized_record
        self._finalized = True
        return finalized_record

    def snapshot(self) -> ExperienceRecord:
        attempts = tuple(_attempt_snapshot(item) for item in self._attempts)
        return replace(self._record, attempts=attempts)

    def _attempt(self, attempt_id: str | None) -> dict[str, Any]:
        self._ensure_open()
        if not self._attempts:
            self.start_attempt()
        active = self._attempts[-1] if attempt_id is None else next((item for item in self._attempts if item["attempt_id"] == attempt_id), None)
        if active is None:
            raise ExperienceRecordValidationError("unknown attempt_id")
        return active

    def _safe(self, value: str) -> str:
        return _safe_text(value, self._manager.limits.max_content_length)

    def _ensure_capacity(self, values: list[Any], limit: int, name: str) -> None:
        if len(values) >= limit:
            raise ExperienceRecordValidationError(f"max_{name}_per_attempt exceeded")

    def _next_id(self, kind: str) -> str:
        return f"{kind}-{self._record.experience_id}-{self._manager._next_sequence()}"

    def _ensure_open(self) -> None:
        if self._finalized:
            raise ExperienceRecordClosedError("experience is finalized")


class ExperienceRecords:
    """Explicit owner for a bounded collection of finalized historical records."""

    def __init__(self, *, limits: ExperienceRecordLimits | None = None, clock: Callable[[], str] | None = None) -> None:
        self.limits = limits or ExperienceRecordLimits()
        self._clock = clock or _utc_now
        self._records: dict[str, ExperienceRecord] = {}
        self._sequence = 0
        self._load_status = ExperienceRecordLoadStatus.MEMORY_MISSING

    @classmethod
    def from_snapshot(cls, snapshot: ExperienceRecordsSnapshot, *, limits: ExperienceRecordLimits | None = None, clock: Callable[[], str] | None = None) -> "ExperienceRecords":
        manager = cls(limits=limits, clock=clock)
        manager._records = {item.experience_id: item for item in snapshot.records}
        manager._sequence = snapshot.sequence
        manager._load_status = snapshot.status
        return manager

    def start_experience(self, task: str, *, project_identity: ExperienceProjectIdentity | None = None, metadata: Mapping[str, Any] | None = None) -> ExperienceSession:
        safe_task = _safe_text(task, self.limits.max_content_length)
        self._sequence += 1
        now = self._clock()
        experience_id = "exp-" + hashlib.sha256(f"{safe_task}|{self._sequence}|{now}".encode("utf-8")).hexdigest()[:24]
        record = ExperienceRecord(experience_id, safe_task, project_identity, now, None, ExperienceLifecycleStatus.STARTED, (), None, None, None, None, None, _safe_metadata(metadata or {}, self.limits))
        return ExperienceSession(self, record, clock=self._clock)

    def get(self, experience_id: str) -> ExperienceRecord | None:
        _validate_id(experience_id, "experience_id")
        return self._records.get(experience_id)

    def list(self, *, project_id: str | None = None, status: ExperienceLifecycleStatus | str | None = None, started_after: str | None = None, started_before: str | None = None) -> tuple[ExperienceRecord, ...]:
        status_value = _coerce_enum(status, ExperienceLifecycleStatus, "lifecycle status") if status is not None else None
        return tuple(item for item in sorted(self._records.values(), key=lambda record: (record.started_at, record.experience_id)) if (project_id is None or (item.project_identity and item.project_identity.project_id == project_id)) and (status_value is None or item.status is status_value) and (started_after is None or item.started_at >= started_after) and (started_before is None or item.started_at <= started_before))

    def snapshot(self) -> ExperienceRecordsSnapshot:
        return ExperienceRecordsSnapshot(tuple(sorted(self._records.values(), key=lambda item: (item.started_at, item.experience_id))), self._load_status, self._sequence)

    def to_json(self) -> str:
        return self.snapshot().to_json()

    def _accept(self, record: ExperienceRecord) -> None:
        if len(self._records) >= self.limits.max_experiences and record.experience_id not in self._records:
            raise ExperienceRecordValidationError("max_experiences exceeded")
        encoded = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > self.limits.max_serialized_record_bytes:
            raise ExperienceRecordValidationError("experience record exceeds serialized size limit")
        self._records[record.experience_id] = record
        if len(self.to_json().encode("utf-8")) > self.limits.max_total_storage_bytes:
            self._records.pop(record.experience_id, None)
            raise ExperienceRecordValidationError("experience collection exceeds total storage limit")

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence


class ExperienceRecordStore:
    """Atomic global persistence separate from both memory stores."""

    def __init__(self, path: Path | str | None = None, *, limits: ExperienceRecordLimits | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_experience_record_path()
        self.limits = limits or ExperienceRecordLimits()
        self._loaded_digest: str | None = None

    @classmethod
    def default(cls, *, limits: ExperienceRecordLimits | None = None) -> "ExperienceRecordStore":
        return cls(None, limits=limits)

    def empty(self, *, clock: Callable[[], str] | None = None) -> ExperienceRecords:
        return ExperienceRecords(limits=self.limits, clock=clock)

    def load(self, *, clock: Callable[[], str] | None = None) -> ExperienceRecordLoadResult:
        try:
            self._validate_storage_location()
        except ExperienceRecordValidationError as exc:
            return ExperienceRecordLoadResult(ExperienceRecordLoadStatus.MEMORY_INVALID, None, str(exc))
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            self._loaded_digest = None
            return ExperienceRecordLoadResult(ExperienceRecordLoadStatus.MEMORY_MISSING, None, "experience record file does not exist")
        except OSError as exc:
            return ExperienceRecordLoadResult(ExperienceRecordLoadStatus.MEMORY_UNAVAILABLE, None, str(exc))
        if len(raw) > self.limits.max_total_storage_bytes:
            return ExperienceRecordLoadResult(ExperienceRecordLoadStatus.MEMORY_INVALID, None, "experience storage exceeds configured byte limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
            snapshot = _snapshot_from_dict(payload, self.limits)
        except UnicodeDecodeError as exc:
            return ExperienceRecordLoadResult(ExperienceRecordLoadStatus.MEMORY_CORRUPTED, None, f"invalid UTF-8: {exc}")
        except json.JSONDecodeError as exc:
            return ExperienceRecordLoadResult(ExperienceRecordLoadStatus.MEMORY_CORRUPTED, None, str(exc))
        except (ExperienceRecordValidationError, TypeError, ValueError) as exc:
            return ExperienceRecordLoadResult(ExperienceRecordLoadStatus.MEMORY_INVALID, None, str(exc))
        self._loaded_digest = _digest(raw)
        return ExperienceRecordLoadResult(ExperienceRecordLoadStatus.LOADED, ExperienceRecords.from_snapshot(snapshot, limits=self.limits, clock=clock), None)

    def _validate_storage_location(self) -> None:
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ExperienceRecordValidationError("experience storage must not use symlinks")

    def save(self, records: ExperienceRecords) -> Path:
        self._validate_storage_location()
        if not isinstance(records, ExperienceRecords):
            raise ExperienceRecordValidationError("records must be ExperienceRecords")
        if self.path.exists():
            current_digest = _digest(self.path.read_bytes())
            if self._loaded_digest is None or current_digest != self._loaded_digest:
                raise ExperienceRecordConflictError("experience records changed since this store loaded them")
        records._load_status = ExperienceRecordLoadStatus.LOADED
        payload = records.to_json().encode("utf-8")
        if len(payload) > self.limits.max_total_storage_bytes:
            raise ExperienceRecordValidationError("experience storage exceeds configured byte limit")
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=directory, prefix=".experience_records.", suffix=".tmp", delete=False) as stream:
                temporary_path = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            try:
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
            self._loaded_digest = _digest(payload)
            return self.path
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass


def default_experience_record_path() -> Path:
    return Path.home() / _DEFAULT_DIRECTORY / _DEFAULT_FILENAME


def _snapshot_from_dict(payload: Any, limits: ExperienceRecordLimits) -> ExperienceRecordsSnapshot:
    if not isinstance(payload, Mapping) or payload.get("format") != EXPERIENCE_RECORD_FORMAT:
        raise ExperienceRecordValidationError("unsupported Experience Record format")
    allowed = {"format", "schema_version", "records", "status", "sequence", "warnings"}
    if set(payload) - allowed:
        raise ExperienceRecordValidationError("unknown Experience Record fields are not accepted")
    if payload.get("schema_version") != EXPERIENCE_RECORD_SCHEMA_VERSION:
        raise ExperienceRecordValidationError("unsupported Experience Record schema version")
    records_payload = payload.get("records", [])
    if not isinstance(records_payload, list) or len(records_payload) > limits.max_experiences:
        raise ExperienceRecordValidationError("invalid or oversized records array")
    records = tuple(_record_from_dict(item, limits) for item in records_payload)
    sequence = payload.get("sequence", 0)
    if not isinstance(sequence, int) or sequence < 0:
        raise ExperienceRecordValidationError("invalid sequence")
    status = _coerce_enum(payload.get("status", ExperienceRecordLoadStatus.LOADED.value), ExperienceRecordLoadStatus, "load status")
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise ExperienceRecordValidationError("invalid warnings")
    return ExperienceRecordsSnapshot(records, status, sequence, tuple(warnings[:_MAX_COLLECTION_ITEMS]))


def _record_from_dict(payload: Any, limits: ExperienceRecordLimits) -> ExperienceRecord:
    if not isinstance(payload, Mapping):
        raise ExperienceRecordValidationError("experience record must be an object")
    allowed = {"experience_id", "task", "project_identity", "started_at", "completed_at", "status", "attempts", "final_solution", "final_summary", "verification", "evaluation", "outcome", "metadata", "schema_version"}
    if set(payload) - allowed:
        raise ExperienceRecordValidationError("unknown experience record fields are not accepted")
    identity_payload = payload.get("project_identity")
    if identity_payload is not None and not isinstance(identity_payload, Mapping):
        raise ExperienceRecordValidationError("project_identity must be an object or null")
    identity = None if identity_payload is None else ExperienceProjectIdentity(str(identity_payload.get("project_id", "")), identity_payload.get("project_root"))
    attempts_payload = payload.get("attempts", [])
    if not isinstance(attempts_payload, list) or len(attempts_payload) > limits.max_attempts_per_experience:
        raise ExperienceRecordValidationError("invalid or oversized attempts")
    attempts = tuple(_attempt_from_dict(item, limits) for item in attempts_payload)
    verification = _verification_from_dict(payload.get("verification")) if payload.get("verification") is not None else None
    evaluation = _evaluation_from_dict(payload.get("evaluation")) if payload.get("evaluation") is not None else None
    outcome = _coerce_enum(payload.get("outcome"), ExperienceOutcomeStatus, "outcome") if payload.get("outcome") is not None else None
    return ExperienceRecord(str(payload.get("experience_id", "")), _safe_text(str(payload.get("task", "")), limits.max_content_length), identity, str(payload.get("started_at", "")), payload.get("completed_at"), _coerce_enum(payload.get("status"), ExperienceLifecycleStatus, "lifecycle status"), attempts, _optional_safe(payload.get("final_solution"), limits), _optional_safe(payload.get("final_summary"), limits), verification, evaluation, outcome, _safe_metadata(payload.get("metadata", {}), limits), str(payload.get("schema_version", "")))


def _attempt_from_dict(payload: Any, limits: ExperienceRecordLimits) -> ExperienceAttempt:
    if not isinstance(payload, Mapping):
        raise ExperienceRecordValidationError("attempt must be an object")
    actions = tuple(_action_from_dict(item, limits) for item in _bounded_list(payload.get("actions", []), limits.max_actions_per_attempt, "actions"))
    observations = tuple(_observation_from_dict(item, limits) for item in _bounded_list(payload.get("observations", []), limits.max_observations_per_attempt, "observations"))
    errors = tuple(_error_from_dict(item, limits) for item in _bounded_list(payload.get("errors", []), limits.max_errors_per_attempt, "errors"))
    corrections = tuple(_correction_from_dict(item, limits) for item in _bounded_list(payload.get("corrections", []), limits.max_corrections_per_attempt, "corrections"))
    return ExperienceAttempt(str(payload.get("attempt_id", "")), str(payload.get("started_at", "")), payload.get("completed_at"), _coerce_enum(payload.get("status"), ExperienceAttemptStatus, "attempt status"), actions, observations, errors, corrections, _optional_safe(payload.get("result"), limits))


def _action_from_dict(payload: Any, limits: ExperienceRecordLimits) -> ExperienceAction:
    return ExperienceAction(str(payload.get("action_id", "")), str(payload.get("attempt_id", "")), _safe_text(str(payload.get("name", "")), limits.max_content_length), _safe_text(str(payload.get("summary", "")), limits.max_content_length), str(payload.get("timestamp", "")), _safe_text(str(payload.get("status", "")), limits.max_content_length), _safe_metadata(payload.get("metadata", {}), limits))


def _observation_from_dict(payload: Any, limits: ExperienceRecordLimits) -> ExperienceObservation:
    return ExperienceObservation(str(payload.get("observation_id", "")), str(payload.get("attempt_id", "")), _safe_text(str(payload.get("summary", "")), limits.max_content_length), _safe_text(str(payload.get("source", "")), limits.max_content_length), str(payload.get("timestamp", "")), _safe_metadata(payload.get("metadata", {}), limits))


def _error_from_dict(payload: Any, limits: ExperienceRecordLimits) -> ExperienceError:
    return ExperienceError(str(payload.get("error_id", "")), str(payload.get("attempt_id", "")), _safe_text(str(payload.get("category", "")), limits.max_content_length), _safe_text(str(payload.get("summary", "")), limits.max_content_length), _safe_text(str(payload.get("source", "")), limits.max_content_length), str(payload.get("timestamp", "")), _coerce_enum(payload.get("status"), ExperienceErrorStatus, "error status"))


def _correction_from_dict(payload: Any, limits: ExperienceRecordLimits) -> ExperienceCorrection:
    error_id = payload.get("error_id")
    return ExperienceCorrection(str(payload.get("correction_id", "")), str(payload.get("attempt_id", "")), str(error_id) if error_id is not None else None, _safe_text(str(payload.get("summary", "")), limits.max_content_length), _safe_text(str(payload.get("outcome", "")), limits.max_content_length), str(payload.get("timestamp", "")))


def _verification_from_dict(payload: Any) -> ExperienceVerification:
    if not isinstance(payload, Mapping):
        raise ExperienceRecordValidationError("verification must be an object")
    return ExperienceVerification(int(payload.get("tests_executed", 0)), int(payload.get("tests_passed", 0)), int(payload.get("tests_failed", 0)), str(payload.get("test_status", "")), str(payload.get("summary", "")), str(payload.get("timestamp", "")), _safe_metadata(payload.get("metadata", {}), ExperienceRecordLimits()))


def _evaluation_from_dict(payload: Any) -> ExperienceEvaluation:
    if not isinstance(payload, Mapping):
        raise ExperienceRecordValidationError("evaluation must be an object")
    criteria = payload.get("criteria", [])
    if not isinstance(criteria, list):
        raise ExperienceRecordValidationError("evaluation criteria must be a list")
    return ExperienceEvaluation(payload.get("score"), str(payload.get("status", "")), str(payload.get("summary", "")), tuple(dict(item) for item in criteria), _safe_metadata(payload.get("evaluator_metadata", {}), ExperienceRecordLimits()))


def _attempt_snapshot(item: dict[str, Any]) -> ExperienceAttempt:
    return ExperienceAttempt(item["attempt_id"], item["started_at"], item["completed_at"], item["status"], tuple(item["actions"]), tuple(item["observations"]), tuple(item["errors"]), tuple(item["corrections"]), item["result"])


def _safe_metadata(value: Mapping[str, Any], limits: ExperienceRecordLimits) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperienceRecordValidationError("metadata must be a mapping")
    safe = _redact_experience_value(dict(value))
    encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > limits.max_metadata_size:
        raise ExperienceRecordValidationError("metadata exceeds configured size bound")
    return _freeze_value(safe)


def _redact_experience_text(value: str) -> str:
    safe = _redact_text(value)
    safe = re.sub(r"(?i)\b(?:bearer|basic)\s+[^\s]+", "[REDACTED]", safe)
    return re.sub(r"(?i)(?:authorization|proxy-authorization)\s*:\s*(?:\[REDACTED\]\s*)?[^\s]+(?:\s+[^\s]+)?", "authorization: [REDACTED]", safe)


def _redact_experience_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        keyed = _redact_value(dict(value))
        return {str(key): _redact_experience_value(item) for key, item in keyed.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_experience_value(item) for item in list(value)[:_MAX_COLLECTION_ITEMS]]
    if isinstance(value, str):
        return _redact_experience_text(value)
    return value


def _safe_text(value: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperienceRecordValidationError("text must contain text")
    safe = _redact_experience_text(value.strip())
    if len(safe) > limit:
        raise ExperienceRecordValidationError("text exceeds configured length bound")
    return safe


def _optional_safe(value: Any, limits: ExperienceRecordLimits) -> str | None:
    return None if value is None else _safe_text(str(value), limits.max_content_length)


def _bounded_list(value: Any, limit: int, name: str) -> list[Any]:
    if not isinstance(value, list) or len(value) > limit:
        raise ExperienceRecordValidationError(f"invalid or oversized {name}")
    return value


def _validate_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ExperienceRecordValidationError(f"{name} must contain text")


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ExperienceRecordValidationError(f"{name} must contain text")


def _coerce_enum(value: Any, enum_type: type[Enum], name: str):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        try:
            return enum_type[value]
        except (KeyError, TypeError) as inner:
            raise ExperienceRecordValidationError(f"invalid {name}: {value!r}") from inner


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in list(value)[:_MAX_COLLECTION_ITEMS])
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_value(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "EXPERIENCE_RECORD_FORMAT",
    "EXPERIENCE_RECORD_SCHEMA_VERSION",
    "ExperienceAction",
    "ExperienceAttempt",
    "ExperienceAttemptStatus",
    "ExperienceCorrection",
    "ExperienceError",
    "ExperienceErrorStatus",
    "ExperienceEvaluation",
    "ExperienceLifecycleStatus",
    "ExperienceOutcomeStatus",
    "ExperienceProjectIdentity",
    "ExperienceRecord",
    "ExperienceRecordClosedError",
    "ExperienceRecordConflictError",
    "ExperienceRecordError",
    "ExperienceRecordLimits",
    "ExperienceRecordLoadResult",
    "ExperienceRecordLoadStatus",
    "ExperienceRecordStore",
    "ExperienceRecordValidationError",
    "ExperienceRecords",
    "ExperienceRecordsSnapshot",
    "ExperienceSession",
    "ExperienceVerification",
    "default_experience_record_path",
]
