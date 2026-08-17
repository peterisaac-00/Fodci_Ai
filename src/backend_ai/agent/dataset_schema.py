"""Canonical, strict, versioned schema for Experience Dataset records.

Phase 10.2 defines a model-agnostic contract after Phase 10.1 extraction.  It
validates structure and integrity only; it does not filter quality, split a
dataset, version releases, tokenize, train, or update a model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any

from backend_ai.agent.experience_dataset import (
    DatasetCandidate,
    DatasetExtractionError,
    _canonical_json,
    _contains_prohibited_secret,
)


DATASET_RECORD_FORMAT = "fodci.experience_dataset_record"
DATASET_RECORD_SCHEMA_VERSION = "1.0"
DATASET_RECORD_ID_PREFIX = "drec-"


class DatasetSchemaError(ValueError):
    """Base error for strict Dataset Schema validation and serialization."""


class DatasetRecordValidationError(DatasetSchemaError):
    """Raised when a Dataset Record or nested payload violates the contract."""


class DatasetOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class DatasetRecordLimits:
    """Finite limits for one canonical record, reusing Phase 10.1 scale."""

    max_task_length: int = 1_024
    max_event_count: int = 512
    max_attempt_count: int = 64
    max_solution_length: int = 1_024
    max_verification_bytes: int = 8_192
    max_evaluation_bytes: int = 8_192
    max_metadata_bytes: int = 4_096
    max_total_serialized_bytes: int = 262_144
    max_nesting_depth: int = 8

    def __post_init__(self) -> None:
        ceilings = {
            "max_task_length": 65_536,
            "max_event_count": 4_096,
            "max_attempt_count": 512,
            "max_solution_length": 65_536,
            "max_verification_bytes": 1_048_576,
            "max_evaluation_bytes": 1_048_576,
            "max_metadata_bytes": 65_536,
            "max_total_serialized_bytes": 8 * 1024 * 1024,
            "max_nesting_depth": 32,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > ceiling:
                raise DatasetSchemaError(f"{name} is outside its configured bound")


@dataclass(frozen=True, slots=True)
class DatasetProjectContext:
    """Project identity explicitly present in the source Experience Record."""

    project_id: str
    project_root: str | None

    def __post_init__(self) -> None:
        _require_text(self.project_id, "project_context.project_id", 512)
        if self.project_root is not None:
            _require_text(self.project_root, "project_context.project_root", 4_096)

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | None) -> "DatasetProjectContext | None":
        if value is None:
            return None
        _expect_mapping(value, "project_context")
        _expect_fields(value, {"project_id", "project_root"}, "project_context")
        return cls(value.get("project_id"), value.get("project_root"))

    def to_dict(self) -> dict[str, Any]:
        return {"project_id": self.project_id, "project_root": self.project_root}


@dataclass(frozen=True, slots=True)
class DatasetRecordProvenance:
    """Mandatory traceability to the originating Experience Record."""

    source_type: str
    experience_id: str
    source_schema_version: str
    source_created_at: str
    completed_at: str
    project_identity: Mapping[str, Any] | None
    original_status: str
    original_outcome: str
    verification_present: bool

    def __post_init__(self) -> None:
        if self.source_type != "experience_record":
            raise DatasetRecordValidationError("provenance.source_type must be experience_record")
        for value, name, limit in (
            (self.experience_id, "provenance.experience_id", 256),
            (self.source_schema_version, "provenance.source_schema_version", 64),
            (self.source_created_at, "provenance.source_created_at", 128),
            (self.completed_at, "provenance.completed_at", 128),
            (self.original_status, "provenance.original_status", 64),
            (self.original_outcome, "provenance.original_outcome", 64),
        ):
            _require_text(value, name, limit)
        _require_timestamp(self.source_created_at, "provenance.source_created_at")
        _require_timestamp(self.completed_at, "provenance.completed_at")
        if self.original_outcome not in {item.value for item in DatasetOutcome}:
            raise DatasetRecordValidationError("provenance.original_outcome is invalid")
        if not isinstance(self.verification_present, bool):
            raise DatasetRecordValidationError("provenance.verification_present must be boolean")
        if self.project_identity is not None:
            _validate_json_value(self.project_identity, "provenance.project_identity", 0, 3)
            object.__setattr__(self, "project_identity", _freeze(self.project_identity))

    @classmethod
    def from_candidate(cls, value: Any) -> "DatasetRecordProvenance":
        if not isinstance(value, Mapping):
            raise DatasetRecordValidationError("candidate provenance must be a mapping")
        _expect_fields(value, {"source_type", "experience_id", "source_schema_version", "started_at", "completed_at", "project_identity", "original_status", "original_outcome", "verification_present"}, "candidate provenance")
        return cls(
            value.get("source_type"),
            value.get("experience_id"),
            value.get("source_schema_version"),
            value.get("started_at"),
            value.get("completed_at"),
            value.get("project_identity"),
            value.get("original_status"),
            value.get("original_outcome"),
            value.get("verification_present"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "DatasetRecordProvenance":
        if not isinstance(value, Mapping):
            raise DatasetRecordValidationError("provenance must be an object")
        _expect_fields(value, {"source_type", "experience_id", "source_schema_version", "source_created_at", "completed_at", "project_identity", "original_status", "original_outcome", "verification_present"}, "provenance")
        return cls(
            value.get("source_type"),
            value.get("experience_id"),
            value.get("source_schema_version"),
            value.get("source_created_at"),
            value.get("completed_at"),
            value.get("project_identity"),
            value.get("original_status"),
            value.get("original_outcome"),
            value.get("verification_present"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "experience_id": self.experience_id,
            "source_schema_version": self.source_schema_version,
            "source_created_at": self.source_created_at,
            "completed_at": self.completed_at,
            "project_identity": _thaw(self.project_identity),
            "original_status": self.original_status,
            "original_outcome": self.original_outcome,
            "verification_present": self.verification_present,
        }


@dataclass(frozen=True, slots=True)
class DatasetVerification:
    """Explicit verification state, preserving the source verification fields."""

    present: bool
    tests_executed: int | None
    tests_passed: int | None
    tests_failed: int | None
    test_status: str | None
    summary: str | None
    timestamp: str | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.present, bool):
            raise DatasetRecordValidationError("verification.present must be boolean")
        for value, name in ((self.tests_executed, "tests_executed"), (self.tests_passed, "tests_passed"), (self.tests_failed, "tests_failed")):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise DatasetRecordValidationError(f"verification.{name} must be a non-negative integer or null")
        if self.present:
            if any(value is None for value in (self.tests_executed, self.tests_passed, self.tests_failed, self.test_status, self.summary, self.timestamp)):
                raise DatasetRecordValidationError("present verification requires all source verification fields")
            if self.tests_passed + self.tests_failed > self.tests_executed:  # type: ignore[operator]
                raise DatasetRecordValidationError("verification test counts are inconsistent")
            _require_text(self.test_status, "verification.test_status", 128)
            _require_text(self.summary, "verification.summary", 65_536)
            _require_timestamp(self.timestamp, "verification.timestamp")
        else:
            if any(value is not None for value in (self.tests_executed, self.tests_passed, self.tests_failed, self.test_status, self.summary, self.timestamp)):
                raise DatasetRecordValidationError("absent verification must not contain verification values")
        _validate_json_value(self.metadata, "verification.metadata", 0, 5)
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @classmethod
    def from_candidate(cls, value: Mapping[str, Any] | None) -> "DatasetVerification":
        if value is None:
            return cls(False, None, None, None, None, None, None, {})
        if "present" not in value:
            value = {"present": True, **dict(value)}
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: Any) -> "DatasetVerification":
        if not isinstance(value, Mapping):
            raise DatasetRecordValidationError("verification must be an object")
        _expect_fields(value, {"present", "tests_executed", "tests_passed", "tests_failed", "test_status", "summary", "timestamp", "metadata"}, "verification")
        return cls(value.get("present"), value.get("tests_executed"), value.get("tests_passed"), value.get("tests_failed"), value.get("test_status"), value.get("summary"), value.get("timestamp"), value.get("metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {"present": self.present, "tests_executed": self.tests_executed, "tests_passed": self.tests_passed, "tests_failed": self.tests_failed, "test_status": self.test_status, "summary": self.summary, "timestamp": self.timestamp, "metadata": _thaw(self.metadata)}


@dataclass(frozen=True, slots=True)
class DatasetEvaluation:
    """Evaluation is distinct from verification and is never LLM-generated here."""

    present: bool
    score: float | None
    status: str | None
    summary: str | None
    criteria: tuple[Mapping[str, Any], ...]
    evaluator_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.present, bool):
            raise DatasetRecordValidationError("evaluation.present must be boolean")
        if self.score is not None and (not isinstance(self.score, (int, float)) or isinstance(self.score, bool) or not math.isfinite(float(self.score))):
            raise DatasetRecordValidationError("evaluation.score must be finite numeric or null")
        if self.present:
            _require_text(self.status, "evaluation.status", 128)
            _require_text(self.summary, "evaluation.summary", 65_536)
        elif any(value is not None for value in (self.score, self.status, self.summary)) or self.criteria:
            raise DatasetRecordValidationError("absent evaluation must not contain evaluation values")
        if not isinstance(self.criteria, tuple) or any(not isinstance(item, Mapping) for item in self.criteria):
            raise DatasetRecordValidationError("evaluation.criteria must be an immutable tuple of objects")
        for item in self.criteria:
            _validate_json_value(item, "evaluation.criteria", 0, 5)
        _validate_json_value(self.evaluator_metadata, "evaluation.evaluator_metadata", 0, 5)
        object.__setattr__(self, "criteria", tuple(_freeze(item) for item in self.criteria))
        object.__setattr__(self, "evaluator_metadata", _freeze(self.evaluator_metadata))

    @classmethod
    def from_candidate(cls, value: Mapping[str, Any] | None) -> "DatasetEvaluation":
        if value is None:
            return cls(False, None, None, None, (), {})
        if "present" not in value:
            value = {"present": True, **dict(value)}
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: Any) -> "DatasetEvaluation":
        if not isinstance(value, Mapping):
            raise DatasetRecordValidationError("evaluation must be an object")
        _expect_fields(value, {"present", "score", "status", "summary", "criteria", "evaluator_metadata"}, "evaluation")
        criteria = value.get("criteria")
        if not isinstance(criteria, (list, tuple)):
            raise DatasetRecordValidationError("evaluation.criteria must be an array")
        return cls(value.get("present"), value.get("score"), value.get("status"), value.get("summary"), tuple(criteria), value.get("evaluator_metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {"present": self.present, "score": self.score, "status": self.status, "summary": self.summary, "criteria": [_thaw(item) for item in self.criteria], "evaluator_metadata": _thaw(self.evaluator_metadata)}


@dataclass(frozen=True, slots=True)
class DatasetSolution:
    """Separate historical solution/result/summary fields."""

    solution: str | None
    final_result: str | None
    final_summary: str | None

    def __post_init__(self) -> None:
        for value, name in ((self.solution, "solution.solution"), (self.final_result, "solution.final_result"), (self.final_summary, "solution.final_summary")):
            if value is not None:
                _require_text(value, name, 65_536)

    @classmethod
    def from_candidate(cls, candidate: DatasetCandidate) -> "DatasetSolution":
        result = None
        if candidate.attempts:
            result = candidate.attempts[-1].get("result")
        return cls(candidate.final_solution, result, candidate.final_summary)

    @classmethod
    def from_dict(cls, value: Any) -> "DatasetSolution":
        if not isinstance(value, Mapping):
            raise DatasetRecordValidationError("solution must be an object")
        _expect_fields(value, {"solution", "final_result", "final_summary"}, "solution")
        return cls(value.get("solution"), value.get("final_result"), value.get("final_summary"))

    def to_dict(self) -> dict[str, Any]:
        return {"solution": self.solution, "final_result": self.final_result, "final_summary": self.final_summary}


@dataclass(frozen=True, slots=True)
class DatasetTrajectory:
    """Ordered, structured execution trajectory without synthetic events."""

    attempts: tuple[Mapping[str, Any], ...]
    actions: tuple[Mapping[str, Any], ...]
    observations: tuple[Mapping[str, Any], ...]
    errors: tuple[Mapping[str, Any], ...]
    corrections: tuple[Mapping[str, Any], ...]
    verification_events: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.attempts, tuple):
            raise DatasetRecordValidationError("trajectory.attempts must be an immutable tuple")
        if len(self.attempts) > 512:
            raise DatasetRecordValidationError("trajectory.attempts exceeds the schema bound")
        _validate_attempts(self.attempts)
        _validate_events(self.actions, "trajectory.actions", {"action_id", "attempt_id", "name", "summary", "timestamp", "status", "metadata"}, "action_id")
        _validate_events(self.observations, "trajectory.observations", {"observation_id", "attempt_id", "summary", "source", "timestamp", "metadata"}, "observation_id")
        _validate_events(self.errors, "trajectory.errors", {"error_id", "attempt_id", "category", "summary", "source", "timestamp", "status"}, "error_id")
        _validate_events(self.corrections, "trajectory.corrections", {"correction_id", "attempt_id", "error_id", "summary", "outcome", "timestamp"}, "correction_id")
        _validate_events(self.verification_events, "trajectory.verification_events", {"event_id", "event_type", "timestamp", "details"}, "event_id")
        object.__setattr__(self, "attempts", tuple(_freeze(item) for item in self.attempts))
        for name in ("actions", "observations", "errors", "corrections", "verification_events"):
            object.__setattr__(self, name, tuple(_freeze(item) for item in getattr(self, name)))

    @classmethod
    def from_candidate(cls, candidate: DatasetCandidate) -> "DatasetTrajectory":
        if not isinstance(candidate, DatasetCandidate):
            raise DatasetRecordValidationError("candidate must be DatasetCandidate")
        return cls(candidate.attempts, candidate.actions, candidate.observations, candidate.errors, candidate.corrections, ())

    @classmethod
    def from_dict(cls, value: Any) -> "DatasetTrajectory":
        if not isinstance(value, Mapping):
            raise DatasetRecordValidationError("trajectory must be an object")
        _expect_fields(value, {"attempts", "actions", "observations", "errors", "corrections", "verification_events"}, "trajectory")
        collections = []
        for name in ("attempts", "actions", "observations", "errors", "corrections", "verification_events"):
            raw = value.get(name)
            if not isinstance(raw, (list, tuple)):
                raise DatasetRecordValidationError(f"trajectory.{name} must be an array")
            collections.append(tuple(raw))
        return cls(*collections)

    def to_dict(self) -> dict[str, Any]:
        return {"attempts": [_thaw(item) for item in self.attempts], "actions": [_thaw(item) for item in self.actions], "observations": [_thaw(item) for item in self.observations], "errors": [_thaw(item) for item in self.errors], "corrections": [_thaw(item) for item in self.corrections], "verification_events": [_thaw(item) for item in self.verification_events]}


@dataclass(frozen=True, slots=True)
class DatasetSchemaValidationResult:
    """Deterministic validation diagnostics for untrusted Dataset payloads."""

    valid: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Canonical model-agnostic Dataset Schema record, version 1.0."""

    format: str
    schema_version: str
    record_id: str
    experience_id: str
    task: str
    project_context: DatasetProjectContext | None
    trajectory: DatasetTrajectory
    solution: DatasetSolution
    verification: DatasetVerification
    evaluation: DatasetEvaluation
    outcome: DatasetOutcome
    provenance: DatasetRecordProvenance
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.format != DATASET_RECORD_FORMAT:
            raise DatasetRecordValidationError("unsupported dataset record format")
        if self.schema_version != DATASET_RECORD_SCHEMA_VERSION:
            raise DatasetRecordValidationError("unsupported dataset record schema_version")
        _require_text(self.experience_id, "experience_id", 256)
        expected_id = derive_dataset_record_id(self.experience_id, self.provenance.source_schema_version)
        if self.record_id != expected_id:
            raise DatasetRecordValidationError("record_id does not match deterministic identity rule")
        _require_text(self.task, "task", 1_024)
        if not isinstance(self.project_context, (DatasetProjectContext, type(None))):
            raise DatasetRecordValidationError("project_context has invalid type")
        if not isinstance(self.trajectory, DatasetTrajectory):
            raise DatasetRecordValidationError("trajectory has invalid type")
        if not isinstance(self.solution, DatasetSolution):
            raise DatasetRecordValidationError("solution has invalid type")
        if not isinstance(self.verification, DatasetVerification):
            raise DatasetRecordValidationError("verification has invalid type")
        if not isinstance(self.evaluation, DatasetEvaluation):
            raise DatasetRecordValidationError("evaluation has invalid type")
        if not isinstance(self.outcome, DatasetOutcome):
            raise DatasetRecordValidationError("outcome has invalid enum value")
        if not isinstance(self.provenance, DatasetRecordProvenance):
            raise DatasetRecordValidationError("provenance has invalid type")
        if self.provenance.experience_id != self.experience_id:
            raise DatasetRecordValidationError("provenance.experience_id does not match experience_id")
        if self.provenance.original_outcome != self.outcome.value:
            raise DatasetRecordValidationError("provenance.original_outcome does not match outcome")
        _validate_json_value(self.metadata, "metadata", 0, 5)
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        self._validate_size(DatasetRecordLimits())

    @classmethod
    def from_candidate(cls, candidate: DatasetCandidate, *, limits: DatasetRecordLimits | None = None) -> "DatasetRecord":
        if not isinstance(candidate, DatasetCandidate):
            raise DatasetRecordValidationError("candidate must be DatasetCandidate")
        provenance = DatasetRecordProvenance.from_candidate(candidate.provenance.to_dict())
        record = cls(
            DATASET_RECORD_FORMAT,
            DATASET_RECORD_SCHEMA_VERSION,
            derive_dataset_record_id(candidate.experience_id, provenance.source_schema_version),
            candidate.experience_id,
            candidate.task,
            DatasetProjectContext.from_value(candidate.project_identity),
            DatasetTrajectory.from_candidate(candidate),
            DatasetSolution.from_candidate(candidate),
            DatasetVerification.from_candidate(candidate.verification),
            DatasetEvaluation.from_candidate(candidate.evaluation),
            _coerce_outcome(candidate.outcome),
            provenance,
            candidate.metadata,
        )
        if limits is not None:
            record._validate_size(limits)
        _assert_safe_record(record)
        return record

    @classmethod
    def from_dict(cls, payload: Any, *, limits: DatasetRecordLimits | None = None) -> "DatasetRecord":
        if not isinstance(payload, Mapping):
            raise DatasetRecordValidationError("dataset record must be an object")
        _expect_fields(payload, {"format", "schema_version", "record_id", "experience_id", "task", "project_context", "trajectory", "solution", "verification", "evaluation", "outcome", "provenance", "metadata"}, "dataset record")
        record_format = payload.get("format")
        schema_version = payload.get("schema_version")
        if schema_version != DATASET_RECORD_SCHEMA_VERSION:
            raise DatasetRecordValidationError("unsupported dataset record schema_version")
        try:
            record = cls(
                record_format,
                schema_version,
                payload.get("record_id"),
                payload.get("experience_id"),
                payload.get("task"),
                DatasetProjectContext.from_value(payload.get("project_context")),
                DatasetTrajectory.from_dict(payload.get("trajectory")),
                DatasetSolution.from_dict(payload.get("solution")),
                DatasetVerification.from_dict(payload.get("verification")),
                DatasetEvaluation.from_dict(payload.get("evaluation")),
                _coerce_outcome(payload.get("outcome")),
                DatasetRecordProvenance.from_dict(payload.get("provenance")),
                payload.get("metadata"),
            )
        except DatasetSchemaError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise DatasetRecordValidationError(f"invalid dataset record structure: {exc}") from exc
        if limits is not None:
            record._validate_size(limits)
        _assert_safe_record(record)
        return record

    @classmethod
    def from_json(cls, payload: str, *, limits: DatasetRecordLimits | None = None) -> "DatasetRecord":
        if not isinstance(payload, str):
            raise DatasetRecordValidationError("dataset JSON must be text")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DatasetRecordValidationError("dataset JSON is malformed") from exc
        return cls.from_dict(value, limits=limits)

    @staticmethod
    def validate_payload(payload: Any, *, limits: DatasetRecordLimits | None = None) -> DatasetSchemaValidationResult:
        try:
            DatasetRecord.from_dict(payload, limits=limits)
        except DatasetSchemaError as exc:
            return DatasetSchemaValidationResult(False, (str(exc),))
        return DatasetSchemaValidationResult(True, ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "experience_id": self.experience_id,
            "task": self.task,
            "project_context": self.project_context.to_dict() if self.project_context else None,
            "trajectory": self.trajectory.to_dict(),
            "solution": self.solution.to_dict(),
            "verification": self.verification.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "outcome": self.outcome.value,
            "provenance": self.provenance.to_dict(),
            "metadata": _thaw(self.metadata),
        }

    def to_json(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._validate_serialized_size(payload, DatasetRecordLimits())
        return payload

    def _validate_size(self, limits: DatasetRecordLimits) -> None:
        if len(self.task) > limits.max_task_length:
            raise DatasetRecordValidationError("task exceeds schema limit")
        if len(self.trajectory.attempts) > limits.max_attempt_count:
            raise DatasetRecordValidationError("trajectory.attempts exceeds schema limit")
        total_events = sum(len(getattr(self.trajectory, name)) for name in ("actions", "observations", "errors", "corrections", "verification_events"))
        if total_events > limits.max_event_count:
            raise DatasetRecordValidationError("trajectory event count exceeds schema limit")
        for value, name in ((self.solution.solution, "solution.solution"), (self.solution.final_result, "solution.final_result"), (self.solution.final_summary, "solution.final_summary")):
            if value is not None and len(value) > limits.max_solution_length:
                raise DatasetRecordValidationError(f"{name} exceeds schema limit")
        self._validate_serialized_size(self.verification.to_dict(), limits, "verification", limits.max_verification_bytes)
        self._validate_serialized_size(self.evaluation.to_dict(), limits, "evaluation", limits.max_evaluation_bytes)
        self._validate_serialized_size(self.metadata, limits, "metadata", limits.max_metadata_bytes)
        self._validate_serialized_size(self.to_dict(), limits, "dataset record", limits.max_total_serialized_bytes)

    @staticmethod
    def _validate_serialized_size(value: Any, limits: DatasetRecordLimits, name: str = "dataset record", bound: int | None = None) -> None:
        encoded = json.dumps(_thaw(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        maximum = bound if bound is not None else limits.max_total_serialized_bytes
        if len(encoded) > maximum:
            raise DatasetRecordValidationError(f"{name} exceeds schema byte limit")


def derive_dataset_record_id(experience_id: str, source_schema_version: str) -> str:
    """Derive stable identity from canonical schema version and source identity."""

    _require_text(experience_id, "experience_id", 256)
    _require_text(source_schema_version, "source_schema_version", 64)
    digest = hashlib.sha256(f"{DATASET_RECORD_SCHEMA_VERSION}|{experience_id}|{source_schema_version}".encode("utf-8")).hexdigest()[:24]
    return f"{DATASET_RECORD_ID_PREFIX}{digest}"


def validate_dataset_record(payload: Any, *, limits: DatasetRecordLimits | None = None) -> DatasetSchemaValidationResult:
    return DatasetRecord.validate_payload(payload, limits=limits)


def _coerce_outcome(value: Any) -> DatasetOutcome:
    if isinstance(value, DatasetOutcome):
        return value
    try:
        return DatasetOutcome(value)
    except (TypeError, ValueError) as exc:
        raise DatasetRecordValidationError("outcome is not a supported DatasetOutcome") from exc


def _validate_attempts(attempts: tuple[Mapping[str, Any], ...]) -> None:
    seen: set[str] = set()
    allowed = {"attempt_id", "started_at", "completed_at", "status", "actions", "observations", "errors", "corrections", "result"}
    for index, item in enumerate(attempts):
        _expect_mapping(item, f"trajectory.attempts[{index}]")
        _expect_fields(item, allowed, f"trajectory.attempts[{index}]")
        attempt_id = item.get("attempt_id")
        _require_text(attempt_id, f"trajectory.attempts[{index}].attempt_id", 256)
        if attempt_id in seen:
            raise DatasetRecordValidationError("duplicate trajectory attempt_id")
        seen.add(attempt_id)
        _require_timestamp(item.get("started_at"), f"trajectory.attempts[{index}].started_at")
        _optional_timestamp(item.get("completed_at"), f"trajectory.attempts[{index}].completed_at")
        _require_text(item.get("status"), f"trajectory.attempts[{index}].status", 128)
        _optional_text(item.get("result"), f"trajectory.attempts[{index}].result", 65_536)
        for name, keys, id_key in (
            ("actions", {"action_id", "attempt_id", "name", "summary", "timestamp", "status", "metadata"}, "action_id"),
            ("observations", {"observation_id", "attempt_id", "summary", "source", "timestamp", "metadata"}, "observation_id"),
            ("errors", {"error_id", "attempt_id", "category", "summary", "source", "timestamp", "status"}, "error_id"),
            ("corrections", {"correction_id", "attempt_id", "error_id", "summary", "outcome", "timestamp"}, "correction_id"),
        ):
            raw = item.get(name)
            if not isinstance(raw, (list, tuple)):
                raise DatasetRecordValidationError(f"trajectory.attempts[{index}].{name} must be an array")
            _validate_events(tuple(raw), f"trajectory.attempts[{index}].{name}", keys, id_key)


def _validate_events(events: tuple[Mapping[str, Any], ...], name: str, allowed: set[str], id_key: str) -> None:
    if not isinstance(events, tuple):
        raise DatasetRecordValidationError(f"{name} must be an immutable tuple")
    if len(events) > 4_096:
        raise DatasetRecordValidationError(f"{name} exceeds schema bound")
    seen: set[str] = set()
    for index, item in enumerate(events):
        _expect_mapping(item, f"{name}[{index}]")
        _expect_fields(item, allowed, f"{name}[{index}]")
        identifier = item.get(id_key)
        _require_text(identifier, f"{name}[{index}].{id_key}", 256)
        if identifier in seen:
            raise DatasetRecordValidationError(f"duplicate {name} identifier")
        seen.add(identifier)
        _validate_event_fields(item, name, index)


def _validate_event_fields(item: Mapping[str, Any], name: str, index: int) -> None:
    for key, value in item.items():
        field_name = f"{name}[{index}].{key}"
        if key in {"action_id", "observation_id", "correction_id", "attempt_id", "name", "summary", "source", "category", "status", "outcome", "event_id", "event_type"}:
            _require_text(value, field_name, 65_536)
        elif key in {"timestamp", "started_at", "completed_at"}:
            if key == "completed_at":
                _optional_timestamp(value, field_name)
            else:
                _require_timestamp(value, field_name)
        elif key == "error_id":
            if value is not None:
                _require_text(value, field_name, 256)
        elif key in {"metadata", "details"}:
            _validate_json_value(value, field_name, 0, 5)
        elif key == "result":
            _optional_text(value, field_name, 65_536)
        else:
            _validate_json_value(value, field_name, 0, 5)


def _expect_mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise DatasetRecordValidationError(f"{name} must be an object")


def _expect_fields(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(str(key) for key in value.keys() if key not in allowed)
    missing = sorted(key for key in allowed if key not in value)
    if unknown:
        raise DatasetRecordValidationError(f"{name} contains unknown fields: {','.join(unknown)}")
    if missing:
        raise DatasetRecordValidationError(f"{name} is missing required fields: {','.join(missing)}")


def _require_text(value: Any, name: str, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DatasetRecordValidationError(f"{name} must be a non-empty string")
    if len(value) > maximum:
        raise DatasetRecordValidationError(f"{name} exceeds maximum length")


def _optional_text(value: Any, name: str, maximum: int) -> None:
    if value is not None:
        _require_text(value, name, maximum)


def _require_timestamp(value: Any, name: str) -> None:
    _require_text(value, name, 128)
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DatasetRecordValidationError(f"{name} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise DatasetRecordValidationError(f"{name} must include a timezone")


def _optional_timestamp(value: Any, name: str) -> None:
    if value is not None:
        _require_timestamp(value, name)


def _validate_json_value(value: Any, name: str, depth: int, maximum_depth: int) -> None:
    if depth > maximum_depth:
        raise DatasetRecordValidationError(f"{name} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetRecordValidationError(f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise DatasetRecordValidationError(f"{name} contains an invalid object key")
            _validate_json_value(item, f"{name}.{key}", depth + 1, maximum_depth)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{name}[{index}]", depth + 1, maximum_depth)
        return
    raise DatasetRecordValidationError(f"{name} contains an unsupported value type")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _assert_safe_record(record: DatasetRecord) -> None:
    payload = _canonical_json(record.to_dict())
    if _contains_prohibited_secret(payload):
        raise DatasetRecordValidationError("dataset record contains prohibited secret material")


__all__ = [
    "DATASET_RECORD_FORMAT",
    "DATASET_RECORD_ID_PREFIX",
    "DATASET_RECORD_SCHEMA_VERSION",
    "DatasetEvaluation",
    "DatasetOutcome",
    "DatasetProjectContext",
    "DatasetRecord",
    "DatasetRecordLimits",
    "DatasetRecordProvenance",
    "DatasetRecordValidationError",
    "DatasetSchemaError",
    "DatasetSchemaValidationResult",
    "DatasetSolution",
    "DatasetTrajectory",
    "DatasetVerification",
    "derive_dataset_record_id",
    "validate_dataset_record",
]
