"""Deterministic, read-only validation for canonical dataset records and splits.

Phase 10.5 validates the outputs of the Dataset Schema, Quality Gates, and
Dataset Splitting layers.  It reports contradictions without repairing,
deleting, persisting, or re-evaluating any source data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any

from backend_ai.agent.dataset_quality import QualityAssessment, QualityCheckStatus, QualityDecision
from backend_ai.agent.dataset_schema import (
    DATASET_RECORD_FORMAT,
    DATASET_RECORD_ID_PREFIX,
    DATASET_RECORD_SCHEMA_VERSION,
    DatasetOutcome,
    DatasetRecord,
    DatasetRecordLimits,
    DatasetRecordProvenance,
    DatasetSchemaError,
    derive_dataset_record_id,
    validate_dataset_record,
)
from backend_ai.agent.dataset_split import (
    DATASET_SPLIT_VERSION,
    DatasetSplitGroup,
    DatasetSplitManifest,
    DatasetSplitResult,
    DatasetSplitError,
    validate_split as validate_existing_split,
)
from backend_ai.agent.experience_dataset import _canonical_json, _contains_prohibited_secret


DATASET_VALIDATION_VERSION = "1.0"
_MAX_MESSAGE_DEFAULT = 512


class DatasetValidationError(ValueError):
    """Invalid validator policy or input configuration."""


class ValidationStatus(str, Enum):
    VALID = "VALID"
    VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
    INVALID = "INVALID"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DatasetDiagnosticCode(str, Enum):
    RECORD_SCHEMA_INVALID = "record_schema_invalid"
    RECORD_IDENTITY_INVALID = "record_identity_invalid"
    DUPLICATE_RECORD = "duplicate_record"
    DUPLICATE_EXPERIENCE = "duplicate_experience"
    EXACT_DUPLICATE_RECORD = "exact_duplicate_record"
    CONTRADICTORY_IDENTITY = "contradictory_identity"
    PROVENANCE_INVALID = "provenance_invalid"
    SECURITY_VIOLATION = "security_violation"
    INTERNAL_CONSISTENCY_ERROR = "internal_consistency_error"
    VERIFICATION_INCONSISTENCY = "verification_inconsistency"
    EVALUATION_INCONSISTENCY = "evaluation_inconsistency"
    MISSING_REQUIRED_DATA = "missing_required_data"
    SPLIT_MANIFEST_MISMATCH = "split_manifest_mismatch"
    PARTITION_OVERLAP = "partition_overlap"
    PARTITION_MISSING_RECORD = "partition_missing_record"
    GROUP_LEAKAGE = "group_leakage"
    EXPERIENCE_LEAKAGE = "experience_leakage"
    PROJECT_LEAKAGE = "project_leakage"
    DATASET_COUNT_MISMATCH = "dataset_count_mismatch"
    QUALITY_DECISION_MISMATCH = "quality_decision_mismatch"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    SPLIT_RATIO_DEVIATION = "split_ratio_deviation"


@dataclass(frozen=True, slots=True)
class DatasetValidationLimits:
    """Finite bounds for validation work and returned diagnostics."""

    max_records: int = 100_000
    max_diagnostics: int = 4_096
    max_diagnostic_length: int = _MAX_MESSAGE_DEFAULT
    max_total_bytes: int = 32 * 1024 * 1024
    schema_limits: DatasetRecordLimits = DatasetRecordLimits()

    def __post_init__(self) -> None:
        ceilings = {
            "max_records": 1_000_000,
            "max_diagnostics": 65_536,
            "max_diagnostic_length": 4_096,
            "max_total_bytes": 512 * 1024 * 1024,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > ceiling:
                raise DatasetValidationError(f"{name} is outside its configured bound")
        if not isinstance(self.schema_limits, DatasetRecordLimits):
            raise DatasetValidationError("schema_limits must be DatasetRecordLimits")


@dataclass(frozen=True, slots=True)
class DatasetValidationProvenance:
    """Safe provenance summary attached to a validation result."""

    record_id: str
    experience_id: str
    source_type: str
    schema_version: str
    source_schema_version: str
    project_id: str | None

    def __post_init__(self) -> None:
        for value, name in ((self.record_id, "record_id"), (self.experience_id, "experience_id"), (self.source_type, "source_type"), (self.schema_version, "schema_version"), (self.source_schema_version, "source_schema_version")):
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise DatasetValidationError(f"provenance {name} is invalid")
        if self.project_id is not None and (not isinstance(self.project_id, str) or not self.project_id.strip() or len(self.project_id) > 512):
            raise DatasetValidationError("provenance project_id is invalid")

    @classmethod
    def from_record(cls, record: DatasetRecord) -> "DatasetValidationProvenance":
        project_id = record.project_context.project_id if record.project_context else None
        return cls(record.record_id, record.experience_id, record.provenance.source_type, record.schema_version, record.provenance.source_schema_version, project_id)

    def to_dict(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "experience_id": self.experience_id, "source_type": self.source_type, "schema_version": self.schema_version, "source_schema_version": self.source_schema_version, "project_id": self.project_id}


@dataclass(frozen=True, slots=True)
class DatasetDiagnostic:
    """Machine-readable, bounded, secret-safe validation finding."""

    code: DatasetDiagnosticCode
    severity: DiagnosticSeverity
    message: str
    record_id: str | None = None
    experience_id: str | None = None
    partition: str | None = None
    path: str | None = None
    provenance: DatasetValidationProvenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, DatasetDiagnosticCode):
            object.__setattr__(self, "code", DatasetDiagnosticCode(self.code))
        if not isinstance(self.severity, DiagnosticSeverity):
            object.__setattr__(self, "severity", DiagnosticSeverity(self.severity))
        if not isinstance(self.message, str) or not self.message.strip() or len(self.message) > 4_096:
            raise DatasetValidationError("diagnostic message is invalid or unbounded")
        for value, name in ((self.record_id, "record_id"), (self.experience_id, "experience_id"), (self.partition, "partition"), (self.path, "path")):
            if value is not None and (not isinstance(value, str) or len(value) > 1_024):
                raise DatasetValidationError(f"diagnostic {name} is invalid")
        if self.provenance is not None and not isinstance(self.provenance, DatasetValidationProvenance):
            raise DatasetValidationError("diagnostic provenance is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "severity": self.severity.value, "message": self.message, "record_id": self.record_id, "experience_id": self.experience_id, "partition": self.partition, "path": self.path, "provenance": self.provenance.to_dict() if self.provenance else None}


@dataclass(frozen=True, slots=True)
class DatasetValidationResult:
    """Immutable aggregate result for one record, dataset, or split validation."""

    validation_status: ValidationStatus
    validation_version: str
    dataset_schema_version: str
    total_records: int
    valid_records: int
    invalid_records: int
    warning_count: int
    error_count: int
    diagnostics: tuple[DatasetDiagnostic, ...]
    provenance: tuple[DatasetValidationProvenance, ...]
    summary: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.validation_status, ValidationStatus):
            object.__setattr__(self, "validation_status", ValidationStatus(self.validation_status))
        if self.validation_version != DATASET_VALIDATION_VERSION:
            raise DatasetValidationError("unsupported validation version")
        if self.dataset_schema_version != DATASET_RECORD_SCHEMA_VERSION:
            raise DatasetValidationError("unsupported dataset schema version")
        for name in ("total_records", "valid_records", "invalid_records", "warning_count", "error_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise DatasetValidationError(f"{name} must be a non-negative integer")
        if not isinstance(self.diagnostics, tuple) or any(not isinstance(item, DatasetDiagnostic) for item in self.diagnostics):
            raise DatasetValidationError("diagnostics must be a tuple of DatasetDiagnostic")
        if not isinstance(self.provenance, tuple) or any(not isinstance(item, DatasetValidationProvenance) for item in self.provenance):
            raise DatasetValidationError("provenance must be a tuple of DatasetValidationProvenance")
        if not isinstance(self.summary, Mapping) or any(not isinstance(key, str) or not isinstance(value, int) or value < 0 for key, value in self.summary.items()):
            raise DatasetValidationError("summary must be a string/integer mapping")
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "summary", MappingProxyType(dict(sorted(self.summary.items()))))

    @property
    def valid(self) -> bool:
        return self.validation_status is not ValidationStatus.INVALID

    def to_dict(self) -> dict[str, Any]:
        return {"validation_status": self.validation_status.value, "validation_version": self.validation_version, "dataset_schema_version": self.dataset_schema_version, "total_records": self.total_records, "valid_records": self.valid_records, "invalid_records": self.invalid_records, "warning_count": self.warning_count, "error_count": self.error_count, "diagnostics": [item.to_dict() for item in self.diagnostics], "provenance": [item.to_dict() for item in self.provenance], "summary": dict(self.summary)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class DatasetValidator:
    """Read-only validator for canonical DatasetRecord and DatasetSplitResult values."""

    def __init__(self, *, limits: DatasetValidationLimits | None = None) -> None:
        self.limits = limits or DatasetValidationLimits()

    def validate_record(self, record: DatasetRecord | Mapping[str, Any]) -> DatasetValidationResult:
        return self.validate_dataset((record,))

    def validate_records(self, records: Sequence[DatasetRecord | Mapping[str, Any]]) -> DatasetValidationResult:
        return self.validate_dataset(records)

    def validate_split(self, split_result: DatasetSplitResult, *, records: Sequence[DatasetRecord] | None = None, quality_assessments: Sequence[QualityAssessment] | Mapping[str, QualityAssessment] | None = None) -> DatasetValidationResult:
        source_records = tuple(records) if records is not None else None
        result = self._validate_collection(source_records or (), include_split=False) if source_records is not None else self._empty_result()
        diagnostics = list(result.diagnostics)
        valid_records = result.valid_records
        invalid_records = result.invalid_records
        provenance = list(result.provenance)
        diagnostics.extend(self._validate_split_structure(split_result, source_records, quality_assessments))
        diagnostics.extend(self._validate_quality_consistency(source_records or (), split_result, quality_assessments))
        return self._result_from(diagnostics, total_records=len(source_records) if source_records is not None else _split_total(split_result), valid_records=valid_records if source_records is not None else _split_total(split_result), invalid_records=invalid_records, provenance=provenance)

    def validate_dataset(self, records: Sequence[DatasetRecord | Mapping[str, Any]], split_result: DatasetSplitResult | None = None, quality_assessments: Sequence[QualityAssessment] | Mapping[str, QualityAssessment] | None = None) -> DatasetValidationResult:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            return self._result_from([self._diagnostic(DatasetDiagnosticCode.RECORD_SCHEMA_INVALID, DiagnosticSeverity.ERROR, "records must be a sequence")], total_records=0, valid_records=0, invalid_records=0)
        collection = self._validate_collection(records, include_split=False)
        diagnostics = list(collection.diagnostics)
        canonical_records = tuple(sorted((item for item in records if isinstance(item, DatasetRecord)), key=lambda item: (item.record_id, item.experience_id)))
        diagnostics.extend(self._validate_quality_consistency(canonical_records, split_result, quality_assessments))
        if split_result is not None:
            diagnostics.extend(self._validate_split_structure(split_result, canonical_records, quality_assessments))
            diagnostics.extend(self._validate_dataset_coverage(canonical_records, split_result, quality_assessments))
        return self._result_from(diagnostics, total_records=len(records), valid_records=collection.valid_records, invalid_records=collection.invalid_records, provenance=list(collection.provenance))

    def _validate_collection(self, records: Sequence[DatasetRecord | Mapping[str, Any]], *, include_split: bool) -> DatasetValidationResult:
        raw_values = tuple(records)
        diagnostics: list[DatasetDiagnostic] = []
        canonical: list[DatasetRecord] = []
        total_bytes = 0
        limit_hit = False
        ordered_inputs = tuple(sorted(raw_values, key=_input_sort_key))
        for index, raw in enumerate(ordered_inputs):
            if index >= self.limits.max_records:
                limit_hit = True
                break
            record, record_diagnostics = self._canonicalize_and_validate(raw)
            diagnostics.extend(record_diagnostics)
            if record is not None:
                canonical.append(record)
                try:
                    total_bytes += len(_canonical_json(record.to_dict()).encode("utf-8"))
                except Exception:
                    pass
                if total_bytes > self.limits.max_total_bytes:
                    limit_hit = True
                    break
        if len(raw_values) > self.limits.max_records:
            limit_hit = True
        if limit_hit:
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.RESOURCE_LIMIT_EXCEEDED, DiagnosticSeverity.ERROR, "validation resource limit exceeded"))
        diagnostics.extend(self._dataset_duplicates(canonical))
        provenance = tuple(DatasetValidationProvenance.from_record(record) for record in sorted(canonical, key=lambda item: (item.record_id, item.experience_id)))
        invalid_ids = {item.record_id for item in canonical if any(d.record_id == item.record_id and d.severity is DiagnosticSeverity.ERROR for d in diagnostics)}
        invalid_records = len(invalid_ids) + max(0, len(raw_values) - len(canonical))
        return self._result_from(diagnostics, total_records=len(raw_values), valid_records=max(0, len(canonical) - len(invalid_ids)), invalid_records=invalid_records, provenance=list(provenance))

    def _canonicalize_and_validate(self, raw: DatasetRecord | Mapping[str, Any]) -> tuple[DatasetRecord | None, list[DatasetDiagnostic]]:
        record: DatasetRecord | None = None
        try:
            if isinstance(raw, DatasetRecord):
                record = raw
                payload = record.to_dict()
                schema_result = validate_dataset_record(payload, limits=self.limits.schema_limits)
                if not schema_result.valid:
                    diagnostics = [self._diagnostic(DatasetDiagnosticCode.RECORD_SCHEMA_INVALID, DiagnosticSeverity.ERROR, _safe_message(schema_result.errors[0]), record)]
                    diagnostics.extend(self._record_checks(record))
                    return record, diagnostics
            elif isinstance(raw, Mapping):
                record = DatasetRecord.from_dict(raw, limits=self.limits.schema_limits)
            else:
                return None, [self._diagnostic(DatasetDiagnosticCode.RECORD_SCHEMA_INVALID, DiagnosticSeverity.ERROR, "input is not a DatasetRecord or mapping")]
        except (DatasetSchemaError, TypeError, ValueError) as exc:
            code = DatasetDiagnosticCode.SECURITY_VIOLATION if "secret" in str(exc).casefold() else DatasetDiagnosticCode.RECORD_SCHEMA_INVALID
            return None, [self._diagnostic(code, DiagnosticSeverity.ERROR, _safe_message(exc), raw=raw)]
        diagnostics = self._record_checks(record)
        return record, diagnostics

    def _record_checks(self, record: DatasetRecord) -> list[DatasetDiagnostic]:
        diagnostics: list[DatasetDiagnostic] = []
        payload = record.to_dict()
        if not re.fullmatch(rf"{re.escape(DATASET_RECORD_ID_PREFIX)}[0-9a-f]{{24}}", record.record_id):
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.RECORD_IDENTITY_INVALID, DiagnosticSeverity.ERROR, "record_id format is invalid", record, path="record_id"))
        try:
            expected_id = derive_dataset_record_id(record.experience_id, record.provenance.source_schema_version)
            if record.record_id != expected_id:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.RECORD_IDENTITY_INVALID, DiagnosticSeverity.ERROR, "record_id is inconsistent with canonical identity", record, path="record_id"))
        except Exception as exc:
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.RECORD_IDENTITY_INVALID, DiagnosticSeverity.ERROR, _safe_message(exc), record, path="record_id"))
        if _contains_prohibited_secret(_canonical_json(payload)):
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.SECURITY_VIOLATION, DiagnosticSeverity.ERROR, "security validation detected prohibited secret material", record))
        provenance = record.provenance
        if not isinstance(provenance, DatasetRecordProvenance) or provenance.source_type != "experience_record":
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.PROVENANCE_INVALID, DiagnosticSeverity.ERROR, "provenance source type is invalid", record, path="provenance.source_type"))
        else:
            if provenance.experience_id != record.experience_id:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.PROVENANCE_INVALID, DiagnosticSeverity.ERROR, "provenance experience_id does not match record", record, path="provenance.experience_id"))
            if provenance.original_outcome != record.outcome.value:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.PROVENANCE_INVALID, DiagnosticSeverity.ERROR, "provenance outcome does not match record outcome", record, path="provenance.original_outcome"))
            if provenance.verification_present != record.verification.present:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.PROVENANCE_INVALID, DiagnosticSeverity.ERROR, "provenance verification flag does not match record", record, path="provenance.verification_present"))
            if record.project_context and provenance.project_identity:
                source_project = provenance.project_identity.get("project_id")
                if source_project is not None and source_project != record.project_context.project_id:
                    diagnostics.append(self._diagnostic(DatasetDiagnosticCode.PROVENANCE_INVALID, DiagnosticSeverity.ERROR, "provenance project identity contradicts project_context", record, path="provenance.project_identity.project_id"))
            if not _status_outcome_consistent(provenance.original_status, record.outcome):
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.PROVENANCE_INVALID, DiagnosticSeverity.ERROR, "provenance status and outcome are inconsistent", record, path="provenance.original_status"))
        diagnostics.extend(self._consistency_checks(record))
        return diagnostics

    def _consistency_checks(self, record: DatasetRecord) -> list[DatasetDiagnostic]:
        diagnostics: list[DatasetDiagnostic] = []
        trajectory = record.trajectory
        attempts = {str(item.get("attempt_id")): item for item in trajectory.attempts}
        if len(attempts) != len(trajectory.attempts):
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.INTERNAL_CONSISTENCY_ERROR, DiagnosticSeverity.ERROR, "attempt identifiers are duplicated", record, path="trajectory.attempts"))
        previous_started: datetime | None = None
        for index, attempt in enumerate(trajectory.attempts):
            started = _parse_timestamp(attempt.get("started_at"))
            completed = _parse_timestamp(attempt.get("completed_at")) if attempt.get("completed_at") else None
            if started is None or (completed is not None and completed < started):
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.INTERNAL_CONSISTENCY_ERROR, DiagnosticSeverity.ERROR, "attempt timestamps are invalid or out of order", record, path=f"trajectory.attempts[{index}]"))
            if previous_started is not None and started is not None and started < previous_started:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.INTERNAL_CONSISTENCY_ERROR, DiagnosticSeverity.ERROR, "attempt ordering is not chronological", record, path="trajectory.attempts"))
            previous_started = started or previous_started
        attempt_ids = set(attempts)
        error_ids = {str(item.get("error_id")) for item in trajectory.errors}
        for collection_name in ("actions", "observations", "errors", "corrections"):
            values = getattr(trajectory, collection_name)
            prior: datetime | None = None
            seen_ids: set[str] = set()
            id_key = {"actions": "action_id", "observations": "observation_id", "errors": "error_id", "corrections": "correction_id"}[collection_name]
            for index, event in enumerate(values):
                identifier = str(event.get(id_key))
                if identifier in seen_ids:
                    diagnostics.append(self._diagnostic(DatasetDiagnosticCode.INTERNAL_CONSISTENCY_ERROR, DiagnosticSeverity.ERROR, f"duplicate {collection_name} identifier", record, path=f"trajectory.{collection_name}[{index}].{id_key}"))
                seen_ids.add(identifier)
                if str(event.get("attempt_id")) not in attempt_ids:
                    diagnostics.append(self._diagnostic(DatasetDiagnosticCode.INTERNAL_CONSISTENCY_ERROR, DiagnosticSeverity.ERROR, "event references a missing attempt", record, path=f"trajectory.{collection_name}[{index}].attempt_id"))
                timestamp = _parse_timestamp(event.get("timestamp"))
                if timestamp is None or (prior is not None and timestamp < prior):
                    diagnostics.append(self._diagnostic(DatasetDiagnosticCode.INTERNAL_CONSISTENCY_ERROR, DiagnosticSeverity.ERROR, "event ordering is invalid", record, path=f"trajectory.{collection_name}[{index}].timestamp"))
                prior = timestamp or prior
            if collection_name == "corrections":
                for index, event in enumerate(values):
                    if event.get("error_id") is not None and str(event.get("error_id")) not in error_ids:
                        diagnostics.append(self._diagnostic(DatasetDiagnosticCode.INTERNAL_CONSISTENCY_ERROR, DiagnosticSeverity.ERROR, "correction references a missing error", record, path=f"trajectory.corrections[{index}].error_id"))
        verification = record.verification
        if record.outcome is DatasetOutcome.SUCCESS:
            if not verification.present:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.VERIFICATION_INCONSISTENCY, DiagnosticSeverity.ERROR, "successful outcome has no verification evidence", record, path="verification"))
            if not any(value and value.strip() for value in (record.solution.solution, record.solution.final_result, record.solution.final_summary)):
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.MISSING_REQUIRED_DATA, DiagnosticSeverity.ERROR, "successful outcome has no final solution or result", record, path="solution"))
            if verification.tests_failed and verification.tests_failed > 0:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.VERIFICATION_INCONSISTENCY, DiagnosticSeverity.ERROR, "successful outcome contains failed verification tests", record, path="verification.tests_failed"))
            if (verification.test_status or "").casefold() in {"fail", "failed", "error"}:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.VERIFICATION_INCONSISTENCY, DiagnosticSeverity.ERROR, "successful outcome has failed verification status", record, path="verification.test_status"))
        if record.outcome is DatasetOutcome.FAILURE and verification.present and verification.tests_failed == 0 and (verification.test_status or "").casefold() in {"pass", "passed", "success", "ok"}:
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.VERIFICATION_INCONSISTENCY, DiagnosticSeverity.ERROR, "failed outcome claims successful verification without explicit failure semantics", record, path="verification"))
        evaluation = record.evaluation
        if evaluation.present and (evaluation.score is None or not 0.0 <= float(evaluation.score) <= 1.0):
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.EVALUATION_INCONSISTENCY, DiagnosticSeverity.ERROR, "evaluation score is outside [0, 1]", record, path="evaluation.score"))
        return diagnostics

    def _dataset_duplicates(self, records: Sequence[DatasetRecord]) -> list[DatasetDiagnostic]:
        diagnostics: list[DatasetDiagnostic] = []
        ordered = tuple(sorted(records, key=lambda item: (item.record_id, item.experience_id, _record_fingerprint(item))))
        by_id: dict[str, DatasetRecord] = {}
        by_experience: dict[str, DatasetRecord] = {}
        by_fingerprint: dict[str, DatasetRecord] = {}
        for record in ordered:
            prior_id = by_id.get(record.record_id)
            if prior_id is not None:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.DUPLICATE_RECORD, DiagnosticSeverity.ERROR, "duplicate record_id detected", record, path="record_id"))
                if _record_fingerprint(prior_id) != _record_fingerprint(record):
                    diagnostics.append(self._diagnostic(DatasetDiagnosticCode.CONTRADICTORY_IDENTITY, DiagnosticSeverity.ERROR, "same record_id has contradictory canonical payloads", record, path="record_id"))
            else:
                by_id[record.record_id] = record
            prior_experience = by_experience.get(record.experience_id)
            if prior_experience is not None and prior_experience.record_id != record.record_id:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.DUPLICATE_EXPERIENCE, DiagnosticSeverity.ERROR, "duplicate experience_id detected", record, path="experience_id"))
            else:
                by_experience[record.experience_id] = record
            fingerprint = _record_fingerprint(record)
            if fingerprint in by_fingerprint:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.EXACT_DUPLICATE_RECORD, DiagnosticSeverity.ERROR, "exact canonical duplicate detected", record))
            else:
                by_fingerprint[fingerprint] = record
        return diagnostics

    def _validate_split_structure(self, split_result: DatasetSplitResult | None, source_records: Sequence[DatasetRecord] | None, quality_assessments: Sequence[QualityAssessment] | Mapping[str, QualityAssessment] | None) -> list[DatasetDiagnostic]:
        if not isinstance(split_result, DatasetSplitResult):
            return [self._diagnostic(DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH, DiagnosticSeverity.ERROR, "split_result is not a DatasetSplitResult")]
        diagnostics: list[DatasetDiagnostic] = []
        try:
            validate_existing_split(split_result)
        except (DatasetSplitError, TypeError, ValueError) as exc:
            diagnostics.append(self._diagnostic(_split_error_code(str(exc)), DiagnosticSeverity.ERROR, _safe_message(exc)))
        partitions = {"train": split_result.train, "validation": split_result.validation, "test": split_result.test}
        all_ids: list[str] = []
        for partition, values in partitions.items():
            for record in values:
                if not isinstance(record, DatasetRecord):
                    diagnostics.append(self._diagnostic(DatasetDiagnosticCode.RECORD_SCHEMA_INVALID, DiagnosticSeverity.ERROR, "split contains a non-canonical record", partition=partition))
                    continue
                all_ids.append(record.record_id)
                diagnostics.extend(self._record_checks(record))
        if len(all_ids) != len(set(all_ids)):
            counts = Counter(all_ids)
            for record_id, count in sorted(counts.items()):
                if count > 1:
                    diagnostics.append(self._diagnostic(DatasetDiagnosticCode.PARTITION_OVERLAP, DiagnosticSeverity.ERROR, "record appears in multiple split partitions", record_id=record_id))
        manifest = split_result.manifest
        if not isinstance(manifest, DatasetSplitManifest):
            return diagnostics + [self._diagnostic(DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH, DiagnosticSeverity.ERROR, "split manifest is invalid")]
        actual_counts = {"total": len(all_ids), "train": len(split_result.train), "validation": len(split_result.validation), "test": len(split_result.test)}
        expected_counts = {"total": manifest.total_records, "train": manifest.train_count, "validation": manifest.validation_count, "test": manifest.test_count}
        if actual_counts != expected_counts:
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.DATASET_COUNT_MISMATCH, DiagnosticSeverity.ERROR, "split manifest counts do not match actual partitions"))
        for partition in ("train", "validation", "test"):
            actual_ids = tuple(record.record_id for record in partitions[partition])
            expected_ids = tuple(manifest.record_ids.get(partition, ()))
            if actual_ids != expected_ids:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH, DiagnosticSeverity.ERROR, "split manifest record IDs do not match partition", partition=partition))
        if manifest.split_version != DATASET_SPLIT_VERSION or manifest.schema_version != DATASET_RECORD_SCHEMA_VERSION or manifest.seed < 0:
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH, DiagnosticSeverity.ERROR, "split manifest version, seed, or schema metadata is invalid"))
        if not _ratios_consistent(manifest.requested_ratios) or not _ratios_consistent(manifest.actual_ratios, allow_zero=True):
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH, DiagnosticSeverity.ERROR, "split manifest ratios are invalid"))
        else:
            for partition in ("train", "validation", "test"):
                actual_ratio = manifest.actual_ratios.get(partition, 0.0)
                calculated = actual_counts[partition] / actual_counts["total"] if actual_counts["total"] else 0.0
                if abs(float(actual_ratio) - calculated) > 1e-9:
                    diagnostics.append(self._diagnostic(DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH, DiagnosticSeverity.ERROR, "split manifest actual ratio does not match count", partition=partition))
                requested = float(manifest.requested_ratios.get(partition, 0.0))
                if abs(actual_ratio - requested) > 1e-9 and manifest.group_by is not DatasetSplitGroup.RECORD:
                    diagnostics.append(self._diagnostic(DatasetDiagnosticCode.SPLIT_RATIO_DEVIATION, DiagnosticSeverity.WARNING, "grouped split actual ratio differs from requested ratio", partition=partition))
        expected_groups = {partition: tuple(_group_key(record, manifest.group_by) for record in partitions[partition]) for partition in partitions}
        for partition in partitions:
            actual_group_ids = tuple(dict.fromkeys(expected_groups[partition]))
            if tuple(manifest.group_ids.get(partition, ())) != actual_group_ids:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH, DiagnosticSeverity.ERROR, "split manifest group IDs do not match partition", partition=partition))
        if manifest.group_by is not DatasetSplitGroup.RECORD:
            seen_groups: dict[str, str] = {}
            for partition, values in partitions.items():
                for record in values:
                    group_key = _group_key(record, manifest.group_by)
                    prior = seen_groups.get(group_key)
                    if prior is not None and prior != partition:
                        code = DatasetDiagnosticCode.PROJECT_LEAKAGE if manifest.group_by is DatasetSplitGroup.PROJECT else DatasetDiagnosticCode.EXPERIENCE_LEAKAGE
                        diagnostics.append(self._diagnostic(code, DiagnosticSeverity.ERROR, "group appears in multiple split partitions", record, partition=partition))
                    seen_groups[group_key] = partition
        return diagnostics

    def _validate_dataset_coverage(self, records: Sequence[DatasetRecord], split_result: DatasetSplitResult, quality_assessments: Sequence[QualityAssessment] | Mapping[str, QualityAssessment] | None) -> list[DatasetDiagnostic]:
        source_ids = {record.record_id for record in records}
        eligible_ids = source_ids
        if quality_assessments is not None:
            assessment_map = _assessment_map(quality_assessments)
            eligible_ids = {record_id for record_id in source_ids if assessment_map.get(record_id) is not None and assessment_map[record_id].decision is QualityDecision.ACCEPT}
        split_ids = {record.record_id for partition in (split_result.train, split_result.validation, split_result.test) for record in partition}
        diagnostics: list[DatasetDiagnostic] = []
        for record_id in sorted(eligible_ids - split_ids):
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.PARTITION_MISSING_RECORD, DiagnosticSeverity.ERROR, "eligible record is missing from split partitions", record_id=record_id))
        for record_id in sorted(split_ids - eligible_ids):
            diagnostics.append(self._diagnostic(DatasetDiagnosticCode.QUALITY_DECISION_MISMATCH if quality_assessments is not None else DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH, DiagnosticSeverity.ERROR, "split contains a record outside the eligible dataset", record_id=record_id))
        return diagnostics

    def _validate_quality_consistency(self, records: Sequence[DatasetRecord], split_result: DatasetSplitResult | None, quality_assessments: Sequence[QualityAssessment] | Mapping[str, QualityAssessment] | None) -> list[DatasetDiagnostic]:
        if quality_assessments is None:
            return []
        diagnostics: list[DatasetDiagnostic] = []
        assessment_map = _assessment_map(quality_assessments, diagnostics=diagnostics)
        record_map = {record.record_id: record for record in records}
        split_ids = {record.record_id for partition in (split_result.train, split_result.validation, split_result.test) for record in partition} if isinstance(split_result, DatasetSplitResult) else set()
        for record_id, assessment in sorted(assessment_map.items()):
            record = record_map.get(record_id)
            if record is None:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.QUALITY_DECISION_MISMATCH, DiagnosticSeverity.ERROR, "quality assessment does not match a supplied record", record_id=record_id))
                continue
            if assessment.experience_id != record.experience_id or assessment.provenance is None or assessment.provenance.experience_id != record.experience_id:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.QUALITY_DECISION_MISMATCH, DiagnosticSeverity.ERROR, "quality assessment provenance does not match DatasetRecord", record))
            if not 0.0 <= float(assessment.score.final_score) <= 1.0:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.QUALITY_DECISION_MISMATCH, DiagnosticSeverity.ERROR, "quality score is outside [0, 1]", record))
            hard_failures = [item for item in assessment.checks if item.status is QualityCheckStatus.FAIL and item.hard_gate]
            if assessment.decision is QualityDecision.ACCEPT and hard_failures:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.QUALITY_DECISION_MISMATCH, DiagnosticSeverity.ERROR, "ACCEPT assessment contains a hard-gate failure", record))
            if record_id in split_ids and assessment.decision is not QualityDecision.ACCEPT:
                diagnostics.append(self._diagnostic(DatasetDiagnosticCode.QUALITY_DECISION_MISMATCH, DiagnosticSeverity.ERROR, "non-ACCEPT quality decision is present in eligible split", record))
        return diagnostics

    def _result_from(self, diagnostics: Sequence[DatasetDiagnostic], *, total_records: int, valid_records: int, invalid_records: int, provenance: Sequence[DatasetValidationProvenance] = ()) -> DatasetValidationResult:
        ordered = sorted(diagnostics, key=_diagnostic_sort_key)
        if len(ordered) > self.limits.max_diagnostics:
            ordered = ordered[: max(0, self.limits.max_diagnostics - 1)]
            ordered.append(self._diagnostic(DatasetDiagnosticCode.RESOURCE_LIMIT_EXCEEDED, DiagnosticSeverity.ERROR, "diagnostic limit exceeded"))
        warning_count = sum(item.severity is DiagnosticSeverity.WARNING for item in ordered)
        error_count = sum(item.severity is DiagnosticSeverity.ERROR for item in ordered)
        if error_count:
            status = ValidationStatus.INVALID
        elif warning_count:
            status = ValidationStatus.VALID_WITH_WARNINGS
        else:
            status = ValidationStatus.VALID
        summary = {"diagnostics": len(ordered), "errors": error_count, "warnings": warning_count, "info": len(ordered) - error_count - warning_count}
        return DatasetValidationResult(status, DATASET_VALIDATION_VERSION, DATASET_RECORD_SCHEMA_VERSION, total_records, valid_records, invalid_records, warning_count, error_count, tuple(ordered), tuple(sorted(provenance, key=lambda item: (item.record_id, item.experience_id))), summary)

    def _empty_result(self) -> DatasetValidationResult:
        return self._result_from([], total_records=0, valid_records=0, invalid_records=0)

    def _diagnostic(self, code: DatasetDiagnosticCode, severity: DiagnosticSeverity, message: str, record: DatasetRecord | None = None, *, record_id: str | None = None, partition: str | None = None, path: str | None = None, raw: Any = None) -> DatasetDiagnostic:
        safe_message = _safe_message(message)[: self.limits.max_diagnostic_length]
        provenance = DatasetValidationProvenance.from_record(record) if record is not None else None
        return DatasetDiagnostic(code, severity, safe_message, record.record_id if record is not None else record_id, record.experience_id if record is not None else _raw_experience_id(raw), partition, path, provenance)


def validate_record(record: DatasetRecord | Mapping[str, Any], *, limits: DatasetValidationLimits | None = None) -> DatasetValidationResult:
    return DatasetValidator(limits=limits).validate_record(record)


def validate_records(records: Sequence[DatasetRecord | Mapping[str, Any]], *, limits: DatasetValidationLimits | None = None) -> DatasetValidationResult:
    return DatasetValidator(limits=limits).validate_records(records)


def validate_split(split_result: DatasetSplitResult, *, records: Sequence[DatasetRecord] | None = None, quality_assessments: Sequence[QualityAssessment] | Mapping[str, QualityAssessment] | None = None, limits: DatasetValidationLimits | None = None) -> DatasetValidationResult:
    return DatasetValidator(limits=limits).validate_split(split_result, records=records, quality_assessments=quality_assessments)


def validate_dataset(records: Sequence[DatasetRecord | Mapping[str, Any]], split_result: DatasetSplitResult | None = None, quality_assessments: Sequence[QualityAssessment] | Mapping[str, QualityAssessment] | None = None, *, limits: DatasetValidationLimits | None = None) -> DatasetValidationResult:
    return DatasetValidator(limits=limits).validate_dataset(records, split_result=split_result, quality_assessments=quality_assessments)


def _record_fingerprint(record: DatasetRecord) -> str:
    return hashlib.sha256(_canonical_json(record.to_dict()).encode("utf-8")).hexdigest()


def _input_sort_key(value: Any) -> tuple[str, str, str]:
    if isinstance(value, DatasetRecord):
        return (value.record_id, value.experience_id, _record_fingerprint(value))
    if isinstance(value, Mapping):
        record_id = str(value.get("record_id", ""))
        experience_id = str(value.get("experience_id", ""))
        try:
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        except Exception:
            serialized = repr(value)
        return (record_id, experience_id, hashlib.sha256(serialized.encode("utf-8")).hexdigest())
    return ("", "", repr(type(value)))


def _diagnostic_sort_key(item: DatasetDiagnostic) -> tuple[str, str, str, str, str]:
    return (item.record_id or "", item.experience_id or "", item.partition or "", item.code.value, item.path or "")


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _status_outcome_consistent(status: str, outcome: DatasetOutcome) -> bool:
    expected = {DatasetOutcome.SUCCESS: {"completed"}, DatasetOutcome.FAILURE: {"failed"}, DatasetOutcome.CANCELLED: {"cancelled"}}
    return status in expected[outcome]


def _group_key(record: DatasetRecord, mode: DatasetSplitGroup) -> str:
    if mode is DatasetSplitGroup.RECORD:
        return f"record:{record.record_id}"
    if mode is DatasetSplitGroup.EXPERIENCE:
        return f"experience:{record.experience_id}"
    project_id = record.project_context.project_id if record.project_context else None
    return f"project:{project_id}" if project_id else f"record:{record.record_id}"


def _ratios_consistent(values: Mapping[str, Any], *, allow_zero: bool = False) -> bool:
    if set(values) != {"train", "validation", "test"}:
        return False
    try:
        numbers = [float(values[name]) for name in ("train", "validation", "test")]
    except (TypeError, ValueError):
        return False
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in numbers):
        return False
    return abs(sum(numbers) - (0.0 if allow_zero and sum(numbers) == 0.0 else 1.0)) <= 1e-9


def _split_error_code(message: str) -> DatasetDiagnosticCode:
    lowered = message.casefold()
    if "overlap" in lowered or "duplicate record" in lowered:
        return DatasetDiagnosticCode.PARTITION_OVERLAP
    if "group" in lowered:
        return DatasetDiagnosticCode.GROUP_LEAKAGE
    if "manifest total" in lowered or "count" in lowered:
        return DatasetDiagnosticCode.DATASET_COUNT_MISMATCH
    return DatasetDiagnosticCode.SPLIT_MANIFEST_MISMATCH


def _assessment_map(values: Sequence[QualityAssessment] | Mapping[str, QualityAssessment], *, diagnostics: list[DatasetDiagnostic] | None = None) -> dict[str, QualityAssessment]:
    if isinstance(values, Mapping):
        items = tuple(values.values())
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        items = tuple(values)
    else:
        if diagnostics is not None:
            diagnostics.append(DatasetDiagnostic(DatasetDiagnosticCode.QUALITY_DECISION_MISMATCH, DiagnosticSeverity.ERROR, "quality_assessments must be a sequence or mapping"))
        return {}
    result: dict[str, QualityAssessment] = {}
    for item in items:
        if not isinstance(item, QualityAssessment):
            if diagnostics is not None:
                diagnostics.append(DatasetDiagnostic(DatasetDiagnosticCode.QUALITY_DECISION_MISMATCH, DiagnosticSeverity.ERROR, "quality assessment has invalid type"))
            continue
        if item.record_id in result:
            if diagnostics is not None:
                diagnostics.append(DatasetDiagnostic(DatasetDiagnosticCode.QUALITY_DECISION_MISMATCH, DiagnosticSeverity.ERROR, "duplicate quality assessment record_id", record_id=item.record_id))
            continue
        result[item.record_id] = item
    return result


def _split_total(value: DatasetSplitResult | None) -> int:
    return len(value.train) + len(value.validation) + len(value.test) if isinstance(value, DatasetSplitResult) else 0


def _raw_experience_id(raw: Any) -> str | None:
    if isinstance(raw, Mapping):
        value = raw.get("experience_id")
        return str(value) if value is not None else None
    return None


def _safe_message(value: Any) -> str:
    text = str(value).strip() or "validation_error"
    text = re.sub(r"(?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie|database_url)\s*(?:=|:)\s*[^,\s}\]]+", "[REDACTED]", text, flags=re.IGNORECASE)
    return text[:4_096]


__all__ = [
    "DATASET_VALIDATION_VERSION",
    "DatasetDiagnostic",
    "DatasetDiagnosticCode",
    "DatasetValidationError",
    "DatasetValidationLimits",
    "DatasetValidationProvenance",
    "DatasetValidationResult",
    "DatasetValidator",
    "DiagnosticSeverity",
    "ValidationStatus",
    "validate_dataset",
    "validate_record",
    "validate_records",
    "validate_split",
]
