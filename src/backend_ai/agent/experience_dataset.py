"""Phase 10.1 extraction of finalized Experience Records into candidates.

This module creates a bounded, deterministic derived representation only.  It
never parses persistence files, creates or mutates experiences, performs
quality scoring, splits/version datasets, or interacts with training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from backend_ai.agent.experience_records import (
    EXPERIENCE_RECORD_SCHEMA_VERSION,
    ExperienceRecord,
    ExperienceRecordLoadStatus,
    ExperienceRecords,
    ExperienceLifecycleStatus,
    _redact_experience_text,
    _redact_experience_value,
)
from backend_ai.agent.memory_governance import MemoryGovernance
from backend_ai.agent.memory_retrieval import MemoryRetrievalItem, RetrievalSource


DATASET_CANDIDATE_SOURCE_TYPE = "experience_record"


class DatasetExtractionError(ValueError):
    """Invalid extraction input or configured extraction bound."""


class DatasetExtractionReason(str, Enum):
    """Stable diagnostic reason values without introducing a filtering taxonomy."""

    INCOMPLETE_EXPERIENCE = "incomplete_experience"
    INVALID_RECORD = "invalid_record"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    SECURITY_VIOLATION = "security_violation"
    UNAVAILABLE_SOURCE = "unavailable_source"
    MISSING_PROVENANCE = "missing_provenance"
    RESOURCE_LIMIT = "resource_limit"


@dataclass(frozen=True, slots=True)
class DatasetExtractionLimits:
    """Finite host-controlled bounds for derived candidate materialization."""

    max_records: int = 128
    max_candidate_bytes: int = 262_144
    max_total_bytes: int = 8 * 1024 * 1024
    max_diagnostic_length: int = 512

    def __post_init__(self) -> None:
        ceilings = {
            "max_records": 1_024,
            "max_candidate_bytes": 8 * 1024 * 1024,
            "max_total_bytes": 32 * 1024 * 1024,
            "max_diagnostic_length": 4_096,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > ceiling:
                raise DatasetExtractionError(f"{name} is outside its configured bound")
        if self.max_candidate_bytes > self.max_total_bytes:
            raise DatasetExtractionError("max_candidate_bytes cannot exceed max_total_bytes")


@dataclass(frozen=True, slots=True)
class DatasetCandidateProvenance:
    """Traceability from a derived candidate back to one Experience Record."""

    source_type: str
    experience_id: str
    source_schema_version: str
    started_at: str
    completed_at: str
    project_identity: Mapping[str, str | None] | None
    original_status: str
    original_outcome: str
    verification_present: bool

    def __post_init__(self) -> None:
        if self.source_type != DATASET_CANDIDATE_SOURCE_TYPE:
            raise DatasetExtractionError("candidate provenance source_type must be experience_record")
        for value, name in ((self.experience_id, "experience_id"), (self.source_schema_version, "source_schema_version"), (self.started_at, "started_at"), (self.completed_at, "completed_at"), (self.original_status, "original_status"), (self.original_outcome, "original_outcome")):
            if not isinstance(value, str) or not value.strip():
                raise DatasetExtractionError(f"provenance {name} must contain text")
        if not isinstance(self.verification_present, bool):
            raise DatasetExtractionError("verification_present must be boolean")
        if self.project_identity is not None:
            if not isinstance(self.project_identity, Mapping):
                raise DatasetExtractionError("project_identity must be a mapping or None")
            object.__setattr__(self, "project_identity", MappingProxyType({str(key): value for key, value in self.project_identity.items()}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "experience_id": self.experience_id,
            "source_schema_version": self.source_schema_version,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "project_identity": dict(self.project_identity) if self.project_identity is not None else None,
            "original_status": self.original_status,
            "original_outcome": self.original_outcome,
            "verification_present": self.verification_present,
        }


@dataclass(frozen=True, slots=True)
class DatasetCandidate:
    """Normalized intermediate representation preserving execution trajectory."""

    experience_id: str
    task: str
    project_identity: Mapping[str, str | None] | None
    attempts: tuple[Mapping[str, Any], ...]
    actions: tuple[Mapping[str, Any], ...]
    observations: tuple[Mapping[str, Any], ...]
    errors: tuple[Mapping[str, Any], ...]
    corrections: tuple[Mapping[str, Any], ...]
    final_solution: str | None
    final_summary: str | None
    verification: Mapping[str, Any] | None
    evaluation: Mapping[str, Any] | None
    outcome: str
    provenance: DatasetCandidateProvenance
    source_schema_version: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.experience_id, str) or not self.experience_id.startswith("exp-"):
            raise DatasetExtractionError("candidate experience_id must be canonical")
        if not isinstance(self.task, str) or not self.task.strip():
            raise DatasetExtractionError("candidate task must contain text")
        for name in ("attempts", "actions", "observations", "errors", "corrections"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, Mapping) for item in values):
                raise DatasetExtractionError(f"candidate {name} must be an immutable tuple of mappings")
            object.__setattr__(self, name, tuple(_freeze_mapping(item) for item in values))
        for name in ("project_identity", "verification", "evaluation", "metadata"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise DatasetExtractionError(f"candidate {name} must be a mapping or None")
            if isinstance(value, Mapping):
                object.__setattr__(self, name, _freeze_mapping(value))
        if self.final_solution is not None and not isinstance(self.final_solution, str):
            raise DatasetExtractionError("candidate final_solution must be text or None")
        if self.final_summary is not None and not isinstance(self.final_summary, str):
            raise DatasetExtractionError("candidate final_summary must be text or None")
        if not isinstance(self.outcome, str) or not self.outcome.strip():
            raise DatasetExtractionError("candidate outcome must contain text")
        if not isinstance(self.provenance, DatasetCandidateProvenance):
            raise DatasetExtractionError("candidate provenance must be DatasetCandidateProvenance")
        if self.provenance.experience_id != self.experience_id:
            raise DatasetExtractionError("candidate provenance experience_id must match candidate")
        if self.source_schema_version != self.provenance.source_schema_version:
            raise DatasetExtractionError("candidate source schema version must match provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "task": self.task,
            "project_identity": _thaw(self.project_identity),
            "attempts": [_thaw(item) for item in self.attempts],
            "actions": [_thaw(item) for item in self.actions],
            "observations": [_thaw(item) for item in self.observations],
            "errors": [_thaw(item) for item in self.errors],
            "corrections": [_thaw(item) for item in self.corrections],
            "final_solution": self.final_solution,
            "final_summary": self.final_summary,
            "verification": _thaw(self.verification),
            "evaluation": _thaw(self.evaluation),
            "outcome": self.outcome,
            "provenance": self.provenance.to_dict(),
            "source_schema_version": self.source_schema_version,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DatasetExtractionDiagnostic:
    """Bounded explanation for one skipped or non-candidate input."""

    experience_id: str | None
    reason: str
    source_status: str
    message: str

    def __post_init__(self) -> None:
        if self.experience_id is not None and (not isinstance(self.experience_id, str) or not self.experience_id.strip()):
            raise DatasetExtractionError("diagnostic experience_id must be text or None")
        if not isinstance(self.reason, DatasetExtractionReason):
            try:
                object.__setattr__(self, "reason", DatasetExtractionReason(self.reason))
            except ValueError as exc:
                raise DatasetExtractionError("diagnostic reason is unsupported") from exc
        for value, name in ((self.source_status, "source_status"), (self.message, "message")):
            if not isinstance(value, str) or not value.strip():
                raise DatasetExtractionError(f"diagnostic {name} must contain text")

    def to_dict(self) -> dict[str, Any]:
        return {"experience_id": self.experience_id, "reason": self.reason.value, "source_status": self.source_status, "message": self.message}


@dataclass(frozen=True, slots=True)
class DatasetExtractionResult:
    """Bounded batch output; valid candidates survive individual record failures."""

    candidates: tuple[DatasetCandidate, ...]
    diagnostics: tuple[DatasetExtractionDiagnostic, ...]
    inspected_count: int
    extracted_count: int
    skipped_count: int
    total_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or any(not isinstance(item, DatasetCandidate) for item in self.candidates):
            raise DatasetExtractionError("candidates must be a tuple of DatasetCandidate")
        if not isinstance(self.diagnostics, tuple) or any(not isinstance(item, DatasetExtractionDiagnostic) for item in self.diagnostics):
            raise DatasetExtractionError("diagnostics must be a tuple of DatasetExtractionDiagnostic")
        if self.inspected_count < 0 or self.extracted_count < 0 or self.skipped_count < 0 or self.total_bytes < 0:
            raise DatasetExtractionError("extraction counters must be non-negative")
        if self.extracted_count != len(self.candidates) or self.skipped_count != len(self.diagnostics):
            raise DatasetExtractionError("extraction counters must match result collections")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "inspected_count": self.inspected_count,
            "extracted_count": self.extracted_count,
            "skipped_count": self.skipped_count,
            "total_bytes": self.total_bytes,
        }


class ExperienceDatasetExtractor:
    """Extract finalized Experience Records without mutating their owner."""

    def __init__(self, *, limits: DatasetExtractionLimits | None = None, governance: MemoryGovernance | None = None) -> None:
        self.limits = limits or DatasetExtractionLimits()
        self.governance = governance or MemoryGovernance()

    def extract(self, record: ExperienceRecord) -> DatasetCandidate:
        if not isinstance(record, ExperienceRecord):
            raise DatasetExtractionError("record must be ExperienceRecord")
        candidate, diagnostic = self._extract_one(record)
        if diagnostic is not None:
            raise DatasetExtractionError(diagnostic.message)
        assert candidate is not None
        return candidate

    def extract_many(self, records: Sequence[ExperienceRecord] | ExperienceRecords) -> DatasetExtractionResult:
        source_status = "IN_MEMORY"
        if isinstance(records, ExperienceRecords):
            source_records = records.list()
        elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            source_records = tuple(records)
        else:
            raise DatasetExtractionError("records must be ExperienceRecords or a sequence of ExperienceRecord")
        if len(source_records) > self.limits.max_records:
            source_records = source_records[: self.limits.max_records]
        ordered = tuple(sorted(source_records, key=lambda item: (getattr(item, "started_at", ""), getattr(item, "experience_id", ""))))
        candidates: list[DatasetCandidate] = []
        diagnostics: list[DatasetExtractionDiagnostic] = []
        total_bytes = 0
        for record in ordered:
            if not isinstance(record, ExperienceRecord):
                diagnostics.append(DatasetExtractionDiagnostic(None, DatasetExtractionReason.INVALID_RECORD, source_status, "input is not an ExperienceRecord"))
                continue
            candidate, diagnostic = self._extract_one(record)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
                continue
            assert candidate is not None
            candidate_bytes = len(_canonical_json(candidate.to_dict()).encode("utf-8"))
            if candidate_bytes > self.limits.max_candidate_bytes or total_bytes + candidate_bytes > self.limits.max_total_bytes:
                diagnostics.append(DatasetExtractionDiagnostic(record.experience_id, DatasetExtractionReason.RESOURCE_LIMIT, source_status, "candidate exceeds the configured extraction resource bound"))
                continue
            candidates.append(candidate)
            total_bytes += candidate_bytes
        return DatasetExtractionResult(tuple(candidates), tuple(diagnostics), len(ordered), len(candidates), len(diagnostics), total_bytes)

    def extract_from_store(self, store: object) -> DatasetExtractionResult:
        """Load through an existing store API; never parses its storage path directly."""

        load = getattr(store, "load", None)
        if not callable(load):
            raise DatasetExtractionError("store must expose a callable load() API")
        try:
            loaded = load()
        except Exception as exc:
            diagnostic = DatasetExtractionDiagnostic(None, DatasetExtractionReason.UNAVAILABLE_SOURCE, "UNAVAILABLE", _safe_message(exc))
            return DatasetExtractionResult((), (diagnostic,), 0, 0, 1, 0)
        status = getattr(loaded, "status", None)
        records = getattr(loaded, "records", None)
        if status is not ExperienceRecordLoadStatus.LOADED or not isinstance(records, ExperienceRecords):
            reason = DatasetExtractionReason.UNAVAILABLE_SOURCE if status in {ExperienceRecordLoadStatus.MEMORY_MISSING, ExperienceRecordLoadStatus.MEMORY_UNAVAILABLE} else DatasetExtractionReason.INVALID_RECORD
            message = _safe_message(getattr(loaded, "error", None) or f"Experience Record source status is {getattr(status, 'value', status)}")
            diagnostic = DatasetExtractionDiagnostic(None, reason, getattr(status, "value", str(status)), message)
            return DatasetExtractionResult((), (diagnostic,), 0, 0, 1, 0)
        return self.extract_many(records)

    def _extract_one(self, record: ExperienceRecord) -> tuple[DatasetCandidate | None, DatasetExtractionDiagnostic | None]:
        if record.schema_version != EXPERIENCE_RECORD_SCHEMA_VERSION:
            return None, DatasetExtractionDiagnostic(record.experience_id, DatasetExtractionReason.UNSUPPORTED_SCHEMA, "IN_MEMORY", "unsupported Experience Record schema")
        if not record.finalized or record.status not in {ExperienceLifecycleStatus.COMPLETED, ExperienceLifecycleStatus.FAILED, ExperienceLifecycleStatus.CANCELLED}:
            return None, DatasetExtractionDiagnostic(record.experience_id, DatasetExtractionReason.INCOMPLETE_EXPERIENCE, "IN_MEMORY", "only finalized experiences can be extracted")
        if record.status is ExperienceLifecycleStatus.CANCELLED and not self._has_final_result(record):
            return None, DatasetExtractionDiagnostic(record.experience_id, DatasetExtractionReason.INCOMPLETE_EXPERIENCE, "IN_MEMORY", "cancelled experience has no sufficient final result")
        if record.completed_at is None or record.outcome is None:
            return None, DatasetExtractionDiagnostic(record.experience_id, DatasetExtractionReason.INVALID_RECORD, "IN_MEMORY", "finalized experience is missing completed_at or outcome")
        if record.metadata.get("governance_invalidated") is True:
            return None, DatasetExtractionDiagnostic(record.experience_id, DatasetExtractionReason.SECURITY_VIOLATION, "IN_MEMORY", "experience is invalidated by memory governance")
        if not self._governance_allows(record):
            return None, DatasetExtractionDiagnostic(record.experience_id, DatasetExtractionReason.SECURITY_VIOLATION, "IN_MEMORY", "experience failed the minimum governance safety check")
        try:
            candidate = self._candidate_from_record(record)
            self._assert_safe_candidate(candidate)
            return candidate, None
        except DatasetExtractionError as exc:
            return None, DatasetExtractionDiagnostic(record.experience_id, DatasetExtractionReason.SECURITY_VIOLATION, "IN_MEMORY", _safe_message(exc))
        except (TypeError, ValueError, KeyError) as exc:
            return None, DatasetExtractionDiagnostic(record.experience_id, DatasetExtractionReason.INVALID_RECORD, "IN_MEMORY", _safe_message(exc))

    @staticmethod
    def _has_final_result(record: ExperienceRecord) -> bool:
        return bool(record.final_solution or record.final_summary or any(attempt.result for attempt in record.attempts))

    def _governance_allows(self, record: ExperienceRecord) -> bool:
        """Apply only extraction-time governance safety, not Phase 10.3 quality gates."""

        item = MemoryRetrievalItem(
            RetrievalSource.EXPERIENCE_RECORDS,
            record.experience_id,
            record.task,
            0.0,
            4 if record.verification is not None and record.outcome is not None and record.outcome.value == "success" else 1,
            record.status.value,
            record.completed_at or record.started_at,
            {
                "verified": record.verification is not None,
                "project_root": record.project_identity.project_root if record.project_identity else None,
                "governance_invalidated": record.metadata.get("governance_invalidated") is True,
            },
            "historical experience dataset extraction",
            record.project_identity.project_id if record.project_identity else None,
        )
        assessment = self.governance.assess(item)
        return assessment.security_status.value == "clear" and assessment.provenance_status.value == "sufficient" and not record.metadata.get("governance_invalidated", False)

    def _candidate_from_record(self, record: ExperienceRecord) -> DatasetCandidate:
        attempts = tuple(_redact_experience_value(item.to_dict()) for item in record.attempts)
        actions = tuple(_redact_experience_value(action.to_dict()) for attempt in record.attempts for action in attempt.actions)
        observations = tuple(_redact_experience_value(observation.to_dict()) for attempt in record.attempts for observation in attempt.observations)
        errors = tuple(_redact_experience_value(error.to_dict()) for attempt in record.attempts for error in attempt.errors)
        corrections = tuple(_redact_experience_value(correction.to_dict()) for attempt in record.attempts for correction in attempt.corrections)
        project_identity = record.project_identity.to_dict() if record.project_identity else None
        provenance = DatasetCandidateProvenance(
            DATASET_CANDIDATE_SOURCE_TYPE,
            record.experience_id,
            record.schema_version,
            record.started_at,
            record.completed_at or "",
            project_identity,
            record.status.value,
            record.outcome.value if record.outcome else "",
            record.verification is not None,
        )
        return DatasetCandidate(
            experience_id=record.experience_id,
            task=record.task,
            project_identity=project_identity,
            attempts=attempts,
            actions=actions,
            observations=observations,
            errors=errors,
            corrections=corrections,
            final_solution=_redact_experience_text(record.final_solution) if record.final_solution else None,
            final_summary=_redact_experience_text(record.final_summary) if record.final_summary else None,
            verification=_redact_experience_value(record.verification.to_dict()) if record.verification else None,
            evaluation=_redact_experience_value(record.evaluation.to_dict()) if record.evaluation else None,
            outcome=record.outcome.value if record.outcome else "",
            provenance=provenance,
            source_schema_version=record.schema_version,
            metadata={"source": DATASET_CANDIDATE_SOURCE_TYPE, "experience_metadata": _redact_experience_value(record.metadata)},
        )

    @staticmethod
    def _assert_safe_candidate(candidate: DatasetCandidate) -> None:
        payload = _canonical_json(candidate.to_dict())
        if _contains_prohibited_secret(payload):
            raise DatasetExtractionError("candidate contains prohibited secret material")


def _freeze_mapping(value: Mapping[str, Any]) -> MappingProxyType:
    return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    import json

    return json.dumps(_thaw(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _contains_prohibited_secret(value: str) -> bool:
    import re

    patterns = (
        re.compile(r"(?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie|database_url)\s*(?:=|:)\s*(?!\[REDACTED\])[^,\s}\]]+", re.IGNORECASE),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    )
    return any(pattern.search(value) for pattern in patterns)


def _safe_message(value: Any) -> str:
    text = str(value)
    import re

    text = re.sub(r"(?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie|database_url)\s*(?:=|:)\s*[^,\s}\]]+", "[REDACTED]", text, flags=re.IGNORECASE)
    return text[:512]


__all__ = [
    "DATASET_CANDIDATE_SOURCE_TYPE",
    "DatasetCandidate",
    "DatasetCandidateProvenance",
    "DatasetExtractionDiagnostic",
    "DatasetExtractionError",
    "DatasetExtractionLimits",
    "DatasetExtractionReason",
    "DatasetExtractionResult",
    "ExperienceDatasetExtractor",
]
