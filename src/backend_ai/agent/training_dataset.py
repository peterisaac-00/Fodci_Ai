"""Phase 11.2 training-dataset preparation over the validated Phase 10 pipeline.

This module converts real, accepted Experience-derived ``DatasetRecord`` values
into immutable model-agnostic training examples.  It reuses the existing
extractor, schema, quality, split, validation, and versioning boundaries.  It
never trains, tokenizes, loads a model, changes model weights, or reads the
benchmark-only test partition through the training loader.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

from backend_ai.agent.dataset_quality import (
    DatasetFilteringResult,
    DatasetQualityEvaluator,
    DatasetQualityPolicy,
    QualityAssessment,
    QualityDecision,
)
from backend_ai.agent.dataset_schema import (
    DATASET_RECORD_SCHEMA_VERSION,
    DatasetRecord,
    DatasetRecordValidationError,
)
from backend_ai.agent.dataset_split import (
    DatasetSplitPolicy,
    DatasetSplitResult,
    DatasetSplitter,
    DatasetSplitError,
    DatasetSplitGroup,
    validate_split,
)
from backend_ai.agent.dataset_validator import (
    DatasetDiagnostic,
    DatasetValidationLimits,
    DatasetValidator,
    DiagnosticSeverity,
    ValidationStatus,
)
from backend_ai.agent.dataset_versioning import (
    DATASET_VERSION_NAME_PATTERN,
    DatasetVersion,
    DatasetVersionError,
    DatasetVersioner,
)
from backend_ai.agent.experience_dataset import (
    DatasetExtractionDiagnostic,
    DatasetExtractionResult,
    DatasetCandidate,
    ExperienceDatasetExtractor,
)
from backend_ai.agent.experience_records import ExperienceRecord, ExperienceRecords


TRAINING_DATASET_FORMAT = "fodci.training_dataset"
TRAINING_DATASET_METADATA_FORMAT = "fodci.training_dataset_metadata"
TRAINING_DATASET_MANIFEST_FORMAT = "fodci.training_dataset_manifest"
TRAINING_DATASET_SCHEMA_VERSION = "11.2"
TRAINING_EXAMPLE_FORMAT = "fodci.training_example"
TRAINING_EXAMPLE_SCHEMA_VERSION = "11.2"
TRAINING_DATASET_ARTIFACT_VERSION = "1.0"
_TRAINING_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_TEXT_LENGTH = 65_536
_MAX_COLLECTION = 100_000
_PARTITION_NAMES = ("train", "validation", "test")


class TrainingDatasetError(ValueError):
    """Base error for bounded training-dataset preparation and artifacts."""


class TrainingDatasetArtifactError(TrainingDatasetError):
    """Malformed or inconsistent on-disk training-dataset artifact."""


class TestSetAccessError(TrainingDatasetError):
    """Raised when a training-purpose loader requests the benchmark test set."""

    __test__ = False


class TrainingDatasetRejectionReason(str, Enum):
    """Stable reasons explaining why source data did not enter training."""

    EXTRACTION_REJECTED = "extraction_rejected"
    SCHEMA_INVALID = "schema_invalid"
    VALIDATION_INVALID = "validation_invalid"
    QUALITY_REJECTED = "quality_rejected"
    QUALITY_REVIEW = "quality_review"
    MISSING_TARGET = "missing_target"
    DUPLICATE_RECORD = "duplicate_record"
    DUPLICATE_EXAMPLE = "duplicate_example"
    RESOURCE_LIMIT = "resource_limit"


class TrainingSplit(str, Enum):
    """The only three persisted partitions in a training dataset artifact."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @classmethod
    def coerce(cls, value: "TrainingSplit | str") -> "TrainingSplit":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except (TypeError, ValueError) as exc:
            raise TrainingDatasetArtifactError("unsupported training dataset split") from exc


@dataclass(frozen=True, slots=True)
class TrainingDatasetConfig:
    """Finite processing configuration recorded in the final manifest.

    ``dataset_version`` follows the existing Phase 10.6 ``dataset-vN`` contract.
    The default split seed is intentionally distinct from the earlier examples so
    the Phase 11.2 build is explicit and reproducible.  The canonical Phase 10
    schema contains one DatasetRecord per ExperienceRecord, so record grouping
    provides exact partitions without allowing experience leakage.
    """

    dataset_version: str = "dataset-v1"
    quality_policy_version: str = "quality-10.3"
    quality_policy: DatasetQualityPolicy = field(default_factory=DatasetQualityPolicy)
    split_policy: DatasetSplitPolicy = field(
        default_factory=lambda: DatasetSplitPolicy(
            seed=2026,
            group_by=DatasetSplitGroup.RECORD,
        )
    )
    validation_limits: DatasetValidationLimits = field(default_factory=DatasetValidationLimits)
    max_training_examples: int = _MAX_COLLECTION
    created_at: str = "deterministic-build"

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_version, str) or not DATASET_VERSION_NAME_PATTERN.fullmatch(self.dataset_version):
            raise TrainingDatasetError("dataset_version must follow the existing dataset-vN contract")
        if not isinstance(self.quality_policy_version, str) or not self.quality_policy_version.strip():
            raise TrainingDatasetError("quality_policy_version must contain text")
        if not isinstance(self.quality_policy, DatasetQualityPolicy):
            raise TrainingDatasetError("quality_policy must be DatasetQualityPolicy")
        if not isinstance(self.split_policy, DatasetSplitPolicy):
            raise TrainingDatasetError("split_policy must be DatasetSplitPolicy")
        if not isinstance(self.validation_limits, DatasetValidationLimits):
            raise TrainingDatasetError("validation_limits must be DatasetValidationLimits")
        if not isinstance(self.max_training_examples, int) or isinstance(self.max_training_examples, bool) or not 0 < self.max_training_examples <= _MAX_COLLECTION:
            raise TrainingDatasetError("max_training_examples is outside the supported bound")
        if not isinstance(self.created_at, str) or not self.created_at.strip() or len(self.created_at) > 128:
            raise TrainingDatasetError("created_at must be bounded text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "quality_policy_version": self.quality_policy_version,
            "quality_policy": self.quality_policy.to_dict(),
            "split_policy": self.split_policy.to_dict(),
            "validation_limits": {
                "max_records": self.validation_limits.max_records,
                "max_diagnostics": self.validation_limits.max_diagnostics,
                "max_diagnostic_length": self.validation_limits.max_diagnostic_length,
                "max_total_bytes": self.validation_limits.max_total_bytes,
            },
            "max_training_examples": self.max_training_examples,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """A deterministic, traceable, model-agnostic instruction/target example."""

    format: str
    schema_version: str
    example_id: str
    source_record_id: str
    source_experience_id: str
    task: str
    context: Mapping[str, Any]
    input: str
    expected_behavior: str | None
    target: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.format != TRAINING_EXAMPLE_FORMAT:
            raise TrainingDatasetError("unsupported training example format")
        if self.schema_version != TRAINING_EXAMPLE_SCHEMA_VERSION:
            raise TrainingDatasetError("unsupported training example schema version")
        for value, name, maximum in (
            (self.example_id, "example_id", 128),
            (self.source_record_id, "source_record_id", 512),
            (self.source_experience_id, "source_experience_id", 512),
            (self.task, "task", _MAX_TEXT_LENGTH),
            (self.input, "input", _MAX_TEXT_LENGTH),
            (self.target, "target", _MAX_TEXT_LENGTH),
        ):
            if not isinstance(value, str) or not value.strip() or len(value) > maximum:
                raise TrainingDatasetError(f"{name} must be bounded non-empty text")
        if self.expected_behavior is not None and (not isinstance(self.expected_behavior, str) or len(self.expected_behavior) > _MAX_TEXT_LENGTH):
            raise TrainingDatasetError("expected_behavior must be bounded text or None")
        if not isinstance(self.context, Mapping) or not isinstance(self.metadata, Mapping):
            raise TrainingDatasetError("context and metadata must be mappings")
        _validate_json_value(self.context, "context", 0, 8)
        _validate_json_value(self.metadata, "metadata", 0, 8)
        if _contains_secret(_canonical_json(self.to_dict())):
            raise TrainingDatasetError("training example contains prohibited secret material")
        object.__setattr__(self, "context", _freeze(self.context))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @classmethod
    def from_record(cls, record: DatasetRecord, assessment: QualityAssessment) -> "TrainingExample":
        if not isinstance(record, DatasetRecord) or not isinstance(assessment, QualityAssessment):
            raise TrainingDatasetError("TrainingExample.from_record requires a DatasetRecord and QualityAssessment")
        if assessment.record_id != record.record_id or assessment.decision is not QualityDecision.ACCEPT:
            raise TrainingDatasetError("training example requires a matching ACCEPT quality assessment")
        target, target_source = _select_target(record)
        if target is None:
            raise TrainingDatasetError("accepted record has no usable target")
        context = {
            "project_context": record.project_context.to_dict() if record.project_context else None,
            "trajectory": record.trajectory.to_dict(),
            "verification": record.verification.to_dict(),
            "evaluation": record.evaluation.to_dict(),
        }
        expected_behavior = _select_expected_behavior(record)
        input_text = _render_input(record.task, context)
        metadata = {
            "record_schema_version": record.schema_version,
            "canonical_record_id": record.record_id,
            "source_schema_version": record.provenance.source_schema_version,
            "target_source": target_source,
            "project_id": record.project_context.project_id if record.project_context else None,
            "quality_score": assessment.score.final_score,
            "quality_checks": [item.to_dict() for item in assessment.checks],
            "quality_reasons": list(assessment.reasons),
            "quality_warnings": list(assessment.warnings),
        }
        return cls(
            TRAINING_EXAMPLE_FORMAT,
            TRAINING_EXAMPLE_SCHEMA_VERSION,
            derive_training_example_id(record.experience_id),
            record.experience_id,
            record.experience_id,
            record.task,
            context,
            input_text,
            expected_behavior,
            target,
            metadata,
        )

    @classmethod
    def from_dict(cls, payload: Any) -> "TrainingExample":
        if not isinstance(payload, Mapping):
            raise TrainingDatasetArtifactError("training example must be an object")
        allowed = {"format", "schema_version", "example_id", "source_record_id", "source_experience_id", "task", "context", "input", "expected_behavior", "target", "metadata"}
        if set(payload) != allowed:
            raise TrainingDatasetArtifactError("training example fields are missing or unknown")
        expected_id = derive_training_example_id(payload.get("source_record_id"))
        if payload.get("example_id") != expected_id:
            raise TrainingDatasetArtifactError("training example ID is not deterministic for its source record")
        if payload.get("source_record_id") != payload.get("source_experience_id"):
            raise TrainingDatasetArtifactError("training example source IDs are inconsistent")
        return cls(
            payload["format"],
            payload["schema_version"],
            payload["example_id"],
            payload["source_record_id"],
            payload["source_experience_id"],
            payload["task"],
            payload["context"],
            payload["input"],
            payload["expected_behavior"],
            payload["target"],
            payload["metadata"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "source_record_id": self.source_record_id,
            "source_experience_id": self.source_experience_id,
            "task": self.task,
            "context": _thaw(self.context),
            "input": self.input,
            "expected_behavior": self.expected_behavior,
            "target": self.target,
            "metadata": _thaw(self.metadata),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def fingerprint(self) -> str:
        return training_example_fingerprint(self)


@dataclass(frozen=True, slots=True)
class TrainingDatasetRejection:
    """One bounded, traceable rejection reason from the build pipeline."""

    source_record_id: str | None
    reason: TrainingDatasetRejectionReason
    stage: str
    message: str

    def __post_init__(self) -> None:
        if self.source_record_id is not None and (not isinstance(self.source_record_id, str) or not self.source_record_id.strip()):
            raise TrainingDatasetError("rejection source_record_id must be text or None")
        if not isinstance(self.reason, TrainingDatasetRejectionReason):
            object.__setattr__(self, "reason", TrainingDatasetRejectionReason(self.reason))
        for value, name in ((self.stage, "stage"), (self.message, "message")):
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise TrainingDatasetError(f"rejection {name} must be bounded text")

    def to_dict(self) -> dict[str, Any]:
        return {"source_record_id": self.source_record_id, "reason": self.reason.value, "stage": self.stage, "message": self.message}


@dataclass(frozen=True, slots=True)
class TrainingDatasetBuildReport:
    """Counters and safe rejection diagnostics for one deterministic build."""

    source_record_count: int
    valid_record_count: int
    accepted_record_count: int
    rejected_record_count: int
    duplicate_count: int
    training_example_count: int
    rejections: tuple[TrainingDatasetRejection, ...]

    def __post_init__(self) -> None:
        for name in ("source_record_count", "valid_record_count", "accepted_record_count", "rejected_record_count", "duplicate_count", "training_example_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise TrainingDatasetError(f"{name} must be a non-negative integer")
        if not isinstance(self.rejections, tuple) or any(not isinstance(item, TrainingDatasetRejection) for item in self.rejections):
            raise TrainingDatasetError("rejections must be an immutable tuple")

    @property
    def rejection_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(sorted(Counter(item.reason.value for item in self.rejections).items())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_record_count": self.source_record_count,
            "valid_record_count": self.valid_record_count,
            "accepted_record_count": self.accepted_record_count,
            "rejected_record_count": self.rejected_record_count,
            "duplicate_count": self.duplicate_count,
            "training_example_count": self.training_example_count,
            "rejection_counts": dict(self.rejection_counts),
            "rejections": [item.to_dict() for item in self.rejections],
        }


@dataclass(frozen=True, slots=True)
class TrainingDatasetManifest:
    """Immutable metadata/manifest contract for the final artifact."""

    format: str
    artifact_version: str
    training_schema_version: str
    dataset_version: str
    source_dataset_version: str
    source_dataset_fingerprint: str
    canonical_dataset_schema_version: str
    split_version: str
    split_seed: int
    split_grouping: str
    dataset_fingerprint: str
    number_of_source_records: int
    number_of_valid_records: int
    number_of_rejected_records: int
    number_of_duplicates: int
    number_of_training_examples: int
    train_count: int
    validation_count: int
    test_count: int
    source_record_ids: tuple[str, ...]
    accepted_record_ids: tuple[str, ...]
    rejected_record_ids: tuple[str, ...]
    duplicate_record_ids: tuple[str, ...]
    example_ids: Mapping[str, tuple[str, ...]]
    source_record_ids_by_split: Mapping[str, tuple[str, ...]]
    rejection_reasons: Mapping[str, tuple[str, ...]]
    processing_configuration: Mapping[str, Any]
    validation_summary: Mapping[str, int]
    created_at: str
    artifacts: Mapping[str, str]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.format != TRAINING_DATASET_MANIFEST_FORMAT or self.artifact_version != TRAINING_DATASET_ARTIFACT_VERSION or self.training_schema_version != TRAINING_DATASET_SCHEMA_VERSION:
            raise TrainingDatasetError("unsupported training dataset manifest version")
        if not isinstance(self.dataset_version, str) or not DATASET_VERSION_NAME_PATTERN.fullmatch(self.dataset_version) or self.source_dataset_version != self.dataset_version:
            raise TrainingDatasetError("training dataset version lineage is invalid")
        for value, name in ((self.source_dataset_fingerprint, "source_dataset_fingerprint"), (self.dataset_fingerprint, "dataset_fingerprint")):
            if not isinstance(value, str) or not _TRAINING_FINGERPRINT_PATTERN.fullmatch(value):
                raise TrainingDatasetError(f"{name} must be a sha256 fingerprint")
        if self.canonical_dataset_schema_version != DATASET_RECORD_SCHEMA_VERSION or not isinstance(self.split_version, str) or not self.split_version.strip():
            raise TrainingDatasetError("canonical dataset or split version is invalid")
        if not isinstance(self.split_seed, int) or isinstance(self.split_seed, bool) or self.split_seed < 0:
            raise TrainingDatasetError("split_seed must be a non-negative integer")
        if not isinstance(self.split_grouping, str) or not self.split_grouping.strip():
            raise TrainingDatasetError("split_grouping must contain text")
        for name in ("number_of_source_records", "number_of_valid_records", "number_of_rejected_records", "number_of_duplicates", "number_of_training_examples", "train_count", "validation_count", "test_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise TrainingDatasetError(f"{name} must be a non-negative integer")
        if self.train_count + self.validation_count + self.test_count != self.number_of_training_examples:
            raise TrainingDatasetError("partition counts do not equal number_of_training_examples")
        for name in ("source_record_ids", "accepted_record_ids", "rejected_record_ids", "duplicate_record_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or tuple(sorted(values)) != values or len(set(values)) != len(values) or any(not isinstance(item, str) or not item.strip() for item in values):
                raise TrainingDatasetError(f"{name} must be unique sorted IDs")
        if not set(self.accepted_record_ids).issubset(self.source_record_ids) or set(self.rejected_record_ids) - set(self.source_record_ids):
            raise TrainingDatasetError("source and accepted/rejected record IDs are inconsistent")
        if set(self.accepted_record_ids) & set(self.rejected_record_ids):
            raise TrainingDatasetError("accepted and rejected records overlap")
        _validate_partition_map(self.example_ids, "example_ids")
        _validate_partition_map(self.source_record_ids_by_split, "source_record_ids_by_split")
        if sum(len(value) for value in self.example_ids.values()) != self.number_of_training_examples:
            raise TrainingDatasetError("example ID manifest does not cover all examples")
        if sum(len(value) for value in self.source_record_ids_by_split.values()) != len(self.accepted_record_ids):
            raise TrainingDatasetError("source record split manifest does not cover accepted records")
        if not isinstance(self.rejection_reasons, Mapping) or any(not isinstance(key, str) or not isinstance(value, tuple) for key, value in self.rejection_reasons.items()):
            raise TrainingDatasetError("rejection_reasons must be a mapping of tuples")
        _validate_json_value(self.processing_configuration, "processing_configuration", 0, 8)
        _validate_json_value(self.validation_summary, "validation_summary", 0, 4)
        _validate_json_value(self.artifacts, "artifacts", 0, 4)
        _validate_json_value(self.metadata, "metadata", 0, 8)
        if not isinstance(self.created_at, str) or not self.created_at.strip() or len(self.created_at) > 128:
            raise TrainingDatasetError("created_at must be bounded text")
        object.__setattr__(self, "source_record_ids", tuple(self.source_record_ids))
        object.__setattr__(self, "accepted_record_ids", tuple(self.accepted_record_ids))
        object.__setattr__(self, "rejected_record_ids", tuple(self.rejected_record_ids))
        object.__setattr__(self, "duplicate_record_ids", tuple(self.duplicate_record_ids))
        object.__setattr__(self, "example_ids", _freeze(self.example_ids))
        object.__setattr__(self, "source_record_ids_by_split", _freeze(self.source_record_ids_by_split))
        object.__setattr__(self, "rejection_reasons", _freeze(self.rejection_reasons))
        object.__setattr__(self, "processing_configuration", _freeze(self.processing_configuration))
        object.__setattr__(self, "validation_summary", _freeze(self.validation_summary))
        object.__setattr__(self, "artifacts", _freeze(self.artifacts))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @classmethod
    def from_dict(cls, payload: Any) -> "TrainingDatasetManifest":
        if not isinstance(payload, Mapping):
            raise TrainingDatasetArtifactError("training dataset manifest must be an object")
        allowed = {
            "format", "artifact_version", "training_schema_version", "dataset_version", "source_dataset_version",
            "source_dataset_fingerprint", "canonical_dataset_schema_version", "split_version", "split_seed",
            "split_grouping", "dataset_fingerprint", "number_of_source_records", "number_of_valid_records",
            "number_of_rejected_records", "number_of_duplicates", "number_of_training_examples", "train_count",
            "validation_count", "test_count", "source_record_ids", "accepted_record_ids", "rejected_record_ids",
            "duplicate_record_ids", "example_ids", "source_record_ids_by_split", "rejection_reasons",
            "processing_configuration", "validation_summary", "created_at", "artifacts", "metadata",
        }
        if set(payload) != allowed:
            raise TrainingDatasetArtifactError("training dataset manifest fields are missing or unknown")
        collections = {name: tuple(payload[name]) for name in ("source_record_ids", "accepted_record_ids", "rejected_record_ids", "duplicate_record_ids")}
        example_ids = {key: tuple(value) for key, value in payload["example_ids"].items()}
        source_by_split = {key: tuple(value) for key, value in payload["source_record_ids_by_split"].items()}
        rejection_reasons = {key: tuple(value) for key, value in payload["rejection_reasons"].items()}
        return cls(
            payload["format"], payload["artifact_version"], payload["training_schema_version"], payload["dataset_version"],
            payload["source_dataset_version"], payload["source_dataset_fingerprint"], payload["canonical_dataset_schema_version"],
            payload["split_version"], payload["split_seed"], payload["split_grouping"], payload["dataset_fingerprint"],
            payload["number_of_source_records"], payload["number_of_valid_records"], payload["number_of_rejected_records"],
            payload["number_of_duplicates"], payload["number_of_training_examples"], payload["train_count"],
            payload["validation_count"], payload["test_count"], collections["source_record_ids"], collections["accepted_record_ids"],
            collections["rejected_record_ids"], collections["duplicate_record_ids"], example_ids, source_by_split,
            rejection_reasons, payload["processing_configuration"], payload["validation_summary"], payload["created_at"],
            payload["artifacts"], payload["metadata"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "artifact_version": self.artifact_version,
            "training_schema_version": self.training_schema_version,
            "dataset_version": self.dataset_version,
            "source_dataset_version": self.source_dataset_version,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "canonical_dataset_schema_version": self.canonical_dataset_schema_version,
            "split_version": self.split_version,
            "split_seed": self.split_seed,
            "split_grouping": self.split_grouping,
            "dataset_fingerprint": self.dataset_fingerprint,
            "number_of_source_records": self.number_of_source_records,
            "number_of_valid_records": self.number_of_valid_records,
            "number_of_rejected_records": self.number_of_rejected_records,
            "number_of_duplicates": self.number_of_duplicates,
            "number_of_training_examples": self.number_of_training_examples,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "test_count": self.test_count,
            "source_record_ids": list(self.source_record_ids),
            "accepted_record_ids": list(self.accepted_record_ids),
            "rejected_record_ids": list(self.rejected_record_ids),
            "duplicate_record_ids": list(self.duplicate_record_ids),
            "example_ids": {key: list(value) for key, value in self.example_ids.items()},
            "source_record_ids_by_split": {key: list(value) for key, value in self.source_record_ids_by_split.items()},
            "rejection_reasons": {key: list(value) for key, value in self.rejection_reasons.items()},
            "processing_configuration": _thaw(self.processing_configuration),
            "validation_summary": _thaw(self.validation_summary),
            "created_at": self.created_at,
            "artifacts": _thaw(self.artifacts),
            "metadata": _thaw(self.metadata),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class TrainingDatasetArtifact:
    """Immutable in-memory artifact with explicit train/validation/test data."""

    manifest: TrainingDatasetManifest
    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]
    test: tuple[TrainingExample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, TrainingDatasetManifest):
            raise TrainingDatasetArtifactError("artifact manifest is invalid")
        for name in _PARTITION_NAMES:
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, TrainingExample) for item in values):
                raise TrainingDatasetArtifactError(f"artifact {name} partition is invalid")
            if len({item.example_id for item in values}) != len(values):
                raise TrainingDatasetArtifactError(f"artifact {name} contains duplicate example IDs")
        partitions = {"train": self.train, "validation": self.validation, "test": self.test}
        all_ids = [item.example_id for values in partitions.values() for item in values]
        if len(all_ids) != len(set(all_ids)):
            raise TrainingDatasetArtifactError("artifact partitions overlap")
        for name, values in partitions.items():
            if tuple(item.example_id for item in values) != tuple(self.manifest.example_ids[name]):
                raise TrainingDatasetArtifactError(f"manifest example IDs do not match {name}")
            if tuple(sorted(item.source_record_id for item in values)) != tuple(self.manifest.source_record_ids_by_split[name]):
                raise TrainingDatasetArtifactError(f"manifest source IDs do not match {name}")
        if _artifact_fingerprint(self.train, self.validation, self.test, self.manifest.processing_configuration, self.manifest.dataset_version, self.manifest.source_dataset_version, self.manifest.source_dataset_fingerprint) != self.manifest.dataset_fingerprint:
            raise TrainingDatasetArtifactError("artifact content does not match manifest fingerprint")

    @property
    def counts(self) -> Mapping[str, int]:
        return MappingProxyType({"train": len(self.train), "validation": len(self.validation), "test": len(self.test), "total": len(self.train) + len(self.validation) + len(self.test)})

    def partition(self, split: TrainingSplit | str) -> tuple[TrainingExample, ...]:
        return getattr(self, TrainingSplit.coerce(split).value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": TRAINING_DATASET_FORMAT,
            "schema_version": TRAINING_DATASET_SCHEMA_VERSION,
            "manifest": self.manifest.to_dict(),
            "splits": {"train": [item.to_dict() for item in self.train], "validation": [item.to_dict() for item in self.validation], "test": [item.to_dict() for item in self.test]},
        }

    def write(self, directory: Path | str) -> Path:
        """Write a complete artifact with atomic JSON file replacement."""

        root = Path(directory).expanduser()
        if root.exists() and root.is_symlink():
            raise TrainingDatasetArtifactError("artifact directory must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        files = {
            "manifest": (root / "manifest.json", self.manifest.to_dict()),
            "metadata": (root / "metadata.json", {"format": TRAINING_DATASET_METADATA_FORMAT, "schema_version": TRAINING_DATASET_SCHEMA_VERSION, "dataset_version": self.manifest.dataset_version, "metadata": _thaw(self.manifest.metadata)}),
            "train": (root / "train.json", _split_payload(self.manifest, TrainingSplit.TRAIN, self.train)),
            "validation": (root / "validation.json", _split_payload(self.manifest, TrainingSplit.VALIDATION, self.validation)),
            "test": (root / "test.json", _split_payload(self.manifest, TrainingSplit.TEST, self.test)),
        }
        for _, (path, payload) in files.items():
            _atomic_write_json(path, payload)
        return root

    @classmethod
    def load(cls, directory: Path | str) -> "TrainingDatasetArtifact":
        root = Path(directory).expanduser()
        if not root.is_dir() or root.is_symlink():
            raise TrainingDatasetArtifactError("artifact directory is unavailable or unsafe")
        try:
            manifest = TrainingDatasetManifest.from_dict(_read_json(root / "manifest.json"))
            metadata = _read_json(root / "metadata.json")
            expected_metadata = {"format", "schema_version", "dataset_version", "metadata"}
            if set(metadata) != expected_metadata or metadata["format"] != TRAINING_DATASET_METADATA_FORMAT or metadata["schema_version"] != TRAINING_DATASET_SCHEMA_VERSION or metadata["dataset_version"] != manifest.dataset_version or metadata["metadata"] != _thaw(manifest.metadata):
                raise TrainingDatasetArtifactError("metadata artifact does not match manifest")
            partitions = {}
            for split in TrainingSplit:
                payload = _read_json(root / f"{split.value}.json")
                partitions[split.value] = _read_split_payload(payload, manifest, split)
            return cls(manifest, partitions["train"], partitions["validation"], partitions["test"])
        except TrainingDatasetError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise TrainingDatasetArtifactError("training dataset artifact is malformed") from exc


class TrainingDatasetLoader:
    """Purpose-aware loader that prevents training code from requesting test data."""

    @staticmethod
    def load_artifact(directory: Path | str) -> TrainingDatasetArtifact:
        return TrainingDatasetArtifact.load(directory)

    @staticmethod
    def load_split(directory: Path | str, split: TrainingSplit | str, *, purpose: str = "training") -> tuple[TrainingExample, ...]:
        selected = TrainingSplit.coerce(split)
        normalized_purpose = purpose.strip().casefold() if isinstance(purpose, str) else ""
        expected = {"training": TrainingSplit.TRAIN, "validation": TrainingSplit.VALIDATION, "benchmark": TrainingSplit.TEST}
        if normalized_purpose not in expected:
            raise TestSetAccessError("purpose must be training, validation, or benchmark")
        if selected is not expected[normalized_purpose]:
            raise TestSetAccessError(f"{normalized_purpose} loader may access only the {expected[normalized_purpose].value} split")
        root = Path(directory).expanduser()
        manifest = TrainingDatasetManifest.from_dict(_read_json(root / "manifest.json"))
        payload = _read_json(root / f"{selected.value}.json")
        return _read_split_payload(payload, manifest, selected)

    @staticmethod
    def load_for_training(directory: Path | str) -> tuple[TrainingExample, ...]:
        return TrainingDatasetLoader.load_split(directory, TrainingSplit.TRAIN, purpose="training")

    @staticmethod
    def load_for_validation(directory: Path | str) -> tuple[TrainingExample, ...]:
        return TrainingDatasetLoader.load_split(directory, TrainingSplit.VALIDATION, purpose="validation")

    @staticmethod
    def load_for_benchmark(directory: Path | str) -> tuple[TrainingExample, ...]:
        return TrainingDatasetLoader.load_split(directory, TrainingSplit.TEST, purpose="benchmark")


@dataclass(frozen=True, slots=True)
class TrainingDatasetBuildResult:
    """Final artifact plus reproducible processing report."""

    artifact: TrainingDatasetArtifact
    report: TrainingDatasetBuildReport
    source_version: DatasetVersion
    split_result: DatasetSplitResult
    validation_status: ValidationStatus

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, TrainingDatasetArtifact) or not isinstance(self.report, TrainingDatasetBuildReport) or not isinstance(self.source_version, DatasetVersion) or not isinstance(self.split_result, DatasetSplitResult) or not isinstance(self.validation_status, ValidationStatus):
            raise TrainingDatasetError("invalid training dataset build result")


class TrainingDatasetBuilder:
    """Compose the existing Experience → Dataset → release pipeline."""

    def __init__(self, *, config: TrainingDatasetConfig | None = None, versioner: DatasetVersioner | None = None) -> None:
        self.config = config or TrainingDatasetConfig()
        self.versioner = versioner or DatasetVersioner()
        self.extractor = ExperienceDatasetExtractor()
        self.quality = DatasetQualityEvaluator(policy=self.config.quality_policy)
        self.validator = DatasetValidator(limits=self.config.validation_limits)
        self.splitter = DatasetSplitter(policy=self.config.split_policy)

    def build_from_experience_records(self, records: ExperienceRecords | Sequence[ExperienceRecord]) -> TrainingDatasetBuildResult:
        if isinstance(records, ExperienceRecords):
            source_values = records.list()
        elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            source_values = tuple(records)
        else:
            raise TrainingDatasetError("records must be ExperienceRecords or a sequence of ExperienceRecord")
        extraction = self.extractor.extract_many(source_values)
        rejections = [_rejection_from_extraction(item) for item in extraction.diagnostics]
        source_ids = tuple(sorted({item.experience_id for item in extraction.candidates} | {item.experience_id for item in extraction.diagnostics if item.experience_id}))
        canonical: list[DatasetRecord] = []
        for candidate in extraction.candidates:
            try:
                canonical.append(DatasetRecord.from_candidate(candidate))
            except (DatasetRecordValidationError, TypeError, ValueError) as exc:
                rejections.append(TrainingDatasetRejection(candidate.experience_id, TrainingDatasetRejectionReason.SCHEMA_INVALID, "schema", _safe_message(exc)))
        return self._finish(tuple(canonical), source_count=extraction.inspected_count, source_ids=source_ids, rejections=rejections)

    def build_from_dataset_records(self, records: Sequence[DatasetRecord | Mapping[str, Any]]) -> TrainingDatasetBuildResult:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TrainingDatasetError("records must be a sequence of DatasetRecord or mappings")
        canonical: list[DatasetRecord] = []
        rejections: list[TrainingDatasetRejection] = []
        source_ids: set[str] = set()
        for index, raw in enumerate(records):
            candidate_id = raw.experience_id if isinstance(raw, DatasetRecord) else (str(raw.get("experience_id")) if isinstance(raw, Mapping) and raw.get("experience_id") else None)
            if candidate_id:
                source_ids.add(candidate_id)
            try:
                canonical.append(raw if isinstance(raw, DatasetRecord) else DatasetRecord.from_dict(raw, limits=self.config.quality_policy.schema_limits))
            except (TypeError, ValueError, DatasetRecordValidationError) as exc:
                rejections.append(TrainingDatasetRejection(candidate_id, TrainingDatasetRejectionReason.SCHEMA_INVALID, "schema", f"record[{index}]: {_safe_message(exc)}"))
        return self._finish(tuple(canonical), source_count=len(records), source_ids=tuple(sorted(source_ids)), rejections=rejections)

    def build_from_store(self, store: object) -> TrainingDatasetBuildResult:
        load = getattr(store, "load", None)
        if not callable(load):
            raise TrainingDatasetError("store must expose a callable load() method")
        loaded = load()
        records = getattr(loaded, "records", None)
        if not isinstance(records, ExperienceRecords):
            raise TrainingDatasetError(f"experience store cannot provide records: {getattr(loaded, 'error', 'unavailable')}")
        return self.build_from_experience_records(records)

    def _finish(self, canonical: tuple[DatasetRecord, ...], *, source_count: int, source_ids: Sequence[str], rejections: list[TrainingDatasetRejection]) -> TrainingDatasetBuildResult:
        deduplicated: list[DatasetRecord] = []
        seen_record_ids: set[str] = set()
        seen_payloads: set[str] = set()
        for record in sorted(canonical, key=lambda item: (item.record_id, item.experience_id)):
            payload_fingerprint = hashlib.sha256(_canonical_json(record.to_dict()).encode("utf-8")).hexdigest()
            if record.record_id in seen_record_ids or payload_fingerprint in seen_payloads:
                rejections.append(TrainingDatasetRejection(record.experience_id, TrainingDatasetRejectionReason.DUPLICATE_RECORD, "deduplication", "duplicate canonical DatasetRecord; first stable record retained"))
                continue
            seen_record_ids.add(record.record_id)
            seen_payloads.add(payload_fingerprint)
            deduplicated.append(record)
        canonical = tuple(deduplicated)
        validation_before_quality = self.validator.validate_records(canonical)
        invalid_ids = {item.record_id for item in validation_before_quality.diagnostics if item.severity is DiagnosticSeverity.ERROR and item.record_id}
        valid_records = tuple(record for record in canonical if record.record_id not in invalid_ids)
        for record_id in sorted(invalid_ids):
            messages = tuple(item.message for item in validation_before_quality.diagnostics if item.record_id == record_id and item.severity is DiagnosticSeverity.ERROR)
            rejections.append(TrainingDatasetRejection(record_id, TrainingDatasetRejectionReason.VALIDATION_INVALID, "validation", "; ".join(messages)[:512] or "record failed validation"))

        filtered = self.quality.filter_many(valid_records)
        assessments = {item.record_id: item for item in filtered.assessments}
        for assessment in filtered.assessments:
            if assessment.decision is QualityDecision.ACCEPT:
                continue
            reason = TrainingDatasetRejectionReason.QUALITY_REJECTED if assessment.decision is QualityDecision.REJECT else TrainingDatasetRejectionReason.QUALITY_REVIEW
            message = "; ".join(assessment.reasons or assessment.warnings) or f"quality decision: {assessment.decision.value}"
            if assessment.duplicate_of:
                reason = TrainingDatasetRejectionReason.DUPLICATE_RECORD
            rejections.append(TrainingDatasetRejection(assessment.experience_id, reason, "quality", message[:512]))

        examples: list[TrainingExample] = []
        example_records: dict[str, DatasetRecord] = {}
        for record in filtered.accepted:
            assessment = assessments[record.record_id]
            try:
                example = TrainingExample.from_record(record, assessment)
            except TrainingDatasetError as exc:
                rejections.append(TrainingDatasetRejection(record.experience_id, TrainingDatasetRejectionReason.MISSING_TARGET, "example", _safe_message(exc)))
                continue
            examples.append(example)
            example_records[example.example_id] = record
        if len(examples) > self.config.max_training_examples:
            for example in examples[self.config.max_training_examples :]:
                rejections.append(TrainingDatasetRejection(example.source_experience_id, TrainingDatasetRejectionReason.RESOURCE_LIMIT, "example", "maximum training example count exceeded"))
            examples = examples[: self.config.max_training_examples]

        seen_examples: dict[str, TrainingExample] = {}
        unique_examples: list[TrainingExample] = []
        duplicate_ids: set[str] = set()
        for example in sorted(examples, key=lambda item: item.source_record_id):
            previous = seen_examples.get(example.fingerprint)
            if previous is not None:
                duplicate_ids.add(example.source_record_id)
                rejections.append(TrainingDatasetRejection(example.source_experience_id, TrainingDatasetRejectionReason.DUPLICATE_EXAMPLE, "deduplication", f"duplicate_of:{previous.source_record_id}"))
                continue
            seen_examples[example.fingerprint] = example
            unique_examples.append(example)
        unique_records = tuple(example_records[item.example_id] for item in sorted(unique_examples, key=lambda item: item.source_record_id))
        unique_assessments = {record.record_id: assessments[record.record_id] for record in unique_records}
        split = self.splitter.split(unique_records, quality_assessments=unique_assessments)
        validate_split(split)
        final_validation = self.validator.validate_dataset(unique_records, split_result=split, quality_assessments=unique_assessments)
        if final_validation.validation_status is not ValidationStatus.VALID or final_validation.error_count or final_validation.invalid_records:
            details = "; ".join(f"{item.code.value}:{item.message}" for item in final_validation.diagnostics[:8])
            raise TrainingDatasetError(f"final training dataset validation did not produce a clean VALID result: {details or final_validation.validation_status.value}")
        source_version = self.versioner.create_version(
            self.config.dataset_version,
            unique_records,
            split,
            final_validation,
            quality_policy=self.config.quality_policy,
            quality_policy_version=self.config.quality_policy_version,
            metadata={"purpose": "phase-11.2-training-dataset", "training_schema_version": TRAINING_DATASET_SCHEMA_VERSION},
        )
        by_canonical_record = {record.record_id: by_record_example for by_record_example, record in ((example, example_records[example.example_id]) for example in unique_examples)}
        partitions = {
            "train": tuple(by_canonical_record[record.record_id] for record in split.train),
            "validation": tuple(by_canonical_record[record.record_id] for record in split.validation),
            "test": tuple(by_canonical_record[record.record_id] for record in split.test),
        }
        rejection_by_id: dict[str, list[str]] = defaultdict(list)
        for item in rejections:
            if item.source_record_id:
                rejection_by_id[item.source_record_id].append(item.reason.value)
        known_source_ids = set(source_ids) | {record.experience_id for record in canonical}
        accepted_ids = tuple(sorted(record.experience_id for record in unique_records))
        rejected_ids = tuple(sorted(known_source_ids - set(accepted_ids)))
        duplicate_record_ids = tuple(sorted({item.source_record_id for item in rejections if item.reason in {TrainingDatasetRejectionReason.DUPLICATE_RECORD, TrainingDatasetRejectionReason.DUPLICATE_EXAMPLE} and item.source_record_id}))
        rejection_reasons = {key: tuple(sorted(set(value))) for key, value in sorted(rejection_by_id.items())}
        processing_configuration = {
            "builder": "TrainingDatasetBuilder",
            "training_schema_version": TRAINING_DATASET_SCHEMA_VERSION,
            "quality_policy_version": self.config.quality_policy_version,
            "quality_policy": self.config.quality_policy.to_dict(),
            "split_policy": self.config.split_policy.to_dict(),
            "validation_limits": self.config.to_dict()["validation_limits"],
            "max_training_examples": self.config.max_training_examples,
        }
        source_record_ids_by_split = {name: tuple(sorted(example.source_experience_id for example in values)) for name, values in partitions.items()}
        example_ids = {name: tuple(item.example_id for item in values) for name, values in partitions.items()}
        final_fingerprint = _artifact_fingerprint(tuple(partitions["train"]), tuple(partitions["validation"]), tuple(partitions["test"]), processing_configuration, self.config.dataset_version, source_version.version, source_version.dataset_fingerprint)
        rejected_record_count = len(rejected_ids) + sum(1 for item in rejections if item.source_record_id is None)
        duplicate_count = len(duplicate_record_ids)
        metadata = {
            "dataset_version": self.config.dataset_version,
            "schema_version": TRAINING_DATASET_SCHEMA_VERSION,
            "number_of_source_records": source_count,
            "number_of_valid_records": len(valid_records),
            "number_of_rejected_records": rejected_record_count,
            "number_of_duplicates": duplicate_count,
            "number_of_training_examples": len(unique_examples),
            "train_count": len(partitions["train"]),
            "validation_count": len(partitions["validation"]),
            "test_count": len(partitions["test"]),
            "split_seed": self.config.split_policy.seed,
            "processing_configuration": processing_configuration,
            "dataset_fingerprint": final_fingerprint,
            "created_at": self.config.created_at,
            "source_record_ids": tuple(sorted(known_source_ids)),
            "accepted_record_ids": accepted_ids,
            "rejected_record_ids": rejected_ids,
            "rejection_reasons": rejection_reasons,
            "rejections": [item.to_dict() for item in rejections],
            "source_dataset_fingerprint": source_version.dataset_fingerprint,
        }
        manifest = TrainingDatasetManifest(
            TRAINING_DATASET_MANIFEST_FORMAT,
            TRAINING_DATASET_ARTIFACT_VERSION,
            TRAINING_DATASET_SCHEMA_VERSION,
            self.config.dataset_version,
            source_version.version,
            source_version.dataset_fingerprint,
            DATASET_RECORD_SCHEMA_VERSION,
            split.manifest.split_version,
            split.manifest.seed,
            split.manifest.group_by.value,
            final_fingerprint,
            source_count,
            len(valid_records),
            rejected_record_count,
            duplicate_count,
            len(unique_examples),
            len(partitions["train"]),
            len(partitions["validation"]),
            len(partitions["test"]),
            tuple(sorted(known_source_ids)),
            accepted_ids,
            rejected_ids,
            duplicate_record_ids,
            example_ids,
            source_record_ids_by_split,
            rejection_reasons,
            processing_configuration,
            dict(final_validation.summary),
            self.config.created_at,
            {"manifest": "manifest.json", "metadata": "metadata.json", "train": "train.json", "validation": "validation.json", "test": "test.json"},
            metadata,
        )
        artifact = TrainingDatasetArtifact(manifest, partitions["train"], partitions["validation"], partitions["test"])
        report = TrainingDatasetBuildReport(source_count, len(valid_records), len(unique_records), rejected_record_count, duplicate_count, len(unique_examples), tuple(rejections))
        return TrainingDatasetBuildResult(artifact, report, source_version, split, final_validation.validation_status)


def build_training_dataset_from_experience_records(records: ExperienceRecords | Sequence[ExperienceRecord], *, config: TrainingDatasetConfig | None = None, versioner: DatasetVersioner | None = None) -> TrainingDatasetBuildResult:
    return TrainingDatasetBuilder(config=config, versioner=versioner).build_from_experience_records(records)


def derive_training_example_id(source_record_id: str) -> str:
    if not isinstance(source_record_id, str) or not source_record_id.strip():
        raise TrainingDatasetError("source_record_id must contain text")
    digest = hashlib.sha256(f"{TRAINING_EXAMPLE_SCHEMA_VERSION}|{source_record_id}".encode("utf-8")).hexdigest()[:24]
    return f"tex-{digest}"


def training_example_fingerprint(example: TrainingExample) -> str:
    if not isinstance(example, TrainingExample):
        raise TrainingDatasetError("training_example_fingerprint requires TrainingExample")
    payload = {"schema_version": example.schema_version, "task": example.task, "context": _thaw(example.context), "input": example.input, "expected_behavior": example.expected_behavior, "target": example.target}
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _artifact_fingerprint(train: Sequence[TrainingExample], validation: Sequence[TrainingExample], test: Sequence[TrainingExample], processing_configuration: Mapping[str, Any], dataset_version: str, source_dataset_version: str, source_dataset_fingerprint: str) -> str:
    payload = {
        "identity_version": TRAINING_DATASET_ARTIFACT_VERSION,
        "training_schema_version": TRAINING_DATASET_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "source_dataset_version": source_dataset_version,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "processing_configuration": _thaw(processing_configuration),
        "splits": {name: [item.to_dict() for item in values] for name, values in (("train", train), ("validation", validation), ("test", test))},
    }
    if _contains_secret(_canonical_json(payload)):
        raise TrainingDatasetError("training dataset fingerprint input contains prohibited secret material")
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _rejection_from_extraction(diagnostic: DatasetExtractionDiagnostic) -> TrainingDatasetRejection:
    return TrainingDatasetRejection(diagnostic.experience_id, TrainingDatasetRejectionReason.EXTRACTION_REJECTED, "extraction", diagnostic.message)


def _select_target(record: DatasetRecord) -> tuple[str | None, str | None]:
    for value, source in ((record.solution.solution, "solution"), (record.solution.final_result, "final_result"), (record.solution.final_summary, "final_summary")):
        if isinstance(value, str) and value.strip():
            return value, source
    return None, None


def _select_expected_behavior(record: DatasetRecord) -> str | None:
    if record.evaluation.present and record.evaluation.summary:
        return record.evaluation.summary
    if record.verification.present and record.verification.summary:
        return record.verification.summary
    return None


def _render_input(task: str, context: Mapping[str, Any]) -> str:
    return f"Task:\n{task}\n\nEvidence context:\n{_canonical_json(context)}"


def _split_payload(manifest: TrainingDatasetManifest, split: TrainingSplit, examples: Sequence[TrainingExample]) -> dict[str, Any]:
    return {
        "format": TRAINING_DATASET_FORMAT,
        "schema_version": TRAINING_DATASET_SCHEMA_VERSION,
        "dataset_version": manifest.dataset_version,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "split": split.value,
        "example_ids": [item.example_id for item in examples],
        "examples": [item.to_dict() for item in examples],
    }


def _read_split_payload(payload: Any, manifest: TrainingDatasetManifest, split: TrainingSplit) -> tuple[TrainingExample, ...]:
    if not isinstance(payload, Mapping) or set(payload) != {"format", "schema_version", "dataset_version", "dataset_fingerprint", "split", "example_ids", "examples"}:
        raise TrainingDatasetArtifactError(f"{split.value} artifact fields are invalid")
    if payload["format"] != TRAINING_DATASET_FORMAT or payload["schema_version"] != TRAINING_DATASET_SCHEMA_VERSION or payload["dataset_version"] != manifest.dataset_version or payload["dataset_fingerprint"] != manifest.dataset_fingerprint or payload["split"] != split.value:
        raise TrainingDatasetArtifactError(f"{split.value} artifact header is invalid")
    if not isinstance(payload["example_ids"], list) or not isinstance(payload["examples"], list) or len(payload["example_ids"]) != len(payload["examples"]):
        raise TrainingDatasetArtifactError(f"{split.value} artifact examples are invalid")
    examples = tuple(TrainingExample.from_dict(item) for item in payload["examples"])
    if tuple(payload["example_ids"]) != tuple(item.example_id for item in examples) or tuple(payload["example_ids"]) != tuple(manifest.example_ids[split.value]):
        raise TrainingDatasetArtifactError(f"{split.value} artifact IDs do not match manifest")
    return examples


def _atomic_write_json(path: Path, payload: Any) -> None:
    if path.exists() and path.is_symlink():
        raise TrainingDatasetArtifactError("artifact file must not be a symlink")
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _read_json(path: Path) -> Any:
    if path.is_symlink():
        raise TrainingDatasetArtifactError("artifact files must not be symlinks")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_partition_map(value: Mapping[str, tuple[str, ...]], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_PARTITION_NAMES):
        raise TrainingDatasetError(f"{name} must contain train, validation, and test")
    all_values: list[str] = []
    for partition in _PARTITION_NAMES:
        items = value[partition]
        if not isinstance(items, tuple) or any(not isinstance(item, str) or not item.strip() for item in items):
            raise TrainingDatasetError(f"{name}.{partition} is invalid")
        all_values.extend(items)
    if len(all_values) != len(set(all_values)):
        raise TrainingDatasetError(f"{name} partitions overlap")


def _validate_json_value(value: Any, name: str, depth: int, maximum_depth: int) -> None:
    if depth > maximum_depth:
        raise TrainingDatasetError(f"{name} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise TrainingDatasetError(f"{name} contains an invalid key")
            _validate_json_value(item, f"{name}.{key}", depth + 1, maximum_depth)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_COLLECTION:
            raise TrainingDatasetError(f"{name} exceeds collection bound")
        for index, item in enumerate(value):
            _validate_json_value(item, f"{name}[{index}]", depth + 1, maximum_depth)
        return
    raise TrainingDatasetError(f"{name} contains an unsupported value type")


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


def _canonical_json(value: Any) -> str:
    return json.dumps(_thaw(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _contains_secret(value: str) -> bool:
    patterns = (
        re.compile(r"(?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie|database_url)\s*(?:=|:)\s*(?!\[REDACTED\])[^,\s}\]]+", re.IGNORECASE),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    )
    return any(pattern.search(value) for pattern in patterns)


def _safe_message(value: Any) -> str:
    text = str(value).strip() or "rejected"
    text = re.sub(r"(?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie|database_url)\s*(?:=|:)\s*[^,\s}\]]+", "[REDACTED]", text, flags=re.IGNORECASE)
    return text[:512]


__all__ = [
    "TRAINING_DATASET_ARTIFACT_VERSION",
    "TRAINING_DATASET_FORMAT",
    "TRAINING_DATASET_MANIFEST_FORMAT",
    "TRAINING_DATASET_METADATA_FORMAT",
    "TRAINING_DATASET_SCHEMA_VERSION",
    "TRAINING_EXAMPLE_FORMAT",
    "TRAINING_EXAMPLE_SCHEMA_VERSION",
    "TestSetAccessError",
    "TrainingDatasetArtifact",
    "TrainingDatasetArtifactError",
    "TrainingDatasetBuildReport",
    "TrainingDatasetBuildResult",
    "TrainingDatasetBuilder",
    "TrainingDatasetConfig",
    "TrainingDatasetError",
    "TrainingDatasetLoader",
    "TrainingDatasetManifest",
    "TrainingDatasetRejection",
    "TrainingDatasetRejectionReason",
    "TrainingExample",
    "TrainingSplit",
    "build_training_dataset_from_experience_records",
    "derive_training_example_id",
    "training_example_fingerprint",
]
