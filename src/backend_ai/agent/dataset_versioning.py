"""Deterministic immutable Dataset Versioning for the completed Phase 10 pipeline.

This module creates explicit local dataset versions from canonical records,
validated splits, and clean validation results.  It never trains, publishes,
mutates source data, or accesses network/external systems.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from backend_ai.agent.dataset_schema import DATASET_RECORD_SCHEMA_VERSION, DatasetRecord
from backend_ai.agent.dataset_split import DATASET_SPLIT_VERSION, DatasetSplitResult, DatasetSplitError, validate_split
from backend_ai.agent.dataset_validator import (
    DatasetValidationResult,
    ValidationStatus,
)
from backend_ai.agent.experience_dataset import _canonical_json, _contains_prohibited_secret


DATASET_VERSIONING_FORMAT = "fodci.dataset_version"
DATASET_VERSIONING_SCHEMA_VERSION = "1.0"
DATASET_VERSIONING_REGISTRY_FORMAT = "fodci.dataset_version_registry"
DATASET_VERSION_NAME_PATTERN = re.compile(r"^dataset-v[0-9]+(?:\.[0-9]+)?$")
_DATASET_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_VERSION_NAME_LENGTH = 64


class DatasetVersionError(ValueError):
    """Invalid version policy, manifest, registry, or precondition."""


class DatasetVersionConflictError(DatasetVersionError):
    """Raised when an immutable version name maps to another fingerprint."""


class DatasetVersionNotFoundError(DatasetVersionError):
    """Raised when a requested local version does not exist."""


class DatasetVersionStorageError(DatasetVersionError):
    """Raised when local registry storage is malformed or unavailable."""


class VersionComparisonStatus(str, Enum):
    IDENTICAL = "IDENTICAL"
    DIFFERENT = "DIFFERENT"


@dataclass(frozen=True, slots=True)
class DatasetVersionLimits:
    """Finite local registry, manifest, lineage, and comparison limits."""

    max_versions: int = 1_024
    max_records_per_version: int = 100_000
    max_manifest_bytes: int = 32 * 1024 * 1024
    max_metadata_bytes: int = 64 * 1024
    max_lineage_depth: int = 64
    max_comparison_items: int = 100_000
    max_record_id_length: int = 512

    def __post_init__(self) -> None:
        ceilings = {
            "max_versions": 100_000,
            "max_records_per_version": 1_000_000,
            "max_manifest_bytes": 512 * 1024 * 1024,
            "max_metadata_bytes": 4 * 1024 * 1024,
            "max_lineage_depth": 512,
            "max_comparison_items": 1_000_000,
            "max_record_id_length": 4_096,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value > ceiling:
                raise DatasetVersionError(f"{name} is outside its configured bound")


@dataclass(frozen=True, slots=True)
class DatasetVersionManifest:
    """Canonical immutable release manifest; creation metadata is not identity input."""

    format: str
    version: str
    version_id: str
    dataset_fingerprint: str
    schema_version: str
    record_count: int
    record_ids: tuple[str, ...]
    record_fingerprints: Mapping[str, str]
    train_record_ids: tuple[str, ...]
    validation_record_ids: tuple[str, ...]
    test_record_ids: tuple[str, ...]
    split_version: str
    split_seed: int
    grouping_policy: str
    quality_policy_version: str | None
    quality_policy: Mapping[str, Any] | None
    validation_status: str
    validation_summary: Mapping[str, int]
    provenance: Mapping[str, Any]
    parent_version: str | None
    metadata: Mapping[str, Any]
    creation_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.format != DATASET_VERSIONING_FORMAT:
            raise DatasetVersionError("unsupported dataset version format")
        _validate_version_name(self.version)
        if self.version_id != self.version:
            raise DatasetVersionError("version_id must match version")
        if not _DATASET_FINGERPRINT_PATTERN.fullmatch(self.dataset_fingerprint):
            raise DatasetVersionError("dataset_fingerprint must be sha256 hexadecimal")
        if self.schema_version != DATASET_RECORD_SCHEMA_VERSION:
            raise DatasetVersionError("unsupported Dataset Schema version")
        if self.split_version != DATASET_SPLIT_VERSION:
            raise DatasetVersionError("unsupported Dataset Split version")
        if not isinstance(self.record_count, int) or self.record_count < 0:
            raise DatasetVersionError("record_count must be a non-negative integer")
        for collection_name in ("record_ids", "train_record_ids", "validation_record_ids", "test_record_ids"):
            values = getattr(self, collection_name)
            if not isinstance(values, tuple) or any(not isinstance(item, str) or not item.strip() or len(item) > 4_096 for item in values):
                raise DatasetVersionError(f"{collection_name} is invalid")
        if len(self.record_ids) != self.record_count or tuple(sorted(self.record_ids)) != self.record_ids or len(set(self.record_ids)) != len(self.record_ids):
            raise DatasetVersionError("record_ids must be unique sorted canonical IDs and match record_count")
        all_partition_ids = self.train_record_ids + self.validation_record_ids + self.test_record_ids
        if len(all_partition_ids) != self.record_count or len(set(all_partition_ids)) != len(all_partition_ids) or set(all_partition_ids) != set(self.record_ids):
            raise DatasetVersionError("partition IDs must be disjoint and cover record_ids")
        for values in (self.train_record_ids, self.validation_record_ids, self.test_record_ids):
            if tuple(sorted(values)) != values:
                raise DatasetVersionError("partition IDs must be sorted")
        if not isinstance(self.record_fingerprints, Mapping) or set(self.record_fingerprints) != set(self.record_ids):
            raise DatasetVersionError("record_fingerprints must cover every record ID")
        if any(not isinstance(key, str) or not _DATASET_FINGERPRINT_PATTERN.fullmatch(value) for key, value in self.record_fingerprints.items()):
            raise DatasetVersionError("record_fingerprints contain invalid values")
        if not isinstance(self.split_seed, int) or isinstance(self.split_seed, bool) or self.split_seed < 0:
            raise DatasetVersionError("split_seed must be a non-negative integer")
        if not isinstance(self.grouping_policy, str) or not self.grouping_policy.strip():
            raise DatasetVersionError("grouping_policy must contain text")
        if self.quality_policy_version is not None and (not isinstance(self.quality_policy_version, str) or not self.quality_policy_version.strip()):
            raise DatasetVersionError("quality_policy_version must be text or None")
        if self.quality_policy is not None and not isinstance(self.quality_policy, Mapping):
            raise DatasetVersionError("quality_policy must be a mapping or None")
        if self.validation_status not in {item.value for item in ValidationStatus}:
            raise DatasetVersionError("validation_status is unsupported")
        _validate_int_mapping(self.validation_summary, "validation_summary")
        for name, value in (("provenance", self.provenance), ("metadata", self.metadata), ("creation_metadata", self.creation_metadata)):
            if not isinstance(value, Mapping):
                raise DatasetVersionError(f"{name} must be a mapping")
            _validate_json_value(value, name, 0, 6)
        if self.parent_version is not None:
            _validate_version_name(self.parent_version)
        if _contains_version_secret(_canonical_json(self.to_dict())):
            raise DatasetVersionError("dataset version manifest contains prohibited secret material")
        object.__setattr__(self, "record_fingerprints", MappingProxyType(dict(sorted(self.record_fingerprints.items()))))
        object.__setattr__(self, "quality_policy", _freeze(self.quality_policy) if self.quality_policy is not None else None)
        object.__setattr__(self, "validation_summary", MappingProxyType(dict(sorted(self.validation_summary.items()))))
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        object.__setattr__(self, "creation_metadata", _freeze(self.creation_metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "version_id": self.version_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "schema_version": self.schema_version,
            "record_count": self.record_count,
            "record_ids": list(self.record_ids),
            "record_fingerprints": dict(self.record_fingerprints),
            "train_record_ids": list(self.train_record_ids),
            "validation_record_ids": list(self.validation_record_ids),
            "test_record_ids": list(self.test_record_ids),
            "split_version": self.split_version,
            "split_seed": self.split_seed,
            "grouping_policy": self.grouping_policy,
            "quality_policy_version": self.quality_policy_version,
            "quality_policy": _thaw(self.quality_policy),
            "validation_status": self.validation_status,
            "validation_summary": dict(self.validation_summary),
            "provenance": _thaw(self.provenance),
            "parent_version": self.parent_version,
            "metadata": _thaw(self.metadata),
            "creation_metadata": _thaw(self.creation_metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Any, *, limits: DatasetVersionLimits | None = None) -> "DatasetVersionManifest":
        if not isinstance(payload, Mapping):
            raise DatasetVersionError("dataset version manifest must be an object")
        allowed = {"format", "version", "version_id", "dataset_fingerprint", "schema_version", "record_count", "record_ids", "record_fingerprints", "train_record_ids", "validation_record_ids", "test_record_ids", "split_version", "split_seed", "grouping_policy", "quality_policy_version", "quality_policy", "validation_status", "validation_summary", "provenance", "parent_version", "metadata", "creation_metadata"}
        if set(payload) != allowed:
            raise DatasetVersionError("dataset version manifest fields are missing or unknown")
        collections = {}
        for name in ("record_ids", "train_record_ids", "validation_record_ids", "test_record_ids"):
            value = payload[name]
            if not isinstance(value, (list, tuple)):
                raise DatasetVersionError(f"{name} must be an array")
            collections[name] = tuple(value)
        manifest = cls(
            payload["format"], payload["version"], payload["version_id"], payload["dataset_fingerprint"], payload["schema_version"], payload["record_count"], collections["record_ids"], payload["record_fingerprints"], collections["train_record_ids"], collections["validation_record_ids"], collections["test_record_ids"], payload["split_version"], payload["split_seed"], payload["grouping_policy"], payload["quality_policy_version"], payload["quality_policy"], payload["validation_status"], payload["validation_summary"], payload["provenance"], payload["parent_version"], payload["metadata"], payload["creation_metadata"],
        )
        if limits is not None and len(manifest.to_json().encode("utf-8")) > limits.max_manifest_bytes:
            raise DatasetVersionError("dataset version manifest exceeds configured byte limit")
        return manifest


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    """Immutable version object backed by a canonical manifest."""

    manifest: DatasetVersionManifest

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, DatasetVersionManifest):
            raise DatasetVersionError("manifest must be DatasetVersionManifest")

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def version_id(self) -> str:
        return self.manifest.version_id

    @property
    def dataset_fingerprint(self) -> str:
        return self.manifest.dataset_fingerprint

    def to_dict(self) -> dict[str, Any]:
        return self.manifest.to_dict()

    def to_json(self) -> str:
        return self.manifest.to_json()


@dataclass(frozen=True, slots=True)
class DatasetVersionCheck:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip() or not isinstance(self.message, str) or not self.message.strip() or len(self.message) > 512:
            raise DatasetVersionError("version check is invalid")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class DatasetVersionVerificationResult:
    valid: bool
    version: str
    expected_fingerprint: str
    actual_fingerprint: str | None
    checks: tuple[DatasetVersionCheck, ...]
    added_record_ids: tuple[str, ...]
    removed_record_ids: tuple[str, ...]
    changed_record_ids: tuple[str, ...]
    train_membership_changed: bool
    validation_membership_changed: bool
    test_membership_changed: bool
    schema_changed: bool
    split_version_changed: bool
    quality_policy_changed: bool
    validation_status_changed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool) or not isinstance(self.version, str) or not _DATASET_FINGERPRINT_PATTERN.fullmatch(self.expected_fingerprint):
            raise DatasetVersionError("version verification result is invalid")
        if self.actual_fingerprint is not None and not _DATASET_FINGERPRINT_PATTERN.fullmatch(self.actual_fingerprint):
            raise DatasetVersionError("actual_fingerprint is invalid")
        if not isinstance(self.checks, tuple) or any(not isinstance(item, DatasetVersionCheck) for item in self.checks):
            raise DatasetVersionError("checks must be DatasetVersionCheck values")
        for name in ("added_record_ids", "removed_record_ids", "changed_record_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or tuple(sorted(values)) != values:
                raise DatasetVersionError(f"{name} must be sorted tuples")
        for name in ("train_membership_changed", "validation_membership_changed", "test_membership_changed", "schema_changed", "split_version_changed", "quality_policy_changed", "validation_status_changed"):
            if not isinstance(getattr(self, name), bool):
                raise DatasetVersionError(f"{name} must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "version": self.version, "expected_fingerprint": self.expected_fingerprint, "actual_fingerprint": self.actual_fingerprint, "checks": [item.to_dict() for item in self.checks], "added_record_ids": list(self.added_record_ids), "removed_record_ids": list(self.removed_record_ids), "changed_record_ids": list(self.changed_record_ids), "train_membership_changed": self.train_membership_changed, "validation_membership_changed": self.validation_membership_changed, "test_membership_changed": self.test_membership_changed, "schema_changed": self.schema_changed, "split_version_changed": self.split_version_changed, "quality_policy_changed": self.quality_policy_changed, "validation_status_changed": self.validation_status_changed}


@dataclass(frozen=True, slots=True)
class DatasetVersionComparison:
    left_version: str
    right_version: str
    status: VersionComparisonStatus
    added_record_ids: tuple[str, ...]
    removed_record_ids: tuple[str, ...]
    changed_record_ids: tuple[str, ...]
    train_membership_changed: bool
    validation_membership_changed: bool
    test_membership_changed: bool
    schema_changed: bool
    split_version_changed: bool
    quality_policy_changed: bool
    validation_status_changed: bool
    fingerprint_changed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.status, VersionComparisonStatus):
            object.__setattr__(self, "status", VersionComparisonStatus(self.status))
        for name in ("added_record_ids", "removed_record_ids", "changed_record_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or tuple(sorted(values)) != values:
                raise DatasetVersionError(f"{name} must be sorted tuples")

    def to_dict(self) -> dict[str, Any]:
        return {"left_version": self.left_version, "right_version": self.right_version, "status": self.status.value, "added_record_ids": list(self.added_record_ids), "removed_record_ids": list(self.removed_record_ids), "changed_record_ids": list(self.changed_record_ids), "train_membership_changed": self.train_membership_changed, "validation_membership_changed": self.validation_membership_changed, "test_membership_changed": self.test_membership_changed, "schema_changed": self.schema_changed, "split_version_changed": self.split_version_changed, "quality_policy_changed": self.quality_policy_changed, "validation_status_changed": self.validation_status_changed, "fingerprint_changed": self.fingerprint_changed}


class DatasetVersionRegistry:
    """Local atomic registry for immutable version manifests; no cloud or network."""

    def __init__(self, path: Path | str | None = None, *, limits: DatasetVersionLimits | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self.limits = limits or DatasetVersionLimits()
        self._versions: dict[str, DatasetVersion] = {}
        self._loaded_digest: str | None = None
        self.reload()

    def reload(self) -> None:
        if self.path is None:
            self._versions = {}
            self._loaded_digest = None
            return
        self._validate_storage_location()
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            self._versions = {}
            self._loaded_digest = None
            return
        except OSError as exc:
            raise DatasetVersionStorageError("dataset version registry is unavailable") from exc
        if len(raw) > self.limits.max_manifest_bytes * self.limits.max_versions:
            raise DatasetVersionStorageError("dataset version registry exceeds configured byte limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, Mapping) or set(payload) != {"format", "schema_version", "versions"} or payload["format"] != DATASET_VERSIONING_REGISTRY_FORMAT or payload["schema_version"] != DATASET_VERSIONING_SCHEMA_VERSION:
                raise DatasetVersionStorageError("dataset version registry header is invalid")
            raw_versions = payload["versions"]
            if not isinstance(raw_versions, Mapping) or len(raw_versions) > self.limits.max_versions:
                raise DatasetVersionStorageError("dataset version registry version count is invalid")
            loaded = {}
            for name in sorted(raw_versions):
                manifest = DatasetVersionManifest.from_dict(raw_versions[name], limits=self.limits)
                if name != manifest.version:
                    raise DatasetVersionStorageError("registry key does not match manifest version")
                loaded[name] = DatasetVersion(manifest)
            self._versions = loaded
            self._validate_lineage_graph()
        except DatasetVersionError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise DatasetVersionStorageError("dataset version registry is malformed") from exc
        self._loaded_digest = _sha256_bytes(raw)

    def list_versions(self) -> tuple[DatasetVersion, ...]:
        return tuple(self._versions[name] for name in sorted(self._versions))

    def get_version(self, version: str) -> DatasetVersion | None:
        _validate_version_name(version)
        return self._versions.get(version)

    def require_version(self, version: str) -> DatasetVersion:
        found = self.get_version(version)
        if found is None:
            raise DatasetVersionNotFoundError(f"dataset version does not exist: {version}")
        return found

    def register(self, version: DatasetVersion) -> DatasetVersion:
        if not isinstance(version, DatasetVersion):
            raise DatasetVersionError("register requires DatasetVersion")
        existing = self._versions.get(version.version)
        if existing is not None:
            if existing.dataset_fingerprint == version.dataset_fingerprint and existing.to_json() == version.to_json():
                return existing
            raise DatasetVersionConflictError("dataset version name already maps to a different immutable manifest")
        if len(self._versions) >= self.limits.max_versions:
            raise DatasetVersionError("maximum dataset version count exceeded")
        if version.manifest.parent_version is not None:
            parent = self._versions.get(version.manifest.parent_version)
            if parent is None:
                raise DatasetVersionError("parent_version does not exist")
            if version.version == parent.version:
                raise DatasetVersionError("version cannot parent itself")
        self._versions[version.version] = version
        try:
            self._persist()
        except Exception:
            self._versions.pop(version.version, None)
            raise
        return version

    def compare(self, left: str, right: str, *, limits: DatasetVersionLimits | None = None) -> DatasetVersionComparison:
        return compare_versions(self.require_version(left), self.require_version(right), limits=limits or self.limits)

    def lineage(self, version: str) -> tuple[DatasetVersion, ...]:
        current = self.require_version(version)
        result: list[DatasetVersion] = []
        seen: set[str] = set()
        while current.manifest.parent_version is not None:
            if current.version in seen or len(result) >= self.limits.max_lineage_depth:
                raise DatasetVersionError("dataset version lineage contains a cycle or exceeds its bound")
            seen.add(current.version)
            parent = self.require_version(current.manifest.parent_version)
            result.append(parent)
            current = parent
        return tuple(result)

    def _validate_lineage_graph(self) -> None:
        for version in self._versions:
            self.lineage(version)

    def _validate_storage_location(self) -> None:
        assert self.path is not None
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise DatasetVersionStorageError("dataset version registry must not use symlinks")

    def _persist(self) -> None:
        if self.path is None:
            self._loaded_digest = None
            return
        self._validate_storage_location()
        if self.path.exists():
            current_digest = _sha256_bytes(self.path.read_bytes())
            if self._loaded_digest is None or current_digest != self._loaded_digest:
                raise DatasetVersionConflictError("dataset version registry changed since it was loaded")
        payload = json.dumps({"format": DATASET_VERSIONING_REGISTRY_FORMAT, "schema_version": DATASET_VERSIONING_SCHEMA_VERSION, "versions": {name: self._versions[name].to_dict() for name in sorted(self._versions)}}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(payload) > self.limits.max_manifest_bytes * self.limits.max_versions:
            raise DatasetVersionError("dataset version registry exceeds configured byte limit")
        directory = self.path.parent
        directory.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=directory, prefix=".dataset_versions.", suffix=".tmp", delete=False) as stream:
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
            self._loaded_digest = _sha256_bytes(payload)
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass


class DatasetVersioner:
    """Explicit version creation, verification, comparison, and registry facade."""

    def __init__(self, *, registry: DatasetVersionRegistry | None = None, limits: DatasetVersionLimits | None = None) -> None:
        self.limits = limits or DatasetVersionLimits()
        self.registry = registry or DatasetVersionRegistry(limits=self.limits)

    def create_version(
        self,
        version: str,
        records: Sequence[DatasetRecord],
        split_result: DatasetSplitResult,
        validation_result: DatasetValidationResult,
        *,
        quality_policy: Mapping[str, Any] | Any | None = None,
        quality_policy_version: str | None = None,
        parent_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        creation_metadata: Mapping[str, Any] | None = None,
    ) -> DatasetVersion:
        _validate_version_name(version)
        canonical = _validate_records(records, self.limits)
        if not isinstance(split_result, DatasetSplitResult):
            raise DatasetVersionError("split_result must be DatasetSplitResult")
        try:
            validate_split(split_result)
        except (DatasetSplitError, TypeError, ValueError) as exc:
            raise DatasetVersionError("split_result is invalid") from exc
        if split_result.manifest.schema_version != DATASET_RECORD_SCHEMA_VERSION or split_result.manifest.split_version != DATASET_SPLIT_VERSION:
            raise DatasetVersionError("split manifest schema or split version is unsupported")
        if not isinstance(validation_result, DatasetValidationResult):
            raise DatasetVersionError("validation_result must be DatasetValidationResult")
        if validation_result.validation_status is not ValidationStatus.VALID:
            raise DatasetVersionError("dataset version requires VALID validation status")
        if validation_result.error_count != 0 or validation_result.invalid_records != 0:
            raise DatasetVersionError("dataset version requires validation without errors or invalid records")
        if validation_result.dataset_schema_version != DATASET_RECORD_SCHEMA_VERSION or validation_result.total_records != len(canonical) or validation_result.valid_records != len(canonical):
            raise DatasetVersionError("validation result does not match the canonical dataset")
        record_ids = tuple(sorted(record.record_id for record in canonical))
        split_ids = tuple(record.record_id for partition in (split_result.train, split_result.validation, split_result.test) for record in partition)
        if set(split_ids) != set(record_ids) or len(split_ids) != len(record_ids):
            raise DatasetVersionError("split does not cover exactly the supplied records")
        if parent_version is not None:
            _validate_version_name(parent_version)
            if parent_version == version:
                raise DatasetVersionError("version cannot parent itself")
            self.registry.require_version(parent_version)
            self.registry.lineage(parent_version)
        policy_payload = _policy_payload(quality_policy)
        safe_metadata = _safe_mapping(metadata or {}, "metadata", self.limits)
        safe_creation_metadata = _safe_mapping(creation_metadata or {}, "creation_metadata", self.limits)
        record_fingerprints = {record.record_id: _record_fingerprint(record) for record in sorted(canonical, key=lambda item: item.record_id)}
        provenance = _dataset_provenance(canonical)
        fingerprint = compute_dataset_fingerprint(canonical, split_result, validation_result, quality_policy=policy_payload, quality_policy_version=quality_policy_version, metadata=safe_metadata)
        manifest = DatasetVersionManifest(DATASET_VERSIONING_FORMAT, version, version, fingerprint, DATASET_RECORD_SCHEMA_VERSION, len(record_ids), record_ids, record_fingerprints, tuple(sorted(record.record_id for record in split_result.train)), tuple(sorted(record.record_id for record in split_result.validation)), tuple(sorted(record.record_id for record in split_result.test)), split_result.manifest.split_version, split_result.manifest.seed, split_result.manifest.group_by.value, quality_policy_version, policy_payload, validation_result.validation_status.value, validation_result.summary, provenance, parent_version, safe_metadata, safe_creation_metadata)
        if len(manifest.to_json().encode("utf-8")) > self.limits.max_manifest_bytes:
            raise DatasetVersionError("dataset version manifest exceeds configured byte limit")
        return self.registry.register(DatasetVersion(manifest))

    def get_version(self, version: str) -> DatasetVersion:
        return self.registry.require_version(version)

    def list_versions(self) -> tuple[DatasetVersion, ...]:
        return self.registry.list_versions()

    def verify_version(
        self,
        version: DatasetVersion | str,
        records: Sequence[DatasetRecord],
        split_result: DatasetSplitResult,
        validation_result: DatasetValidationResult,
        *,
        quality_policy: Mapping[str, Any] | Any | None = None,
        quality_policy_version: str | None = None,
    ) -> DatasetVersionVerificationResult:
        expected = self._resolve(version)
        canonical = _validate_records(records, self.limits)
        actual_ids = {record.record_id for record in canonical}
        expected_ids = set(expected.manifest.record_ids)
        added = tuple(sorted(actual_ids - expected_ids))
        removed = tuple(sorted(expected_ids - actual_ids))
        current_fingerprints = {record.record_id: _record_fingerprint(record) for record in canonical}
        changed = tuple(sorted(record_id for record_id in actual_ids & expected_ids if current_fingerprints[record_id] != expected.manifest.record_fingerprints[record_id]))
        checks: list[DatasetVersionCheck] = []
        if added:
            checks.append(DatasetVersionCheck("extra_records", "current dataset contains records absent from the version"))
        if removed:
            checks.append(DatasetVersionCheck("missing_records", "current dataset is missing records from the version"))
        if changed:
            checks.append(DatasetVersionCheck("changed_records", "record content fingerprints changed"))
        actual_partitions = {"train": tuple(sorted(record.record_id for record in split_result.train)), "validation": tuple(sorted(record.record_id for record in split_result.validation)), "test": tuple(sorted(record.record_id for record in split_result.test))}
        expected_partitions = {"train": expected.manifest.train_record_ids, "validation": expected.manifest.validation_record_ids, "test": expected.manifest.test_record_ids}
        membership = {name: actual_partitions[name] != expected_partitions[name] for name in ("train", "validation", "test")}
        for name, changed_flag in membership.items():
            if changed_flag:
                checks.append(DatasetVersionCheck(f"{name}_membership_changed", f"{name} partition membership changed"))
        schema_changed = any(record.schema_version != expected.manifest.schema_version for record in canonical) or not canonical
        split_version_changed = split_result.manifest.split_version != expected.manifest.split_version or split_result.manifest.seed != expected.manifest.split_seed or split_result.manifest.group_by.value != expected.manifest.grouping_policy
        policy_payload = _policy_payload(quality_policy)
        quality_changed = quality_policy_version != expected.manifest.quality_policy_version or _canonical_json(policy_payload) != _canonical_json(expected.manifest.quality_policy)
        validation_changed = validation_result.validation_status.value != expected.manifest.validation_status or validation_result.dataset_schema_version != expected.manifest.schema_version or validation_result.total_records != expected.manifest.record_count
        if schema_changed:
            checks.append(DatasetVersionCheck("schema_changed", "Dataset Schema version changed"))
        if split_version_changed:
            checks.append(DatasetVersionCheck("split_version_changed", "Dataset Split version changed"))
        if quality_changed:
            checks.append(DatasetVersionCheck("quality_policy_changed", "quality policy or version changed"))
        if validation_changed:
            checks.append(DatasetVersionCheck("validation_status_changed", "validation status changed"))
        try:
            actual_fingerprint = compute_dataset_fingerprint(canonical, split_result, validation_result, quality_policy=policy_payload, quality_policy_version=quality_policy_version, metadata=expected.manifest.metadata)
        except DatasetVersionError as exc:
            actual_fingerprint = None
            checks.append(DatasetVersionCheck("fingerprint_unavailable", str(exc)))
        if actual_fingerprint != expected.dataset_fingerprint:
            checks.append(DatasetVersionCheck("fingerprint_mismatch", "current content does not match the immutable version fingerprint"))
        return DatasetVersionVerificationResult(not checks, expected.version, expected.dataset_fingerprint, actual_fingerprint, tuple(checks), added, removed, changed, membership["train"], membership["validation"], membership["test"], schema_changed, split_version_changed, quality_changed, validation_changed)

    def compare_versions(self, left: DatasetVersion | str, right: DatasetVersion | str) -> DatasetVersionComparison:
        return compare_versions(self._resolve(left), self._resolve(right), limits=self.limits)

    def verify_manifest(self, version: DatasetVersion | str) -> None:
        manifest = self._resolve(version).manifest
        DatasetVersionManifest.from_dict(manifest.to_dict(), limits=self.limits)

    def _resolve(self, value: DatasetVersion | str) -> DatasetVersion:
        if isinstance(value, DatasetVersion):
            return value
        return self.registry.require_version(value)


def compute_dataset_fingerprint(
    records: Sequence[DatasetRecord],
    split_result: DatasetSplitResult,
    validation_result: DatasetValidationResult,
    *,
    quality_policy: Mapping[str, Any] | None = None,
    quality_policy_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    canonical = _validate_records(records, DatasetVersionLimits(max_records_per_version=max(1, len(records))))
    record_payloads = [{"record_id": record.record_id, "content": record.to_dict()} for record in sorted(canonical, key=lambda item: item.record_id)]
    payload = {
        "identity_version": DATASET_VERSIONING_SCHEMA_VERSION,
        "schema_version": split_result.manifest.schema_version,
        "records": record_payloads,
        "split": {
            "split_version": split_result.manifest.split_version,
            "seed": split_result.manifest.seed,
            "grouping_policy": split_result.manifest.group_by.value,
            "train_record_ids": sorted(record.record_id for record in split_result.train),
            "validation_record_ids": sorted(record.record_id for record in split_result.validation),
            "test_record_ids": sorted(record.record_id for record in split_result.test),
        },
        "quality_policy_version": quality_policy_version,
        "quality_policy": _thaw(quality_policy),
        "validation": {"status": validation_result.validation_status.value, "schema_version": validation_result.dataset_schema_version, "total_records": validation_result.total_records, "valid_records": validation_result.valid_records, "invalid_records": validation_result.invalid_records, "warning_count": validation_result.warning_count, "error_count": validation_result.error_count, "summary": dict(validation_result.summary)},
        "metadata": _thaw(metadata),
    }
    if _contains_version_secret(_canonical_json(payload)):
        raise DatasetVersionError("dataset fingerprint input contains prohibited secret material")
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def compare_versions(left: DatasetVersion, right: DatasetVersion, *, limits: DatasetVersionLimits | None = None) -> DatasetVersionComparison:
    if not isinstance(left, DatasetVersion) or not isinstance(right, DatasetVersion):
        raise DatasetVersionError("compare_versions requires DatasetVersion values")
    active_limits = limits or DatasetVersionLimits()
    left_ids = set(left.manifest.record_ids)
    right_ids = set(right.manifest.record_ids)
    added = tuple(sorted(right_ids - left_ids))
    removed = tuple(sorted(left_ids - right_ids))
    changed = tuple(sorted(record_id for record_id in left_ids & right_ids if left.manifest.record_fingerprints[record_id] != right.manifest.record_fingerprints[record_id]))
    for collection in (added, removed, changed):
        if len(collection) > active_limits.max_comparison_items:
            raise DatasetVersionError("comparison output exceeds configured limit")
    train_changed = left.manifest.train_record_ids != right.manifest.train_record_ids
    validation_membership_changed = left.manifest.validation_record_ids != right.manifest.validation_record_ids
    test_changed = left.manifest.test_record_ids != right.manifest.test_record_ids
    schema_changed = left.manifest.schema_version != right.manifest.schema_version
    split_changed = left.manifest.split_version != right.manifest.split_version or left.manifest.split_seed != right.manifest.split_seed or left.manifest.grouping_policy != right.manifest.grouping_policy
    quality_changed = left.manifest.quality_policy_version != right.manifest.quality_policy_version or _canonical_json(left.manifest.quality_policy) != _canonical_json(right.manifest.quality_policy)
    validation_status_changed = left.manifest.validation_status != right.manifest.validation_status
    fingerprint_changed = left.dataset_fingerprint != right.dataset_fingerprint
    different = bool(added or removed or changed or train_changed or validation_membership_changed or test_changed or schema_changed or split_changed or quality_changed or validation_status_changed or fingerprint_changed)
    return DatasetVersionComparison(left.version, right.version, VersionComparisonStatus.DIFFERENT if different else VersionComparisonStatus.IDENTICAL, added, removed, changed, train_changed, validation_membership_changed, test_changed, schema_changed, split_changed, quality_changed, validation_status_changed, fingerprint_changed)


def _validate_records(records: Sequence[DatasetRecord], limits: DatasetVersionLimits) -> tuple[DatasetRecord, ...]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise DatasetVersionError("records must be a sequence of DatasetRecord")
    if len(records) > limits.max_records_per_version:
        raise DatasetVersionError("records exceed version limit")
    canonical = tuple(records)
    if any(not isinstance(record, DatasetRecord) for record in canonical):
        raise DatasetVersionError("versioning accepts canonical DatasetRecord objects only")
    ids = [record.record_id for record in canonical]
    if len(ids) != len(set(ids)):
        raise DatasetVersionError("records contain duplicate record IDs")
    return tuple(sorted(canonical, key=lambda item: item.record_id))


def _record_fingerprint(record: DatasetRecord) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(record.to_dict()).encode("utf-8")).hexdigest()


def _dataset_provenance(records: Sequence[DatasetRecord]) -> dict[str, Any]:
    return {"source_type": "experience_record", "experience_ids": sorted({record.experience_id for record in records}), "project_ids": sorted({record.project_context.project_id for record in records if record.project_context}), "schema_versions": sorted({record.schema_version for record in records}), "record_count": len(records)}


def _policy_payload(policy: Mapping[str, Any] | Any | None) -> Mapping[str, Any] | None:
    if policy is None:
        return None
    if hasattr(policy, "to_dict") and callable(policy.to_dict):
        policy = policy.to_dict()
    if not isinstance(policy, Mapping):
        raise DatasetVersionError("quality_policy must be a mapping or expose to_dict()")
    return _freeze(policy)


def _safe_mapping(value: Mapping[str, Any], name: str, limits: DatasetVersionLimits) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetVersionError(f"{name} must be a mapping")
    _validate_json_value(value, name, 0, 6)
    frozen = _freeze(value)
    if _contains_version_secret(_canonical_json(frozen)):
        raise DatasetVersionError(f"{name} contains prohibited secret material")
    if len(_canonical_json(frozen).encode("utf-8")) > limits.max_metadata_bytes:
        raise DatasetVersionError(f"{name} exceeds metadata byte limit")
    return frozen


def _validate_version_name(value: Any) -> None:
    if not isinstance(value, str) or len(value) > _MAX_VERSION_NAME_LENGTH or not DATASET_VERSION_NAME_PATTERN.fullmatch(value):
        raise DatasetVersionError("version must match dataset-vN or dataset-vN.M")


def _validate_int_mapping(value: Mapping[str, Any], name: str) -> None:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) or not isinstance(item, int) or item < 0 for key, item in value.items()):
        raise DatasetVersionError(f"{name} must map strings to non-negative integers")


def _validate_json_value(value: Any, name: str, depth: int, maximum_depth: int) -> None:
    if depth > maximum_depth:
        raise DatasetVersionError(f"{name} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetVersionError(f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise DatasetVersionError(f"{name} contains an invalid key")
            _validate_json_value(item, f"{name}.{key}", depth + 1, maximum_depth)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{name}[{index}]", depth + 1, maximum_depth)
        return
    raise DatasetVersionError(f"{name} contains an unsupported value")


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


def _contains_version_secret(value: str) -> bool:
    if _contains_prohibited_secret(value):
        return True
    quoted_key = re.compile(r'["\'](?:password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key|credential|cookie|database_url)["\']\s*:\s*["\'][^"\']+["\']', re.IGNORECASE)
    return quoted_key.search(value) is not None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "DATASET_VERSIONING_FORMAT",
    "DATASET_VERSIONING_REGISTRY_FORMAT",
    "DATASET_VERSIONING_SCHEMA_VERSION",
    "DATASET_VERSION_NAME_PATTERN",
    "DatasetVersion",
    "DatasetVersionCheck",
    "DatasetVersionComparison",
    "DatasetVersionConflictError",
    "DatasetVersionError",
    "DatasetVersionLimits",
    "DatasetVersionManifest",
    "DatasetVersionNotFoundError",
    "DatasetVersionRegistry",
    "DatasetVersionStorageError",
    "DatasetVersionVerificationResult",
    "DatasetVersioner",
    "VersionComparisonStatus",
    "compare_versions",
    "compute_dataset_fingerprint",
]
