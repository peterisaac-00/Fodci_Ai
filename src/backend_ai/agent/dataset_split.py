"""Deterministic train/validation/test splitting for accepted DatasetRecord values.

Phase 10.4 partitions canonical records only.  It does not re-evaluate quality,
persist releases, tokenize, call models, or modify any source object.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from enum import Enum
from typing import Any

from backend_ai.agent.dataset_quality import (
    DatasetFilteringResult,
    QualityAssessment,
    QualityDecision,
)
from backend_ai.agent.dataset_schema import (
    DATASET_RECORD_SCHEMA_VERSION,
    DatasetRecord,
)


DATASET_SPLIT_VERSION = "1.0"
_RATIO_TOLERANCE = 1e-9
_MAX_SEED = (1 << 63) - 1


class DatasetSplitError(ValueError):
    """Invalid input, policy, or impossible split constraint."""


class DuplicateDatasetRecordError(DatasetSplitError):
    """Raised when input contains duplicate canonical record IDs."""


class DatasetSplitGroup(str, Enum):
    RECORD = "record"
    EXPERIENCE = "experience"
    PROJECT = "project"


@dataclass(frozen=True, slots=True)
class DatasetSplitPolicy:
    """Immutable, inspectable deterministic split contract."""

    train_ratio: float = 0.80
    validation_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42
    split_version: str = DATASET_SPLIT_VERSION
    group_by: DatasetSplitGroup = DatasetSplitGroup.RECORD
    minimum_train_records: int = 0
    minimum_validation_records: int = 0
    minimum_test_records: int = 0
    require_non_empty_partitions: bool = False
    max_records: int = 100_000

    def __post_init__(self) -> None:
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) for value in ratios):
            raise DatasetSplitError("split ratios must be finite numbers")
        if any(float(value) < 0.0 or float(value) > 1.0 for value in ratios):
            raise DatasetSplitError("split ratios must be within [0, 1]")
        if abs(sum(float(value) for value in ratios) - 1.0) > _RATIO_TOLERANCE:
            raise DatasetSplitError("split ratios must sum to 1.0")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed <= _MAX_SEED:
            raise DatasetSplitError("seed must be an integer in the supported deterministic range")
        if not isinstance(self.split_version, str) or not self.split_version.strip() or len(self.split_version) > 32:
            raise DatasetSplitError("split_version must be bounded text")
        if not isinstance(self.group_by, DatasetSplitGroup):
            try:
                object.__setattr__(self, "group_by", DatasetSplitGroup(self.group_by))
            except (TypeError, ValueError) as exc:
                raise DatasetSplitError("unsupported grouping policy") from exc
        for name in ("minimum_train_records", "minimum_validation_records", "minimum_test_records", "max_records"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise DatasetSplitError(f"{name} must be a non-negative integer")
        if self.max_records <= 0 or self.max_records > 1_000_000:
            raise DatasetSplitError("max_records is outside the supported bound")
        if not isinstance(self.require_non_empty_partitions, bool):
            raise DatasetSplitError("require_non_empty_partitions must be boolean")
        if self.require_non_empty_partitions:
            minimums = (self.minimum_train_records, self.minimum_validation_records, self.minimum_test_records)
            if any(value == 0 for value in minimums):
                object.__setattr__(self, "minimum_train_records", max(1, self.minimum_train_records))
                object.__setattr__(self, "minimum_validation_records", max(1, self.minimum_validation_records))
                object.__setattr__(self, "minimum_test_records", max(1, self.minimum_test_records))

    @property
    def ratios(self) -> tuple[float, float, float]:
        return (float(self.train_ratio), float(self.validation_ratio), float(self.test_ratio))

    @property
    def minimums(self) -> tuple[int, int, int]:
        return (self.minimum_train_records, self.minimum_validation_records, self.minimum_test_records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "split_version": self.split_version,
            "group_by": self.group_by.value,
            "minimum_train_records": self.minimum_train_records,
            "minimum_validation_records": self.minimum_validation_records,
            "minimum_test_records": self.minimum_test_records,
            "require_non_empty_partitions": self.require_non_empty_partitions,
            "max_records": self.max_records,
        }


@dataclass(frozen=True, slots=True)
class DatasetSplitManifest:
    """In-memory reproducibility and audit metadata; never persisted automatically."""

    split_version: str
    seed: int
    policy: Mapping[str, Any]
    schema_version: str
    group_by: DatasetSplitGroup
    total_records: int
    train_count: int
    validation_count: int
    test_count: int
    requested_ratios: Mapping[str, float]
    actual_ratios: Mapping[str, float]
    record_ids: Mapping[str, tuple[str, ...]]
    group_ids: Mapping[str, tuple[str, ...]]
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split_version != DATASET_SPLIT_VERSION:
            raise DatasetSplitError("unsupported split_version")
        if self.schema_version != DATASET_RECORD_SCHEMA_VERSION:
            raise DatasetSplitError("unsupported dataset schema_version in split manifest")
        if not isinstance(self.group_by, DatasetSplitGroup):
            raise DatasetSplitError("manifest group_by must be DatasetSplitGroup")
        for value in (self.total_records, self.train_count, self.validation_count, self.test_count):
            if not isinstance(value, int) or value < 0:
                raise DatasetSplitError("manifest counts must be non-negative integers")
        if self.train_count + self.validation_count + self.test_count != self.total_records:
            raise DatasetSplitError("manifest counts do not sum to total_records")
        _validate_ratio_map(self.requested_ratios, "requested_ratios")
        _validate_ratio_map(self.actual_ratios, "actual_ratios", allow_sum_tolerance=True)
        if not isinstance(self.record_ids, Mapping) or not isinstance(self.group_ids, Mapping):
            raise DatasetSplitError("manifest IDs must be mappings")
        if any(not isinstance(value, tuple) for value in self.record_ids.values()) or any(not isinstance(value, tuple) for value in self.group_ids.values()):
            raise DatasetSplitError("manifest ID collections must be tuples")
        if any(not isinstance(item, str) or not item.strip() for item in self.diagnostics):
            raise DatasetSplitError("manifest diagnostics must contain text")
        object.__setattr__(self, "policy", MappingProxyType(dict(self.policy)))
        object.__setattr__(self, "requested_ratios", MappingProxyType(dict(self.requested_ratios)))
        object.__setattr__(self, "actual_ratios", MappingProxyType(dict(self.actual_ratios)))
        object.__setattr__(self, "record_ids", MappingProxyType({key: tuple(value) for key, value in self.record_ids.items()}))
        object.__setattr__(self, "group_ids", MappingProxyType({key: tuple(value) for key, value in self.group_ids.items()}))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_version": self.split_version,
            "seed": self.seed,
            "policy": dict(self.policy),
            "schema_version": self.schema_version,
            "group_by": self.group_by.value,
            "counts": {"total": self.total_records, "train": self.train_count, "validation": self.validation_count, "test": self.test_count},
            "requested_ratios": dict(self.requested_ratios),
            "actual_ratios": dict(self.actual_ratios),
            "record_ids": {key: list(value) for key, value in self.record_ids.items()},
            "group_ids": {key: list(value) for key, value in self.group_ids.items()},
            "diagnostics": list(self.diagnostics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class DatasetSplitResult:
    """Immutable train/validation/test partitions and their reproducibility manifest."""

    train: tuple[DatasetRecord, ...]
    validation: tuple[DatasetRecord, ...]
    test: tuple[DatasetRecord, ...]
    manifest: DatasetSplitManifest
    excluded_record_ids: tuple[str, ...] = ()
    quality_decisions: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, DatasetRecord) for item in values):
                raise DatasetSplitError(f"{name} must be a tuple of DatasetRecord")
        if not isinstance(self.manifest, DatasetSplitManifest):
            raise DatasetSplitError("manifest must be DatasetSplitManifest")
        if any(not isinstance(item, str) or not item.strip() for item in self.excluded_record_ids):
            raise DatasetSplitError("excluded_record_ids must contain text")
        decisions = self.quality_decisions or {}
        if not isinstance(decisions, Mapping) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in decisions.items()):
            raise DatasetSplitError("quality_decisions must be a string mapping")
        object.__setattr__(self, "quality_decisions", MappingProxyType(dict(sorted(decisions.items()))))
        validate_split(self)

    @property
    def total_records(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    @property
    def counts(self) -> Mapping[str, int]:
        return MappingProxyType({"train": len(self.train), "validation": len(self.validation), "test": len(self.test), "total": self.total_records})

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": [item.to_dict() for item in self.train],
            "validation": [item.to_dict() for item in self.validation],
            "test": [item.to_dict() for item in self.test],
            "manifest": self.manifest.to_dict(),
            "excluded_record_ids": list(self.excluded_record_ids),
            "quality_decisions": dict(self.quality_decisions),
        }


class DatasetSplitter:
    """Deterministic splitter for canonical records already accepted by Phase 10.3."""

    def __init__(self, *, policy: DatasetSplitPolicy | None = None) -> None:
        self.policy = policy or DatasetSplitPolicy()

    def split(
        self,
        records: Sequence[DatasetRecord],
        *,
        quality_assessments: Sequence[QualityAssessment] | Mapping[str, QualityAssessment] | None = None,
    ) -> DatasetSplitResult:
        canonical = _validate_input_records(records, self.policy.max_records)
        quality_map, excluded = _quality_filter(canonical, quality_assessments)
        eligible = tuple(record for record in canonical if record.record_id not in excluded)
        if len(eligible) < sum(self.policy.minimums):
            raise DatasetSplitError("dataset cannot satisfy configured minimum partition counts")
        groups = _build_groups(eligible, self.policy.group_by)
        if self.policy.group_by is not DatasetSplitGroup.RECORD and self.policy.require_non_empty_partitions and len(groups) < 3:
            raise DatasetSplitError("grouped dataset has fewer groups than required non-empty partitions")
        counts = _allocate_counts(len(eligible), self.policy.ratios, self.policy.minimums)
        assigned = _assign_groups(groups, counts, self.policy)
        partitions = {name: tuple(record for group in assigned[name] for record in group.records) for name in _PARTITIONS}
        diagnostics: list[str] = []
        if self.policy.group_by is not DatasetSplitGroup.RECORD and len(groups) < 3:
            diagnostics.append("grouped_dataset_has_fewer_than_three_groups; empty partitions may be possible")
        actual = {name: (len(partitions[name]) / len(eligible) if eligible else 0.0) for name in _PARTITIONS}
        record_ids = {name: tuple(record.record_id for record in partitions[name]) for name in _PARTITIONS}
        group_ids = {name: tuple(group.key for group in assigned[name]) for name in _PARTITIONS}
        manifest = DatasetSplitManifest(
            DATASET_SPLIT_VERSION,
            self.policy.seed,
            self.policy.to_dict(),
            DATASET_RECORD_SCHEMA_VERSION,
            self.policy.group_by,
            len(eligible),
            len(partitions["train"]),
            len(partitions["validation"]),
            len(partitions["test"]),
            {"train": self.policy.train_ratio, "validation": self.policy.validation_ratio, "test": self.policy.test_ratio},
            actual,
            record_ids,
            group_ids,
            tuple(diagnostics),
        )
        return DatasetSplitResult(partitions["train"], partitions["validation"], partitions["test"], manifest, tuple(sorted(excluded)), quality_map)

    def split_accepted(self, filtered: DatasetFilteringResult) -> DatasetSplitResult:
        if not isinstance(filtered, DatasetFilteringResult):
            raise DatasetSplitError("split_accepted requires DatasetFilteringResult")
        assessments = {item.record_id: item for item in filtered.assessments}
        result = self.split(filtered.accepted, quality_assessments=assessments)
        excluded = tuple(sorted(record_id for record_id, assessment in assessments.items() if assessment.decision is not QualityDecision.ACCEPT))
        return DatasetSplitResult(result.train, result.validation, result.test, result.manifest, excluded, assessments_to_decisions(assessments))


def validate_split(result: DatasetSplitResult) -> None:
    """Validate partition disjointness, coverage, group isolation, and metadata."""

    if not isinstance(result, DatasetSplitResult):
        raise DatasetSplitError("result must be DatasetSplitResult")
    partitions = {"train": result.train, "validation": result.validation, "test": result.test}
    ids = {name: [record.record_id for record in records] for name, records in partitions.items()}
    all_ids = [record_id for values in ids.values() for record_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise DatasetSplitError("split contains duplicate record IDs or overlap")
    if set(all_ids) != set(result.manifest.record_ids["train"] + result.manifest.record_ids["validation"] + result.manifest.record_ids["test"]):
        raise DatasetSplitError("manifest record IDs do not match partitions")
    if tuple(all_ids) != result.manifest.record_ids["train"] + result.manifest.record_ids["validation"] + result.manifest.record_ids["test"]:
        raise DatasetSplitError("manifest record ordering does not match partitions")
    for name in _PARTITIONS:
        if tuple(ids[name]) != result.manifest.record_ids[name]:
            raise DatasetSplitError(f"manifest IDs do not match {name} partition")
    if result.manifest.total_records != len(all_ids):
        raise DatasetSplitError("manifest total does not match partitions")
    if result.manifest.group_by is not DatasetSplitGroup.RECORD:
        groups: dict[str, str] = {}
        for name, records in partitions.items():
            for record in records:
                key = _group_key(record, result.manifest.group_by)
                previous = groups.get(key)
                if previous is not None and previous != name:
                    raise DatasetSplitError("group crosses split partitions")
                groups[key] = name


def _validate_input_records(records: Sequence[DatasetRecord], maximum: int) -> tuple[DatasetRecord, ...]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise DatasetSplitError("records must be a bounded sequence of DatasetRecord")
    if len(records) > maximum:
        raise DatasetSplitError("records exceed splitter resource bound")
    canonical = tuple(records)
    if any(not isinstance(record, DatasetRecord) for record in canonical):
        raise DatasetSplitError("splitter accepts canonical DatasetRecord objects only")
    ids = [record.record_id for record in canonical]
    if len(ids) != len(set(ids)):
        raise DuplicateDatasetRecordError("duplicate DatasetRecord record_id in input")
    return tuple(sorted(canonical, key=lambda record: record.record_id))


def _quality_filter(records: tuple[DatasetRecord, ...], assessments: Sequence[QualityAssessment] | Mapping[str, QualityAssessment] | None) -> tuple[dict[str, str], set[str]]:
    if assessments is None:
        return {}, set()
    if isinstance(assessments, Mapping):
        values = tuple(assessments.values())
    elif isinstance(assessments, Sequence) and not isinstance(assessments, (str, bytes)):
        values = tuple(assessments)
    else:
        raise DatasetSplitError("quality_assessments must be a sequence or mapping")
    by_id: dict[str, QualityAssessment] = {}
    for item in values:
        if not isinstance(item, QualityAssessment):
            raise DatasetSplitError("quality_assessments must contain QualityAssessment")
        if item.record_id in by_id:
            raise DatasetSplitError("duplicate quality assessment record_id")
        by_id[item.record_id] = item
    record_ids = {record.record_id for record in records}
    if not record_ids.issubset(by_id):
        raise DatasetSplitError("quality assessments must cover every supplied record")
    scoped = {record_id: by_id[record_id] for record_id in record_ids}
    excluded = {record_id for record_id, assessment in scoped.items() if assessment.decision is not QualityDecision.ACCEPT}
    decisions = assessments_to_decisions(scoped)
    return decisions, excluded


def assessments_to_decisions(assessments: Mapping[str, QualityAssessment]) -> dict[str, str]:
    return {record_id: assessments[record_id].decision.value for record_id in sorted(assessments)}


@dataclass(frozen=True, slots=True)
class _DatasetGroup:
    key: str
    records: tuple[DatasetRecord, ...]


def _build_groups(records: tuple[DatasetRecord, ...], mode: DatasetSplitGroup) -> tuple[_DatasetGroup, ...]:
    grouped: dict[str, list[DatasetRecord]] = {}
    for record in records:
        grouped.setdefault(_group_key(record, mode), []).append(record)
    return tuple(_DatasetGroup(key, tuple(sorted(values, key=lambda record: record.record_id))) for key, values in sorted(grouped.items()))


def _group_key(record: DatasetRecord, mode: DatasetSplitGroup) -> str:
    if mode is DatasetSplitGroup.RECORD:
        return f"record:{record.record_id}"
    if mode is DatasetSplitGroup.EXPERIENCE:
        return f"experience:{record.experience_id}"
    project_id = record.project_context.project_id if record.project_context is not None else None
    return f"project:{project_id}" if project_id else f"record:{record.record_id}"


def _allocate_counts(total: int, ratios: tuple[float, float, float], minimums: tuple[int, int, int]) -> tuple[int, int, int]:
    if total < sum(minimums):
        raise DatasetSplitError("minimum partition counts exceed eligible dataset size")
    base = list(minimums)
    remaining = total - sum(base)
    raw = [remaining * ratio for ratio in ratios]
    additions = [int(value) for value in raw]
    for index, value in enumerate(additions):
        base[index] += value
    remainder = remaining - sum(additions)
    order = sorted(range(3), key=lambda index: (-(raw[index] - int(raw[index])), index))
    for index in order[:remainder]:
        base[index] += 1
    if sum(base) != total:
        raise DatasetSplitError("count allocation did not account for every record")
    return tuple(base)  # type: ignore[return-value]


def _assign_groups(groups: tuple[_DatasetGroup, ...], target_counts: tuple[int, int, int], policy: DatasetSplitPolicy) -> dict[str, tuple[_DatasetGroup, ...]]:
    if policy.group_by is DatasetSplitGroup.RECORD:
        shuffled = list(groups)
        random.Random(policy.seed).shuffle(shuffled)
        partitions: dict[str, list[_DatasetGroup]] = {name: [] for name in _PARTITIONS}
        cursor = 0
        for name, count in zip(_PARTITIONS, target_counts):
            partitions[name] = shuffled[cursor: cursor + count]
            cursor += count
        return {name: tuple(value) for name, value in partitions.items()}
    shuffled = list(groups)
    random.Random(policy.seed).shuffle(shuffled)
    assigned: dict[str, list[_DatasetGroup]] = {name: [] for name in _PARTITIONS}
    counts = {name: 0 for name in _PARTITIONS}
    targets = dict(zip(_PARTITIONS, target_counts))
    for group in shuffled:
        available = [name for name in _PARTITIONS if counts[name] < targets[name] or not any(assigned.values())]
        if not available:
            available = list(_PARTITIONS)
        selected = max(available, key=lambda name: (targets[name] - counts[name], -_PARTITIONS.index(name)))
        assigned[selected].append(group)
        counts[selected] += len(group.records)
    return {name: tuple(value) for name, value in assigned.items()}


def _validate_ratio_map(values: Mapping[str, float], name: str, *, allow_sum_tolerance: bool = False) -> None:
    if set(values) != set(_PARTITIONS):
        raise DatasetSplitError(f"{name} must contain train, validation, and test")
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0 for value in values.values()):
        raise DatasetSplitError(f"{name} contains invalid ratios")
    if not allow_sum_tolerance and abs(sum(float(values[name]) for name in _PARTITIONS) - 1.0) > _RATIO_TOLERANCE:
        raise DatasetSplitError(f"{name} must sum to 1.0")


_PARTITIONS = ("train", "validation", "test")


__all__ = [
    "DATASET_SPLIT_VERSION",
    "DatasetFilteringResult",
    "DatasetSplitError",
    "DatasetSplitGroup",
    "DatasetSplitManifest",
    "DatasetSplitPolicy",
    "DatasetSplitResult",
    "DatasetSplitter",
    "DuplicateDatasetRecordError",
    "validate_split",
]
