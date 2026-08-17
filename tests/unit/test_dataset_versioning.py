from __future__ import annotations

from dataclasses import replace

import pytest

from backend_ai.agent.dataset_quality import DatasetQualityEvaluator, QualityDecision
from backend_ai.agent.dataset_split import DatasetSplitPolicy, DatasetSplitter
from backend_ai.agent.dataset_validator import DatasetValidator, ValidationStatus
from backend_ai.agent.dataset_versioning import (
    DatasetVersion,
    DatasetVersionConflictError,
    DatasetVersionError,
    DatasetVersionLimits,
    DatasetVersionManifest,
    DatasetVersionRegistry,
    DatasetVersioner,
    VersionComparisonStatus,
    compare_versions,
    compute_dataset_fingerprint,
)

from tests.unit.test_dataset_split import _record, _records


def _pipeline(count: int = 9, *, seed: int = 42):
    records = _records(count)
    assessments = tuple(DatasetQualityEvaluator().evaluate(record) for record in records)
    assert all(item.decision is QualityDecision.ACCEPT for item in assessments)
    split = DatasetSplitter(policy=DatasetSplitPolicy(train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, seed=seed)).split(records, quality_assessments=assessments)
    validation = DatasetValidator().validate_dataset(records, split_result=split, quality_assessments=assessments)
    assert validation.validation_status is ValidationStatus.VALID
    return records, assessments, split, validation


def test_valid_version_creation_has_content_identity_and_manifest_round_trip() -> None:
    records, _, split, validation = _pipeline()
    versioner = DatasetVersioner()
    version = versioner.create_version("dataset-v1", records, split, validation, quality_policy_version="quality-1", metadata={"purpose": "evaluation"})
    assert version.version == "dataset-v1"
    assert version.dataset_fingerprint.startswith("sha256:")
    assert version.manifest.record_count == len(records)
    assert DatasetVersionManifest.from_dict(version.manifest.to_dict()).to_json() == version.to_json()
    with pytest.raises(TypeError):
        version.manifest.metadata["new"] = "value"  # type: ignore[index]


def test_fingerprint_is_deterministic_and_changes_for_content_split_policy_and_quality_inputs() -> None:
    records, assessments, split, validation = _pipeline()
    same = compute_dataset_fingerprint(tuple(reversed(records)), split, validation, quality_policy={"threshold": 0.75}, quality_policy_version="quality-1")
    repeat = compute_dataset_fingerprint(records, split, validation, quality_policy={"threshold": 0.75}, quality_policy_version="quality-1")
    assert same == repeat
    changed_record = replace(records[0], task="Fix a different backend API behavior")
    changed = compute_dataset_fingerprint(changed_record and (changed_record,) + records[1:], split, validation, quality_policy={"threshold": 0.75}, quality_policy_version="quality-1")
    assert changed != same
    changed_split = DatasetSplitter(policy=DatasetSplitPolicy(train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, seed=43)).split(records, quality_assessments=assessments)
    changed_split_fp = compute_dataset_fingerprint(records, changed_split, validation, quality_policy={"threshold": 0.75}, quality_policy_version="quality-1")
    assert changed_split_fp != same
    changed_quality = compute_dataset_fingerprint(records, split, validation, quality_policy={"threshold": 0.80}, quality_policy_version="quality-1")
    assert changed_quality != same
    changed_schema = compute_dataset_fingerprint(records, split, validation, quality_policy={"threshold": 0.75}, quality_policy_version="quality-2")
    assert changed_schema != same


def test_invalid_validation_cannot_create_version() -> None:
    records, _, split, validation = _pipeline()
    invalid = replace(validation, validation_status=ValidationStatus.INVALID, error_count=1)
    with pytest.raises(DatasetVersionError):
        DatasetVersioner().create_version("dataset-v1", records, split, invalid)


def test_collision_is_rejected_and_same_manifest_is_idempotent() -> None:
    records, _, split, validation = _pipeline()
    versioner = DatasetVersioner()
    first = versioner.create_version("dataset-v1", records, split, validation)
    second = versioner.create_version("dataset-v1", records, split, validation)
    assert first.to_json() == second.to_json()
    changed = replace(records[0], task="Changed backend task")
    changed_records = (changed,) + records[1:]
    changed_assessments = tuple(DatasetQualityEvaluator().evaluate(record) for record in changed_records)
    changed_split = DatasetSplitter(policy=DatasetSplitPolicy(train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, seed=42)).split(changed_records, quality_assessments=changed_assessments)
    changed_validation = DatasetValidator().validate_dataset(changed_records, split_result=changed_split, quality_assessments=changed_assessments)
    with pytest.raises(DatasetVersionConflictError):
        versioner.create_version("dataset-v1", changed_records, changed_split, changed_validation)


def test_registry_reload_and_parent_lineage_are_deterministic(tmp_path) -> None:
    records, _, split, validation = _pipeline()
    path = tmp_path / ".fodci" / "datasets.json"
    registry = DatasetVersionRegistry(path)
    versioner = DatasetVersioner(registry=registry)
    v1 = versioner.create_version("dataset-v1", records, split, validation)
    v2 = versioner.create_version("dataset-v2", records, split, validation, parent_version="dataset-v1", metadata={"note": "same content, new release identity"})
    assert [item.version for item in registry.lineage("dataset-v2")] == ["dataset-v1"]
    reloaded = DatasetVersionRegistry(path)
    assert [item.version for item in reloaded.list_versions()] == ["dataset-v1", "dataset-v2"]
    assert reloaded.require_version("dataset-v1").to_json() == v1.to_json()
    assert reloaded.require_version("dataset-v2").manifest.parent_version == "dataset-v1"
    with pytest.raises(DatasetVersionError):
        DatasetVersioner(registry=reloaded).create_version("dataset-v3", records, split, validation, parent_version="dataset-v9")


def test_verification_detects_unchanged_changed_and_split_membership() -> None:
    records, assessments, split, validation = _pipeline()
    versioner = DatasetVersioner()
    version = versioner.create_version("dataset-v1", records, split, validation, quality_policy_version="quality-1")
    verified = versioner.verify_version(version, records, split, validation, quality_policy_version="quality-1")
    assert verified.valid is True
    changed = replace(records[0], task="Changed backend task")
    changed_records = (changed,) + records[1:]
    changed_assessments = tuple(DatasetQualityEvaluator().evaluate(record) for record in changed_records)
    changed_split = DatasetSplitter(policy=DatasetSplitPolicy(train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, seed=43)).split(changed_records, quality_assessments=changed_assessments)
    changed_validation = DatasetValidator().validate_dataset(changed_records, split_result=changed_split, quality_assessments=changed_assessments)
    failed = versioner.verify_version(version, changed_records, changed_split, changed_validation, quality_policy_version="quality-1")
    assert failed.valid is False
    assert changed.record_id in failed.changed_record_ids
    assert failed.train_membership_changed or failed.validation_membership_changed or failed.test_membership_changed


def test_comparison_detects_added_removed_and_partition_changes() -> None:
    records, _, split, validation = _pipeline(9, seed=42)
    versioner = DatasetVersioner()
    v1 = versioner.create_version("dataset-v1", records, split, validation)
    records2, _, split2, validation2 = _pipeline(10, seed=43)
    v2 = versioner.create_version("dataset-v2", records2, split2, validation2, parent_version="dataset-v1")
    comparison = compare_versions(v1, v2)
    assert comparison.status is VersionComparisonStatus.DIFFERENT
    assert set(comparison.added_record_ids) == {records2[-1].record_id}
    assert comparison.fingerprint_changed is True
    assert comparison.train_membership_changed or comparison.validation_membership_changed or comparison.test_membership_changed


def test_secrets_and_resource_limits_are_rejected() -> None:
    records, _, split, validation = _pipeline()
    with pytest.raises(DatasetVersionError):
        DatasetVersioner(limits=DatasetVersionLimits(max_metadata_bytes=32)).create_version("dataset-v1", records, split, validation, metadata={"description": "this is too long for the configured bound"})
    with pytest.raises(DatasetVersionError):
        DatasetVersioner().create_version("dataset-v1", records, split, validation, metadata={"token": "runtime-secret"})


def test_malformed_version_names_and_manifest_future_schema_fail() -> None:
    records, _, split, validation = _pipeline()
    with pytest.raises(DatasetVersionError):
        DatasetVersioner().create_version("v1", records, split, validation)
    version = DatasetVersioner().create_version("dataset-v1", records, split, validation)
    payload = version.manifest.to_dict()
    payload["schema_version"] = "99.0"
    with pytest.raises(DatasetVersionError):
        DatasetVersionManifest.from_dict(payload)
