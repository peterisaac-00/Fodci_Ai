from __future__ import annotations

from dataclasses import replace

import pytest

from backend_ai.agent.dataset_quality import DatasetQualityEvaluator, QualityDecision
from backend_ai.agent.dataset_schema import DatasetProjectContext, DatasetRecord, derive_dataset_record_id
from backend_ai.agent.dataset_split import (
    DATASET_SPLIT_VERSION,
    DatasetSplitError,
    DatasetSplitGroup,
    DatasetSplitPolicy,
    DatasetSplitter,
    DuplicateDatasetRecordError,
    validate_split,
)

from tests.unit.test_dataset_schema import _candidate


def _record(index: int, project: str = "project-a") -> DatasetRecord:
    base = DatasetRecord.from_candidate(_candidate())
    experience_id = f"experience-{index}"
    provenance = replace(
        base.provenance,
        experience_id=experience_id,
        project_identity={"project_id": project, "project_root": f"/{project}"},
    )
    return replace(
        base,
        record_id=derive_dataset_record_id(experience_id, provenance.source_schema_version),
        experience_id=experience_id,
        task=f"Fix API validation for service {index}",
        project_context=DatasetProjectContext(project, f"/{project}"),
        provenance=provenance,
    )


def _records(count: int) -> tuple[DatasetRecord, ...]:
    return tuple(_record(index) for index in range(count))


def test_empty_and_small_record_level_datasets_are_explicit_and_conservative() -> None:
    splitter = DatasetSplitter()
    empty = splitter.split(())
    assert empty.counts == {"train": 0, "validation": 0, "test": 0, "total": 0}
    one = splitter.split((_record(1),))
    assert one.counts == {"train": 1, "validation": 0, "test": 0, "total": 1}
    two = splitter.split(_records(2))
    assert two.total_records == 2
    validate_split(two)


def test_largest_remainder_allocation_accounts_for_every_record() -> None:
    result = DatasetSplitter(policy=DatasetSplitPolicy(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)).split(_records(10))
    assert result.counts == {"train": 8, "validation": 1, "test": 1, "total": 10}
    result_70 = DatasetSplitter(policy=DatasetSplitPolicy(train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15)).split(_records(20))
    assert result_70.counts == {"train": 14, "validation": 3, "test": 3, "total": 20}
    assert result.manifest.total_records == result.counts["total"] == 10


def test_same_records_policy_seed_and_split_version_produce_identical_manifest() -> None:
    policy = DatasetSplitPolicy(seed=42)
    first = DatasetSplitter(policy=policy).split(_records(24))
    second = DatasetSplitter(policy=policy).split(tuple(reversed(_records(24))))
    assert first.manifest.to_json() == second.manifest.to_json()
    assert first.to_dict()["manifest"] == second.to_dict()["manifest"]
    assert first.manifest.split_version == DATASET_SPLIT_VERSION


def test_different_seed_is_deterministic_and_can_change_membership() -> None:
    first = DatasetSplitter(policy=DatasetSplitPolicy(seed=42)).split(_records(30))
    second = DatasetSplitter(policy=DatasetSplitPolicy(seed=43)).split(_records(30))
    repeat = DatasetSplitter(policy=DatasetSplitPolicy(seed=43)).split(_records(30))
    assert first.manifest.to_json() != second.manifest.to_json()
    assert second.manifest.to_json() == repeat.manifest.to_json()


def test_partitions_are_disjoint_and_cover_all_eligible_records() -> None:
    records = _records(31)
    result = DatasetSplitter().split(records)
    partition_ids = [record.record_id for partition in (result.train, result.validation, result.test) for record in partition]
    assert len(partition_ids) == len(set(partition_ids))
    assert set(partition_ids) == {record.record_id for record in records}
    assert set(result.manifest.record_ids["train"]).isdisjoint(result.manifest.record_ids["validation"])
    assert set(result.manifest.record_ids["train"]).isdisjoint(result.manifest.record_ids["test"])
    assert set(result.manifest.record_ids["validation"]).isdisjoint(result.manifest.record_ids["test"])
    validate_split(result)


def test_duplicate_ids_and_invalid_inputs_fail_explicitly() -> None:
    record = _record(1)
    with pytest.raises(DuplicateDatasetRecordError):
        DatasetSplitter().split((record, record))
    with pytest.raises(DatasetSplitError):
        DatasetSplitter().split((record.to_dict(),))  # type: ignore[arg-type]
    with pytest.raises(DatasetSplitError):
        DatasetSplitter().split(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_ratio": -0.1, "validation_ratio": 0.5, "test_ratio": 0.6},
        {"train_ratio": 0.8, "validation_ratio": 0.3, "test_ratio": 0.1},
        {"train_ratio": float("nan"), "validation_ratio": 0.1, "test_ratio": 0.9},
        {"train_ratio": float("inf"), "validation_ratio": 0.0, "test_ratio": 0.0},
        {"train_ratio": 0.0, "validation_ratio": 0.0, "test_ratio": 0.0},
    ],
)
def test_invalid_ratios_fail_deterministically(kwargs: dict[str, float]) -> None:
    with pytest.raises(DatasetSplitError):
        DatasetSplitPolicy(**kwargs)


def test_minimum_counts_and_non_empty_policy_are_explicit() -> None:
    policy = DatasetSplitPolicy(minimum_train_records=1, minimum_validation_records=1, minimum_test_records=1)
    with pytest.raises(DatasetSplitError):
        DatasetSplitter(policy=policy).split(_records(2))
    result = DatasetSplitter(policy=policy).split(_records(3))
    assert all(result.counts[name] >= 1 for name in ("train", "validation", "test"))
    with pytest.raises(DatasetSplitError):
        DatasetSplitter(policy=DatasetSplitPolicy(require_non_empty_partitions=True)).split(_records(2))


def test_quality_assessments_exclude_review_and_reject_without_re_evaluation() -> None:
    records = _records(3)
    assessments = [DatasetQualityEvaluator().evaluate(record) for record in records]
    assessments[1] = replace(assessments[1], decision=QualityDecision.REVIEW, warnings=("manual_review",))
    assessments[2] = replace(assessments[2], decision=QualityDecision.REJECT, reasons=("failed_policy",))
    result = DatasetSplitter().split(records, quality_assessments=assessments)
    assert result.total_records == 1
    assert set(result.excluded_record_ids) == {records[1].record_id, records[2].record_id}
    assert result.quality_decisions[records[1].record_id] == "REVIEW"
    assert result.quality_decisions[records[2].record_id] == "REJECT"


def test_split_accepted_preserves_all_quality_decisions_and_only_accepts_eligible_records() -> None:
    records = _records(4)
    assessments = tuple(DatasetQualityEvaluator().evaluate(record) for record in records)
    from backend_ai.agent.dataset_quality import DatasetFilteringResult

    accepted = (records[0], records[1])
    review = replace(assessments[2], decision=QualityDecision.REVIEW, warnings=("manual_review",))
    rejected = replace(assessments[3], decision=QualityDecision.REJECT, reasons=("failed_policy",))
    filtered_assessments = (assessments[0], assessments[1], review, rejected)
    filtered = DatasetFilteringResult(accepted, (rejected,), (review,), filtered_assessments, ("manual_review", "failed_policy"))
    result = DatasetSplitter().split_accepted(filtered)
    assert result.total_records == 2
    assert set(result.excluded_record_ids) == {records[2].record_id, records[3].record_id}
    assert set(result.quality_decisions.values()) == {"ACCEPT", "REVIEW", "REJECT"}


def test_project_grouping_prevents_cross_partition_leakage_and_reports_actual_ratios() -> None:
    records = tuple(_record(index, f"project-{index // 3}") for index in range(9))
    policy = DatasetSplitPolicy(group_by=DatasetSplitGroup.PROJECT, train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, require_non_empty_partitions=True)
    result = DatasetSplitter(policy=policy).split(records)
    assert result.manifest.group_by is DatasetSplitGroup.PROJECT
    assert len(result.manifest.group_ids["train"]) == 1
    assert len(result.manifest.group_ids["validation"]) == 1
    assert len(result.manifest.group_ids["test"]) == 1
    groups = {}
    for name, partition in (("train", result.train), ("validation", result.validation), ("test", result.test)):
        for record in partition:
            project = record.project_context.project_id
            assert groups.setdefault(project, name) == name
    assert sum(result.manifest.actual_ratios.values()) == pytest.approx(1.0)


def test_small_grouped_dataset_has_explicit_error_when_non_empty_partitions_required() -> None:
    records = (_record(1, "project-a"), _record(2, "project-b"))
    policy = DatasetSplitPolicy(group_by=DatasetSplitGroup.PROJECT, require_non_empty_partitions=True)
    with pytest.raises(DatasetSplitError):
        DatasetSplitter(policy=policy).split(records)


def test_split_is_read_only_and_preserves_full_record_provenance() -> None:
    records = _records(8)
    before = tuple(record.to_json() for record in records)
    result = DatasetSplitter().split(records)
    after = tuple(record.to_json() for record in records)
    assert before == after
    for partition in (result.train, result.validation, result.test):
        for record in partition:
            assert record.record_id
            assert record.experience_id == record.provenance.experience_id
            assert record.schema_version == "1.0"
            assert record.provenance.source_type == "experience_record"
