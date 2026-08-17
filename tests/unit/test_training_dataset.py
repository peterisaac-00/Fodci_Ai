from __future__ import annotations

import pytest

from backend_ai.agent.experience_dataset import ExperienceDatasetExtractor
from backend_ai.agent.dataset_schema import DatasetRecord
from backend_ai.agent.experience_records import (
    ExperienceLifecycleStatus,
    ExperienceOutcomeStatus,
    ExperienceRecords,
    ExperienceVerification,
)
from backend_ai.agent.dataset_split import DatasetSplitGroup, DatasetSplitPolicy
from backend_ai.agent.training_dataset import (
    TestSetAccessError,
    TrainingDatasetArtifact,
    TrainingDatasetBuilder,
    TrainingDatasetConfig,
    TrainingDatasetError,
    TrainingDatasetLoader,
    TrainingDatasetRejectionReason,
    TrainingSplit,
    derive_training_example_id,
)

STAMP = "2026-01-01T00:00:00+00:00"


def _successful_records(count: int = 3) -> ExperienceRecords:
    records = ExperienceRecords(clock=lambda: STAMP)
    for index in range(count):
        session = records.start_experience(f"Fix FastAPI authentication issue {index}")
        attempt_id = session.start_attempt()
        session.record_action("inspect", "inspect backend endpoint", attempt_id=attempt_id)
        session.record_attempt_result("implemented and verified", attempt_id=attempt_id)
        session.record_verification(ExperienceVerification(1, 1, 0, "passed", "all tests passed", STAMP))
        session.finalize(
            status=ExperienceLifecycleStatus.COMPLETED,
            outcome=ExperienceOutcomeStatus.SUCCESS,
            final_solution=f"Updated the FastAPI handler safely for case {index}.",
            final_summary="The endpoint now returns the documented response.",
        )
    return records


def _config() -> TrainingDatasetConfig:
    return TrainingDatasetConfig(
        dataset_version="dataset-v1",
        split_policy=DatasetSplitPolicy(
            seed=2026,
            group_by=DatasetSplitGroup.RECORD,
            minimum_train_records=1,
            minimum_validation_records=1,
            minimum_test_records=1,
            require_non_empty_partitions=True,
        ),
        created_at="fixed-test-build",
    )


def test_training_dataset_build_is_traceable_and_deterministic() -> None:
    first = TrainingDatasetBuilder(config=_config()).build_from_experience_records(_successful_records())
    second = TrainingDatasetBuilder(config=_config()).build_from_experience_records(_successful_records())

    assert first.report.source_record_count == 3
    assert first.report.accepted_record_count == 3
    assert first.report.rejected_record_count == 0
    assert first.report.duplicate_count == 0
    assert first.report.training_example_count == 3
    assert first.artifact.manifest.dataset_fingerprint == second.artifact.manifest.dataset_fingerprint
    assert first.artifact.to_dict() == second.artifact.to_dict()
    assert {item.source_record_id for item in first.artifact.train + first.artifact.validation + first.artifact.test} == set(first.artifact.manifest.accepted_record_ids)
    assert all(item.example_id == derive_training_example_id(item.source_record_id) for item in first.artifact.train + first.artifact.validation + first.artifact.test)
    assert set(first.artifact.manifest.source_record_ids_by_split) == {"train", "validation", "test"}


def test_failed_and_duplicate_source_records_are_rejected_with_reasons() -> None:
    records = _successful_records(1)
    duplicate = DatasetRecord.from_candidate(ExperienceDatasetExtractor().extract(records.list()[0]))
    failed_records = ExperienceRecords(clock=lambda: STAMP)
    failed_session = failed_records.start_experience("Fix FastAPI failed deployment")
    failed_attempt = failed_session.start_attempt()
    failed_session.record_attempt_result("could not complete", attempt_id=failed_attempt)
    failed_session.finalize(status=ExperienceLifecycleStatus.FAILED, outcome=ExperienceOutcomeStatus.FAILURE, final_summary="failed")

    result = TrainingDatasetBuilder(config=TrainingDatasetConfig(created_at="fixed-test-build")).build_from_dataset_records((duplicate, duplicate))
    assert result.report.duplicate_count == 1
    assert any(item.reason is TrainingDatasetRejectionReason.DUPLICATE_RECORD for item in result.report.rejections)

    failed_result = TrainingDatasetBuilder(config=TrainingDatasetConfig(created_at="fixed-test-build")).build_from_experience_records(failed_records)
    assert failed_result.report.accepted_record_count == 0
    assert any(item.reason is TrainingDatasetRejectionReason.QUALITY_REJECTED for item in failed_result.report.rejections)


def test_invalid_mapping_is_rejected_without_entering_artifact() -> None:
    result = TrainingDatasetBuilder(config=TrainingDatasetConfig(created_at="fixed-test-build")).build_from_dataset_records(({"experience_id": "exp-invalid"},))
    assert result.report.source_record_count == 1
    assert result.report.training_example_count == 0
    assert any(item.reason is TrainingDatasetRejectionReason.SCHEMA_INVALID for item in result.report.rejections)


def test_artifact_round_trip_and_test_set_access_control(tmp_path) -> None:
    result = TrainingDatasetBuilder(config=_config()).build_from_experience_records(_successful_records())
    artifact_dir = result.artifact.write(tmp_path / "training" / "dataset-v1")
    loaded = TrainingDatasetArtifact.load(artifact_dir)
    assert loaded.to_dict() == result.artifact.to_dict()
    assert TrainingDatasetLoader.load_for_training(artifact_dir) == loaded.train
    assert TrainingDatasetLoader.load_for_validation(artifact_dir) == loaded.validation
    assert TrainingDatasetLoader.load_for_benchmark(artifact_dir) == loaded.test
    with pytest.raises(TestSetAccessError):
        TrainingDatasetLoader.load_split(artifact_dir, TrainingSplit.TEST, purpose="training")
    with pytest.raises(TestSetAccessError):
        TrainingDatasetLoader.load_split(artifact_dir, TrainingSplit.TEST, purpose="validation")


def test_manifest_rejects_changed_dataset_payload(tmp_path) -> None:
    result = TrainingDatasetBuilder(config=_config()).build_from_experience_records(_successful_records())
    artifact_dir = result.artifact.write(tmp_path / "dataset")
    train_path = artifact_dir / "train.json"
    train_path.write_text(train_path.read_text(encoding="utf-8").replace("Fix FastAPI", "Change FastAPI", 1), encoding="utf-8")
    with pytest.raises(TrainingDatasetError):
        TrainingDatasetArtifact.load(artifact_dir)
