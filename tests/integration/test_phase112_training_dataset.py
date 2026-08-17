from __future__ import annotations

from backend_ai.agent.dataset_versioning import DatasetVersionRegistry, DatasetVersioner
from backend_ai.agent.training_dataset import TrainingDatasetArtifact, TrainingDatasetBuilder, TrainingDatasetConfig, TrainingDatasetLoader

from tests.unit.test_training_dataset import _config, _successful_records


def test_phase112_end_to_end_training_artifact(tmp_path) -> None:
    config = _config()
    registry = DatasetVersionRegistry(tmp_path / ".fodci" / "datasets.json")
    versioner = DatasetVersioner(registry=registry)
    builder = TrainingDatasetBuilder(config=config, versioner=versioner)

    result = builder.build_from_experience_records(_successful_records())
    version = registry.require_version(config.dataset_version)

    assert result.validation_status.value == "VALID"
    assert result.report.source_record_count == 3
    assert result.report.accepted_record_count == 3
    assert result.report.rejected_record_count == 0
    assert result.artifact.manifest.source_dataset_fingerprint == version.dataset_fingerprint
    assert version.manifest.train_record_ids or version.manifest.validation_record_ids or version.manifest.test_record_ids
    canonical = tuple(result.split_result.train + result.split_result.validation + result.split_result.test)
    validation = builder.validator.validate_dataset(canonical, split_result=result.split_result)
    assert versioner.verify_version(version, canonical, result.split_result, validation, quality_policy=config.quality_policy, quality_policy_version=config.quality_policy_version).valid is True

    artifact_path = result.artifact.write(tmp_path / "training" / config.dataset_version)
    loaded = TrainingDatasetArtifact.load(artifact_path)
    assert loaded.manifest.dataset_fingerprint == result.artifact.manifest.dataset_fingerprint
    assert TrainingDatasetLoader.load_for_training(artifact_path) == loaded.train
    assert TrainingDatasetLoader.load_for_validation(artifact_path) == loaded.validation
    assert TrainingDatasetLoader.load_for_benchmark(artifact_path) == loaded.test
