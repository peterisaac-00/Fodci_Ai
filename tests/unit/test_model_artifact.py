from __future__ import annotations

from pathlib import Path

import pytest

from backend_ai.model_artifact import (
    EvaluationReference,
    ModelArtifact,
    ModelArtifactConflictError,
    ModelArtifactIntegrityError,
    ModelArtifactRegistry,
    ModelArtifactStorageError,
    compute_training_config_fingerprint,
)
from backend_ai.training import FineTuningRunner

from tests.unit.test_fine_tuning import _base_checkpoint, _fine_config
from tests.unit.test_training_dataset import _config as dataset_config
from tests.unit.test_training_dataset import _successful_records
from backend_ai.agent.training_dataset import TrainingDatasetBuilder


def _completed_run(tmp_path: Path, run_id: str = "artifact-run"):
    dataset_result = TrainingDatasetBuilder(config=dataset_config()).build_from_experience_records(_successful_records())
    dataset_directory = dataset_result.artifact.write(tmp_path / "dataset" / "dataset-v1")
    base_checkpoint = _base_checkpoint(tmp_path)
    return FineTuningRunner.from_paths(
        base_checkpoint=base_checkpoint,
        dataset_directory=dataset_directory,
        config=_fine_config(tmp_path, run_id),
    ).run()


def test_model_artifact_creation_has_required_metadata_and_round_trip(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    artifact = run.create_model_artifact(
        tmp_path / "models" / "candidate-v1",
        model_id="fodci-backend-candidate-v1",
        evaluation_reference=EvaluationReference.recorded("baseline-fodci-tiny-v1-2026-08-17-1", protocol_version="11.1"),
        created_at="fixed-artifact-time",
    )

    assert artifact.metadata.model_version == "candidate-v1"
    assert artifact.metadata.model_id == "fodci-backend-candidate-v1"
    assert artifact.metadata.base_model.model_fingerprint is not None
    assert artifact.metadata.dataset_version == run.dataset.dataset_version
    assert artifact.metadata.dataset_fingerprint == run.dataset.dataset_fingerprint
    assert artifact.metadata.training_config_fingerprint == compute_training_config_fingerprint(run.configuration)
    assert artifact.metadata.checkpoint_fingerprint.startswith("sha256:")
    assert artifact.metadata.evaluation_reference.status == "RECORDED"
    assert artifact.fingerprint.startswith("sha256:")
    assert (artifact.root / "metadata.json").is_file()
    assert (artifact.root / "evaluation.json").is_file()
    assert artifact.checkpoint_path.is_file()
    assert artifact.verify().valid

    loaded = ModelArtifact.load(artifact.root)
    assert loaded.to_dict() == artifact.to_dict()
    assert loaded.fingerprint == artifact.fingerprint


def test_artifact_fingerprint_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    run = _completed_run(tmp_path, "deterministic-run")
    first = run.create_model_artifact(tmp_path / "models" / "first", model_id="fodci-backend-deterministic", created_at="fixed")
    second = run.create_model_artifact(tmp_path / "models" / "second", model_id="fodci-backend-deterministic", created_at="fixed")
    assert first.fingerprint == second.fingerprint
    assert first.metadata.to_json() == second.metadata.to_json()

    changed = run.create_model_artifact(tmp_path / "models" / "changed", model_id="fodci-backend-changed", created_at="fixed")
    assert changed.fingerprint != first.fingerprint


def test_artifact_tampering_is_detected(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    artifact = run.create_model_artifact(tmp_path / "models" / "tampered", model_id="fodci-backend-tampered", created_at="fixed")
    artifact.checkpoint_path.write_bytes(artifact.checkpoint_path.read_bytes() + b"tamper")
    verification = artifact.verify()
    assert not verification.valid
    with pytest.raises(ModelArtifactIntegrityError):
        ModelArtifact.load(artifact.root)


def test_artifact_and_registry_collisions_are_rejected(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    artifact = run.create_model_artifact(tmp_path / "models" / "collision", model_id="fodci-backend-collision", created_at="fixed")
    with pytest.raises(ModelArtifactConflictError):
        run.create_model_artifact(tmp_path / "models" / "collision", model_id="fodci-backend-collision", created_at="fixed")

    registry = ModelArtifactRegistry(tmp_path / "registry.json")
    entry = registry.register(artifact)
    assert entry.status == "UNASSIGNED"
    assert registry.current_candidate() is None
    assert registry.current_official() is None
    with pytest.raises(ModelArtifactConflictError):
        registry.register(artifact)
    second = run.create_model_artifact(tmp_path / "models" / "collision-second", model_id="fodci-backend-collision-second", created_at="fixed")
    with pytest.raises(ModelArtifactConflictError):
        registry.register(second)
    candidate = registry.set_current_candidate(entry.model_id)
    assert candidate.status == "CANDIDATE"
    assert registry.current_candidate().model_id == entry.model_id
    assert registry.current_official() is None
    assert registry.load_artifact(entry.model_id).fingerprint == artifact.fingerprint

    reloaded = ModelArtifactRegistry(tmp_path / "registry.json")
    assert reloaded.current_candidate().model_id == entry.model_id
    assert reloaded.current_official() is None


def test_invalid_metadata_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "invalid"
    directory.mkdir()
    (directory / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelArtifactStorageError):
        ModelArtifact.load(directory)
