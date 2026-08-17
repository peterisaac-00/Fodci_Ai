from __future__ import annotations

from pathlib import Path

from backend_ai.model_artifact import EvaluationReference, ModelArtifact, ModelArtifactRegistry

from tests.unit.test_model_artifact import _completed_run


def test_phase114_end_to_end_candidate_artifact_and_registry(tmp_path: Path) -> None:
    run = _completed_run(tmp_path, "phase114-integration")
    artifact = run.create_model_artifact(
        tmp_path / "models" / "candidate-v1",
        model_id="fodci-backend-candidate-v1",
        evaluation_reference=EvaluationReference.not_evaluated(),
        created_at="fixed-integration-time",
    )
    registry = ModelArtifactRegistry(tmp_path / "models.json")
    entry = registry.register(artifact)

    loaded = ModelArtifact.load(artifact.root)
    assert loaded.verify().valid
    assert entry.model_id == loaded.model_id
    assert entry.model_version == loaded.model_version
    assert entry.artifact_fingerprint == loaded.fingerprint
    assert registry.current_candidate() is None
    assert registry.current_official() is None
