from __future__ import annotations

from pathlib import Path

from backend_ai.agent.training_dataset import TrainingDatasetBuilder
from backend_ai.training import FineTuningRunner, FineTuningStatus, load_run_result

from tests.unit.test_fine_tuning import _base_checkpoint, _fine_config
from tests.unit.test_training_dataset import _config as dataset_config
from tests.unit.test_training_dataset import _successful_records


def test_phase113_end_to_end_candidate_run(tmp_path: Path) -> None:
    dataset_result = TrainingDatasetBuilder(config=dataset_config()).build_from_experience_records(_successful_records())
    dataset_directory = dataset_result.artifact.write(tmp_path / "dataset" / "dataset-v1")
    base_checkpoint = _base_checkpoint(tmp_path)
    config = _fine_config(tmp_path, "integration-smoke")

    result = FineTuningRunner.from_paths(
        base_checkpoint=base_checkpoint,
        dataset_directory=dataset_directory,
        config=config,
    ).run()

    assert result.status is FineTuningStatus.COMPLETED
    assert result.base_model.model_fingerprint is not None
    assert result.dataset.dataset_fingerprint == dataset_result.artifact.manifest.dataset_fingerprint
    assert result.candidate_model is not None
    assert result.candidate_model.model_version == "candidate-v1"
    assert Path(result.candidate_model.model_path or "").is_file()
    loaded = load_run_result(Path(result.run_directory) / "run.json")
    assert loaded.to_json() == result.to_json()
