from __future__ import annotations

from pathlib import Path

import pytest
import torch

from backend_ai.agent.training_dataset import TrainingDatasetBuilder
from backend_ai.checkpoint import CheckpointManager
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import FodciTokenizer
from backend_ai.dataset import TrainingExample
from backend_ai.training import (
    FineTuningConfig,
    FineTuningConfigurationError,
    FineTuningDatasetError,
    FineTuningModelError,
    FineTuningRunner,
    FineTuningStatus,
    FodciModelAdapter,
    load_run_result,
)
from backend_ai.training.config import TrainingConfig
from backend_ai.training import FodciTrainer

from tests.unit.test_training_dataset import _config as dataset_config
from tests.unit.test_training_dataset import _successful_records


def _artifact(tmp_path: Path):
    result = TrainingDatasetBuilder(config=dataset_config()).build_from_experience_records(_successful_records())
    path = result.artifact.write(tmp_path / "dataset")
    return result.artifact, path


def _base_checkpoint(tmp_path: Path) -> Path:
    model = FodciModel(
        ModelConfig(
            vocab_size=300,
            context_length=2048,
            hidden_size=16,
            num_layers=1,
            num_attention_heads=4,
            feed_forward_size=32,
            dropout=0.0,
            seed=7,
        )
    )
    manager = CheckpointManager(tmp_path / "base", model_version="fodci-tiny-v1", tokenizer_version=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return manager.save(model, optimizer, TrainingConfig(epochs=1, output_dir=tmp_path / "base"), epoch=0, global_step=0, path=tmp_path / "base" / "base.pt")


def _fine_config(tmp_path: Path, run_id: str, epochs: int = 1) -> FineTuningConfig:
    return FineTuningConfig(
        run_id=run_id,
        candidate_model_version="candidate-v1",
        epochs=epochs,
        max_steps=1 if epochs == 1 else None,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        device="cpu",
        checkpoint_interval=1,
        validation_interval=1,
        output_directory=tmp_path / "runs",
    )


def test_fine_tuning_config_validates_and_records_effective_values(tmp_path: Path) -> None:
    config = _fine_config(tmp_path, "config-test")
    assert config.to_training_config().gradient_accumulation_steps == 1
    assert config.to_dict()["candidate_model_version"] == "candidate-v1"
    with pytest.raises(FineTuningConfigurationError):
        FineTuningConfig(run_id="bad id")
    with pytest.raises(FineTuningConfigurationError):
        FineTuningConfig(run_id="bad", gradient_accumulation_steps=0)
    with pytest.raises(FineTuningConfigurationError):
        FineTuningConfig(run_id="bad", candidate_model_version="fodci-v1")


def test_fine_tuning_cpu_smoke_creates_traceable_candidate_and_metrics(tmp_path: Path) -> None:
    _, dataset_path = _artifact(tmp_path)
    base_path = _base_checkpoint(tmp_path)
    result = FineTuningRunner.from_paths(
        base_checkpoint=base_path,
        dataset_directory=dataset_path,
        config=_fine_config(tmp_path, "smoke"),
    ).run()

    assert result.status is FineTuningStatus.COMPLETED
    assert result.candidate_model is not None
    assert result.dataset.training_examples == 1
    assert result.dataset.validation_examples == 1
    assert result.metrics
    assert result.metrics[0]["train_loss"] == pytest.approx(result.metrics[0]["train_loss"])
    assert result.checkpoints
    final_checkpoint = Path(result.candidate_model.model_path or "")
    assert final_checkpoint.is_file()
    assert (Path(result.run_directory) / "run.json").is_file()
    assert (Path(result.run_directory) / "metrics.json").is_file()
    loaded = load_run_result(Path(result.run_directory) / "run.json")
    assert loaded.run_id == result.run_id
    assert loaded.dataset.dataset_fingerprint == result.dataset.dataset_fingerprint


def test_resume_preserves_phase113_lineage(tmp_path: Path) -> None:
    _, dataset_path = _artifact(tmp_path)
    base_path = _base_checkpoint(tmp_path)
    first = FineTuningRunner.from_paths(
        base_checkpoint=base_path,
        dataset_directory=dataset_path,
        config=_fine_config(tmp_path, "first"),
    ).run()
    checkpoint = Path(first.candidate_model.model_path or "")
    resumed = FineTuningRunner.from_paths(
        base_checkpoint=base_path,
        dataset_directory=dataset_path,
        config=_fine_config(tmp_path, "resumed", epochs=2),
    ).run(resume_checkpoint=checkpoint)

    assert resumed.status is FineTuningStatus.COMPLETED
    assert resumed.resumed_from == str(checkpoint)
    assert resumed.metrics[0]["epoch"] == 2


def test_missing_or_incompatible_inputs_fail_explicitly(tmp_path: Path) -> None:
    with pytest.raises(FineTuningDatasetError):
        FineTuningRunner.from_paths(
            base_checkpoint=tmp_path / "missing.pt",
            dataset_directory=tmp_path / "missing-dataset",
            config=_fine_config(tmp_path, "missing"),
        )
    with pytest.raises(FineTuningModelError):
        FodciModelAdapter.from_checkpoint(tmp_path / "missing.pt")


def test_legacy_checkpoint_resume_is_explicitly_failed(tmp_path: Path) -> None:
    _, dataset_path = _artifact(tmp_path)
    base_path = _base_checkpoint(tmp_path)
    result = FineTuningRunner.from_paths(
        base_checkpoint=base_path,
        dataset_directory=dataset_path,
        config=_fine_config(tmp_path, "legacy-resume"),
    ).run(resume_checkpoint=base_path)
    assert result.status is FineTuningStatus.FAILED
    assert result.candidate_model is None
    assert result.error is not None
    assert "Phase 11.3 checkpoint" in result.error
    assert (Path(result.run_directory) / "run.json").is_file()


def test_gradient_accumulation_updates_in_bounded_optimizer_steps(tmp_path: Path) -> None:
    model = FodciModel(ModelConfig(vocab_size=32, context_length=8, hidden_size=16, num_layers=1, num_attention_heads=4, feed_forward_size=32, seed=7))
    examples = [TrainingExample((1, 2, 3, 4), (2, 3, 4, 5), f"example-{index}") for index in range(4)]
    trainer = FodciTrainer(
        model,
        lambda: iter(examples),
        config=TrainingConfig(epochs=1, batch_size=1, gradient_accumulation_steps=2, output_dir=tmp_path, checkpoint_interval=0),
    )
    result = trainer.train()
    assert result.global_step == 2
    assert result.final_metrics is not None
    assert result.final_metrics.training_steps == 2


def test_checkpoint_tokenizer_mismatch_is_rejected(tmp_path: Path) -> None:
    base_path = _base_checkpoint(tmp_path)
    tokenizer_path = tmp_path / "wrong-tokenizer.json"
    FodciTokenizer(vocab_size=301).save(tokenizer_path)
    with pytest.raises(FineTuningModelError, match="vocabulary"):
        FodciModelAdapter.from_checkpoint(base_path, tokenizer_path=tokenizer_path)
