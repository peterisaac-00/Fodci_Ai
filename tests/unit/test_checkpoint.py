from __future__ import annotations

from pathlib import Path

import pytest
import torch

from backend_ai.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointFormatError,
    CheckpointManager,
)
from backend_ai.dataset import TrainingExample
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.training import FodciTrainer, TrainingConfig


def _model(**overrides: object) -> FodciModel:
    values: dict[str, object] = {
        "vocab_size": 32,
        "context_length": 8,
        "hidden_size": 16,
        "num_layers": 2,
        "num_attention_heads": 4,
        "feed_forward_size": 32,
        "dropout": 0.0,
        "seed": 7,
    }
    values.update(overrides)
    return FodciModel(ModelConfig(**values))


def _optimizer_with_state(model: FodciModel) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    target_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(
        model(input_ids).reshape(-1, model.config.vocab_size),
        target_ids.reshape(-1),
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def _config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(epochs=2, batch_size=1, output_dir=tmp_path, seed=2026)


def test_save_inspect_load_restores_model_optimizer_and_metadata(tmp_path: Path) -> None:
    model = _model()
    optimizer = _optimizer_with_state(model)
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.save(
        model,
        optimizer,
        _config(tmp_path),
        epoch=2,
        global_step=7,
        metrics={"validation_loss": 1.25},
        path=tmp_path / "v1.pt",
    )
    info = manager.inspect(checkpoint)

    restored_model = _model(seed=19)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    loaded = manager.load(
        checkpoint,
        restored_model,
        restored_optimizer,
        device=torch.device("cpu"),
    )

    assert info.metadata.model_version == "fodci-tiny-v1"
    assert info.metadata.format_version == 2
    assert info.metadata.vocabulary_size == 32
    assert info.metadata.context_length == 8
    assert info.metadata.epoch == 2
    assert info.metadata.global_step == 7
    assert info.metadata.metrics["validation_loss"] == 1.25
    assert loaded.epoch == 2
    assert loaded.global_step == 7
    assert loaded.metrics["validation_loss"] == 1.25
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(model.parameters(), restored_model.parameters())
    )
    assert restored_optimizer.state


def test_list_latest_and_best_use_metadata_not_filename_order(tmp_path: Path) -> None:
    model = _model()
    optimizer = _optimizer_with_state(model)
    manager = CheckpointManager(tmp_path)
    for filename, step, loss in (("z.pt", 2, 1.5), ("a.pt", 8, 1.1), ("m.pt", 5, 0.9)):
        manager.save(
            model,
            optimizer,
            _config(tmp_path),
            epoch=step,
            global_step=step,
            metrics={"validation_loss": loss},
            path=tmp_path / filename,
        )

    listed = manager.list()

    assert [info.metadata.global_step for info in listed] == [2, 5, 8]
    assert manager.latest() is not None
    assert manager.latest().metadata.global_step == 8
    assert manager.best() is not None
    assert manager.best().metadata.global_step == 5
    assert manager.exists(tmp_path / "a.pt")
    assert not manager.exists(tmp_path / "missing.pt")


def test_compatibility_rejects_model_and_tokenizer_mismatches(tmp_path: Path) -> None:
    model = _model()
    optimizer = _optimizer_with_state(model)
    manager = CheckpointManager(tmp_path)
    path = manager.save(model, optimizer, _config(tmp_path), epoch=1, global_step=1)

    with pytest.raises(CheckpointCompatibilityError, match="vocabulary size"):
        manager.load(path, _model(vocab_size=33), torch.optim.AdamW(_model(vocab_size=33).parameters()), device=torch.device("cpu"))
    with pytest.raises(CheckpointCompatibilityError, match="hidden_size"):
        incompatible = _model(hidden_size=32, num_attention_heads=4, feed_forward_size=32)
        manager.load(path, incompatible, torch.optim.AdamW(incompatible.parameters()), device=torch.device("cpu"))
    with pytest.raises(CheckpointCompatibilityError, match="Tokenizer version"):
        wrong_tokenizer = CheckpointManager(tmp_path, tokenizer_version=999)
        wrong_tokenizer.load(path, _model(), torch.optim.AdamW(_model().parameters()), device=torch.device("cpu"))
    with pytest.raises(CheckpointCompatibilityError, match="Model version"):
        wrong_version = CheckpointManager(tmp_path, model_version="other-model")
        wrong_version.load(path, _model(), torch.optim.AdamW(_model().parameters()), device=torch.device("cpu"))


def test_corrupt_missing_and_unsupported_checkpoints_fail_explicitly(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        manager.inspect(tmp_path / "missing.pt")

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointFormatError, match="Unable to read"):
        manager.inspect(corrupt)

    unsupported = tmp_path / "unsupported.pt"
    torch.save(
        {"metadata": {"format": "fodci.checkpoint", "format_version": 999}},
        unsupported,
    )
    with pytest.raises(CheckpointFormatError, match="missing required fields|Unsupported"):
        manager.inspect(unsupported)


def test_atomic_save_removes_temporary_file_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = CheckpointManager(tmp_path)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def fail_save(*args, **kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated interrupted write"):
        manager.save(model, optimizer, _config(tmp_path), epoch=1, global_step=1, path=tmp_path / "safe.pt")

    assert not (tmp_path / "safe.pt").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_cpu_resume_restores_state_and_continues_training(tmp_path: Path) -> None:
    examples = [
        TrainingExample((1, 2, 3, 4), (2, 3, 4, 5), "one"),
        TrainingExample((2, 3, 4, 5), (3, 4, 5, 6), "two"),
    ]
    source = lambda: iter(examples)
    first = FodciTrainer(
        _model(),
        source,
        source,
        TrainingConfig(epochs=1, batch_size=1, output_dir=tmp_path, checkpoint_interval=1),
    )
    first_result = first.train()
    checkpoint = Path(first_result.last_checkpoint or "")
    assert checkpoint.is_file()

    second_model = _model(seed=31)
    second = FodciTrainer(
        second_model,
        source,
        source,
        TrainingConfig(epochs=2, batch_size=1, output_dir=tmp_path / "resumed"),
    )
    before = {name: parameter.detach().clone() for name, parameter in second_model.named_parameters()}
    loaded = second.resume(checkpoint)
    result = second.train()

    assert loaded.epoch == 1
    assert loaded.global_step == 2
    assert result.global_step == 4
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in second_model.named_parameters()
    )
