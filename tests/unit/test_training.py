from __future__ import annotations

from pathlib import Path

import pytest
import torch

from backend_ai.dataset import TrainingExample
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.training import FodciTrainer, TrainingConfig, perplexity


def _model() -> FodciModel:
    return FodciModel(
        ModelConfig(
            vocab_size=32,
            context_length=8,
            hidden_size=16,
            num_layers=2,
            num_attention_heads=4,
            feed_forward_size=32,
            dropout=0.0,
            seed=7,
        ),
    )


def _examples() -> list[TrainingExample]:
    return [
        TrainingExample((1, 2, 3, 4), (2, 3, 4, 5), "a"),
        TrainingExample((2, 3, 4, 5), (3, 4, 5, 6), "b"),
        TrainingExample((3, 4, 5, 6), (4, 5, 6, 7), "c"),
        TrainingExample((4, 5, 6, 7), (5, 6, 7, 8), "d"),
    ]


def _source(examples: list[TrainingExample]):
    return lambda: iter(examples)


def test_training_config_validates_values_and_resolves_cpu(tmp_path: Path) -> None:
    config = TrainingConfig(
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        output_dir=tmp_path,
    )

    assert config.resolve_device() == torch.device("cpu")
    assert config.to_dict()["output_dir"] == str(tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=0.0)


def test_explicit_cuda_fails_clearly_when_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        TrainingConfig(device="cuda").resolve_device()


def test_batching_and_training_update_parameters_with_finite_metrics(tmp_path: Path) -> None:
    model = _model()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    trainer = FodciTrainer(
        model,
        _source(_examples()),
        _source(_examples()),
        TrainingConfig(
            epochs=2,
            batch_size=2,
            output_dir=tmp_path,
            seed=11,
            checkpoint_interval=1,
        ),
    )

    result = trainer.train()

    assert result.global_step == 4
    assert len(result.history) == 2
    assert result.final_metrics is not None
    assert result.final_metrics.training_steps == 2
    assert result.final_metrics.validation_steps == 2
    assert torch.isfinite(torch.tensor(result.final_metrics.train_loss))
    assert torch.isfinite(torch.tensor(result.final_metrics.validation_loss))
    assert result.final_metrics.train_perplexity > 0.0
    assert result.last_checkpoint is not None
    assert Path(result.last_checkpoint).is_file()
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )


def test_validation_does_not_update_parameters_or_global_step(tmp_path: Path) -> None:
    model = _model()
    trainer = FodciTrainer(
        model,
        _source(_examples()),
        _source(_examples()),
        TrainingConfig(epochs=1, output_dir=tmp_path),
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss, steps, tokens = trainer._validate_epoch(1)

    assert loss is not None and loss > 0.0
    assert steps == 2
    assert tokens == 16
    assert trainer.global_step == 0
    assert all(torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters())


def test_checkpoint_load_and_resume_continue_from_next_epoch(tmp_path: Path) -> None:
    first_model = _model()
    first = FodciTrainer(
        first_model,
        _source(_examples()),
        _source(_examples()),
        TrainingConfig(epochs=1, batch_size=2, output_dir=tmp_path, checkpoint_interval=1),
    )
    first_result = first.train()
    checkpoint = Path(first_result.last_checkpoint or "")
    assert checkpoint.is_file()

    resumed_model = _model()
    resumed = FodciTrainer(
        resumed_model,
        _source(_examples()),
        _source(_examples()),
        TrainingConfig(epochs=2, batch_size=2, output_dir=tmp_path / "resumed"),
    )
    state = resumed.resume(checkpoint)
    result = resumed.train()

    assert state.epoch == 1
    assert state.global_step == 2
    assert result.history[0].epoch == 2
    assert result.global_step == 4


def test_deterministic_cpu_training_reproduces_parameters(tmp_path: Path) -> None:
    first_model = _model()
    second_model = _model()
    first = FodciTrainer(
        first_model,
        _source(_examples()),
        config=TrainingConfig(epochs=1, batch_size=2, output_dir=tmp_path / "first", seed=23),
    )
    second = FodciTrainer(
        second_model,
        _source(_examples()),
        config=TrainingConfig(epochs=1, batch_size=2, output_dir=tmp_path / "second", seed=23),
    )

    first.train()
    second.train()

    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(first_model.parameters(), second_model.parameters())
    )


def test_invalid_training_examples_fail_before_forward(tmp_path: Path) -> None:
    trainer = FodciTrainer(
        _model(),
        _source([TrainingExample((1, 2, 3), (2, 3), "bad-length")]),
        config=TrainingConfig(epochs=1, output_dir=tmp_path),
    )
    with pytest.raises(ValueError, match="equal input and target lengths"):
        trainer.train()

    invalid_ids = FodciTrainer(
        _model(),
        _source([TrainingExample((1, 2, 99, 4), (2, 3, 4, 5), "bad-vocab")]),
        config=TrainingConfig(epochs=1, output_dir=tmp_path),
    )
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        invalid_ids.train()


def test_perplexity_guards_overflow() -> None:
    assert perplexity(0.0) == 1.0
    assert perplexity(None) is None
    assert perplexity(float("inf")) == float("inf")


def test_tiny_end_to_end_smoke_uses_existing_dataset_pipeline(tmp_path: Path) -> None:
    from backend_ai.dataset import DatasetConfig, FodciDatasetPipeline
    from backend_ai.tokenizer import FodciTokenizer

    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    (train_dir / "train.txt").write_text("backend engineering " * 20, encoding="utf-8")
    (validation_dir / "validation.txt").write_text("backend validation " * 20, encoding="utf-8")

    train_pipeline = FodciDatasetPipeline(
        DatasetConfig(train_dir, context_length=8),
        FodciTokenizer(),
    )
    validation_pipeline = FodciDatasetPipeline(
        DatasetConfig(validation_dir, context_length=8),
        FodciTokenizer(),
    )
    model = FodciModel(
        ModelConfig(
            vocab_size=260,
            context_length=8,
            hidden_size=16,
            num_layers=2,
            num_attention_heads=4,
            feed_forward_size=32,
            dropout=0.0,
            seed=7,
        ),
    )
    trainer = FodciTrainer(
        model,
        train_pipeline.iter_samples,
        validation_pipeline.iter_samples,
        TrainingConfig(
            epochs=1,
            batch_size=2,
            output_dir=tmp_path / "checkpoints",
            checkpoint_interval=1,
            seed=17,
        ),
    )

    result = trainer.train()

    assert result.final_metrics is not None
    assert result.final_metrics.training_steps > 0
    assert result.final_metrics.validation_steps > 0
    assert result.final_metrics.train_loss == pytest.approx(result.final_metrics.train_loss)
    assert result.last_checkpoint is not None
    assert Path(result.last_checkpoint).exists()


def test_gradient_clipping_is_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        calls.append(float(max_norm))
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr("backend_ai.training.trainer.torch.nn.utils.clip_grad_norm_", recording_clip)
    trainer = FodciTrainer(
        _model(),
        _source(_examples()),
        config=TrainingConfig(epochs=1, batch_size=2, max_grad_norm=0.25, output_dir=tmp_path),
    )

    trainer.train()

    assert calls == [0.25, 0.25]


def test_context_length_and_missing_checkpoint_fail_clearly(tmp_path: Path) -> None:
    trainer = FodciTrainer(
        _model(),
        _source([TrainingExample(tuple(range(9)), tuple(range(1, 10)), "too-long")]),
        config=TrainingConfig(epochs=1, output_dir=tmp_path),
    )
    with pytest.raises(ValueError, match="within model context length"):
        trainer.train()
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        trainer.resume(tmp_path / "missing.pt")


def test_max_steps_caps_training_budget(tmp_path: Path) -> None:
    trainer = FodciTrainer(
        _model(),
        _source(_examples()),
        _source(_examples()),
        TrainingConfig(
            epochs=3,
            max_steps=1,
            batch_size=2,
            output_dir=tmp_path,
        ),
    )

    result = trainer.train()

    assert result.global_step == 1
    assert len(result.history) == 1
    assert result.final_metrics is not None
    assert result.final_metrics.training_steps == 1
    assert result.final_metrics.training_tokens == 8
