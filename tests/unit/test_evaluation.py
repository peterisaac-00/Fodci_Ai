from __future__ import annotations

from pathlib import Path

import pytest
import torch

from backend_ai.checkpoint import CheckpointFormatError, CheckpointManager
from backend_ai.dataset import TrainingExample
from backend_ai.evaluation import EvaluationConfig, FodciEvaluator
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.training import TrainingConfig


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


def _examples() -> list[TrainingExample]:
    return [
        TrainingExample((1, 2, 3, 4), (2, 3, 4, 5), "one"),
        TrainingExample((2, 3, 4, 5), (3, 4, 5, 6), "two"),
    ]


def _config(tmp_path: Path) -> EvaluationConfig:
    return EvaluationConfig(
        batch_size=1,
        device="cpu",
        seed=2026,
        dataset_path=tmp_path / "validation",
        checkpoint_dir=tmp_path / "checkpoints",
    )


def test_evaluation_is_eval_no_grad_and_does_not_change_model_or_optimizer(tmp_path: Path) -> None:
    model = _model()
    model.train()
    evaluator = FodciEvaluator(model, _config(tmp_path))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer_state_before = dict(evaluator._optimizer.state)

    result = evaluator.evaluate(lambda: iter(_examples()))

    assert not model.training
    assert result.evaluation_examples == 2
    assert result.evaluated_tokens == 8
    assert result.loss > 0.0 and torch.isfinite(torch.tensor(result.loss))
    assert result.perplexity > 0.0 and torch.isfinite(torch.tensor(result.perplexity))
    assert evaluator._optimizer.state == optimizer_state_before
    assert all(torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluation_is_deterministic_for_same_seed_and_source(tmp_path: Path) -> None:
    first = FodciEvaluator(_model(), _config(tmp_path))
    second = FodciEvaluator(_model(), _config(tmp_path))

    first_result = first.evaluate(lambda: iter(_examples()))
    second_result = second.evaluate(lambda: iter(_examples()))

    assert first_result.loss == second_result.loss
    assert first_result.perplexity == second_result.perplexity
    assert first_result.evaluation_examples == second_result.evaluation_examples
    assert first_result.evaluated_tokens == second_result.evaluated_tokens


def test_compare_reports_loss_and_perplexity_improvement() -> None:
    from backend_ai.evaluation import ModelEvaluationResult

    common = {
        "checkpoint_path": None,
        "model_version": "fodci-tiny-v1",
        "tokenizer_version": 1,
        "device": "cpu",
        "dataset_path": "validation",
        "dataset_split": "validation",
        "dataset_hash": "hash",
        "document_count": 2,
        "evaluation_examples": 2,
        "evaluated_tokens": 8,
        "evaluation_seconds": 0.1,
    }
    baseline = ModelEvaluationResult(checkpoint_id="random", loss=4.0, perplexity=54.0, **common)
    trained = ModelEvaluationResult(checkpoint_id="trained", loss=2.0, perplexity=7.0, epoch=1, global_step=2, **common)

    comparison = FodciEvaluator.compare(baseline, trained)

    assert comparison.loss_delta == -2.0
    assert comparison.loss_improvement == 2.0
    assert comparison.loss_relative_improvement_percent == 50.0
    assert comparison.perplexity_delta == -47.0
    assert comparison.perplexity_improvement == 47.0


def test_checkpoint_evaluation_validates_and_restores_trained_state(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path / "checkpoints")
    checkpoint = manager.save(
        model,
        optimizer,
        TrainingConfig(output_dir=tmp_path / "checkpoints"),
        epoch=3,
        global_step=11,
        metrics={"validation_loss": 1.5},
        path=tmp_path / "checkpoints" / "trained.pt",
    )

    evaluator = FodciEvaluator(_model(seed=99), _config(tmp_path))
    result = evaluator.evaluate_checkpoint(checkpoint, lambda: iter(_examples()))

    assert result.checkpoint_id == "trained"
    assert result.epoch == 3
    assert result.global_step == 11
    assert result.checkpoint_path == str(checkpoint)
    assert result.evaluation_examples == 2


def test_multiple_and_best_checkpoint_evaluation_use_same_validation_source(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path / "checkpoints")
    first = manager.save(
        model, optimizer, TrainingConfig(output_dir=tmp_path / "checkpoints"),
        epoch=1, global_step=2, metrics={"validation_loss": 2.0}, path=tmp_path / "checkpoints" / "first.pt",
    )
    second = manager.save(
        model, optimizer, TrainingConfig(output_dir=tmp_path / "checkpoints"),
        epoch=2, global_step=4, metrics={"validation_loss": 1.0}, path=tmp_path / "checkpoints" / "second.pt",
    )
    evaluator = FodciEvaluator(_model(), _config(tmp_path))
    results = evaluator.evaluate_checkpoints((first, second), lambda: iter(_examples()))
    best = evaluator.evaluate_best(lambda: iter(_examples()), manager=manager)

    assert [result.global_step for result in results] == [2, 4]
    assert best.checkpoint_id == "best"
    assert best.global_step == 4


def test_missing_and_corrupt_checkpoint_fail_clearly(tmp_path: Path) -> None:
    evaluator = FodciEvaluator(_model(), _config(tmp_path))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        evaluator.evaluate_checkpoint(tmp_path / "missing.pt", lambda: iter(_examples()))

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"bad checkpoint")
    with pytest.raises(CheckpointFormatError, match="Unable to read"):
        evaluator.evaluate_checkpoint(corrupt, lambda: iter(_examples()))


def test_response_only_evaluation_reports_response_loss(tmp_path: Path) -> None:
    from backend_ai.dataset import TrainingExample

    masked_examples = [
        TrainingExample((1, 2, 3, 4), (2, 3, 4, 5), "masked", (False, False, False, True)),
    ]
    evaluator = FodciEvaluator(
        _model(),
        _config(tmp_path),
        dataset_metadata={"loss_type": "response_only"},
    )

    result = evaluator.evaluate(lambda: iter(masked_examples))

    assert result.loss_type == "response_only"
    assert result.response_loss == result.loss
    assert result.evaluated_tokens == 1
