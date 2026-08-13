from __future__ import annotations

from pathlib import Path

import pytest
import torch

from backend_ai.dataset import (
    DatasetConfig,
    InstructionDatasetManifestBuilder,
    InstructionDatasetPipeline,
    InstructionExample,
    InstructionFormatError,
    InstructionManifestError,
)
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import FodciTokenizer
from backend_ai.training import FodciTrainer, TrainingConfig


def _text(instruction: str = "Do the task.", input_text: str = "Use Python.", response: str = "Return a safe result.") -> str:
    return f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n{response}\n"


def test_instruction_serialization_and_parser_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "example.txt"
    text = _text("Create an endpoint.", "The service uses Flask.", "Validate input first.")
    source.write_text(text, encoding="utf-8")

    first = InstructionExample.parse(text, source)
    second = InstructionExample.parse(first.serialize(), source)

    assert first == second
    assert first.serialize() == text
    assert first.content_sha256 == second.content_sha256
    assert first.instruction == "Create an endpoint."
    assert first.input_text == "The service uses Flask."
    assert first.response == "Validate input first."


@pytest.mark.parametrize(
    "text, reason",
    [
        ("", "empty"),
        ("### Instruction\nOnly instruction\n", "missing response"),
        ("### Instruction\n\n### Input\ncontext\n### Response\nanswer\n", "missing instruction"),
        ("### Instruction\ntask\n### Input\n\n### Response\nanswer\n", "empty input"),
        ("### Response\nanswer\n### Input\ncontext\n### Instruction\ntask\n", "order"),
    ],
)
def test_malformed_instruction_is_rejected(text: str, reason: str) -> None:
    with pytest.raises(InstructionFormatError, match=reason):
        InstructionExample.parse(text)


def test_instruction_pipeline_marks_only_response_targets(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text(
        _text("Explain a route.", "The route receives an ID.", "Return 404 when absent."),
        encoding="utf-8",
    )
    pipeline = InstructionDatasetPipeline(
        DatasetConfig(tmp_path, supported_extensions=frozenset({".txt"}), context_length=16),
        FodciTokenizer(),
    )

    samples = list(pipeline.iter_training_examples())

    assert samples
    assert all(sample.loss_mask is not None for sample in samples)
    assert all(any(sample.loss_mask) for sample in samples if sample.loss_mask is not None)
    assert any(
        not all(sample.loss_mask) for sample in samples if sample.loss_mask is not None
    )
    assert all(len(sample.input_ids) == 16 for sample in samples)
    assert all(len(sample.target_ids) == len(sample.loss_mask or ()) for sample in samples)


def test_instruction_manifest_rejects_duplicates_and_cross_split_leakage(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    same = _text("Same task.", "Same context.", "Same response.")
    (tmp_path / "train" / "one.txt").write_text(same, encoding="utf-8")
    (tmp_path / "train" / "two.txt").write_text(same, encoding="utf-8")
    (tmp_path / "validation" / "heldout.txt").write_text(
        _text("Different task.", "Different context.", "Different response."),
        encoding="utf-8",
    )
    diagnostic = InstructionDatasetManifestBuilder(tmp_path, strict=False).build()
    assert diagnostic.train.duplicate_count == 1
    with pytest.raises(InstructionManifestError, match="duplicate"):
        InstructionDatasetManifestBuilder(tmp_path, strict=True).build()

    (tmp_path / "train" / "two.txt").unlink()
    (tmp_path / "validation" / "heldout.txt").write_text(same, encoding="utf-8")
    with pytest.raises(InstructionManifestError, match="leakage"):
        InstructionDatasetManifestBuilder(tmp_path, strict=False).build()


def test_instruction_manifest_rejects_wrong_tokenizer(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    (tmp_path / "train" / "one.txt").write_text(_text(), encoding="utf-8")
    (tmp_path / "validation" / "two.txt").write_text(_text("Other task."), encoding="utf-8")
    with pytest.raises(InstructionManifestError, match="vocabulary"):
        InstructionDatasetManifestBuilder(tmp_path, tokenizer=FodciTokenizer(vocab_size=300)).build()


def test_response_masking_integrates_with_existing_trainer(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    (tmp_path / "train" / "one.txt").write_text(_text(), encoding="utf-8")
    (tmp_path / "validation" / "two.txt").write_text(_text("Other task."), encoding="utf-8")
    train_pipeline = InstructionDatasetPipeline(
        DatasetConfig(tmp_path / "train", supported_extensions=frozenset({".txt"}), context_length=16),
        FodciTokenizer(),
    )
    validation_pipeline = InstructionDatasetPipeline(
        DatasetConfig(tmp_path / "validation", supported_extensions=frozenset({".txt"}), context_length=16),
        FodciTokenizer(),
    )
    model = FodciModel(
        ModelConfig(
            vocab_size=10_000,
            context_length=16,
            hidden_size=16,
            num_layers=2,
            num_attention_heads=4,
            feed_forward_size=32,
            dropout=0.0,
            seed=7,
        )
    )
    trainer = FodciTrainer(
        model,
        train_pipeline.iter_training_examples,
        validation_pipeline.iter_training_examples,
        TrainingConfig(
            epochs=1,
            max_steps=1,
            batch_size=1,
            output_dir=tmp_path / "checkpoints",
            seed=2026,
        ),
    )

    result = trainer.train()

    assert result.global_step == 1
    assert result.final_metrics is not None
    assert result.final_metrics.training_tokens > 0
    assert result.final_metrics.training_tokens < 16
    assert result.final_metrics.validation_tokens > 0
    assert torch.isfinite(torch.tensor(result.final_metrics.train_loss))


def test_real_instruction_manifest_has_reproducible_identity() -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "fodci_instructions"
    manifest = InstructionDatasetManifestBuilder(root).build()

    assert manifest.dataset_sha256 == "c42a4ad2552bb832ced35603eebe15d2df430fec7f463d054df60806ff46af5c"
    assert manifest.train.instruction_count == 8
    assert manifest.validation.instruction_count == 3
    assert manifest.train.response_tokens == 2994
    assert manifest.validation.response_tokens == 901
    assert manifest.train.training_example_count == 15
    assert manifest.validation.training_example_count == 4
    assert manifest.train_validation_leakage_count == 0
