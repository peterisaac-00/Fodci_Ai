from __future__ import annotations

import json
from pathlib import Path

from scripts.train_stage1 import split_examples


ROOT = Path(__file__).parents[2]


def test_stage1_split_is_deterministic_and_non_empty() -> None:
    from backend_ai.dataset.config import DatasetConfig
    from backend_ai.dataset.instructions import InstructionDatasetPipeline
    from backend_ai.tokenizer import FodciTokenizer

    pipeline = InstructionDatasetPipeline(
        DatasetConfig(
            ROOT / "training_data" / "fundamentals",
            supported_extensions=frozenset({".txt"}),
            context_length=256,
            use_eos_document_boundaries=False,
        ),
        FodciTokenizer(),
    )
    examples = list(pipeline.iter_training_examples())
    train_a, validation_a, split_a = split_examples(examples)
    train_b, validation_b, split_b = split_examples(examples)
    assert train_a and validation_a
    assert [example.document_id for example in train_a] == [example.document_id for example in train_b]
    assert [example.document_id for example in validation_a] == [example.document_id for example in validation_b]
    assert split_a == split_b
    assert set(example.document_id for example in train_a).isdisjoint(example.document_id for example in validation_a)


def test_stage1_training_report_records_all_gates() -> None:
    report_path = ROOT / "artifacts" / "evaluation" / "stage1_training.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["format"] == "fodci.stage1_training"
    assert report["phase"] == "13.3"
    assert report["model_parameters"] == 11_424_400
    assert report["training_result"]["global_step"] == 4
    assert report["evaluation"]["trained"]["loss"] < report["evaluation"]["baseline"]["loss"]
    assert all(report["validation_gates"].values())
    assert Path(report["checkpoint_path"]).is_file()


def test_stage1_training_markdown_contains_before_after_metrics() -> None:
    report_path = ROOT / "docs" / "experiments" / "phase133_stage1_training.md"
    content = report_path.read_text(encoding="utf-8")
    assert "Before/after validation loss" in content
    assert "Checkpoint reload succeeds" in content
    assert "Pipeline validation | `True`" in content
