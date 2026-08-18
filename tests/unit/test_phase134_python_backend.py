from __future__ import annotations

import json
from pathlib import Path

from scripts.train_phase134_python_backend import split_examples


ROOT = Path(__file__).parents[2]


def test_python_backend_specialist_dataset_is_balanced_and_parsed() -> None:
    from backend_ai.dataset.config import DatasetConfig
    from backend_ai.dataset.instructions import InstructionDatasetLoader

    root = ROOT / "training_data" / "python_backend"
    train = InstructionDatasetLoader(DatasetConfig(root / "train", supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False)).load()
    validation = InstructionDatasetLoader(DatasetConfig(root / "validation", supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False)).load()
    assert len(train.examples) == 32
    assert len(validation.examples) == 8
    assert not train.issues
    assert not validation.issues


def test_phase134_training_report_has_all_validation_gates() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase134_python_backend_training.json").read_text(encoding="utf-8"))
    assert report["format"] == "fodci.phase134_python_backend_training"
    assert report["base_model_version"] == "fodci-stage1-v1"
    assert report["model_version"] == "fodci-python-backend-v1"
    assert report["split"]["train_examples"] >= report["split"]["train_documents"]
    assert report["split"]["validation_examples"] >= report["split"]["validation_documents"]
    assert report["split"]["documents"] == 32
    assert report["evaluation"]["trained"]["loss"] < report["evaluation"]["base"]["loss"]
    assert report["validation_gates"]["all_passed"] is True
    assert all(report["validation_gates"].values())


def test_phase134_benchmark_is_held_out_and_has_specialist_identity() -> None:
    benchmark = json.loads((ROOT / "artifacts" / "evaluation" / "phase134_python_backend_benchmark.json").read_text(encoding="utf-8"))
    assert benchmark["model"]["version"] == "fodci-python-backend-v1"
    assert benchmark["run_id"].startswith("phase134-python-backend-")
    assert benchmark["dataset"]["records"] == 8
    assert benchmark["dataset"]["split"] == "benchmark"
    assert benchmark["evaluation"]["aggregate"]["items"] == 8


def test_split_examples_does_not_overlap_document_ids() -> None:
    from backend_ai.dataset.config import DatasetConfig
    from backend_ai.dataset.instructions import InstructionDatasetPipeline
    from backend_ai.tokenizer import FodciTokenizer

    root = ROOT / "training_data" / "python_backend" / "train"
    examples = list(InstructionDatasetPipeline(DatasetConfig(root, supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False), FodciTokenizer()).iter_training_examples())
    train, validation, _ = split_examples(examples)
    assert train and validation
    assert {example.document_id for example in train}.isdisjoint({example.document_id for example in validation})
