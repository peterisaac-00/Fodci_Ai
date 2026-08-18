from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_testing_qa_dataset_is_balanced_and_valid() -> None:
    from backend_ai.dataset.config import DatasetConfig
    from backend_ai.dataset.instructions import InstructionDatasetLoader

    root = ROOT / "training_data" / "testing_qa"
    train = InstructionDatasetLoader(DatasetConfig(root / "train", supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False)).load()
    validation = InstructionDatasetLoader(DatasetConfig(root / "validation", supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False)).load()
    assert len(train.examples) == 32
    assert len(validation.examples) == 8
    assert not train.issues
    assert not validation.issues
    assert any("unit test" in example.instruction.lower() for example in train.examples)
    assert any("integration" in example.instruction.lower() for example in train.examples)
    assert any("fixture" in example.instruction.lower() for example in train.examples)
    assert any("coverage" in example.instruction.lower() for example in train.examples)


def test_phase139_training_report_has_testing_lineage_and_all_gates() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase139_testing_qa_training.json").read_text(encoding="utf-8"))
    assert report["format"] == "fodci.phase139_testing_qa_training"
    assert report["phase"] == "13.9"
    assert report["base_model_version"] == "fodci-security-auth-v1"
    assert report["model_version"] == "fodci-testing-qa-v1"
    assert report["split"]["documents"] >= 31
    assert report["split"]["train_examples"] >= report["split"]["train_documents"]
    assert report["split"]["validation_examples"] >= report["split"]["validation_documents"]
    assert report["evaluation"]["trained"]["loss"] < report["evaluation"]["base"]["loss"]
    assert report["validation_gates"]["all_passed"] is True
    assert all(report["validation_gates"].values())


def test_phase139_benchmark_is_held_out_and_uses_testing_identity() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase139_testing_qa_benchmark.json").read_text(encoding="utf-8"))
    assert report["model"]["version"] == "fodci-testing-qa-v1"
    assert report["run_id"].startswith("phase139-testing-qa-")
    assert report["dataset"]["records"] == 8
    assert report["dataset"]["split"] == "benchmark"
    assert report["evaluation"]["aggregate"]["items"] == 8


def test_phase139_markdown_reports_testing_scope_and_limits() -> None:
    content = (ROOT / "docs" / "experiments" / "phase139_testing_qa_training.md").read_text(encoding="utf-8")
    assert "Testing & Quality Assurance" in content
    assert "Pytest" in content
    assert "coverage" in content.lower()
    assert "All gates passed" in content
    assert "does not claim" in content
