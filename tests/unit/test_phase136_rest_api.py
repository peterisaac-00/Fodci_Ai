from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_rest_api_dataset_is_balanced_and_valid() -> None:
    from backend_ai.dataset.config import DatasetConfig
    from backend_ai.dataset.instructions import InstructionDatasetLoader

    root = ROOT / "training_data" / "rest_api"
    train = InstructionDatasetLoader(DatasetConfig(root / "train", supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False)).load()
    validation = InstructionDatasetLoader(DatasetConfig(root / "validation", supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False)).load()
    assert len(train.examples) == 32
    assert len(validation.examples) == 8
    assert not train.issues
    assert not validation.issues


def test_phase136_training_report_has_rest_lineage_and_all_gates() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase136_rest_api_training.json").read_text(encoding="utf-8"))
    assert report["format"] == "fodci.phase136_rest_api_training"
    assert report["phase"] == "13.6"
    assert report["base_model_version"] == "fodci-sql-database-v1"
    assert report["model_version"] == "fodci-rest-api-v1"
    assert report["split"]["documents"] >= 31
    assert report["split"]["train_examples"] >= report["split"]["train_documents"]
    assert report["split"]["validation_examples"] >= report["split"]["validation_documents"]
    assert report["evaluation"]["trained"]["loss"] < report["evaluation"]["base"]["loss"]
    assert report["validation_gates"]["all_passed"] is True
    assert all(report["validation_gates"].values())


def test_phase136_benchmark_is_held_out_and_uses_rest_identity() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase136_rest_api_benchmark.json").read_text(encoding="utf-8"))
    assert report["model"]["version"] == "fodci-rest-api-v1"
    assert report["run_id"].startswith("phase136-rest-api-")
    assert report["dataset"]["records"] == 8
    assert report["dataset"]["split"] == "benchmark"
    assert report["evaluation"]["aggregate"]["items"] == 8


def test_phase136_markdown_reports_rest_scope_and_pipeline_gates() -> None:
    content = (ROOT / "docs" / "experiments" / "phase136_rest_api_training.md").read_text(encoding="utf-8")
    assert "RESTful API Design" in content
    assert "checkpoint reload succeeds" in content.lower()
    assert "All gates passed" in content
