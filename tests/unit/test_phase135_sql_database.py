from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_sql_database_dataset_is_balanced_and_valid() -> None:
    from backend_ai.dataset.config import DatasetConfig
    from backend_ai.dataset.instructions import InstructionDatasetLoader

    root = ROOT / "training_data" / "sql_database"
    train = InstructionDatasetLoader(DatasetConfig(root / "train", supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False)).load()
    validation = InstructionDatasetLoader(DatasetConfig(root / "validation", supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False)).load()
    assert len(train.examples) == 32
    assert len(validation.examples) == 8
    assert not train.issues
    assert not validation.issues


def test_phase135_training_report_has_sql_lineage_and_all_gates() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase135_sql_database_training.json").read_text(encoding="utf-8"))
    assert report["format"] == "fodci.phase135_sql_database_training"
    assert report["phase"] == "13.5"
    assert report["base_model_version"] == "fodci-python-backend-v1"
    assert report["model_version"] == "fodci-sql-database-v1"
    assert report["split"]["documents"] >= 31
    assert report["split"]["train_examples"] >= report["split"]["train_documents"]
    assert report["split"]["validation_examples"] >= report["split"]["validation_documents"]
    assert report["evaluation"]["trained"]["loss"] < report["evaluation"]["base"]["loss"]
    assert report["validation_gates"]["all_passed"] is True
    assert all(report["validation_gates"].values())


def test_phase135_benchmark_is_held_out_and_uses_sql_identity() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase135_sql_database_benchmark.json").read_text(encoding="utf-8"))
    assert report["model"]["version"] == "fodci-sql-database-v1"
    assert report["run_id"].startswith("phase135-sql-database-")
    assert report["dataset"]["records"] == 8
    assert report["dataset"]["split"] == "benchmark"
    assert report["evaluation"]["aggregate"]["items"] == 8


def test_phase135_markdown_reports_sql_scope_and_pipeline_gates() -> None:
    content = (ROOT / "docs" / "experiments" / "phase135_sql_database_training.md").read_text(encoding="utf-8")
    assert "SQL & Database Reasoning" in content
    assert "checkpoint reload succeeds" in content.lower()
    assert "All gates passed" in content
