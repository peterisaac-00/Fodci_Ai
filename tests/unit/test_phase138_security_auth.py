from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_security_auth_dataset_is_balanced_and_valid() -> None:
    from backend_ai.dataset.config import DatasetConfig
    from backend_ai.dataset.instructions import InstructionDatasetLoader

    root = ROOT / "training_data" / "security_auth"
    train = InstructionDatasetLoader(DatasetConfig(root / "train", supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False)).load()
    validation = InstructionDatasetLoader(DatasetConfig(root / "validation", supported_extensions=frozenset({".txt"}), context_length=256, use_eos_document_boundaries=False)).load()
    assert len(train.examples) == 32
    assert len(validation.examples) == 8
    assert not train.issues
    assert not validation.issues
    assert any("JWT" in example.instruction for example in train.examples)
    assert any("OAuth2" in example.instruction for example in train.examples)
    assert any("password" in example.instruction.lower() for example in train.examples)
    assert any("middleware" in example.instruction.lower() for example in train.examples)


def test_phase138_training_report_has_security_lineage_and_all_gates() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase138_security_auth_training.json").read_text(encoding="utf-8"))
    assert report["format"] == "fodci.phase138_security_auth_training"
    assert report["phase"] == "13.8"
    assert report["base_model_version"] == "fodci-debugging-v1"
    assert report["model_version"] == "fodci-security-auth-v1"
    assert report["split"]["documents"] >= 31
    assert report["split"]["train_examples"] >= report["split"]["train_documents"]
    assert report["split"]["validation_examples"] >= report["split"]["validation_documents"]
    assert report["evaluation"]["trained"]["loss"] < report["evaluation"]["base"]["loss"]
    assert report["validation_gates"]["all_passed"] is True
    assert all(report["validation_gates"].values())


def test_phase138_benchmark_is_held_out_and_uses_security_identity() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase138_security_auth_benchmark.json").read_text(encoding="utf-8"))
    assert report["model"]["version"] == "fodci-security-auth-v1"
    assert report["run_id"].startswith("phase138-security-auth-")
    assert report["dataset"]["records"] == 8
    assert report["dataset"]["split"] == "benchmark"
    assert report["evaluation"]["aggregate"]["items"] == 8


def test_phase138_markdown_reports_security_scope_and_limits() -> None:
    content = (ROOT / "docs" / "experiments" / "phase138_security_auth_training.md").read_text(encoding="utf-8")
    assert "Security & Authentication" in content
    assert "JWT" in content
    assert "OAuth2" in content
    assert "All gates passed" in content
    assert "does not claim" in content
