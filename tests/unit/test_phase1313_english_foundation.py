"""Focused regression tests for the Phase 13.13 English experiments."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = ROOT / "tokenizers" / "fodci-english-v4.json"
FOUNDATION_REPORT = ROOT / "artifacts" / "evaluation" / "phase1313_english_foundation.json"
DOLLY_REPORT = ROOT / "artifacts" / "evaluation" / "phase1313_dolly_instruction_tuning.json"


def _read_json(path: Path) -> dict:
    assert path.is_file(), f"required Phase 13.13 artifact is missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase1313_v4_tokenizer_has_expected_bounded_merge_count() -> None:
    tokenizer = _read_json(TOKENIZER_PATH)

    assert tokenizer["format"] == "fodci-byte-bpe"
    assert tokenizer["vocab_size"] == 10_000
    assert len(tokenizer["merges"]) == 512


def test_phase1313_foundation_report_passes_structural_gates() -> None:
    report = _read_json(FOUNDATION_REPORT)

    assert report["phase"] == "13.13"
    assert report["language"] == "en"
    assert report["tokenizer_path"].endswith("fodci-english-v4.json")
    assert report["all_training_gates_passed"] is True
    assert report["stable_runtime_replaced"] is False
    assert report["data"]["train_examples"] > 0
    assert report["data"]["validation_examples"] > 0
    assert all(model["checkpoint_reload"] for model in report["models"].values())
    assert all(model["finite_loss"] for model in report["models"].values())
    assert all(model["parameters_changed"] for model in report["models"].values())


def test_phase1313_dolly_report_passes_structural_gates_and_improves_loss() -> None:
    report = _read_json(DOLLY_REPORT)

    assert report["format"] == "fodci.phase1313_dolly_instruction_tuning"
    assert report["language"] == "en"
    assert report["license"] == "CC-BY-SA-3.0"
    assert report["tokenizer_path"].endswith("fodci-english-v4.json")
    assert report["structural_gates_passed"] is True
    assert report["checkpoint_reload"] is True
    assert report["finite_loss"] is True
    assert report["parameters_changed"] is True
    assert report["non_empty_split"] is True
    assert report["heldout_loss_improved"] is True
    assert report["trained_validation_loss"] < report["baseline_validation_loss"]


def test_phase1313_experiments_do_not_replace_stable_runtime() -> None:
    foundation = _read_json(FOUNDATION_REPORT)
    dolly = _read_json(DOLLY_REPORT)

    assert foundation["stable_runtime"] == "fodci-testing-qa-v1"
    assert foundation["stable_runtime_replaced"] is False
    assert dolly["base_checkpoint"].endswith("fodci-english-25m-v1.pt")
    assert "testing-qa" not in dolly["checkpoint_path"]
