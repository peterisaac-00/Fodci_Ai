from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_phase1311_synergy_report_has_all_gates() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase1311_multi_agent_synergy.json").read_text(encoding="utf-8"))
    assert report["format"] == "fodci.phase1311_multi_agent_synergy"
    assert report["phase"] == "13.11"
    assert report["model_version"] == "fodci-testing-qa-v1"
    assert report["experimental_scaling_model"] == "fodci-scaling-25m-experimental-v1"
    assert report["synergy_gates"]["all_passed"] is True
    assert all(report["synergy_gates"].values())


def test_phase1311_uses_the_real_checkpoint_through_the_provider_boundary() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase1311_multi_agent_synergy.json").read_text(encoding="utf-8"))
    provider = report["model_provider"]
    assert provider["checkpoint_model_version"] == "fodci-testing-qa-v1"
    assert provider["checkpoint_identity"].endswith("artifacts/checkpoints/fodci-testing-qa-v1.pt")
    assert provider["max_new_tokens"] == 4
    assert provider["model_vocab_size"] == 10_000


def test_phase1311_multi_agent_and_memory_evidence_is_persistent() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase1311_multi_agent_synergy.json").read_text(encoding="utf-8"))
    orchestration = report["multi_agent"]
    memory = report["memory"]
    assert orchestration["status"] == "COMPLETED"
    assert orchestration["completed_subtasks"] == orchestration["total_subtasks"] == 4
    assert len(orchestration["completed_steps"]) == 4
    assert memory["records_after_orchestration"] >= 5
    assert memory["records_after_reload"] >= memory["records_after_orchestration"]
    assert memory["retrieved_records"] >= 1
    assert memory["retrieved_after_autonomy"] >= 1


def test_phase1311_autonomy_is_completed_and_bounded() -> None:
    report = json.loads((ROOT / "artifacts" / "evaluation" / "phase1311_multi_agent_synergy.json").read_text(encoding="utf-8"))
    autonomy = report["autonomy"]
    budget = report["autonomy_budget"]
    assert autonomy["lifecycle_state"] == "COMPLETED"
    assert autonomy["progress"]["completed_subtasks"] == 2
    assert autonomy["progress"]["total_subtasks"] == 2
    assert autonomy["checkpoints_count"] <= budget["max_iterations"]
    assert report["synergy_gates"]["default_runtime_preserved"] is True
