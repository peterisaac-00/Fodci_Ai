from __future__ import annotations

from pathlib import Path

from backend_ai.evaluation.acceptance import AcceptanceDecision, AcceptanceStore, ModelAcceptanceEvaluator, render_acceptance_report
from tests.unit.test_acceptance_phase116 import _request


def test_phase116_improved_model_acceptance_is_persisted(tmp_path: Path) -> None:
    request = _request(tmp_path, base_passed=False, candidate_passed=True, comparison_id="phase116-accept", training_config={"epochs": 1, "learning_rate": 0.001})
    report = ModelAcceptanceEvaluator().evaluate(request)
    assert report.decision is AcceptanceDecision.ACCEPT
    store = AcceptanceStore(tmp_path / "acceptance.json")
    store.save(report)
    reloaded = AcceptanceStore(tmp_path / "acceptance.json").get("phase116-accept")
    assert reloaded is not None
    assert reloaded.decision is AcceptanceDecision.ACCEPT
    text = render_acceptance_report(reloaded)
    assert "Decision: ACCEPT" in text
    assert "FINAL DECISION" in text


def test_phase116_regressed_model_is_rejected_even_when_evidence_exists(tmp_path: Path) -> None:
    request = _request(tmp_path, base_passed=True, candidate_passed=False, comparison_id="phase116-reject", training_config={"epochs": 1})
    report = ModelAcceptanceEvaluator().evaluate(request)
    assert report.decision is AcceptanceDecision.REJECT
    assert report.regressions
    assert any(item.critical for item in report.regressions)
    assert "critical_regression" in report.reason
    store = AcceptanceStore(tmp_path / "acceptance.json")
    store.save(report)
    assert AcceptanceStore(tmp_path / "acceptance.json").get("phase116-reject").decision is AcceptanceDecision.REJECT
