from __future__ import annotations

import json
from pathlib import Path

from backend_ai.core.contracts import LLMRequest, LLMResponse
from backend_ai.distillation import PromotionPolicy, ShadowMode
from scripts import run_phase156_shadow_promotion as phase156


class _FakeProvider:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.text)


def _negative_evaluation() -> dict[str, object]:
    return {
        "phase_gates_passed": True,
        "all_cases_completed": True,
        "response_quality_accepted": False,
        "stable_runtime_replaced": False,
        "stable": {
            "understandable_heuristic_rate": 0.0,
            "average_keyword_coverage": 0.0,
            "average_repeated_token_rate": 0.327777625,
        },
        "distilled": {
            "understandable_heuristic_rate": 0.0,
            "average_keyword_coverage": 0.0,
            "average_repeated_token_rate": 0.696994,
        },
    }


def test_shadow_mode_returns_primary_response_and_runs_candidate() -> None:
    primary = _FakeProvider("stable response")
    candidate = _FakeProvider("candidate response")
    mode = ShadowMode(primary, candidate)

    response, result = mode.generate(LLMRequest.from_prompt("backend question"))

    assert response.text == "stable response"
    assert result.primary_text == "stable response"
    assert result.candidate_text == "candidate response"
    assert primary.calls == 1
    assert candidate.calls == 1


def test_promotion_policy_rejects_phase155_candidate() -> None:
    decision = PromotionPolicy().decide(_negative_evaluation(), human_approved=False)

    assert decision.eligible is False
    assert decision.stable_runtime_replaced is False
    assert "human approval is required" in decision.reasons
    assert "candidate repetition is worse than stable" in decision.reasons
    assert "response quality has not been accepted" in decision.reasons


def test_phase156_runner_gates_preserve_stable_runtime(tmp_path: Path, monkeypatch) -> None:
    evaluation_path = tmp_path / "phase155.json"
    output_path = tmp_path / "phase156.json"
    evaluation_path.write_text(json.dumps(_negative_evaluation()), encoding="utf-8")
    providers = iter((_FakeProvider("stable"), _FakeProvider("candidate")))

    def fake_from_checkpoint(cls, checkpoint_path, **kwargs):
        return next(providers)

    monkeypatch.setattr(phase156.FodciLocalProvider, "from_checkpoint", classmethod(fake_from_checkpoint))
    report = phase156.run_phase156(
        root=tmp_path,
        evaluation_report_path=evaluation_path,
        output_path=output_path,
        prompt="test prompt",
    )

    assert report["phase_gates_passed"] is True
    assert report["stable_runtime_replaced"] is False
    assert report["shadow"]["returned_response_source"] == "stable-primary"
    assert report["promotion_decision"]["eligible"] is False
    assert json.loads(output_path.read_text(encoding="utf-8"))["phase"] == "15.6"
