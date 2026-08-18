"""Phase 15.6 shadow execution and controlled promotion policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend_ai.core.contracts import LLMProvider, LLMRequest, LLMResponse


@dataclass(frozen=True, slots=True)
class ShadowResult:
    primary_text: str
    candidate_text: str
    primary_error: str | None
    candidate_error: str | None
    candidate_would_be_promoted: bool


class ShadowMode:
    """Run candidate inference for comparison while returning the primary response."""

    def __init__(self, primary: LLMProvider, candidate: LLMProvider) -> None:
        self.primary = primary
        self.candidate = candidate

    def generate(self, request: LLMRequest, *, candidate_would_be_promoted: bool = False) -> tuple[LLMResponse, ShadowResult]:
        primary_text = ""
        candidate_text = ""
        primary_error = None
        candidate_error = None
        try:
            primary_text = self.primary.generate(request).text
        except Exception as exc:
            primary_error = str(exc)
        try:
            candidate_text = self.candidate.generate(request).text
        except Exception as exc:
            candidate_error = str(exc)
        result = ShadowResult(primary_text, candidate_text, primary_error, candidate_error, candidate_would_be_promoted)
        if primary_error is not None:
            raise RuntimeError(f"primary provider failed in shadow mode: {primary_error}")
        return LLMResponse(text=primary_text), result


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    eligible: bool
    reasons: tuple[str, ...]
    stable_runtime_replaced: bool = False


class PromotionPolicy:
    """Conservative, report-driven promotion gate."""

    def decide(self, evaluation_report: dict[str, Any], *, human_approved: bool = False) -> PromotionDecision:
        reasons: list[str] = []
        if not evaluation_report.get("phase_gates_passed", False):
            reasons.append("held-out evaluation gates did not pass")
        if not evaluation_report.get("all_cases_completed", False):
            reasons.append("candidate evaluation is incomplete")
        if not human_approved:
            reasons.append("human approval is required")
        distilled = evaluation_report.get("distilled", {})
        stable = evaluation_report.get("stable", {})
        if distilled.get("understandable_heuristic_rate", 0.0) < stable.get("understandable_heuristic_rate", 0.0):
            reasons.append("candidate readability is not better than stable")
        if distilled.get("average_keyword_coverage", 0.0) < stable.get("average_keyword_coverage", 0.0):
            reasons.append("candidate keyword coverage is not better than stable")
        if distilled.get("average_repeated_token_rate", 1.0) > stable.get("average_repeated_token_rate", 0.0):
            reasons.append("candidate repetition is worse than stable")
        if evaluation_report.get("response_quality_accepted", False) is not True:
            reasons.append("response quality has not been accepted")
        return PromotionDecision(not reasons, tuple(reasons), False)


__all__ = ["PromotionDecision", "PromotionPolicy", "ShadowMode", "ShadowResult"]
