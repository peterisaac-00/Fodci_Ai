"""Run Phase 15.6 shadow mode and conservative promotion gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backend_ai.core.contracts import LLMRequest
from backend_ai.distillation import PromotionPolicy, ShadowMode
from backend_ai.llm.fodci_provider import FodciLocalProvider


DEFAULT_PROMPT = "Explain why a backend API should validate request data before business logic."


def _resolve_path(root: Path, value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def run_phase156(
    *,
    root: Path,
    evaluation_report_path: Path,
    output_path: Path,
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, Any]:
    evaluation = json.loads(evaluation_report_path.read_text(encoding="utf-8"))
    stable_checkpoint = _resolve_path(
        root,
        evaluation.get("stable_checkpoint"),
        root / "artifacts/checkpoints/fodci-testing-qa-v1.pt",
    )
    candidate_checkpoint = _resolve_path(
        root,
        evaluation.get("checkpoint"),
        root / "artifacts/checkpoints/fodci-distilled-phase154-v1.pt",
    )

    primary = FodciLocalProvider.from_checkpoint(stable_checkpoint)
    candidate = FodciLocalProvider.from_checkpoint(candidate_checkpoint)
    shadow = ShadowMode(primary, candidate)
    request = LLMRequest.from_prompt(prompt)
    primary_response, shadow_result = shadow.generate(request)

    decision = PromotionPolicy().decide(evaluation, human_approved=False)
    if decision.eligible:
        raise RuntimeError("Phase 15.6 safety gate failed: rejected candidate became eligible")

    report: dict[str, Any] = {
        "format": "fodci.phase156_shadow_promotion",
        "schema_version": "1.0",
        "phase": "15.6",
        "protocol": {
            "device": "cpu",
            "primary_checkpoint": str(stable_checkpoint),
            "candidate_checkpoint": str(candidate_checkpoint),
            "returns_primary_response": True,
            "human_approved": False,
        },
        "shadow": {
            "prompt": prompt,
            "primary_response": primary_response.text,
            "candidate_response": shadow_result.candidate_text,
            "candidate_ran": shadow_result.candidate_error is None,
            "primary_error": shadow_result.primary_error,
            "candidate_error": shadow_result.candidate_error,
            "returned_response_source": "stable-primary",
        },
        "promotion_decision": asdict(decision),
        "stable_runtime_replaced": False,
        "phase_gates": {
            "primary_response_returned": True,
            "candidate_ran_in_shadow": shadow_result.candidate_error is None,
            "candidate_rejected": not decision.eligible,
            "human_approval_required": True,
            "stable_runtime_preserved": True,
            "no_automatic_replacement": True,
        },
    }
    report["phase_gates_passed"] = all(report["phase_gates"].values())
    if not report["phase_gates_passed"]:
        raise RuntimeError(f"Phase 15.6 gates failed: {report['phase_gates']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=None,
        help="Phase 15.5 JSON report; defaults to artifacts/evaluation/phase155_heldout_evaluation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON; defaults to artifacts/evaluation/phase156_shadow_promotion.json",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    root = args.root.resolve()
    evaluation = (args.evaluation_report or root / "artifacts/evaluation/phase155_heldout_evaluation.json").resolve()
    output = (args.output or root / "artifacts/evaluation/phase156_shadow_promotion.json").resolve()
    report = run_phase156(root=root, evaluation_report_path=evaluation, output_path=output, prompt=args.prompt)
    print(json.dumps({"output": str(output), "phase_gates_passed": report["phase_gates_passed"]}, indent=2))


if __name__ == "__main__":
    main()
