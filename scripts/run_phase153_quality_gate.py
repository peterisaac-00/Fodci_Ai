#!/usr/bin/env python3
"""Validate Phase 15.3 quality and verification gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.distillation.contract import QualityStatus, RecordSplit, RedactionStatus, TeacherStudentExample, VerificationStatus  # noqa: E402
from backend_ai.distillation.quality import QualityFilter, VerificationGate  # noqa: E402

DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase153_quality_gate.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase153_quality_gate.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 15.3 quality and verification gates.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def make_example(prompt: str, response: str, index: int) -> TeacherStudentExample:
    return TeacherStudentExample.create(
        prompt=prompt,
        response=response,
        teacher_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        teacher_model_fingerprint="sha256:" + str(index) * 64,
    )


def main() -> int:
    args = parse_args()
    quality_filter = QualityFilter()
    gate = VerificationGate(quality_filter)
    clean = make_example("How should FastAPI validate a JWT?", "Use FastAPI to validate the JWT signature, issuer, audience, and expiration.", 8)
    secret = make_example("Store a secret", "Use api_key=sk_test_123456789012345678 in the backend.", 9)
    repetitive = make_example("Explain a backend error", "the the the the the the the the", 10)
    clean_pending, clean_assessment = gate.verify(clean, split=RecordSplit.TRAIN)
    clean_accepted, accepted_assessment = gate.verify(clean, split=RecordSplit.TRAIN, execution_passed=True, redaction_status=RedactionStatus.CLEAN, execution_evidence={"tests_passed": 2})
    secret_result, secret_assessment = gate.verify(secret, split=RecordSplit.TRAIN, execution_passed=True)
    repetitive_result, repetitive_assessment = gate.verify(repetitive, split=RecordSplit.TRAIN, execution_passed=True)
    report = {
        "format": "fodci.phase153_quality_gate",
        "schema_version": "1.0",
        "phase": "15.3",
        "clean_pending": {"quality": clean_pending.quality_status.value, "verification": clean_pending.verification_status.value, "training_eligible": clean_pending.training_eligible, "assessment_accepted": clean_assessment.accepted},
        "clean_accepted": {"quality": clean_accepted.quality_status.value, "verification": clean_accepted.verification_status.value, "training_eligible": clean_accepted.training_eligible, "assessment_accepted": accepted_assessment.accepted},
        "secret_rejected": {"quality": secret_result.quality_status.value, "redaction": secret_result.redaction_status.value, "secret_detected": secret_assessment.secret_detected, "training_eligible": secret_result.training_eligible},
        "repetitive_rejected": {"quality": repetitive_result.quality_status.value, "reasons": repetitive_assessment.reasons, "training_eligible": repetitive_result.training_eligible},
        "automatic_training_performed": False,
        "phase_gates": {
            "clean_without_evidence_stays_pending": clean_pending.quality_status is QualityStatus.PENDING and clean_pending.training_eligible is False,
            "execution_verified_clean_record_accepted": clean_accepted.quality_status is QualityStatus.ACCEPTED and clean_accepted.training_eligible is True,
            "secret_record_rejected": secret_result.quality_status is QualityStatus.REJECTED and secret_result.redaction_status is RedactionStatus.BLOCKED,
            "repetitive_record_rejected": repetitive_result.quality_status is QualityStatus.REJECTED,
            "no_automatic_training": True,
        },
    }
    report["phase_gates_passed"] = all(report["phase_gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": report["phase"], "phase_gates_passed": report["phase_gates_passed"], "automatic_training_performed": report["automatic_training_performed"], "report": str(args.report)}, ensure_ascii=False, indent=2))
    return 0 if report["phase_gates_passed"] else 1


def render_markdown(report: dict) -> str:
    return "\n".join([
        "# Phase 15.3 — Quality Filter and Verification Gate",
        "",
        "> Quality heuristics decide eligibility; they do not claim semantic correctness.",
        "",
        "| Case | Result |",
        "|---|---|",
        f"| Clean without evidence | `{report['clean_pending']['quality']}/{report['clean_pending']['verification']}`; eligible `{report['clean_pending']['training_eligible']}` |",
        f"| Clean with execution evidence | `{report['clean_accepted']['quality']}/{report['clean_accepted']['verification']}`; eligible `{report['clean_accepted']['training_eligible']}` |",
        f"| Secret-like response | `{report['secret_rejected']['quality']}`; redaction `{report['secret_rejected']['redaction']}` |",
        f"| Repetitive response | `{report['repetitive_rejected']['quality']}` |",
        f"| Automatic training performed | `{report['automatic_training_performed']}` |",
        f"| All phase gates | `{report['phase_gates_passed']}` |",
        "",
        "Only clean records with positive execution or human evidence can move to a training split. Rejected records never enter Fodci training.",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
