#!/usr/bin/env python3
"""Validate Phase 15.1 Teacher–Student contract gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.distillation.contract import (  # noqa: E402
    ContractError,
    QualityStatus,
    RecordSplit,
    RedactionStatus,
    TeacherStudentExample,
    VerificationStatus,
)

DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase151_contract.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase151_contract.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Phase 15.1 Teacher–Student data contract.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = TeacherStudentExample.create(
        prompt="How should a FastAPI endpoint validate a JWT?",
        response="Validate the signature, issuer, audience, and expiration before using the claims.",
        teacher_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        teacher_model_fingerprint="sha256:" + "1" * 64,
    )
    accepted = raw.mark_verified(
        verification_status=VerificationStatus.HUMAN_APPROVED,
        quality_status=QualityStatus.ACCEPTED,
        split=RecordSplit.TRAIN,
        redaction_status=RedactionStatus.CLEAN,
        user_approved=True,
        execution_evidence={"status": "reviewed", "tests": 0},
    )
    round_trip = TeacherStudentExample.from_dict(json.loads(accepted.to_json()))
    blocked_record_rejected = False
    try:
        raw.mark_verified(
            verification_status=VerificationStatus.UNVERIFIED,
            quality_status=QualityStatus.ACCEPTED,
            split=RecordSplit.TRAIN,
            redaction_status=RedactionStatus.NOT_REVIEWED,
        )
    except ContractError:
        blocked_record_rejected = True
    report = {
        "format": "fodci.phase151_contract",
        "schema_version": "1.0",
        "phase": "15.1",
        "raw_record_id": raw.record_id,
        "accepted_record_id": accepted.record_id,
        "record_id_stable_after_verification": raw.record_id == accepted.record_id,
        "round_trip_preserved": round_trip.to_json() == accepted.to_json(),
        "accepted_training_eligible": accepted.training_eligible,
        "raw_training_eligible": raw.training_eligible,
        "blocked_unverified_acceptance_rejected": blocked_record_rejected,
        "automatic_training_performed": False,
        "external_api_used": False,
        "phase_gates": {
            "record_id_is_sha256": accepted.record_id.startswith("sha256:") and len(accepted.record_id) == 71,
            "backend_domain_only": accepted.domain == "backend",
            "provenance_present": bool(accepted.teacher_model and accepted.teacher_model_fingerprint),
            "round_trip_preserved": round_trip.to_json() == accepted.to_json(),
            "unverified_acceptance_rejected": blocked_record_rejected,
            "raw_record_not_training_eligible": raw.training_eligible is False,
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
        "# Phase 15.1 — Teacher–Student Data Contract",
        "",
        "> Interaction data is captured as provenance-rich records, but no record is training-eligible until verification and redaction gates pass.",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| Stable record identity | `{report['record_id_stable_after_verification']}` |",
        f"| JSON round-trip | `{report['round_trip_preserved']}` |",
        f"| Accepted record training-eligible | `{report['accepted_training_eligible']}` |",
        f"| Raw record training-eligible | `{report['raw_training_eligible']}` |",
        f"| Unverified acceptance rejected | `{report['blocked_unverified_acceptance_rejected']}` |",
        f"| Automatic training performed | `{report['automatic_training_performed']}` |",
        f"| All phase gates | `{report['phase_gates_passed']}` |",
        "",
        "The contract prevents raw teacher output from becoming training data automatically. Later phases add capture, verification, offline training, and promotion controls.",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
