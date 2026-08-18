from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend_ai.distillation.contract import (
    ContractError,
    QualityStatus,
    RecordSplit,
    RedactionStatus,
    TeacherStudentExample,
    VerificationStatus,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "artifacts" / "evaluation" / "phase151_contract.json"


def test_phase151_contract_report_passes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["format"] == "fodci.phase151_contract"
    assert report["phase_gates_passed"] is True
    assert report["automatic_training_performed"] is False
    assert report["raw_training_eligible"] is False
    assert report["accepted_training_eligible"] is True


def test_phase151_raw_record_requires_verification_before_acceptance() -> None:
    raw = TeacherStudentExample.create(
        prompt="How do I test a FastAPI endpoint?",
        response="Use pytest and a test client.",
        teacher_model="teacher",
        teacher_model_fingerprint="sha256:" + "2" * 64,
    )

    assert raw.training_eligible is False
    with pytest.raises(ContractError):
        raw.mark_verified(
            verification_status=VerificationStatus.UNVERIFIED,
            quality_status=QualityStatus.ACCEPTED,
            split=RecordSplit.TRAIN,
            redaction_status=RedactionStatus.NOT_REVIEWED,
        )


def test_phase151_accepted_record_is_round_trip_stable() -> None:
    raw = TeacherStudentExample.create(
        prompt="What is a SQL transaction?",
        response="A transaction groups changes and supports commit or rollback.",
        teacher_model="teacher",
        teacher_model_fingerprint="sha256:" + "3" * 64,
    )
    accepted = raw.mark_verified(
        verification_status=VerificationStatus.EXECUTION_PASS,
        quality_status=QualityStatus.ACCEPTED,
        split=RecordSplit.TRAIN,
        redaction_status=RedactionStatus.CLEAN,
    )

    restored = TeacherStudentExample.from_dict(json.loads(accepted.to_json()))
    assert accepted.record_id == raw.record_id
    assert restored.to_json() == accepted.to_json()
    assert restored.training_eligible is True
