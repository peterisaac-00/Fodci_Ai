from __future__ import annotations

from backend_ai.distillation.contract import QualityStatus, RecordSplit, RedactionStatus, TeacherStudentExample, VerificationStatus
from backend_ai.distillation.quality import QualityFilter, VerificationGate


def _example(response: str) -> TeacherStudentExample:
    return TeacherStudentExample.create(
        prompt="How should a FastAPI backend validate a JWT?",
        response=response,
        teacher_model="teacher",
        teacher_model_fingerprint="sha256:" + "7" * 64,
    )


def test_quality_filter_accepts_readable_backend_answer() -> None:
    assessment = QualityFilter().assess(
        "How should FastAPI validate a JWT?",
        "Validate the JWT signature, issuer, audience, and expiration in the backend before using its claims.",
    )

    assert assessment.accepted is True
    assert assessment.secret_detected is False
    assert assessment.backend_signal is True


def test_quality_filter_rejects_secret_and_repetition() -> None:
    secret = QualityFilter().assess("Store this", "api_key=sk_test_123456789012345678")
    repeated = QualityFilter().assess("Backend", "the the the the the the the the")

    assert secret.accepted is False
    assert secret.secret_detected is True
    assert repeated.accepted is False
    assert "repetitive" in " ".join(repeated.reasons)


def test_verification_gate_keeps_clean_but_unverified_example_pending() -> None:
    example = _example("Use FastAPI to validate the JWT signature and expiration.")

    result, assessment = VerificationGate().verify(example, split=RecordSplit.TRAIN)

    assert assessment.accepted is True
    assert result.quality_status is QualityStatus.PENDING
    assert result.verification_status is VerificationStatus.UNVERIFIED
    assert result.training_eligible is False


def test_verification_gate_accepts_execution_verified_example() -> None:
    example = _example("Use FastAPI to validate the JWT signature and expiration.")

    result, assessment = VerificationGate().verify(
        example,
        split=RecordSplit.TRAIN,
        execution_passed=True,
        redaction_status=RedactionStatus.CLEAN,
        execution_evidence={"tests_passed": 2},
    )

    assert assessment.accepted is True
    assert result.quality_status is QualityStatus.ACCEPTED
    assert result.verification_status is VerificationStatus.EXECUTION_PASS
    assert result.training_eligible is True


def test_verification_gate_rejects_secret_example() -> None:
    example = _example("Use api_key=sk_test_123456789012345678 in the FastAPI backend.")

    result, assessment = VerificationGate().verify(example, split=RecordSplit.TRAIN, execution_passed=True)

    assert assessment.secret_detected is True
    assert result.quality_status is QualityStatus.REJECTED
    assert result.redaction_status is RedactionStatus.BLOCKED
    assert result.training_eligible is False
