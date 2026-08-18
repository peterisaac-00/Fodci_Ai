from __future__ import annotations

import pytest

from backend_ai.distillation.buffer import BufferError, InteractionBuffer
from backend_ai.distillation.contract import TeacherStudentExample


def _example(index: int = 1) -> TeacherStudentExample:
    return TeacherStudentExample.create(
        prompt=f"How do I validate a JWT in FastAPI? {index}",
        response="Validate the signature, issuer, audience, and expiration.",
        teacher_model="teacher",
        teacher_model_fingerprint="sha256:" + str(index) * 64,
    )


def test_interaction_buffer_appends_and_round_trips(tmp_path) -> None:
    buffer = InteractionBuffer(tmp_path / "buffer.jsonl", max_records=2)
    first = buffer.append(_example())

    assert buffer.read_all() == (first,)
    assert buffer.stats().total_records == 1
    assert buffer.stats().pending_records == 1
    assert buffer.stats().eligible_records == 0


def test_interaction_buffer_rejects_duplicate_and_bound(tmp_path) -> None:
    buffer = InteractionBuffer(tmp_path / "buffer.jsonl", max_records=1)
    first = _example()
    buffer.append(first)

    with pytest.raises(BufferError, match="duplicate"):
        buffer.append(first)
    with pytest.raises(BufferError, match="bound"):
        buffer.append(_example(2))


def test_interaction_buffer_exposes_only_verified_training_records(tmp_path) -> None:
    from backend_ai.distillation.contract import QualityStatus, RecordSplit, RedactionStatus, VerificationStatus

    buffer = InteractionBuffer(tmp_path / "buffer.jsonl")
    accepted = _example().mark_verified(
        verification_status=VerificationStatus.EXECUTION_PASS,
        quality_status=QualityStatus.ACCEPTED,
        split=RecordSplit.TRAIN,
        redaction_status=RedactionStatus.CLEAN,
    )
    buffer.append(accepted)

    assert buffer.pending() == ()
    assert buffer.training_eligible() == (accepted,)
    assert buffer.export_split(RecordSplit.TRAIN) == (accepted,)
