"""Phase 15.1 Teacher–Student data contract.

Records are immutable, JSONL-safe, provenance-rich, and explicitly separated
from model training. A record is eligible for training only after later quality
gates mark it accepted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Any


EXAMPLE_FORMAT = "fodci.teacher_student_example"
SCHEMA_VERSION = "1.0"
RECORD_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when a Teacher–Student record violates the data contract."""


class RecordSource(str, Enum):
    TEACHER_INTERACTION = "teacher_interaction"
    HUMAN_CORRECTION = "human_correction"
    EXECUTION_FEEDBACK = "execution_feedback"


class QualityStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    HEURISTIC_PASS = "heuristic_pass"
    EXECUTION_PASS = "execution_pass"
    HUMAN_APPROVED = "human_approved"
    REJECTED = "rejected"


class RecordSplit(str, Enum):
    BUFFER = "buffer"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RedactionStatus(str, Enum):
    NOT_REVIEWED = "not_reviewed"
    CLEAN = "clean"
    REDACTED = "redacted"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class TeacherStudentExample:
    """One provenance-rich interaction, not yet a training example by default."""

    record_id: str
    prompt: str
    response: str
    domain: str
    teacher_model: str
    teacher_model_fingerprint: str
    source: RecordSource
    quality_status: QualityStatus = QualityStatus.PENDING
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    split: RecordSplit = RecordSplit.BUFFER
    redaction_status: RedactionStatus = RedactionStatus.NOT_REVIEWED
    user_approved: bool = False
    execution_evidence: Mapping[str, Any] = None  # type: ignore[assignment]
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]
    created_at: str = ""

    def __post_init__(self) -> None:
        if not RECORD_ID_PATTERN.fullmatch(self.record_id):
            raise ContractError("record_id must be a sha256 fingerprint")
        for name in ("prompt", "response", "domain", "teacher_model", "teacher_model_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{name} must contain text")
        if self.domain != "backend":
            raise ContractError("only backend domain records are accepted")
        if not self.teacher_model_fingerprint.startswith("sha256:"):
            raise ContractError("teacher_model_fingerprint must be a sha256 fingerprint")
        if not isinstance(self.source, RecordSource) or not isinstance(self.quality_status, QualityStatus):
            raise ContractError("record source or quality status is invalid")
        if not isinstance(self.verification_status, VerificationStatus) or not isinstance(self.split, RecordSplit):
            raise ContractError("verification status or split is invalid")
        if not isinstance(self.redaction_status, RedactionStatus):
            raise ContractError("redaction status is invalid")
        if self.redaction_status in {RedactionStatus.NOT_REVIEWED, RedactionStatus.BLOCKED} and self.quality_status == QualityStatus.ACCEPTED:
            raise ContractError("accepted records must be reviewed for secrets")
        if self.quality_status == QualityStatus.ACCEPTED and self.verification_status in {VerificationStatus.UNVERIFIED, VerificationStatus.REJECTED}:
            raise ContractError("accepted records require positive verification")
        if self.quality_status == QualityStatus.ACCEPTED and self.split == RecordSplit.BUFFER:
            raise ContractError("accepted records must be assigned to a training or evaluation split")
        if self.user_approved and self.quality_status == QualityStatus.REJECTED:
            raise ContractError("rejected records cannot be user approved")
        created_at = self.created_at or _utc_now()
        _validate_timestamp(created_at)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "execution_evidence", _freeze_mapping(self.execution_evidence or {}))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata or {}))

    @classmethod
    def create(
        cls,
        *,
        prompt: str,
        response: str,
        teacher_model: str,
        teacher_model_fingerprint: str,
        source: RecordSource = RecordSource.TEACHER_INTERACTION,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TeacherStudentExample":
        identity = {
            "format": EXAMPLE_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "prompt": prompt,
            "response": response,
            "domain": "backend",
            "teacher_model": teacher_model,
            "teacher_model_fingerprint": teacher_model_fingerprint,
            "source": source.value,
        }
        record_id = "sha256:" + hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        return cls(
            record_id=record_id,
            prompt=prompt,
            response=response,
            domain="backend",
            teacher_model=teacher_model,
            teacher_model_fingerprint=teacher_model_fingerprint,
            source=source,
            metadata=metadata or {},
        )

    def mark_verified(
        self,
        *,
        verification_status: VerificationStatus,
        quality_status: QualityStatus,
        split: RecordSplit,
        redaction_status: RedactionStatus = RedactionStatus.CLEAN,
        user_approved: bool = False,
        execution_evidence: Mapping[str, Any] | None = None,
    ) -> "TeacherStudentExample":
        return TeacherStudentExample(
            record_id=self.record_id,
            prompt=self.prompt,
            response=self.response,
            domain=self.domain,
            teacher_model=self.teacher_model,
            teacher_model_fingerprint=self.teacher_model_fingerprint,
            source=self.source,
            quality_status=quality_status,
            verification_status=verification_status,
            split=split,
            redaction_status=redaction_status,
            user_approved=user_approved,
            execution_evidence=execution_evidence or self.execution_evidence,
            metadata=self.metadata,
            created_at=self.created_at,
        )

    @property
    def training_eligible(self) -> bool:
        return (
            self.quality_status is QualityStatus.ACCEPTED
            and self.verification_status in {VerificationStatus.EXECUTION_PASS, VerificationStatus.HUMAN_APPROVED}
            and self.redaction_status in {RedactionStatus.CLEAN, RedactionStatus.REDACTED}
            and self.split in {RecordSplit.TRAIN, RecordSplit.VALIDATION, RecordSplit.TEST}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": EXAMPLE_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "record_id": self.record_id,
            "prompt": self.prompt,
            "response": self.response,
            "domain": self.domain,
            "teacher_model": self.teacher_model,
            "teacher_model_fingerprint": self.teacher_model_fingerprint,
            "source": self.source.value,
            "quality_status": self.quality_status.value,
            "verification_status": self.verification_status.value,
            "split": self.split.value,
            "redaction_status": self.redaction_status.value,
            "user_approved": self.user_approved,
            "execution_evidence": dict(self.execution_evidence),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "training_eligible": self.training_eligible,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeacherStudentExample":
        required = {
            "format", "schema_version", "record_id", "prompt", "response", "domain",
            "teacher_model", "teacher_model_fingerprint", "source", "quality_status",
            "verification_status", "split", "redaction_status", "user_approved",
            "execution_evidence", "metadata", "created_at",
        }
        if set(payload) - (required | {"training_eligible"}) != set() or set(payload) != required | ({"training_eligible"} if "training_eligible" in payload else set()):
            raise ContractError("record fields are invalid")
        if payload["format"] != EXAMPLE_FORMAT or payload["schema_version"] != SCHEMA_VERSION:
            raise ContractError("record format/schema is invalid")
        record = cls(
            record_id=str(payload["record_id"]),
            prompt=str(payload["prompt"]),
            response=str(payload["response"]),
            domain=str(payload["domain"]),
            teacher_model=str(payload["teacher_model"]),
            teacher_model_fingerprint=str(payload["teacher_model_fingerprint"]),
            source=RecordSource(str(payload["source"])),
            quality_status=QualityStatus(str(payload["quality_status"])),
            verification_status=VerificationStatus(str(payload["verification_status"])),
            split=RecordSplit(str(payload["split"])),
            redaction_status=RedactionStatus(str(payload["redaction_status"])),
            user_approved=bool(payload["user_approved"]),
            execution_evidence=payload["execution_evidence"],
            metadata=payload["metadata"],
            created_at=str(payload["created_at"]),
        )
        if "training_eligible" in payload and bool(payload["training_eligible"]) != record.training_eligible:
            raise ContractError("training_eligible does not match record state")
        return record


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_timestamp(value: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("created_at must be an ISO-8601 timestamp") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("metadata/evidence must be mappings")
    return dict(value)


__all__ = [
    "EXAMPLE_FORMAT",
    "SCHEMA_VERSION",
    "ContractError",
    "RecordSource",
    "QualityStatus",
    "VerificationStatus",
    "RecordSplit",
    "RedactionStatus",
    "TeacherStudentExample",
]
