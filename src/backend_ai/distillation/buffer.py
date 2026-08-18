"""Phase 15.2 local append-only Teacher–Student interaction buffer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

from backend_ai.distillation.contract import (
    ContractError,
    QualityStatus,
    RecordSplit,
    TeacherStudentExample,
)


class BufferError(ValueError):
    """Raised when the local interaction buffer is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class BufferStats:
    total_records: int
    pending_records: int
    eligible_records: int
    rejected_records: int


class InteractionBuffer:
    """A bounded JSONL store; it never invokes training or external services."""

    def __init__(self, path: Path | str, *, max_records: int = 100_000) -> None:
        self.path = Path(path).expanduser()
        if self.path.exists() and self.path.is_symlink():
            raise BufferError("interaction buffer must not be a symlink")
        if not 1 <= max_records <= 1_000_000:
            raise ValueError("max_records is outside the safety bound")
        self.max_records = max_records

    def append(self, example: TeacherStudentExample) -> TeacherStudentExample:
        if not isinstance(example, TeacherStudentExample):
            raise BufferError("only TeacherStudentExample records can be appended")
        records = self.read_all()
        if any(item.record_id == example.record_id for item in records):
            raise BufferError("duplicate record_id cannot be appended")
        if len(records) >= self.max_records:
            raise BufferError("interaction buffer reached its record bound")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink():
            raise BufferError("interaction buffer parent must not be a symlink")
        with self.path.open("ab") as stream:
            stream.write((example.to_json() + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        return example

    def read_all(self) -> tuple[TeacherStudentExample, ...]:
        if not self.path.exists():
            return ()
        if self.path.is_symlink():
            raise BufferError("interaction buffer must not be a symlink")
        records: list[TeacherStudentExample] = []
        try:
            for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    records.append(TeacherStudentExample.from_dict(payload))
                except (json.JSONDecodeError, ContractError, TypeError, ValueError) as exc:
                    raise BufferError(f"malformed interaction record at line {line_number}") from exc
        except UnicodeDecodeError as exc:
            raise BufferError("interaction buffer must be UTF-8") from exc
        if len(records) > self.max_records:
            raise BufferError("interaction buffer exceeds its record bound")
        if len({item.record_id for item in records}) != len(records):
            raise BufferError("interaction buffer contains duplicate record IDs")
        return tuple(records)

    def pending(self) -> tuple[TeacherStudentExample, ...]:
        return tuple(item for item in self.read_all() if item.quality_status is QualityStatus.PENDING)

    def training_eligible(self) -> tuple[TeacherStudentExample, ...]:
        return tuple(item for item in self.read_all() if item.training_eligible)

    def stats(self) -> BufferStats:
        records = self.read_all()
        return BufferStats(
            total_records=len(records),
            pending_records=sum(item.quality_status is QualityStatus.PENDING for item in records),
            eligible_records=sum(item.training_eligible for item in records),
            rejected_records=sum(item.quality_status is QualityStatus.REJECTED for item in records),
        )

    def export_split(self, split: RecordSplit) -> tuple[TeacherStudentExample, ...]:
        return tuple(item for item in self.training_eligible() if item.split is split)


__all__ = ["BufferError", "BufferStats", "InteractionBuffer"]
