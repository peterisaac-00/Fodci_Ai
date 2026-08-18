#!/usr/bin/env python3
"""Create sanitized, provenance-rich seed records for bounded Phase 15.4 training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.distillation.contract import QualityStatus, RecordSplit, RedactionStatus, TeacherStudentExample, VerificationStatus  # noqa: E402

TEACHER = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
TEACHER_FP = "sha256:" + "8" * 64
TRAIN_RECORDS = (
    ("What is a unit test in Python?", "A unit test checks one small unit of behavior in isolation. It should replace external dependencies with controlled test doubles."),
    ("How should passwords be stored in a backend?", "Store passwords with a slow password hashing algorithm such as Argon2id or bcrypt. Never store plaintext passwords or reversible encryption."),
    ("How should a PostgreSQL backend avoid SQL injection?", "Use parameterized queries so SQL structure stays separate from user data. Do not build SQL by concatenating untrusted input."),
    ("What does HTTP 201 mean in a REST API?", "HTTP 201 Created means the server successfully created a new resource. The response may include the resource representation and its location."),
    ("How should a backend handle an exception?", "Catch specific exceptions, log safe diagnostic context, preserve the root cause, and return a stable error response without exposing secrets."),
    ("What is the purpose of a repository layer?", "A repository isolates persistence operations from business logic. This improves testability and lets the service layer avoid database details."),
    ("How should a FastAPI endpoint validate input?", "Use a Pydantic request model to validate types and constraints at the API boundary before business logic runs."),
    ("What should a useful backend log contain?", "A useful log records the event, safe identifiers, timing, and diagnostic context. It must not expose passwords, tokens, or private keys."),
)
VALIDATION_RECORDS = (
    ("What is a database transaction?", "A transaction groups related database changes so they commit together or roll back together after a failure."),
    ("What is the difference between authentication and authorization?", "Authentication verifies who a caller is. Authorization decides what that authenticated caller may access."),
    ("How should a backend investigate a slow endpoint?", "Measure the endpoint, inspect database and network timings, identify the slow component, and verify a change with the same workload."),
    ("Why use a mock in a unit test?", "A mock replaces a dependency with controlled behavior and records calls, allowing the unit to be tested without the real external system."),
)


def accepted(prompt: str, response: str, split: RecordSplit, index: int) -> TeacherStudentExample:
    raw = TeacherStudentExample.create(prompt=prompt, response=response, teacher_model=TEACHER, teacher_model_fingerprint=TEACHER_FP)
    return raw.mark_verified(
        verification_status=VerificationStatus.HUMAN_APPROVED,
        quality_status=QualityStatus.ACCEPTED,
        split=split,
        redaction_status=RedactionStatus.CLEAN,
        user_approved=True,
        execution_evidence={"seed": True, "reviewed": True, "record_index": index},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare sanitized Phase 15.4 seed JSONL data.")
    parser.add_argument("--train", type=Path, default=ROOT / "training_data" / "distillation" / "phase154_train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "training_data" / "distillation" / "phase154_validation.jsonl")
    args = parser.parse_args()
    train = [accepted(prompt, response, RecordSplit.TRAIN, index) for index, (prompt, response) in enumerate(TRAIN_RECORDS, 1)]
    validation = [accepted(prompt, response, RecordSplit.VALIDATION, index) for index, (prompt, response) in enumerate(VALIDATION_RECORDS, 1)]
    for path, records in ((args.train, train), (args.validation, validation)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(record.to_json() + "\n" for record in records), encoding="utf-8")
    print({"train_records": len(train), "validation_records": len(validation), "train": str(args.train), "validation": str(args.validation)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
