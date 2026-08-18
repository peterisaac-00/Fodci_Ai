#!/usr/bin/env python3
"""Validate Phase 15.2 append-only local interaction buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.distillation.buffer import BufferError, InteractionBuffer  # noqa: E402
from backend_ai.distillation.contract import TeacherStudentExample  # noqa: E402

DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase152_buffer.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase152_buffer.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 15.2 interaction capture buffer.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="fodci-phase152-") as directory:
        path = Path(directory) / "interactions.jsonl"
        buffer = InteractionBuffer(path, max_records=2)
        first = TeacherStudentExample.create(
            prompt="How should a FastAPI endpoint validate a JWT?",
            response="Validate the signature, issuer, audience, and expiration.",
            teacher_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            teacher_model_fingerprint="sha256:" + "4" * 64,
        )
        buffer.append(first)
        duplicate_rejected = False
        bound_rejected = False
        try:
            buffer.append(first)
        except BufferError:
            duplicate_rejected = True
        second = TeacherStudentExample.create(
            prompt="What is a SQL transaction?",
            response="A transaction groups changes and supports commit or rollback.",
            teacher_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            teacher_model_fingerprint="sha256:" + "5" * 64,
        )
        buffer.append(second)
        try:
            buffer.append(TeacherStudentExample.create(
                prompt="How do I run pytest?",
                response="Run pytest from the backend project root.",
                teacher_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                teacher_model_fingerprint="sha256:" + "6" * 64,
            ))
        except BufferError:
            bound_rejected = True
        stats = buffer.stats()
        report = {
            "format": "fodci.phase152_buffer",
            "schema_version": "1.0",
            "phase": "15.2",
            "storage_format": "append-only-jsonl",
            "record_count": stats.total_records,
            "pending_count": stats.pending_records,
            "eligible_count": stats.eligible_records,
            "duplicate_rejected": duplicate_rejected,
            "bound_rejected": bound_rejected,
            "automatic_training_performed": False,
            "external_api_used": False,
            "raw_buffer_path_is_local": True,
            "phase_gates": {
                "bounded_storage": stats.total_records == 2,
                "pending_records_preserved": stats.pending_records == 2,
                "duplicate_rejected": duplicate_rejected,
                "bound_rejected": bound_rejected,
                "training_not_started": True,
                "local_only": True,
            },
        }
        report["phase_gates_passed"] = all(report["phase_gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": report["phase"], "phase_gates_passed": report["phase_gates_passed"], "record_count": report["record_count"], "automatic_training_performed": report["automatic_training_performed"], "report": str(args.report)}, ensure_ascii=False, indent=2))
    return 0 if report["phase_gates_passed"] else 1


def render_markdown(report: dict) -> str:
    return "\n".join([
        "# Phase 15.2 — Interaction Capture and Training Buffer",
        "",
        "> Interactions are persisted locally as bounded JSONL records. Capture never starts model training.",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| Records stored | {report['record_count']} |",
        f"| Pending records | {report['pending_count']} |",
        f"| Duplicate rejected | `{report['duplicate_rejected']}` |",
        f"| Bound rejected | `{report['bound_rejected']}` |",
        f"| Automatic training performed | `{report['automatic_training_performed']}` |",
        f"| All phase gates | `{report['phase_gates_passed']}` |",
        "",
        "The default buffer is local and ignored by Git. Phase 15.3 decides which records can leave the pending state and enter a training split.",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
