#!/usr/bin/env python3
"""Bounded offline Teacher–Student distillation into an experimental Fodci checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.checkpoint import CheckpointManager  # noqa: E402
from backend_ai.dataset.samples import TrainingExample  # noqa: E402
from backend_ai.distillation.contract import TeacherStudentExample  # noqa: E402
from backend_ai.model import FodciModel, ModelConfig  # noqa: E402
from backend_ai.tokenizer import EOS_ID, FodciTokenizer  # noqa: E402
from backend_ai.training import FodciTrainer, TrainingConfig  # noqa: E402

DEFAULT_BASE = ROOT / "artifacts" / "checkpoints" / "fodci-testing-qa-v1.pt"
DEFAULT_TRAIN = ROOT / "training_data" / "distillation" / "phase154_train.jsonl"
DEFAULT_VALIDATION = ROOT / "training_data" / "distillation" / "phase154_validation.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "checkpoints" / "fodci-distilled-phase154-v1.pt"
DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase154_offline_distillation.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase154_offline_distillation.md"
CONTEXT_LENGTH = 256
SEED = 2026


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_records(path: Path) -> list[TeacherStudentExample]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = TeacherStudentExample.from_dict(json.loads(line))
        if not record.training_eligible:
            raise ValueError(f"record at {path}:{line_number} is not training eligible")
        records.append(record)
    if not records:
        raise ValueError(f"distillation split is empty: {path}")
    return records


def record_example(tokenizer: FodciTokenizer, record: TeacherStudentExample) -> TrainingExample:
    text = f"### Instruction\nAnswer clearly in English.\n\n### Input\n{record.prompt}\n\n### Response\n{record.response}\n"
    token_ids = tokenizer.encode(text) + [EOS_ID]
    if len(token_ids) < 2 or len(token_ids) > CONTEXT_LENGTH + 1:
        raise ValueError(f"record is outside context bound: {record.record_id}")
    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]
    response_start = len(tokenizer.encode(f"### Instruction\nAnswer clearly in English.\n\n### Input\n{record.prompt}\n\n### Response\n"))
    loss_mask = (False,) * max(0, response_start - 1) + (True,) * max(1, len(token_ids) - response_start - 1)
    width = CONTEXT_LENGTH
    input_ids = input_ids + [0] * (width - len(input_ids))
    target_ids = target_ids + [0] * (width - len(target_ids))
    loss_mask = (loss_mask[: len(token_ids) - 1] + (False,) * width)[:width]
    return TrainingExample(tuple(input_ids), tuple(target_ids), record.record_id, tuple(loss_mask))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an experimental Fodci checkpoint offline from verified Teacher–Student records.")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_steps <= 0 or args.repeats <= 0 or args.learning_rate <= 0:
        raise ValueError("max_steps, repeats, and learning_rate must be positive")
    for path in (args.base_checkpoint, args.train, args.validation):
        if not path.is_file():
            raise FileNotFoundError(path)
    tokenizer = FodciTokenizer()
    train_records = read_records(args.train)
    validation_records = read_records(args.validation)
    train_examples = [record_example(tokenizer, record) for record in train_records] * args.repeats
    validation_examples = [record_example(tokenizer, record) for record in validation_records]
    inspection = CheckpointManager(args.base_checkpoint.parent, model_version="phase154-inspection").inspect(args.base_checkpoint)
    model_config = ModelConfig(**inspection.metadata.model_config)
    base_model = FodciModel(model_config)
    base_manager = CheckpointManager(args.base_checkpoint.parent, model_version=inspection.metadata.model_version)
    base_manager.load_model(args.base_checkpoint, base_model, device=torch.device("cpu"))
    trainer = FodciTrainer(
        base_model,
        train_examples,
        validation_examples,
        TrainingConfig(epochs=1, max_steps=args.max_steps, batch_size=2, learning_rate=args.learning_rate, weight_decay=0.01, max_grad_norm=1.0, device="cpu", seed=SEED, validation_interval=1, checkpoint_interval=0, output_dir=args.output.parent),
        model_version="fodci-distilled-phase154-v1",
        checkpoint_run_metadata={"phase": "15.4", "stage": "offline-distillation", "base_checkpoint": str(args.base_checkpoint), "base_checkpoint_sha256": sha256(args.base_checkpoint), "teacher_model": train_records[0].teacher_model, "train_records": len(train_records), "validation_records": len(validation_records)},
    )
    baseline_loss, baseline_steps, baseline_tokens = trainer.evaluate(validation_examples)
    before = [parameter.detach().clone() for parameter in base_model.parameters()]
    started = time.perf_counter()
    result = trainer.train()
    trained_loss, validation_steps, validation_tokens = trainer.evaluate(validation_examples)
    saved = trainer.save_checkpoint(args.output, run_metadata={"phase": "15.4", "stage": "offline-distillation", "base_checkpoint": str(args.base_checkpoint), "base_checkpoint_sha256": sha256(args.base_checkpoint), "teacher_model": train_records[0].teacher_model, "train_records": len(train_records), "validation_records": len(validation_records)})
    reloaded_model = FodciModel(model_config)
    loaded = CheckpointManager(saved.parent, model_version="fodci-distilled-phase154-v1").load_model(saved, reloaded_model, device=torch.device("cpu"))
    reloaded_trainer = FodciTrainer(reloaded_model, train_examples[:2], validation_examples, TrainingConfig(epochs=1, max_steps=1, batch_size=2, device="cpu", seed=SEED, checkpoint_interval=0, output_dir=saved.parent), model_version="fodci-distilled-phase154-v1")
    reloaded_loss, reload_steps, reload_tokens = reloaded_trainer.evaluate(validation_examples)
    parameters_changed = any(not torch.equal(old, new) for old, new in zip(before, base_model.parameters(), strict=True))
    finite_loss = all(torch.isfinite(torch.tensor(value)) for value in (baseline_loss, trained_loss, reloaded_loss))
    report = {
        "format": "fodci.phase154_offline_distillation",
        "schema_version": "1.0",
        "phase": "15.4",
        "model_version": "fodci-distilled-phase154-v1",
        "parameter_count": base_model.num_parameters,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "checkpoint_path": str(saved),
        "teacher_model": train_records[0].teacher_model,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_examples": len(train_examples),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "global_step": result.global_step,
        "training_seconds": time.perf_counter() - started,
        "baseline_validation_loss": baseline_loss,
        "trained_validation_loss": trained_loss,
        "reloaded_validation_loss": reloaded_loss,
        "validation_improvement": baseline_loss - trained_loss,
        "checkpoint_exists": saved.is_file(),
        "checkpoint_reload": loaded.metadata.model_version == "fodci-distilled-phase154-v1",
        "finite_loss": finite_loss,
        "parameters_changed": parameters_changed,
        "non_empty_splits": bool(train_examples and validation_examples),
        "stable_runtime_replaced": False,
        "automatic_online_training": False,
        "training_gates_passed": all((saved.is_file(), loaded.metadata.model_version == "fodci-distilled-phase154-v1", finite_loss, parameters_changed, bool(train_examples and validation_examples))),
        "validation_quality_gate_passed": trained_loss < baseline_loss,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": report["phase"], "parameter_count": report["parameter_count"], "global_step": report["global_step"], "baseline_validation_loss": report["baseline_validation_loss"], "trained_validation_loss": report["trained_validation_loss"], "training_gates_passed": report["training_gates_passed"], "validation_quality_gate_passed": report["validation_quality_gate_passed"], "stable_runtime_replaced": report["stable_runtime_replaced"], "checkpoint": str(saved)}, ensure_ascii=False, indent=2))
    return 0 if report["training_gates_passed"] else 1


def render_markdown(report: dict) -> str:
    return "\n".join([
        "# Phase 15.4 — Offline Distillation Training",
        "",
        "> This is a bounded offline update from verified Teacher–Student records. It does not train during user interaction and does not replace the stable runtime.",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Model parameters | {report['parameter_count']:,} |",
        f"| Verified train records | {report['train_records']} |",
        f"| Validation records | {report['validation_records']} |",
        f"| Training steps | {report['global_step']} |",
        f"| Validation loss | {report['baseline_validation_loss']:.6f} → {report['trained_validation_loss']:.6f} |",
        f"| Parameters changed | `{report['parameters_changed']}` |",
        f"| Checkpoint reload | `{report['checkpoint_reload']}` |",
        f"| Training gates | `{report['training_gates_passed']}` |",
        f"| Validation quality gate | `{report['validation_quality_gate_passed']}` |",
        f"| Stable runtime replaced | `{report['stable_runtime_replaced']}` |",
        "",
        "The distilled checkpoint is experimental. A loss improvement is not sufficient to claim natural language quality; Phase 15.5 evaluates held-out responses and regression behavior.",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
