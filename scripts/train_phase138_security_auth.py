from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import torch

from backend_ai.checkpoint import CheckpointManager
from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.instructions import InstructionDatasetPipeline
from backend_ai.evaluation import EvaluationConfig, FodciEvaluator
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import FodciTokenizer
from backend_ai.training import FodciTrainer, TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "training_data" / "security_auth"
DEFAULT_BASE = ROOT / "artifacts" / "checkpoints" / "fodci-debugging-v1.pt"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-security-auth-v1.pt"
DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase138_security_auth_training.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase138_security_auth_training.md"
SEED = 2026
BASE_VERSION = "fodci-debugging-v1"
MODEL_VERSION = "fodci-security-auth-v1"
DATASET_VERSION = "security-auth-specialist-v1"


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def make_pipeline(path: Path, tokenizer: FodciTokenizer) -> InstructionDatasetPipeline:
    return InstructionDatasetPipeline(
        DatasetConfig(path, supported_extensions=frozenset({".txt"}), context_length=64, use_eos_document_boundaries=False),
        tokenizer,
    )


def source(items: list[Any]) -> Callable[[], list[Any]]:
    return lambda: list(items)


def split_examples(examples: list[Any]) -> tuple[list[Any], list[Any], dict[str, int]]:
    document_ids = sorted({example.document_id for example in examples})
    if len(document_ids) < 2:
        raise ValueError("specialist dataset requires at least two documents")
    validation_count = max(1, round(len(document_ids) * 0.2))
    validation_ids = set(document_ids[-validation_count:])
    train = [example for example in examples if example.document_id not in validation_ids]
    validation = [example for example in examples if example.document_id in validation_ids]
    if not train or not validation:
        raise ValueError("specialist train/validation split is empty")
    return train, validation, {"documents": len(document_ids), "train_documents": len(document_ids) - len(validation_ids), "validation_documents": len(validation_ids), "train_examples": len(train), "validation_examples": len(validation)}


def evaluate(model: FodciModel, examples: list[Any], dataset_path: Path) -> dict[str, Any]:
    config = EvaluationConfig(batch_size=2, device="cpu", seed=SEED, dataset_path=dataset_path, dataset_split="security-auth-validation", checkpoint_dir=DEFAULT_CHECKPOINT.parent, model_version=MODEL_VERSION)
    evaluator = FodciEvaluator(model, config, dataset_metadata={"dataset_version": DATASET_VERSION, "dataset_sha256": sha256_directory(dataset_path), "loss_type": "response_only"})
    return evaluator.evaluate(source(examples)).to_dict()


def render_markdown(report: dict[str, Any]) -> str:
    base = report["evaluation"]["base"]
    trained = report["evaluation"]["trained"]
    gates = report["validation_gates"]
    config = report["training_config"]
    return f"""# Phase 13.8 — Security & Authentication Patterns Training

> This is a bounded specialization experiment. It validates transfer from the Phase 13.7 checkpoint to JWT, OAuth2, password hashing, and security middleware patterns; it does not claim production security certification or complete threat-model coverage.

## Specialization scope

The dataset contains **32 training records** and **8 validation records** covering JWT validation, OAuth2 flows, password hashing, and authentication middleware. The curriculum emphasizes fail-closed behavior, least privilege, redaction, bounded lifetimes, and tenant-aware authorization. The validation documents are separate from the training documents and are selected deterministically by sorted document identity.

## Reproducibility

| Field | Value |
|---|---|
| Base checkpoint | `{report['base_checkpoint']}` |
| Base model version | `{report['base_model_version']}` |
| Specialist checkpoint | `{report['checkpoint_path']}` |
| Specialist model version | `{report['model_version']}` |
| Dataset version | `{report['dataset_version']}` |
| Dataset SHA-256 | `{report['dataset_sha256']}` |
| Seed | `{report['seed']}` |
| Parameters | `{report['model_parameters']:,}` |

## Training configuration

| Field | Value |
|---|---:|
| Device | `{config['device']}` |
| Epochs | {config['epochs']} |
| Maximum steps | {config['max_steps']} |
| Batch size | {config['batch_size']} |
| Learning rate | {config['learning_rate']} |
| Weight decay | {config['weight_decay']} |
| Training seconds | {report['training_seconds']:.4f} |
| Global step | {report['training_result']['global_step']} |

## Before/after validation

| Metric | Phase 13.7 base | After Security specialization |
|---|---:|---:|
| Validation loss | {base['loss']:.9f} | {trained['loss']:.9f} |
| Response loss | {base['response_loss']:.9f} | {trained['response_loss']:.9f} |
| Perplexity | {base['perplexity']:.6f} | {trained['perplexity']:.6f} |
| Evaluation examples | {base['evaluation_examples']} | {trained['evaluation_examples']} |
| Response tokens | {base['evaluated_tokens']} | {trained['evaluated_tokens']} |

## Validation gates

| Gate | Result |
|---|---|
| Base checkpoint version is Phase 13.7 | `{gates['base_checkpoint_compatible']}` |
| Specialist dataset produced examples | `{gates['dataset_examples']}` |
| Train/validation split non-empty | `{gates['split_non_empty']}` |
| Base evaluation finite | `{gates['base_evaluation_finite']}` |
| Training loss finite | `{gates['finite_training_loss']}` |
| Specialist checkpoint exists | `{gates['checkpoint_exists']}` |
| Specialist checkpoint reload succeeds | `{gates['checkpoint_reload']}` |
| Parameters changed | `{gates['parameters_changed']}` |
| All gates passed | `{gates['all_passed']}` |

The objective loss result validates the specialization pipeline and must be read together with the held-out generation benchmark at `training_data/security_auth/evaluation/phase_138.jsonl`. Keyword coverage is a conservative proxy and is not a semantic judge; security correctness still requires threat modeling, implementation review, and execution-aware tests.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the Fodci Phase 13.7 checkpoint on security and authentication patterns.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    if min(args.max_steps, args.epochs, args.batch_size) <= 0 or args.learning_rate <= 0:
        raise ValueError("training limits must be positive")
    random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.perf_counter()
    dataset_path = args.dataset.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    specialist_checkpoint = args.checkpoint.resolve()
    base_manager = CheckpointManager(base_checkpoint.parent, model_version=BASE_VERSION)
    base_info = base_manager.inspect(base_checkpoint)
    base_checkpoint_compatible = base_info.metadata.model_version == BASE_VERSION
    if not base_checkpoint_compatible:
        raise ValueError(f"expected {BASE_VERSION} checkpoint, got {base_info.metadata.model_version}")
    tokenizer = FodciTokenizer()
    examples = list(make_pipeline(dataset_path / "train", tokenizer).iter_training_examples())
    train_examples, validation_examples, split = split_examples(examples)
    base_model = FodciModel(ModelConfig(seed=SEED))
    base_manager.load_model(base_checkpoint, base_model, device=torch.device("cpu"))
    base_eval = evaluate(base_model, validation_examples, dataset_path / "validation")
    training_model = FodciModel(ModelConfig(seed=SEED))
    base_manager.load_model(base_checkpoint, training_model, device=torch.device("cpu"))
    before = {name: value.detach().clone() for name, value in training_model.named_parameters()}
    training_config = TrainingConfig(epochs=args.epochs, max_steps=args.max_steps, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=0.01, max_grad_norm=1.0, device="cpu", seed=SEED, validation_interval=1, checkpoint_interval=0, output_dir=specialist_checkpoint.parent)
    trainer = FodciTrainer(training_model, source(train_examples), source(validation_examples), training_config, model_version=MODEL_VERSION, checkpoint_run_metadata={"phase": "13.8", "base_checkpoint": str(base_checkpoint), "dataset_version": DATASET_VERSION})
    training_result = trainer.train()
    checkpoint = trainer.save_checkpoint(specialist_checkpoint)
    trained_model = FodciModel(ModelConfig(seed=SEED))
    CheckpointManager(checkpoint.parent, model_version=MODEL_VERSION).load_model(checkpoint, trained_model, device=torch.device("cpu"))
    trained_eval = evaluate(trained_model, validation_examples, dataset_path / "validation")
    reloaded_eval = evaluate(trained_model, validation_examples, dataset_path / "validation")
    changed = any(not torch.equal(before[name], parameter.detach().cpu()) for name, parameter in training_model.named_parameters())
    gates = {
        "base_checkpoint_compatible": base_checkpoint_compatible,
        "dataset_examples": bool(examples),
        "split_non_empty": bool(train_examples and validation_examples),
        "base_evaluation_finite": bool(torch.isfinite(torch.tensor(base_eval["loss"]))),
        "finite_training_loss": bool(training_result.history and all(torch.isfinite(torch.tensor(metric.train_loss)) for metric in training_result.history)),
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_reload": abs(float(reloaded_eval["loss"]) - float(trained_eval["loss"])) < 1e-8,
        "parameters_changed": changed,
    }
    gates["all_passed"] = all(gates.values())
    if not gates["all_passed"]:
        raise RuntimeError(f"Phase 13.8 validation failed: {gates}")
    report = {
        "format": "fodci.phase138_security_auth_training",
        "schema_version": "1.0",
        "phase": "13.8",
        "base_checkpoint": str(base_checkpoint),
        "base_model_version": base_info.metadata.model_version,
        "checkpoint_path": str(checkpoint),
        "model_version": MODEL_VERSION,
        "model_parameters": sum(parameter.numel() for parameter in trained_model.parameters()),
        "dataset_version": DATASET_VERSION,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_directory(dataset_path),
        "seed": SEED,
        "split": split,
        "training_config": training_config.to_dict(),
        "training_seconds": training_result.elapsed_seconds,
        "training_result": {"global_step": training_result.global_step, "history": [metric.to_dict() for metric in training_result.history]},
        "evaluation": {"base": base_eval, "trained": trained_eval, "reloaded": reloaded_eval},
        "validation_gates": gates,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": "13.8", "checkpoint": str(checkpoint), "base_loss": base_eval["loss"], "trained_loss": trained_eval["loss"], "global_step": training_result.global_step, "validation_gates": gates, "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
