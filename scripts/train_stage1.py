from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import torch

from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.instructions import InstructionDatasetPipeline
from backend_ai.evaluation import EvaluationConfig, FodciEvaluator
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import FodciTokenizer
from backend_ai.training import FodciTrainer, TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "training_data" / "fundamentals"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-stage1-v1.pt"
DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "stage1_training.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase133_stage1_training.md"
SEED = 2026
MODEL_VERSION = "fodci-stage1-v1"
DATASET_VERSION = "stage1-fundamentals-v1"


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def make_pipeline(dataset_dir: Path, tokenizer: FodciTokenizer) -> InstructionDatasetPipeline:
    return InstructionDatasetPipeline(
        DatasetConfig(
            dataset_dir,
            supported_extensions=frozenset({".txt"}),
            context_length=256,
            use_eos_document_boundaries=False,
        ),
        tokenizer,
    )


def split_examples(examples: list[Any], validation_fraction: float = 0.2) -> tuple[list[Any], list[Any], dict[str, Any]]:
    document_ids = sorted({example.document_id for example in examples})
    if len(document_ids) < 2:
        raise ValueError("Stage 1 training requires at least two instruction documents")
    validation_count = max(1, int(round(len(document_ids) * validation_fraction)))
    validation_ids = set(document_ids[-validation_count:])
    train = [example for example in examples if example.document_id not in validation_ids]
    validation = [example for example in examples if example.document_id in validation_ids]
    if not train or not validation:
        raise ValueError("deterministic train/validation split produced an empty partition")
    return train, validation, {
        "document_count": len(document_ids),
        "train_documents": len(document_ids) - len(validation_ids),
        "validation_documents": len(validation_ids),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "validation_rule": "sorted document IDs; final 20 percent reserved for validation",
    }


def make_model(seed: int) -> FodciModel:
    return FodciModel(ModelConfig(seed=seed))


def source(items: list[Any]) -> Callable[[], list[Any]]:
    return lambda: list(items)


def evaluate_model(model: FodciModel, validation_examples: list[Any], checkpoint_path: Path | None, dataset_dir: Path, *, seed: int) -> dict[str, Any]:
    evaluation_config = EvaluationConfig(
        batch_size=2,
        device="cpu",
        seed=seed,
        dataset_path=dataset_dir,
        dataset_split="stage1-validation",
        checkpoint_dir=DEFAULT_CHECKPOINT.parent,
        model_version=MODEL_VERSION,
    )
    evaluator = FodciEvaluator(model, evaluation_config, dataset_metadata={"dataset_version": DATASET_VERSION, "dataset_sha256": sha256_directory(dataset_dir), "loss_type": "response_only"})
    if checkpoint_path is None:
        return evaluator.evaluate(source(validation_examples)).to_dict()
    return evaluator.evaluate_checkpoint(checkpoint_path, source(validation_examples), checkpoint_id="stage1-trained").to_dict()


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["evaluation"]["baseline"]
    trained = report["evaluation"]["trained"]
    comparison = report["evaluation"]["comparison"]
    config = report["training_config"]
    checks = report["validation_gates"]
    return f"""# Phase 13.3 — Stage 1 Training & Pipeline Validation

> This is a bounded CPU training experiment. It validates the engineering pipeline and does not claim general language capability or production readiness.

## Reproducibility

| Field | Value |
|---|---|
| Model version | `{report['model_version']}` |
| Parameters | `{report['model_parameters']:,}` |
| Dataset version | `{report['dataset_version']}` |
| Dataset SHA-256 | `{report['dataset_sha256']}` |
| Seed | `{report['seed']}` |
| Device | `{config['device']}` |
| Checkpoint | `{report['checkpoint_path']}` |

## Dataset split

| Metric | Value |
|---|---:|
| Documents | {report['split']['document_count']} |
| Train documents | {report['split']['train_documents']} |
| Validation documents | {report['split']['validation_documents']} |
| Train examples | {report['split']['train_examples']} |
| Validation examples | {report['split']['validation_examples']} |

## Training configuration

| Field | Value |
|---|---:|
| Epochs | {config['epochs']} |
| Maximum steps | {config['max_steps']} |
| Batch size | {config['batch_size']} |
| Learning rate | {config['learning_rate']} |
| Weight decay | {config['weight_decay']} |
| Gradient clipping | {config['max_grad_norm']} |
| Training time (seconds) | {report['training_seconds']:.4f} |
| Global step | {report['training_result']['global_step']} |

## Before/after validation loss

| Metric | Random baseline | After Stage 1 training |
|---|---:|---:|
| Validation loss | {baseline['loss']:.9f} | {trained['loss']:.9f} |
| Response loss | {baseline['response_loss']:.9f} | {trained['response_loss']:.9f} |
| Perplexity | {baseline['perplexity']:.6f} | {trained['perplexity']:.6f} |
| Evaluation examples | {baseline['evaluation_examples']} | {trained['evaluation_examples']} |
| Response tokens | {baseline['evaluated_tokens']} | {trained['evaluated_tokens']} |

| Comparison | Value |
|---|---:|
| Loss improvement | {comparison['loss_improvement']:.9f} |
| Relative loss improvement | {comparison['loss_relative_improvement_percent']:.4f}% |
| Perplexity improvement | {comparison['perplexity_improvement']:.6f} |
| Relative perplexity improvement | {comparison['perplexity_relative_improvement_percent']:.4f}% |

## Validation gates

| Gate | Result |
|---|---|
| Dataset produced examples | `{checks['dataset_examples']}` |
| Train/validation split non-empty | `{checks['split_non_empty']}` |
| Finite training loss | `{checks['finite_training_loss']}` |
| Validation loss available | `{checks['validation_loss_available']}` |
| Checkpoint exists | `{checks['checkpoint_exists']}` |
| Checkpoint reload succeeds | `{checks['checkpoint_reload']}` |
| Parameters changed | `{checks['parameters_changed']}` |
| Pipeline validation | `{checks['all_passed']}` |

The workflow validates dataset loading, response-only masking, model forward pass, loss calculation, backpropagation, optimizer updates, checkpoint writing, checkpoint compatibility, and validation measurement. It deliberately does not run generation or modify the normal interactive agent runtime.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded Stage 1 training and validate the local Fodci pipeline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("training limits must be positive")
    random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.perf_counter()
    dataset_dir = args.dataset.resolve()
    tokenizer = FodciTokenizer()
    pipeline = make_pipeline(dataset_dir, tokenizer)
    examples = list(pipeline.iter_training_examples())
    train_examples, validation_examples, split = split_examples(examples)
    if not examples:
        raise ValueError("Stage 1 dataset produced no training examples")
    model_parameters = make_model(SEED).num_parameters
    baseline = evaluate_model(make_model(SEED), validation_examples, None, dataset_dir, seed=SEED)
    training_config = TrainingConfig(
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        device="cpu",
        seed=SEED,
        validation_interval=1,
        checkpoint_interval=0,
        output_dir=args.checkpoint.parent,
    )
    before_parameters = {name: value.detach().clone() for name, value in make_model(SEED).named_parameters()}
    training_model = make_model(SEED)
    trainer = FodciTrainer(training_model, source(train_examples), source(validation_examples), training_config, model_version=MODEL_VERSION, checkpoint_run_metadata={"phase": "13.3", "dataset_version": DATASET_VERSION})
    training_result = trainer.train()
    checkpoint = trainer.save_checkpoint(args.checkpoint)
    trained = evaluate_model(make_model(SEED), validation_examples, checkpoint, dataset_dir, seed=SEED)
    comparison = FodciEvaluator.compare_dicts(baseline, trained) if hasattr(FodciEvaluator, "compare_dicts") else {
        "loss_improvement": baseline["loss"] - trained["loss"],
        "loss_relative_improvement_percent": ((baseline["loss"] - trained["loss"]) / baseline["loss"] * 100) if baseline["loss"] else 0.0,
        "perplexity_improvement": baseline["perplexity"] - trained["perplexity"],
        "perplexity_relative_improvement_percent": ((baseline["perplexity"] - trained["perplexity"]) / baseline["perplexity"] * 100) if baseline["perplexity"] else 0.0,
    }
    reloaded_model = make_model(SEED)
    reload_evaluator = evaluate_model(reloaded_model, validation_examples, checkpoint, dataset_dir, seed=SEED)
    changed = any(not torch.equal(before_parameters[name], parameter.detach().cpu()) for name, parameter in training_model.named_parameters())
    validation_gates = {
        "dataset_examples": bool(examples),
        "split_non_empty": bool(train_examples and validation_examples),
        "finite_training_loss": bool(training_result.history and all(torch.isfinite(torch.tensor(metric.train_loss)) for metric in training_result.history)),
        "validation_loss_available": bool(training_result.history and training_result.final_metrics.validation_loss is not None),
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_reload": abs(float(reload_evaluator["loss"]) - float(trained["loss"])) < 1e-8,
        "parameters_changed": changed,
    }
    validation_gates["all_passed"] = all(validation_gates.values())
    if not validation_gates["all_passed"]:
        raise RuntimeError(f"Stage 1 pipeline validation failed: {validation_gates}")
    report = {
        "format": "fodci.stage1_training",
        "schema_version": "1.0",
        "phase": "13.3",
        "model_version": MODEL_VERSION,
        "model_parameters": model_parameters,
        "dataset_version": DATASET_VERSION,
        "dataset_path": str(dataset_dir),
        "dataset_sha256": sha256_directory(dataset_dir),
        "seed": SEED,
        "checkpoint_path": str(args.checkpoint),
        "training_config": training_config.to_dict(),
        "split": split,
        "training_seconds": training_result.elapsed_seconds,
        "training_result": {"global_step": training_result.global_step, "last_checkpoint": training_result.last_checkpoint, "history": [metric.to_dict() for metric in training_result.history]},
        "evaluation": {"baseline": baseline, "trained": trained, "reloaded": reload_evaluator, "comparison": comparison},
        "validation_gates": validation_gates,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": "13.3", "global_step": training_result.global_step, "checkpoint": str(checkpoint), "baseline_loss": baseline["loss"], "trained_loss": trained["loss"], "validation_gates": validation_gates, "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
