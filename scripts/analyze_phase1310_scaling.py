from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from backend_ai.checkpoint import CheckpointManager
from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.instructions import InstructionDatasetPipeline
from backend_ai.evaluation import EvaluationConfig, FodciEvaluator
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import FodciTokenizer
from backend_ai.training import FodciTrainer, TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-testing-qa-v1.pt"
DEFAULT_DATASET = ROOT / "training_data" / "testing_qa"
DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase1310_scaling_analysis.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase1310_scaling_analysis.md"
SEED = 2026
BASE_VERSION = "fodci-testing-qa-v1"
SCALED_VERSION = "fodci-scaling-48m-experimental-v1"
DATASET_VERSION = "testing-qa-specialist-v1"

DEFAULT_CONFIG = ModelConfig(seed=SEED)
SCALED_CONFIG = ModelConfig(
    vocab_size=10_000,
    context_length=256,
    hidden_size=608,
    num_layers=8,
    num_attention_heads=8,
    feed_forward_size=2_432,
    dropout=0.0,
    seed=SEED,
)


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def source(items: list[Any]) -> Callable[[], list[Any]]:
    return lambda: list(items)


def load_examples(dataset_path: Path) -> tuple[list[Any], list[Any], dict[str, int]]:
    tokenizer = FodciTokenizer()
    config = DatasetConfig(
        dataset_path / "train",
        supported_extensions=frozenset({".txt"}),
        context_length=64,
        use_eos_document_boundaries=False,
    )
    examples = list(InstructionDatasetPipeline(config, tokenizer).iter_training_examples())
    document_ids = sorted({example.document_id for example in examples})
    validation_count = max(1, round(len(document_ids) * 0.2))
    validation_ids = set(document_ids[-validation_count:])
    train = [example for example in examples if example.document_id not in validation_ids]
    validation = [example for example in examples if example.document_id in validation_ids]
    if not train or not validation:
        raise ValueError("scaling analysis dataset split is empty")
    return train, validation, {
        "documents": len(document_ids),
        "train_documents": len(document_ids) - len(validation_ids),
        "validation_documents": len(validation_ids),
        "train_examples": len(train),
        "validation_examples": len(validation),
    }


def evaluate(model: FodciModel, examples: list[Any], dataset_path: Path, model_version: str) -> dict[str, Any]:
    config = EvaluationConfig(
        batch_size=2,
        device="cpu",
        seed=SEED,
        dataset_path=dataset_path,
        dataset_split="testing-qa-validation",
        checkpoint_dir=DEFAULT_CHECKPOINT.parent,
        model_version=model_version,
    )
    evaluator = FodciEvaluator(
        model,
        config,
        dataset_metadata={
            "dataset_version": DATASET_VERSION,
            "dataset_sha256": sha256_directory(dataset_path.parent),
            "loss_type": "response_only",
        },
    )
    return evaluator.evaluate(source(examples)).to_dict()


def rss_megabytes() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0


def measure_forward_backward(model: FodciModel, examples: list[Any]) -> dict[str, Any]:
    example = examples[0]
    input_ids = torch.tensor([example.input_ids], dtype=torch.long)
    target_ids = torch.tensor([example.target_ids], dtype=torch.long)
    started = time.perf_counter()
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
    loss.backward()
    elapsed = time.perf_counter() - started
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    return {
        "sequence_length": len(example.input_ids),
        "loss": float(loss.detach().item()),
        "elapsed_seconds": elapsed,
        "gradients_finite": gradients_finite,
        "rss_megabytes_after_backward": rss_megabytes(),
    }


def run_short_training(model: FodciModel, train_examples: list[Any], validation_examples: list[Any], *, model_version: str) -> dict[str, Any]:
    config = TrainingConfig(
        epochs=1,
        max_steps=2,
        batch_size=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        device="cpu",
        seed=SEED,
        validation_interval=1,
        checkpoint_interval=0,
        output_dir=DEFAULT_CHECKPOINT.parent,
    )
    trainer = FodciTrainer(
        model,
        source(train_examples),
        source(validation_examples),
        config,
        model_version=model_version,
        checkpoint_run_metadata={"phase": "13.10", "dataset_version": DATASET_VERSION},
    )
    result = trainer.train()
    losses = [metric.train_loss for metric in result.history]
    return {
        "global_step": result.global_step,
        "elapsed_seconds": result.elapsed_seconds,
        "finite_losses": bool(losses) and all(torch.isfinite(torch.tensor(loss)) for loss in losses),
        "first_train_loss": float(losses[0]) if losses else None,
        "last_train_loss": float(losses[-1]) if losses else None,
    }


def render_markdown(report: dict[str, Any]) -> str:
    default = report["models"]["default_11m"]
    scaled = report["models"]["scaled_candidate"]
    return f"""# Phase 13.10 — Model Scaling Analysis

> This is a bounded CPU experiment. The approximately 48M-parameter candidate is experimental only; the default Fodci runtime and all existing checkpoints remain on the 11.4M-parameter architecture.

## Objective

The experiment measures whether a larger configuration is technically feasible on CPU and whether the available evidence justifies replacing the default model. It does not claim that parameter count alone produces better language, code, security, or testing behavior.

## Configurations

| Field | Default 11.4M | Scaled candidate |
|---|---:|---:|
| Model version | `{default['model_version']}` | `{scaled['model_version']}` |
| Parameters | {default['parameter_count']:,} | {scaled['parameter_count']:,} |
| Hidden size | {default['config']['hidden_size']} | {scaled['config']['hidden_size']} |
| Transformer layers | {default['config']['num_layers']} | {scaled['config']['num_layers']} |
| Attention heads | {default['config']['num_attention_heads']} | {scaled['config']['num_attention_heads']} |
| Feed-forward size | {default['config']['feed_forward_size']} | {scaled['config']['feed_forward_size']} |
| Context length | {default['config']['context_length']} | {scaled['config']['context_length']} |
| Parameter multiplier | 1.00× | {report['comparison']['parameter_multiplier']:.2f}× |

## Resource and execution measurements

| Metric | Default 11.4M | Scaled candidate |
|---|---:|---:|
| Forward/backward loss | {default['forward_backward']['loss']:.6f} | {scaled['forward_backward']['loss']:.6f} |
| Forward/backward seconds | {default['forward_backward']['elapsed_seconds']:.4f} | {scaled['forward_backward']['elapsed_seconds']:.4f} |
| RSS after backward (MB) | {default['forward_backward']['rss_megabytes_after_backward']:.2f} | {scaled['forward_backward']['rss_megabytes_after_backward']:.2f} |
| Short training steps | {default['short_training']['global_step']} | {scaled['short_training']['global_step']} |
| Short training seconds | {default['short_training']['elapsed_seconds']:.4f} | {scaled['short_training']['elapsed_seconds']:.4f} |
| Gradients/loss finite | `{default['forward_backward']['gradients_finite']}` / `{default['short_training']['finite_losses']}` | `{scaled['forward_backward']['gradients_finite']}` / `{scaled['short_training']['finite_losses']}` |

## Validation loss evidence

| Metric | Default 11.4M checkpoint | Scaled candidate after short run |
|---|---:|---:|
| Validation loss | {default['validation']['loss']:.9f} | {scaled['validation']['loss']:.9f} |
| Evaluation examples | {default['validation']['evaluation_examples']} | {scaled['validation']['evaluation_examples']} |
| Dataset version | `{report['dataset_version']}` | `{report['dataset_version']}` |

The scaled candidate starts from random initialization because the 11.4M checkpoint is not shape-compatible with the larger configuration. Therefore the loss comparison is diagnostic, not a fair capability comparison. A valid quality comparison requires a larger-model training run with the same data, protocol, compute budget, and held-out tasks.

## Decision

> **Decision: retain the 11.4M model as the default.**

The scaled candidate is technically runnable, but the current evidence does not demonstrate a semantic or benchmark advantage. The candidate checkpoint is intentionally not saved or wired into the normal runtime. A future scaling decision should require equivalent specialist training, repeatable benchmark gains, acceptable CPU/memory budgets, and execution-aware task improvements.

## Reproducibility

| Field | Value |
|---|---|
| Base checkpoint | `{report['base_checkpoint']}` |
| Base checkpoint SHA-256 | `{report['base_checkpoint_sha256']}` |
| Dataset SHA-256 | `{report['dataset_sha256']}` |
| Seed | `{report['seed']}` |
| CPU threads | `{report['torch_num_threads']}` |
| Scaled checkpoint saved | `{report['scaled_checkpoint_saved']}` |
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded Phase 13.10 model scaling analysis.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    started = time.perf_counter()
    checkpoint = args.checkpoint.resolve()
    dataset_path = args.dataset.resolve()
    manager = CheckpointManager(checkpoint.parent, model_version=BASE_VERSION)
    checkpoint_info = manager.inspect(checkpoint)
    if checkpoint_info.metadata.model_version != BASE_VERSION:
        raise ValueError(f"expected {BASE_VERSION} checkpoint, got {checkpoint_info.metadata.model_version}")
    train_examples, validation_examples, split = load_examples(dataset_path)

    default_model = FodciModel(DEFAULT_CONFIG)
    manager.load_model(checkpoint, default_model, device=torch.device("cpu"))
    default_validation = evaluate(default_model, validation_examples, dataset_path / "validation", BASE_VERSION)
    default_forward_backward = measure_forward_backward(default_model, validation_examples)
    default_short_training = run_short_training(default_model, train_examples, validation_examples, model_version=BASE_VERSION)

    scaled_model = FodciModel(SCALED_CONFIG)
    scaled_validation_before = evaluate(scaled_model, validation_examples, dataset_path / "validation", SCALED_VERSION)
    scaled_forward_backward = measure_forward_backward(scaled_model, validation_examples)
    scaled_short_training = run_short_training(scaled_model, train_examples, validation_examples, model_version=SCALED_VERSION)
    scaled_validation = evaluate(scaled_model, validation_examples, dataset_path / "validation", SCALED_VERSION)

    default_parameters = default_model.num_parameters
    scaled_parameters = scaled_model.num_parameters
    report = {
        "format": "fodci.phase1310_scaling_analysis",
        "schema_version": "1.0",
        "phase": "13.10",
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_directory(dataset_path),
        "dataset_version": DATASET_VERSION,
        "seed": SEED,
        "torch_num_threads": torch.get_num_threads(),
        "split": split,
        "scaled_checkpoint_saved": False,
        "models": {
            "default_11m": {
                "model_version": BASE_VERSION,
                "parameter_count": default_parameters,
                "config": DEFAULT_CONFIG.__dict__ if hasattr(DEFAULT_CONFIG, "__dict__") else {
                    "vocab_size": DEFAULT_CONFIG.vocab_size,
                    "context_length": DEFAULT_CONFIG.context_length,
                    "hidden_size": DEFAULT_CONFIG.hidden_size,
                    "num_layers": DEFAULT_CONFIG.num_layers,
                    "num_attention_heads": DEFAULT_CONFIG.num_attention_heads,
                    "feed_forward_size": DEFAULT_CONFIG.feed_forward_size,
                },
                "validation": default_validation,
                "forward_backward": default_forward_backward,
                "short_training": default_short_training,
            },
            "scaled_candidate": {
                "model_version": SCALED_VERSION,
                "parameter_count": scaled_parameters,
                "config": {
                    "vocab_size": SCALED_CONFIG.vocab_size,
                    "context_length": SCALED_CONFIG.context_length,
                    "hidden_size": SCALED_CONFIG.hidden_size,
                    "num_layers": SCALED_CONFIG.num_layers,
                    "num_attention_heads": SCALED_CONFIG.num_attention_heads,
                    "feed_forward_size": SCALED_CONFIG.feed_forward_size,
                },
                "validation_before_training": scaled_validation_before,
                "validation": scaled_validation,
                "forward_backward": scaled_forward_backward,
                "short_training": scaled_short_training,
            },
        },
        "comparison": {
            "parameter_multiplier": scaled_parameters / default_parameters,
            "checkpoint_shape_compatible": False,
            "default_runtime_changed": False,
            "semantic_advantage_proven": False,
            "decision": "retain_11m_default",
        },
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": "13.10", "default_parameters": default_parameters, "scaled_parameters": scaled_parameters, "parameter_multiplier": report["comparison"]["parameter_multiplier"], "decision": report["comparison"]["decision"], "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
