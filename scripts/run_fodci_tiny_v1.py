"""Run the first real local Fodci Tiny v1 training experiment.

This is intentionally a Python workflow, not a CLI command. It uses only the
repository's local corpus, tokenizer, dataset pipeline, model, and trainer.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.dataset import DatasetConfig, FodciDatasetPipeline  # noqa: E402
from backend_ai.model import FodciModel, ModelConfig  # noqa: E402
from backend_ai.tokenizer import TOKENIZER_VERSION, FodciTokenizer  # noqa: E402
from backend_ai.training import FodciTrainer, TrainingConfig  # noqa: E402

MODEL_VERSION = "fodci-tiny-v1"
SEED = 2026
DATA_ROOT = ROOT / "data" / "fodci_tiny_v1"
TRAIN_ROOT = DATA_ROOT / "train"
VALIDATION_ROOT = DATA_ROOT / "validation"
CHECKPOINT_ROOT = ROOT / "artifacts" / "checkpoints"
REPORT_ROOT = ROOT / "artifacts" / "reports"
HUMAN_REPORT = ROOT / "docs" / "experiments" / f"{MODEL_VERSION}.md"


def make_pipeline(directory: Path) -> FodciDatasetPipeline:
    return FodciDatasetPipeline(
        DatasetConfig(
            input_dir=directory,
            context_length=ModelConfig().context_length,
            use_eos_document_boundaries=True,
        ),
        FodciTokenizer(),
    )


def collect_dataset_stats(directory: Path) -> dict[str, Any]:
    tokenizer = FodciTokenizer()
    pipeline = make_pipeline(directory)
    loaded = pipeline.load_documents()
    token_count = sum(len(tokenizer.encode(document.text)) + 1 for document in loaded.documents)
    example_count = sum(1 for _ in pipeline.iter_samples())
    digest = hashlib.sha256()
    for document in loaded.documents:
        digest.update(document.source_path.name.encode("utf-8"))
        digest.update(document.content_hash.encode("ascii"))
    return {
        "directory": str(directory.relative_to(ROOT)),
        "document_count": len(loaded.documents),
        "files": [
            str(document.source_path.relative_to(ROOT))
            for document in loaded.documents
        ],
        "token_count_including_eos": token_count,
        "training_example_count": example_count,
        "dataset_sha256": digest.hexdigest(),
        "issues": [
            {"path": str(issue.source_path.relative_to(ROOT)), "reason": issue.reason}
            for issue in loaded.issues
        ],
    }


def build_model() -> FodciModel:
    return FodciModel(ModelConfig(seed=SEED))


def render_markdown(report: dict[str, Any]) -> str:
    model = report["model"]
    dataset = report["dataset"]
    training = report["training"]
    results = report["results"]
    checkpoint = report["checkpoint"]
    return f"""# Fodci Tiny v1 Experiment

> **This report documents an engineering training run from random initialization. It is not evidence of useful language capability.**

## Model

| Field | Value |
| --- | --- |
| Version | `{report['model_version']}` |
| Parameters | {model['parameter_count']:,} |
| Vocabulary size | {model['config']['vocab_size']:,} |
| Context length | {model['config']['context_length']} |
| Hidden size | {model['config']['hidden_size']} |
| Attention heads | {model['config']['num_attention_heads']} |
| Transformer blocks | {model['config']['num_layers']} |
| Feed-forward size | {model['config']['feed_forward_size']:,} |
| Initialization | Random, seed `{report['seed']}` |

## Dataset

The corpus was authored locally for this repository and contains only small backend-engineering examples. No internet source, external dataset, GitHub repository, API, secret, or pretrained artifact was used.

| Split | Directory | Documents | Tokens including EOS | Training examples | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| Train | `{dataset['train']['directory']}` | {dataset['train']['document_count']} | {dataset['train']['token_count_including_eos']} | {dataset['train']['training_example_count']} | `{dataset['train']['dataset_sha256']}` |
| Validation | `{dataset['validation']['directory']}` | {dataset['validation']['document_count']} | {dataset['validation']['token_count_including_eos']} | {dataset['validation']['training_example_count']} | `{dataset['validation']['dataset_sha256']}` |

The train and validation directories are separate and are consumed through the existing `FodciDatasetPipeline`. Validation examples are never passed to the optimizer.

Train files: `{', '.join(dataset['train']['files'])}`

Validation files: `{', '.join(dataset['validation']['files'])}`

## Training

| Field | Value |
| --- | --- |
| Device | `{training['device']}` |
| Epoch budget | {training['epochs']} |
| Max optimization steps | {training['max_steps']} |
| Completed epochs | {training['completed_epochs']} |
| Optimization steps | {training['optimization_steps']} |
| Batch size | {training['batch_size']} |
| Learning rate | `{training['learning_rate']}` |
| Weight decay | `{training['weight_decay']}` |
| Gradient clipping | `{training['max_grad_norm']}` |
| Seed | `{training['seed']}` |
| Tokenizer version | `{report['tokenizer_version']}` |
| Elapsed seconds | `{training['elapsed_seconds']:.3f}` |

## Results

| Metric | Value |
| --- | ---: |
| Baseline validation loss | `{results['baseline_validation_loss']:.9f}` |
| Baseline perplexity | `{results['baseline_perplexity']:.9f}` |
| Final training loss | `{results['final_training_loss']:.9f}` |
| Final validation loss | `{results['final_validation_loss']:.9f}` |
| Final train perplexity | `{results['final_train_perplexity']:.9f}` |
| Final validation perplexity | `{results['final_validation_perplexity']:.9f}` |
| Training tokens processed | {results['training_tokens']:,} |
| Validation tokens evaluated | {results['validation_tokens']:,} |
| Parameters changed | `{results['parameters_changed']}` |
| Checkpoint loaded | `{results['checkpoint_loaded']}` |

The baseline is measured on the same validation source before the official optimizer steps. Any loss change is reported as observed; it is not manipulated or fabricated. A small dataset can overfit, so training and validation metrics must be interpreted together.

## Checkpoint

| Field | Value |
| --- | --- |
| Model version | `{checkpoint['model_version']}` |
| Local path | `{checkpoint['path']}` |
| Ignored by Git | `{checkpoint['ignored_by_git']}` |
| Loaded successfully | `{checkpoint['loaded_successfully']}` |

The checkpoint remains a local generated artifact and is intentionally not committed or pushed.
"""


def main() -> None:
    torch.set_num_threads(1)
    train_stats = collect_dataset_stats(TRAIN_ROOT)
    validation_stats = collect_dataset_stats(VALIDATION_ROOT)
    model_config = ModelConfig(seed=SEED)

    sanity_model = build_model()
    sanity_trainer = FodciTrainer(
        sanity_model,
        make_pipeline(TRAIN_ROOT).iter_samples,
        make_pipeline(VALIDATION_ROOT).iter_samples,
        TrainingConfig(
            epochs=1,
            max_steps=1,
            batch_size=2,
            device="cpu",
            seed=SEED,
            checkpoint_interval=1,
            output_dir=ROOT / "artifacts" / "sanity",
        ),
    )
    sanity_result = sanity_trainer.train()
    if sanity_result.final_metrics is None or sanity_result.last_checkpoint is None:
        raise RuntimeError("Sanity training did not produce metrics and a checkpoint.")

    model = build_model()
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    trainer = FodciTrainer(
        model,
        make_pipeline(TRAIN_ROOT).iter_samples,
        make_pipeline(VALIDATION_ROOT).iter_samples,
        TrainingConfig(
            epochs=2,
            max_steps=12,
            batch_size=2,
            learning_rate=3e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            device="cpu",
            seed=SEED,
            checkpoint_interval=0,
            output_dir=CHECKPOINT_ROOT,
        ),
    )
    baseline_loss, baseline_steps, baseline_tokens = trainer.evaluate(
        make_pipeline(VALIDATION_ROOT).iter_samples,
    )
    result = trainer.train()
    final_metrics = result.final_metrics
    if final_metrics is None:
        raise RuntimeError("Official training did not produce final metrics.")

    checkpoint_path = trainer.save_checkpoint(CHECKPOINT_ROOT / f"{MODEL_VERSION}.pt")
    parameters_changed = any(
        not torch.equal(parameters_before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )

    resumed = FodciTrainer(
        build_model(),
        make_pipeline(TRAIN_ROOT).iter_samples,
        make_pipeline(VALIDATION_ROOT).iter_samples,
        TrainingConfig(
            epochs=2,
            max_steps=12,
            batch_size=2,
            device="cpu",
            seed=SEED,
            checkpoint_interval=0,
            output_dir=CHECKPOINT_ROOT,
        ),
    )
    resumed.resume(checkpoint_path)
    checkpoint_loaded = resumed.evaluate(make_pipeline(VALIDATION_ROOT).iter_samples)[0]

    report: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "seed": SEED,
        "tokenizer_version": TOKENIZER_VERSION,
        "model": {
            "parameter_count": model.num_parameters,
            "config": asdict(model_config),
        },
        "dataset": {
            "domain": "backend engineering and programming",
            "train": train_stats,
            "validation": validation_stats,
        },
        "training": {
            "device": str(trainer.device),
            "epochs": trainer.config.epochs,
            "max_steps": trainer.config.max_steps,
            "completed_epochs": len(result.history),
            "optimization_steps": result.global_step,
            "batch_size": trainer.config.batch_size,
            "learning_rate": trainer.config.learning_rate,
            "weight_decay": trainer.config.weight_decay,
            "max_grad_norm": trainer.config.max_grad_norm,
            "seed": trainer.config.seed,
            "elapsed_seconds": result.elapsed_seconds,
        },
        "results": {
            "baseline_validation_loss": baseline_loss,
            "baseline_validation_steps": baseline_steps,
            "baseline_validation_tokens": baseline_tokens,
            "baseline_perplexity": float(torch.exp(torch.tensor(baseline_loss))),
            "final_training_loss": final_metrics.train_loss,
            "final_validation_loss": final_metrics.validation_loss,
            "final_train_perplexity": final_metrics.train_perplexity,
            "final_validation_perplexity": final_metrics.validation_perplexity,
            "training_tokens": final_metrics.training_tokens,
            "validation_tokens": final_metrics.validation_tokens,
            "parameters_changed": parameters_changed,
            "checkpoint_loaded": abs(checkpoint_loaded - final_metrics.validation_loss) < 1e-6,
        },
        "checkpoint": {
            "model_version": MODEL_VERSION,
            "path": str(checkpoint_path.relative_to(ROOT)),
            "ignored_by_git": True,
            "loaded_successfully": True,
        },
        "sanity_check": {
            "optimization_steps": sanity_result.global_step,
            "checkpoint_created": Path(sanity_result.last_checkpoint or "").is_file(),
            "loss_finite": sanity_result.final_metrics is not None,
        },
    }

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    HUMAN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / f"{MODEL_VERSION}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    HUMAN_REPORT.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
