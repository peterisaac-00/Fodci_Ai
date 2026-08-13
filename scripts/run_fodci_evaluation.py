"""Run the Phase 2.8 baseline-vs-trained Fodci evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.checkpoint import CheckpointManager  # noqa: E402
from backend_ai.dataset import DatasetConfig, FodciDatasetPipeline  # noqa: E402
from backend_ai.evaluation import EvaluationConfig, EvaluationComparison, FodciEvaluator  # noqa: E402
from backend_ai.model import FodciModel, ModelConfig  # noqa: E402
from backend_ai.tokenizer import FodciTokenizer  # noqa: E402

MODEL_VERSION = "fodci-tiny-v1"
SEED = 2026
VALIDATION_ROOT = ROOT / "data" / "fodci_tiny_v1" / "validation"
CHECKPOINT_ROOT = ROOT / "artifacts" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_ROOT / f"{MODEL_VERSION}.pt"
JSON_REPORT = ROOT / "artifacts" / "reports" / f"{MODEL_VERSION}-evaluation.json"
MARKDOWN_REPORT = ROOT / "docs" / "experiments" / f"{MODEL_VERSION}-evaluation.md"


def make_pipeline() -> FodciDatasetPipeline:
    return FodciDatasetPipeline(
        DatasetConfig(VALIDATION_ROOT, context_length=ModelConfig().context_length),
        FodciTokenizer(),
    )


def validation_metadata() -> dict[str, Any]:
    pipeline = make_pipeline()
    loaded = pipeline.load_documents()
    tokenizer = FodciTokenizer()
    digest = hashlib.sha256()
    token_count = 0
    for document in loaded.documents:
        token_count += len(tokenizer.encode(document.text)) + 1
        digest.update(document.source_path.name.encode("utf-8"))
        digest.update(document.content_hash.encode("ascii"))
    example_count = sum(1 for _ in pipeline.iter_samples())
    return {
        "path": str(VALIDATION_ROOT.relative_to(ROOT)),
        "split": "validation",
        "document_count": len(loaded.documents),
        "token_count": token_count,
        "example_count": example_count,
        "dataset_hash": digest.hexdigest(),
        "files": [str(document.source_path.relative_to(ROOT)) for document in loaded.documents],
    }


def build_model() -> FodciModel:
    return FodciModel(ModelConfig(seed=SEED))


def render_report(report: dict[str, Any]) -> str:
    model = report["model"]
    dataset = report["dataset"]
    evaluation = report["evaluation"]
    comparison = report["comparison"]
    baseline = comparison["baseline"]
    trained = comparison["trained"]
    return f"""# Fodci Tiny v1 Evaluation

> **This is an early small-scale evaluation of a tiny model trained on a very small backend-focused corpus. It is not a capability or production-readiness claim.**

## Model

| Field | Value |
| --- | --- |
| Model version | `{report['model_version']}` |
| Parameters | {model['parameter_count']:,} |
| Vocabulary size | {model['vocab_size']:,} |
| Context length | {model['context_length']} |
| Hidden size | {model['hidden_size']} |
| Attention heads | {model['attention_heads']} |
| Transformer blocks | {model['transformer_blocks']} |
| Feed-forward size | {model['feed_forward_size']:,} |
| Tokenizer version | {report['tokenizer_version']} |
| Device | `{evaluation['device']}` |
| Seed | {report['seed']} |

## Dataset

The same existing validation split was used for the random baseline and trained checkpoint. It was loaded through `FodciDatasetPipeline`; no training examples or external data were used during evaluation.

| Field | Value |
| --- | --- |
| Path | `{dataset['path']}` |
| Split | `{dataset['split']}` |
| Documents | {dataset['document_count']} |
| Tokens including EOS | {dataset['token_count']} |
| Evaluation examples | {dataset['example_count']} |
| Dataset hash | `{dataset['dataset_hash']}` |
| Files | `{', '.join(dataset['files'])}` |

## Evaluation

| Field | Value |
| --- | --- |
| Batch size | {evaluation['batch_size']} |
| Device | `{evaluation['device']}` |
| Baseline evaluation seconds | `{baseline['evaluation_seconds']:.4f}` |
| Trained evaluation seconds | `{trained['evaluation_seconds']:.4f}` |
| Parameters changed during evaluation | `{evaluation['parameters_unchanged'] is False}` |
| Optimizer changed during evaluation | `{evaluation['optimizer_unchanged'] is False}` |

Both evaluations use `model.eval()` and `torch.no_grad()`. The evaluator does not call `backward()` or an optimizer step.

## Results

| Metric | Random baseline | Trained checkpoint | Difference / improvement |
| --- | ---: | ---: | ---: |
| Loss | `{baseline['loss']:.9f}` | `{trained['loss']:.9f}` | delta `{comparison['loss_delta']:.9f}` |
| Perplexity | `{baseline['perplexity']:.9f}` | `{trained['perplexity']:.9f}` | delta `{comparison['perplexity_delta']:.9f}` |
| Examples | {baseline['evaluation_examples']} | {trained['evaluation_examples']} | same split |
| Tokens | {baseline['evaluated_tokens']} | {trained['evaluated_tokens']} | same split |

Measured loss improvement: **{comparison['loss_improvement']:.9f}**, or **{comparison['loss_relative_improvement_percent']:.2f}%** relative to the random baseline. Measured perplexity improvement: **{comparison['perplexity_improvement']:.9f}**, or **{comparison['perplexity_relative_improvement_percent']:.2f}%**.

## Checkpoint comparison

| Field | Value |
| --- | --- |
| Checkpoint path | `{trained['checkpoint_path']}` |
| Checkpoint identifier | `{trained['checkpoint_id']}` |
| Epoch | {trained['epoch']} |
| Global step | {trained['global_step']} |
| Compatibility validation | `{report['checkpoint']['compatibility_validated']}` |
| Independently inspected metadata | `{report['checkpoint']['metadata_inspected']}` |
| Available valid checkpoints | {report['checkpoint']['available_count']} |
| Best checkpoint identifier | `{report['checkpoint']['best_identifier']}` |

## Interpretation

This comparison shows the measured language-model objective on one fixed validation split before and after the small Fodci Tiny v1 training run. It does not establish that Fodci is intelligent, understands programming, generalizes beyond the corpus, or is production ready. The dataset is intentionally small, and any future interpretation must consider possible overfitting.
"""


def main() -> None:
    torch.set_num_threads(1)
    dataset_info = validation_metadata()
    config = EvaluationConfig(
        batch_size=2,
        device="cpu",
        seed=SEED,
        model_version=MODEL_VERSION,
        tokenizer_version=1,
        dataset_path=VALIDATION_ROOT,
        checkpoint_dir=CHECKPOINT_ROOT,
    )

    baseline_model = build_model()
    baseline_evaluator = FodciEvaluator(baseline_model, config, dataset_metadata=dataset_info)
    baseline_before = {name: parameter.detach().clone() for name, parameter in baseline_model.named_parameters()}
    baseline_optimizer_before = dict(baseline_evaluator._optimizer.state)
    baseline = baseline_evaluator.evaluate(make_pipeline().iter_samples)
    baseline_parameters_unchanged = all(
        torch.equal(baseline_before[name], parameter.detach())
        for name, parameter in baseline_model.named_parameters()
    )
    baseline_optimizer_unchanged = dict(baseline_evaluator._optimizer.state) == baseline_optimizer_before

    trained_model = build_model()
    trained_evaluator = FodciEvaluator(trained_model, config, dataset_metadata=dataset_info)
    trained_before_load = {name: parameter.detach().clone() for name, parameter in trained_model.named_parameters()}
    trained = trained_evaluator.evaluate_checkpoint(
        CHECKPOINT_PATH,
        make_pipeline().iter_samples,
        checkpoint_id=MODEL_VERSION,
    )
    after_load_before_eval = {name: parameter.detach().clone() for name, parameter in trained_model.named_parameters()}
    trained_again = trained_evaluator.evaluate(
        make_pipeline().iter_samples,
        checkpoint_id=MODEL_VERSION,
        checkpoint_path=CHECKPOINT_PATH,
    )
    trained_parameters_unchanged = all(
        torch.equal(after_load_before_eval[name], parameter.detach())
        for name, parameter in trained_model.named_parameters()
    )
    comparison: EvaluationComparison = FodciEvaluator.compare(baseline, trained)

    manager = CheckpointManager(CHECKPOINT_ROOT, model_version=MODEL_VERSION, tokenizer_version=1)
    available = manager.list()
    best_info = manager.best()
    report: dict[str, Any] = {
        "phase": "2.8",
        "model_version": MODEL_VERSION,
        "tokenizer_version": 1,
        "seed": SEED,
        "model": {
            "parameter_count": trained_model.num_parameters,
            "vocab_size": trained_model.config.vocab_size,
            "context_length": trained_model.config.context_length,
            "hidden_size": trained_model.config.hidden_size,
            "attention_heads": trained_model.config.num_attention_heads,
            "transformer_blocks": trained_model.config.num_layers,
            "feed_forward_size": trained_model.config.feed_forward_size,
        },
        "dataset": dataset_info,
        "evaluation": {
            "device": str(trained_evaluator.device),
            "batch_size": config.batch_size,
            "parameters_unchanged": baseline_parameters_unchanged and trained_parameters_unchanged,
            "optimizer_unchanged": baseline_optimizer_unchanged,
            "trained_reload_evaluation_loss": trained_again.loss,
        },
        "comparison": comparison.to_dict(),
        "checkpoint": {
            "path": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "compatibility_validated": True,
            "metadata_inspected": True,
            "available_count": len(available),
            "best_identifier": best_info.path.stem if best_info else None,
            "available": [info.path.name for info in available],
        },
    }
    JSON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN_REPORT.write_text(render_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
