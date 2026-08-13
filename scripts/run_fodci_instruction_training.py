"""Run the bounded Phase 2.10 instruction-training experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.dataset import (  # noqa: E402
    DatasetConfig,
    InstructionDatasetManifestBuilder,
    InstructionDatasetPipeline,
)
from backend_ai.evaluation import EvaluationConfig, FodciEvaluator  # noqa: E402
from backend_ai.model import FodciModel, ModelConfig  # noqa: E402
from backend_ai.tokenizer import FodciTokenizer  # noqa: E402
from backend_ai.training import FodciTrainer, TrainingConfig  # noqa: E402

DATA_ROOT = Path("data/fodci_instructions")
CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-instruction-v1.pt"
REPORT_JSON = ROOT / "artifacts" / "reports" / "fodci-instruction-training.json"
REPORT_MD = ROOT / "docs" / "experiments" / "fodci-instruction-training.md"
LOSS_TYPE = "response_only"
MODEL_VERSION = "fodci-tiny-v1"
DATASET_VERSION = "fodci-instructions-v1"
SEED = 2026


def make_pipeline(split: str, tokenizer: FodciTokenizer) -> InstructionDatasetPipeline:
    return InstructionDatasetPipeline(
        DatasetConfig(
            DATA_ROOT / split,
            supported_extensions=frozenset({".txt"}),
            context_length=256,
            use_eos_document_boundaries=False,
        ),
        tokenizer,
    )


def make_model() -> FodciModel:
    return FodciModel(ModelConfig(seed=SEED))


def render_report(payload: dict) -> str:
    comparison = payload["comparison"]
    baseline = comparison["baseline"]
    trained = comparison["trained"]
    config = payload["training_config"]
    return f"""# Fodci Instruction Training — Tiny v1

> **This is a bounded from-scratch engineering experiment. It does not claim intelligence, useful general coding ability, or production readiness.**

## Objective and format

The dataset uses ordinary textual delimiters, not new tokenizer special tokens:

```text
### Instruction
{{instruction}}

### Input
{{context}}

### Response
{{response}}
```

The existing causal language-model training engine is reused. **Response-only loss masking is implemented**: instruction and input tokens provide conditioning context, while response target tokens and the response EOS boundary contribute to cross-entropy. The reported loss and perplexity below are therefore response-only metrics.

## Reproducibility

| Field | Value |
| --- | --- |
| Model version | `{payload['model_version']}` |
| Model parameters | {payload['model_parameters']:,} |
| Dataset version | `{payload['dataset_version']}` |
| Dataset SHA-256 | `{payload['dataset_sha256']}` |
| Tokenizer version | {payload['tokenizer_version']} |
| Vocabulary size | {payload['vocabulary_size']:,} |
| Context length | {payload['context_length']} |
| Seed | {payload['seed']} |
| Device | `{config['device']}` |

## Training configuration

| Field | Value |
| --- | ---: |
| Batch size | {config['batch_size']} |
| Learning rate | {config['learning_rate']} |
| Weight decay | {config['weight_decay']} |
| Gradient clipping | {config['max_grad_norm']} |
| Epochs | {config['epochs']} |
| Optimization steps | {config['max_steps']} |
| Training time (seconds) | {payload['training_seconds']:.4f} |
| Checkpoint | `{payload['checkpoint_path']}` |

## Dataset

| Split | Instructions | Serialized tokens | Response tokens | Training examples |
| --- | ---: | ---: | ---: | ---: |
| Train | {payload['dataset_stats']['train']['instruction_count']} | {payload['dataset_stats']['train']['total_tokens']:,} | {payload['dataset_stats']['train']['response_tokens']:,} | {payload['dataset_stats']['train']['training_example_count']} |
| Validation | {payload['dataset_stats']['validation']['instruction_count']} | {payload['dataset_stats']['validation']['total_tokens']:,} | {payload['dataset_stats']['validation']['response_tokens']:,} | {payload['dataset_stats']['validation']['training_example_count']} |

## Before/after evaluation

Both states were evaluated on the same validation instruction source and the same response-only mask.

| Metric | Random Fodci Tiny v1 | After instruction training |
| --- | ---: | ---: |
| Validation loss | {baseline['loss']:.9f} | {trained['loss']:.9f} |
| Response loss | {baseline['response_loss']:.9f} | {trained['response_loss']:.9f} |
| Perplexity | {baseline['perplexity']:.6f} | {trained['perplexity']:.6f} |
| Evaluated examples | {baseline['evaluation_examples']} | {trained['evaluation_examples']} |
| Evaluated response tokens | {baseline['evaluated_tokens']} | {trained['evaluated_tokens']} |
| Checkpoint identity | `{baseline['checkpoint_id']}` | `{trained['checkpoint_id']}` |
| Global step | `{baseline['global_step']}` | `{trained['global_step']}` |

| Comparison | Value |
| --- | ---: |
| Loss improvement | {comparison['loss_improvement']:.9f} |
| Relative loss improvement | {comparison['loss_relative_improvement_percent']:.4f}% |
| Perplexity improvement | {comparison['perplexity_improvement']:.6f} |
| Relative perplexity improvement | {comparison['perplexity_relative_improvement_percent']:.4f}% |

The result validates the data path, masking path, optimizer update, checkpoint compatibility, and objective measurement on a tiny local dataset. It is not evidence that the model can reliably follow arbitrary instructions or write production backend code.

No pretrained model, tokenizer, or weights were used. No external data was downloaded. No generation, inference, CLI integration, Agent behavior, or Phase 3 functionality is part of Phase 2.10.
"""


def main() -> None:
    started = time.perf_counter()
    manifest = InstructionDatasetManifestBuilder(DATA_ROOT, strict=True).build()
    tokenizer = FodciTokenizer()
    train_pipeline = make_pipeline("train", tokenizer)
    validation_pipeline = make_pipeline("validation", tokenizer)
    metadata = {
        "dataset_hash": manifest.dataset_sha256,
        "document_count": manifest.validation.instruction_count,
        "loss_type": LOSS_TYPE,
    }
    evaluation_config = EvaluationConfig(
        batch_size=2,
        device="cpu",
        seed=SEED,
        dataset_path=DATA_ROOT / "validation",
        dataset_split="validation",
        checkpoint_dir=CHECKPOINT.parent,
        model_version=MODEL_VERSION,
    )

    baseline_evaluator = FodciEvaluator(make_model(), evaluation_config, dataset_metadata=metadata)
    baseline = baseline_evaluator.evaluate(validation_pipeline.iter_training_examples)

    trainer = FodciTrainer(
        make_model(),
        train_pipeline.iter_training_examples,
        validation_pipeline.iter_training_examples,
        TrainingConfig(
            epochs=1,
            max_steps=4,
            batch_size=2,
            learning_rate=3e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            device="cpu",
            seed=SEED,
            validation_interval=1,
            checkpoint_interval=0,
            output_dir=CHECKPOINT.parent,
        ),
        model_version=MODEL_VERSION,
    )
    training_result = trainer.train()
    checkpoint = trainer.save_checkpoint(CHECKPOINT)

    trained_evaluator = FodciEvaluator(make_model(), evaluation_config, dataset_metadata=metadata)
    trained = trained_evaluator.evaluate_checkpoint(
        checkpoint,
        validation_pipeline.iter_training_examples,
        checkpoint_id="instruction-trained",
    )
    comparison = FodciEvaluator.compare(baseline, trained)
    comparison_payload = comparison.to_dict()
    comparison_payload["trained"]["checkpoint_path"] = str(CHECKPOINT.relative_to(ROOT))
    payload = {
        "phase": "2.10",
        "model_version": MODEL_VERSION,
        "model_parameters": make_model().num_parameters,
        "dataset_version": DATASET_VERSION,
        "dataset_sha256": manifest.dataset_sha256,
        "tokenizer_version": manifest.tokenizer_version,
        "vocabulary_size": manifest.vocabulary_size,
        "context_length": manifest.context_length,
        "seed": SEED,
        "loss_type": LOSS_TYPE,
        "training_config": trainer.config.to_dict(),
        "training_seconds": training_result.elapsed_seconds,
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)),
        "dataset_stats": {
            "train": manifest.train.to_dict(),
            "validation": manifest.validation.to_dict(),
        },
        "comparison": comparison_payload,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"elapsed_total_seconds={time.perf_counter() - started:.4f}")


if __name__ == "__main__":
    main()
