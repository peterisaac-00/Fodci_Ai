from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from backend_ai.checkpoint import CheckpointManager
from backend_ai.dataset.samples import TrainingExample
from backend_ai.inference import InferenceConfig, InferenceEngine
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import EOS_ID, FodciTokenizer
from backend_ai.training import FodciTrainer, TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "training_data" / "english_foundation"
TOKENIZER_PATH = ROOT / "tokenizers" / "fodci-english-v4.json"
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"
REPORT_PATH = ROOT / "artifacts" / "evaluation" / "phase1313_english_foundation.json"
MARKDOWN_PATH = ROOT / "docs" / "experiments" / "phase1313_english_foundation.md"
SEED = 2026
DATASET_VERSION = "english-foundation-v1"
MODEL_11M_VERSION = "fodci-english-11m-v1"
MODEL_25M_VERSION = "fodci-english-25m-v1"
MODEL_11M_CHECKPOINT = CHECKPOINT_DIR / "fodci-english-11m-v1.pt"
MODEL_25M_CHECKPOINT = CHECKPOINT_DIR / "fodci-english-25m-v1.pt"
DEFAULT_MAX_STEPS = 128
CONTEXT_LENGTH = 256

CONFIG_11M = ModelConfig(seed=SEED, context_length=CONTEXT_LENGTH)
CONFIG_25M = ModelConfig(
    vocab_size=10_000,
    context_length=CONTEXT_LENGTH,
    hidden_size=448,
    num_layers=7,
    num_attention_heads=7,
    feed_forward_size=1_792,
    dropout=0.0,
    seed=SEED,
)


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def chunk_text(tokenizer: FodciTokenizer, text: str, document_id: str, width: int = CONTEXT_LENGTH) -> list[TrainingExample]:
    token_ids = tokenizer.encode(text) + [EOS_ID]
    examples: list[TrainingExample] = []
    for start in range(0, len(token_ids) - 1, width):
        window = token_ids[start : start + width + 1]
        if len(window) != width + 1:
            continue
        examples.append(TrainingExample(tuple(window[:-1]), tuple(window[1:]), document_id, (True,) * width))
    return examples


def padded_instruction(tokenizer: FodciTokenizer, text: str, document_id: str, width: int = CONTEXT_LENGTH) -> TrainingExample:
    token_ids = tokenizer.encode(text) + [EOS_ID]
    if len(token_ids) < 2:
        raise ValueError(f"instruction record is too short: {document_id}")
    actual_targets = min(width, len(token_ids) - 1)
    input_ids = token_ids[:-1][:width]
    target_ids = token_ids[1:][:width]
    input_ids = input_ids + [0] * (width - len(input_ids))
    target_ids = target_ids + [0] * (width - len(target_ids))
    loss_mask = (True,) * actual_targets + (False,) * (width - actual_targets)
    return TrainingExample(tuple(input_ids), tuple(target_ids), document_id, loss_mask)


def load_data(tokenizer: FodciTokenizer, *, limit_examples: int) -> tuple[list[TrainingExample], list[TrainingExample], dict[str, Any]]:
    train_examples: list[TrainingExample] = []
    validation_examples: list[TrainingExample] = []
    train_chars = 0
    validation_chars = 0
    train_documents = sorted((DATA_ROOT / "train").glob("*.txt"))
    validation_documents = sorted((DATA_ROOT / "validation").glob("*.txt"))
    for path in train_documents:
        text = path.read_text(encoding="utf-8")
        train_chars += len(text)
        train_examples.extend(chunk_text(tokenizer, text, f"raw:{path.name}"))
    for path in validation_documents:
        text = path.read_text(encoding="utf-8")
        validation_chars += len(text)
        validation_examples.extend(chunk_text(tokenizer, text, f"raw:{path.name}"))

    instruction_dir = DATA_ROOT / "instructions" / "train"
    instruction_examples_list: list[TrainingExample] = []
    for path in sorted(instruction_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        instruction_examples_list.append(padded_instruction(tokenizer, text, f"instruction:{path.name}"))

    available_train = len(train_examples) + len(instruction_examples_list)
    if available_train < 256 or len(validation_examples) < 16:
        raise ValueError(f"English foundation corpus produced too few fixed-length examples: train={available_train}, validation={len(validation_examples)}")
    effective_limit = min(limit_examples, available_train)
    raw_limit = max(0, effective_limit - len(instruction_examples_list))
    train_examples = train_examples[:raw_limit] + instruction_examples_list
    instruction_examples = len(instruction_examples_list)
    validation_examples = validation_examples[: max(16, min(256, len(validation_examples)))]
    stats = {
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "train_characters": train_chars,
        "validation_characters": validation_chars,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "instruction_chunks_in_train": instruction_examples,
        "context_length": CONTEXT_LENGTH,
        "tokenizer_merges": len(tokenizer.merges),
    }
    return train_examples, validation_examples, stats


def train_one(
    *,
    config: ModelConfig,
    model_version: str,
    checkpoint_path: Path,
    train_examples: list[TrainingExample],
    validation_examples: list[TrainingExample],
    max_steps: int,
) -> dict[str, Any]:
    model = FodciModel(config)
    initial_validation_model = FodciModel(config)
    initial_validation_model.load_state_dict(model.state_dict())
    trainer = FodciTrainer(
        model,
        train_examples,
        validation_examples,
        TrainingConfig(
            epochs=1,
            max_steps=max_steps,
            batch_size=2,
            learning_rate=3e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            device="cpu",
            seed=SEED,
            validation_interval=1,
            checkpoint_interval=0,
            output_dir=CHECKPOINT_DIR,
        ),
        model_version=model_version,
        checkpoint_run_metadata={"phase": "13.13", "dataset_version": DATASET_VERSION, "language": "en"},
    )
    baseline_loss, baseline_steps, baseline_tokens = trainer.evaluate(validation_examples)
    started = time.perf_counter()
    result = trainer.train()
    trained_loss, validation_steps, validation_tokens = trainer.evaluate(validation_examples)
    saved = trainer.save_checkpoint(checkpoint_path, run_metadata={"phase": "13.13", "dataset_version": DATASET_VERSION, "language": "en"})

    reloaded_model = FodciModel(config)
    loaded = CheckpointManager(checkpoint_path.parent, model_version=model_version).load_model(saved, reloaded_model, device=torch.device("cpu"))
    reloaded_trainer = FodciTrainer(
        reloaded_model,
        train_examples,
        validation_examples,
        TrainingConfig(epochs=1, max_steps=1, batch_size=2, device="cpu", seed=SEED, checkpoint_interval=0, output_dir=CHECKPOINT_DIR),
        model_version=model_version,
    )
    reloaded_loss, _, _ = reloaded_trainer.evaluate(validation_examples)
    finite_loss = all(torch.isfinite(torch.tensor(value)) for value in (baseline_loss, trained_loss, reloaded_loss))
    parameters_changed = any(not torch.equal(before, after) for before, after in zip(initial_validation_model.parameters(), model.parameters(), strict=True))
    report = {
        "model_version": model_version,
        "parameter_count": model.num_parameters,
        "config": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_attention_heads": config.num_attention_heads,
            "feed_forward_size": config.feed_forward_size,
        },
        "checkpoint_path": str(saved),
        "checkpoint_exists": saved.is_file(),
        "checkpoint_reload": loaded.metadata.model_version == model_version,
        "max_steps": max_steps,
        "global_step": result.global_step,
        "training_seconds": time.perf_counter() - started,
        "baseline_validation_loss": baseline_loss,
        "trained_validation_loss": trained_loss,
        "reloaded_validation_loss": reloaded_loss,
        "validation_improvement": baseline_loss - trained_loss,
        "finite_loss": finite_loss,
        "parameters_changed": parameters_changed,
        "non_empty_split": bool(train_examples and validation_examples),
        "baseline_validation_steps": baseline_steps,
        "validation_steps": validation_steps,
        "validation_tokens": validation_tokens,
        "all_gates_passed": all((saved.is_file(), loaded.metadata.model_version == model_version, finite_loss, parameters_changed, bool(train_examples and validation_examples), trained_loss < baseline_loss)),
    }
    if not report["all_gates_passed"]:
        raise RuntimeError(f"English foundation gates failed for {model_version}: {report}")
    return report


def response_probe(checkpoint: Path, model_version: str, config: ModelConfig, tokenizer: FodciTokenizer) -> dict[str, Any]:
    engine = InferenceEngine(
        FodciModel(config),
        tokenizer,
        InferenceConfig(max_new_tokens=32, device="cpu", seed=SEED, model_version=model_version, checkpoint_path=checkpoint),
    )
    prompts = (
        "What is a unit test in Python?",
        "Explain what an API is in one short paragraph.",
        "Hello. Please introduce yourself in clear English.",
    )
    outputs = []
    for prompt in prompts:
        result = engine.generate(f"### Instruction\nAnswer clearly in English.\n\n### Input\n{prompt}\n\n### Response\n")
        outputs.append({"prompt": prompt, "text": result.generated_text, "non_empty": bool(result.generated_text.strip()), "generated_tokens": result.generated_token_count, "stopped_reason": result.stopped_reason})
    return {"model_version": model_version, "checkpoint": str(checkpoint), "outputs": outputs, "all_non_empty": all(item["non_empty"] for item in outputs)}


def render_markdown(report: dict[str, Any]) -> str:
    rows = []
    for name, item in report["models"].items():
        rows.append(f"| {name} | {item['parameter_count']:,} | {item['baseline_validation_loss']:.6f} | {item['trained_validation_loss']:.6f} | {item['validation_improvement']:.6f} | `{item['all_gates_passed']}` |")
    probe_sections = []
    for name, probe in report["response_probes"].items():
        probe_sections.append(f"### {name}\n\n" + "\n".join(f"- **Prompt:** {item['prompt']}\n  **Output:** `{item['text']}`" for item in probe["outputs"]))
    return f"""# Phase 13.13 — English Language Foundation & 25M Training

> This experiment is English-only and keeps the existing `fodci-testing-qa-v1` release untouched until a new checkpoint demonstrates both finite training evidence and understandable held-out responses.

## Dataset and tokenizer

The corpus uses five verified English Project Gutenberg UTF-8 sources with provenance recorded in `training_data/english_foundation/manifest.json`. Four books are used for training and one held-out book is used for validation. A deterministic byte-BPE tokenizer with 256 learned English merges is saved at `tokenizers/fodci-english-v2.json`, eliminating the previous inference mismatch where a trained tokenizer artifact was not loaded.

| Field | Value |
|---|---:|
| Train documents | {report['data']['train_documents']} |
| Validation documents | {report['data']['validation_documents']} |
| Train characters | {report['data']['train_characters']:,} |
| Validation characters | {report['data']['validation_characters']:,} |
| Fixed context length | {report['data']['context_length']} |
| Tokenizer merges | {report['data']['tokenizer_merges']} |
| Curated instruction chunks | {report['data']['instruction_chunks_in_train']} |

## Matched model comparison

| Model | Parameters | Baseline loss | Trained loss | Improvement | Gates |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Both candidates use the same English tokenizer, corpus split, CPU optimizer settings, seed, and bounded step budget. The 25M candidate is a real trained checkpoint, not only a parameter-count probe.

## Natural-response probes

{chr(10).join(probe_sections)}

These probes are diagnostic and must not be mistaken for a human-quality evaluation. The release decision remains conservative: the current stable backend checkpoint is not replaced unless the English candidate is both statistically improved and visibly understandable on held-out prompts.

## Decision

**Current stable runtime preserved:** `{report['stable_runtime']}`.

The English checkpoints are experimental until the response probes and broader held-out evaluations demonstrate clear improvement over the current behavior. The training pipeline and tokenizer artifact are now reproducible, English-only, and ready for a longer training run if CPU/GPU resources permit.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Train matched English-only 11M and 25M Fodci foundation candidates.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--train-examples", type=int, default=4096)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.train_examples < 256:
        raise ValueError("max-steps must be positive and train-examples must be at least 256")
    tokenizer = FodciTokenizer.load(TOKENIZER_PATH)
    train_examples, validation_examples, data_stats = load_data(tokenizer, limit_examples=args.train_examples)
    started = time.perf_counter()
    model_reports = {
        "english_11m": train_one(config=CONFIG_11M, model_version=MODEL_11M_VERSION, checkpoint_path=MODEL_11M_CHECKPOINT, train_examples=train_examples, validation_examples=validation_examples, max_steps=args.max_steps),
        "english_25m": train_one(config=CONFIG_25M, model_version=MODEL_25M_VERSION, checkpoint_path=MODEL_25M_CHECKPOINT, train_examples=train_examples, validation_examples=validation_examples, max_steps=args.max_steps),
    }
    probes = {
        "english_11m": response_probe(MODEL_11M_CHECKPOINT, MODEL_11M_VERSION, CONFIG_11M, tokenizer),
        "english_25m": response_probe(MODEL_25M_CHECKPOINT, MODEL_25M_VERSION, CONFIG_25M, tokenizer),
    }
    report = {
        "format": "fodci.phase1313_english_foundation",
        "schema_version": "1.0",
        "phase": "13.13",
        "language": "en",
        "dataset_version": DATASET_VERSION,
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_sha256": sha256(TOKENIZER_PATH),
        "data": data_stats,
        "models": model_reports,
        "response_probes": probes,
        "stable_runtime": "fodci-testing-qa-v1",
        "stable_runtime_replaced": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    report["all_training_gates_passed"] = all(item["all_gates_passed"] for item in model_reports.values())
    report["all_response_probes_non_empty"] = all(item["all_non_empty"] for item in probes.values())
    if not report["all_training_gates_passed"]:
        raise RuntimeError("At least one English foundation model failed its training gates.")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_PATH.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": "13.13", "models": {name: item["parameter_count"] for name, item in model_reports.items()}, "all_training_gates_passed": report["all_training_gates_passed"], "all_response_probes_non_empty": report["all_response_probes_non_empty"], "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
