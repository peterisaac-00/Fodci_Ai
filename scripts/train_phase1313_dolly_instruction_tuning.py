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
DATA_PATH = ROOT / "training_data" / "english_foundation" / "dolly" / "databricks-dolly-15k.jsonl"
TOKENIZER_PATH = ROOT / "tokenizers" / "fodci-english-v4.json"
BASE_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-english-25m-v1.pt"
OUTPUT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-english-25m-dolly-v1.pt"
REPORT_PATH = ROOT / "artifacts" / "evaluation" / "phase1313_dolly_instruction_tuning.json"
MARKDOWN_PATH = ROOT / "docs" / "experiments" / "phase1313_dolly_instruction_tuning.md"
MODEL_VERSION = "fodci-english-25m-dolly-v1"
CONTEXT_LENGTH = 256
SEED = 2026


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_example(tokenizer: FodciTokenizer, record_id: str, instruction: str, context: str, response: str, width: int = CONTEXT_LENGTH) -> TrainingExample:
    prompt = f"### Instruction\nAnswer clearly and accurately in English.\n\n### Input\n{instruction}"
    if context.strip():
        prompt += f"\n\nReference context:\n{context}"
    prompt += "\n\n### Response\n"
    prefix_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response) + [EOS_ID]
    if not response_ids:
        raise ValueError(f"empty response: {record_id}")
    response_ids = response_ids[: min(len(response_ids), width - 8)]
    context_budget = max(1, width + 1 - len(response_ids))
    if len(prefix_ids) > context_budget:
        prefix_ids = prefix_ids[-context_budget:]
    token_ids = prefix_ids + response_ids
    token_ids = token_ids[: width + 1]
    actual_response_start = len(prefix_ids)
    target_count = len(token_ids) - 1
    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]
    active_start = max(0, actual_response_start - 1)
    loss_mask = (False,) * active_start + (True,) * max(1, target_count - active_start)
    input_ids += [0] * (width - len(input_ids))
    target_ids += [0] * (width - len(target_ids))
    loss_mask = loss_mask[:width] + (False,) * max(0, width - len(loss_mask))
    return TrainingExample(tuple(input_ids), tuple(target_ids), record_id, tuple(loss_mask[:width]))


def load_splits(tokenizer: FodciTokenizer, max_records: int, validation_modulus: int) -> tuple[list[TrainingExample], list[TrainingExample], dict[str, Any]]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dolly dataset is missing: {DATA_PATH}")
    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    categories: dict[str, int] = {}
    total = 0
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if total >= max_records:
            break
        row = json.loads(line)
        instruction = row.get("instruction")
        context = row.get("context", "")
        response = row.get("response")
        category = row.get("category", "unknown")
        if not all(isinstance(value, str) for value in (instruction, context, response)):
            continue
        record_id = hashlib.sha256(f"{instruction}\0{context}\0{response}".encode("utf-8")).hexdigest()
        example = build_example(tokenizer, record_id, instruction, context, response)
        categories[category] = categories.get(category, 0) + 1
        bucket = int(record_id[:8], 16) % validation_modulus
        (validation if bucket == 0 else train).append(example)
        total += 1
    if len(train) < 128 or len(validation) < 16:
        raise ValueError(f"Dolly split is too small: train={len(train)}, validation={len(validation)}")
    return train, validation, {"source": "databricks-dolly-15k", "language": "en", "records_read": total, "train_examples": len(train), "validation_examples": len(validation), "validation_modulus": validation_modulus, "categories": categories, "context_length": CONTEXT_LENGTH}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the 25M English foundation checkpoint on Dolly 15K instructions.")
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--max-records", type=int, default=12000)
    parser.add_argument("--validation-modulus", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_records < 256 or args.validation_modulus < 2 or args.learning_rate <= 0:
        raise ValueError("invalid Dolly training arguments")

    tokenizer = FodciTokenizer.load(TOKENIZER_PATH)
    base_info = CheckpointManager(BASE_CHECKPOINT.parent, model_version="checkpoint-inspection").inspect(BASE_CHECKPOINT)
    model_config = ModelConfig(**base_info.metadata.model_config)
    train_examples, validation_examples, data_stats = load_splits(tokenizer, args.max_records, args.validation_modulus)
    model = FodciModel(model_config)
    CheckpointManager(BASE_CHECKPOINT.parent, model_version=base_info.metadata.model_version).load_model(BASE_CHECKPOINT, model, device=torch.device("cpu"))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    trainer = FodciTrainer(model, train_examples, validation_examples, TrainingConfig(epochs=1, max_steps=args.max_steps, batch_size=2, learning_rate=args.learning_rate, weight_decay=0.01, max_grad_norm=1.0, device="cpu", seed=SEED, validation_interval=1, checkpoint_interval=0, output_dir=OUTPUT_CHECKPOINT.parent), model_version=MODEL_VERSION, checkpoint_run_metadata={"phase": "13.13", "language": "en", "stage": "dolly-instruction-tuning", "base_checkpoint": str(BASE_CHECKPOINT), "dataset_sha256": sha256(DATA_PATH), "tokenizer_sha256": sha256(TOKENIZER_PATH)})
    baseline_loss, baseline_steps, baseline_tokens = trainer.evaluate(validation_examples)
    started = time.perf_counter()
    result = trainer.train()
    trained_loss, validation_steps, validation_tokens = trainer.evaluate(validation_examples)
    saved = trainer.save_checkpoint(OUTPUT_CHECKPOINT, run_metadata={"phase": "13.13", "language": "en", "stage": "dolly-instruction-tuning", "base_checkpoint": str(BASE_CHECKPOINT), "dataset_sha256": sha256(DATA_PATH), "tokenizer_sha256": sha256(TOKENIZER_PATH)})
    reloaded_model = FodciModel(model_config)
    loaded = CheckpointManager(saved.parent, model_version=MODEL_VERSION).load_model(saved, reloaded_model, device=torch.device("cpu"))
    parameters_changed = any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))

    prompts = ("What is a unit test in Python?", "Explain what an API is in one short paragraph.", "Hello. Please introduce yourself in clear English.", "How should passwords be stored?", "What does HTTP 201 mean?")
    engine = InferenceEngine(FodciModel(model_config), tokenizer, InferenceConfig(max_new_tokens=48, device="cpu", seed=SEED, model_version=MODEL_VERSION, checkpoint_path=saved))
    outputs = []
    for prompt in prompts:
        result_probe = engine.generate(f"### Instruction\nAnswer clearly and accurately in English.\n\n### Input\n{prompt}\n\n### Response\n")
        outputs.append({"prompt": prompt, "text": result_probe.generated_text, "non_empty": bool(result_probe.generated_text.strip()), "generated_tokens": result_probe.generated_token_count, "stopped_reason": result_probe.stopped_reason})

    report = {"format": "fodci.phase1313_dolly_instruction_tuning", "schema_version": "1.0", "phase": "13.13", "language": "en", "model_version": MODEL_VERSION, "parameter_count": model.num_parameters, "base_checkpoint": str(BASE_CHECKPOINT), "checkpoint_path": str(saved), "tokenizer_path": str(TOKENIZER_PATH), "tokenizer_sha256": sha256(TOKENIZER_PATH), "dataset_path": str(DATA_PATH), "dataset_sha256": sha256(DATA_PATH), "license": "CC-BY-SA-3.0", "data": data_stats, "max_steps": args.max_steps, "learning_rate": args.learning_rate, "global_step": result.global_step, "training_seconds": time.perf_counter() - started, "baseline_validation_loss": baseline_loss, "trained_validation_loss": trained_loss, "validation_improvement": baseline_loss - trained_loss, "checkpoint_exists": saved.is_file(), "checkpoint_reload": loaded.metadata.model_version == MODEL_VERSION, "finite_loss": all(torch.isfinite(torch.tensor(value)) for value in (baseline_loss, trained_loss)), "parameters_changed": parameters_changed, "non_empty_split": bool(train_examples and validation_examples), "response_probes": outputs, "all_response_probes_non_empty": all(item["non_empty"] for item in outputs), "structural_gates_passed": all((saved.is_file(), loaded.metadata.model_version == MODEL_VERSION, parameters_changed, bool(train_examples and validation_examples))), "heldout_loss_improved": trained_loss < baseline_loss}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 13.13 — Dolly English Instruction Tuning", "", "This is an English-only response-generation experiment using the Databricks Dolly 15K dataset under CC-BY-SA-3.0. The 25M foundation checkpoint remains separate from the stable runtime.", "", f"- Base checkpoint: `{BASE_CHECKPOINT.name}`", f"- Output checkpoint: `{saved.name}`", f"- Dataset records used: `{data_stats['records_read']}`", f"- Training steps: `{result.global_step}`", f"- Held-out validation loss: `{baseline_loss:.6f}` → `{trained_loss:.6f}`", f"- Structural gates: `{report['structural_gates_passed']}`", f"- Held-out loss improved: `{report['heldout_loss_improved']}`", "", "## Response probes", ""]
    for item in outputs:
        lines.extend([f"### {item['prompt']}", "", f"```text\n{item['text']}\n```", ""])
    lines.extend(["A non-empty response is not sufficient for a quality claim. This report intentionally separates structural training success, held-out loss, and human-readable response quality.", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"model_version": MODEL_VERSION, "parameter_count": model.num_parameters, "records": data_stats["records_read"], "global_step": result.global_step, "baseline_validation_loss": baseline_loss, "trained_validation_loss": trained_loss, "structural_gates_passed": report["structural_gates_passed"], "heldout_loss_improved": report["heldout_loss_improved"], "checkpoint": str(saved)}, indent=2))


if __name__ == "__main__":
    main()
