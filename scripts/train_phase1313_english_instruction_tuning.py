from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from backend_ai.checkpoint import CheckpointManager
from backend_ai.dataset.instructions import InstructionExample
from backend_ai.dataset.samples import TrainingExample
from backend_ai.inference import InferenceConfig, InferenceEngine
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import EOS_ID, FodciTokenizer
from backend_ai.training import FodciTrainer, TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training_data" / "english_foundation" / "instructions" / "train"
TOKENIZER_PATH = ROOT / "tokenizers" / "fodci-english-v4.json"
BASE_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-english-25m-v1.pt"
OUTPUT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-english-25m-instruct-v1.pt"
REPORT_PATH = ROOT / "artifacts" / "evaluation" / "phase1313_english_instruction_tuning.json"
MARKDOWN_PATH = ROOT / "docs" / "experiments" / "phase1313_english_instruction_tuning.md"
MODEL_VERSION = "fodci-english-25m-instruct-v1"
CONTEXT_LENGTH = 256
SEED = 2026


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def instruction_example(tokenizer: FodciTokenizer, path: Path, width: int = CONTEXT_LENGTH) -> TrainingExample:
    parsed = InstructionExample.parse(path.read_text(encoding="utf-8"), path)
    context = f"### Instruction\n{parsed.instruction}\n\n### Input\n{parsed.input_text}\n\n### Response\n"
    token_ids = tokenizer.encode(context) + tokenizer.encode(parsed.response) + [EOS_ID]
    if len(token_ids) < 2:
        raise ValueError(f"instruction is too short: {path}")
    if len(token_ids) > width + 1:
        raise ValueError(f"instruction exceeds context window: {path}")
    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]
    response_start = len(tokenizer.encode(context))
    active_targets = max(1, len(token_ids) - response_start - 1)
    input_ids = input_ids + [0] * (width - len(input_ids))
    target_ids = target_ids + [0] * (width - len(target_ids))
    loss_mask = (False,) * max(0, response_start - 1) + (True,) * active_targets
    loss_mask = loss_mask[:width] + (False,) * max(0, width - len(loss_mask))
    return TrainingExample(tuple(input_ids), tuple(target_ids), parsed.example_id, tuple(loss_mask[:width]))


HELD_OUT_RECORDS = (
    ("heldout_http_201", "Explain an HTTP status code.", "What does HTTP 201 mean?", "HTTP 201 means that a request succeeded and created a new resource. The response often includes the new resource location or representation."),
    ("heldout_transaction", "Explain a backend concept clearly.", "What is a database transaction?", "A database transaction groups related changes so they commit together or roll back together when a failure occurs."),
    ("heldout_retry", "Answer with a safe practical rule.", "When is a retry safe?", "A retry is safest for a transient failure when the operation is idempotent or protected by an idempotency key."),
    ("heldout_mock", "Explain a testing concept.", "Why use a mock in a unit test?", "A mock isolates the unit by replacing a dependency and recording how it was called. It should not hide the behavior that the test is meant to verify."),
    ("heldout_logging", "Give a concise engineering explanation.", "What should a useful log contain?", "A useful log records the event, relevant identifiers, timing, and safe diagnostic context without exposing passwords, tokens, or other secrets."),
    ("heldout_clarification", "Ask for missing information instead of guessing.", "Deploy it now.", "Which environment, commit, deployment target, and verification checks should be used? I should not deploy without those details."),
    ("heldout_coverage", "Explain a quality metric accurately.", "What does branch coverage measure?", "Branch coverage measures whether the possible decision paths in conditional logic were executed. It does not prove that the assertions or requirements are correct."),
    ("heldout_summary", "Summarize the request in one sentence.", "I need validation, persistence, and a clear error response.", "You need a boundary validation step, a persistence operation, and a stable error contract for failure cases."),
)


def fields_example(tokenizer: FodciTokenizer, record_id: str, instruction: str, input_text: str, response: str) -> TrainingExample:
    text = f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n{response}\n"
    token_ids = tokenizer.encode(text)
    temporary = DATA_DIR / f".{record_id}.txt"
    temporary.write_text(text, encoding="utf-8")
    try:
        return instruction_example(tokenizer, temporary)
    finally:
        temporary.unlink(missing_ok=True)


def load_splits(tokenizer: FodciTokenizer, repeats: int) -> tuple[list[TrainingExample], list[TrainingExample], dict[str, Any]]:
    paths = sorted(DATA_DIR.glob("*.txt"))
    if len(paths) < 16:
        raise ValueError("English instruction tuning requires at least 16 records.")
    train_base = [instruction_example(tokenizer, path) for path in paths]
    validation = [fields_example(tokenizer, *record) for record in HELD_OUT_RECORDS]
    train = train_base * repeats
    return train, validation, {"total_records": len(paths), "train_records": len(paths), "validation_records": len(validation), "repeat_factor": repeats, "train_examples": len(train), "validation_examples": len(validation), "held_out": True, "context_length": CONTEXT_LENGTH}


def main() -> None:
    parser = argparse.ArgumentParser(description="Response-only English instruction tuning for the 25M foundation checkpoint.")
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.repeats <= 0 or args.learning_rate <= 0:
        raise ValueError("max-steps, repeats, and learning-rate must be positive")

    tokenizer = FodciTokenizer.load(TOKENIZER_PATH)
    base_info = CheckpointManager(BASE_CHECKPOINT.parent, model_version="checkpoint-inspection").inspect(BASE_CHECKPOINT)
    model_config = ModelConfig(**base_info.metadata.model_config)
    train_examples, validation_examples, data_stats = load_splits(tokenizer, args.repeats)
    base_model = FodciModel(model_config)
    base_manager = CheckpointManager(BASE_CHECKPOINT.parent, model_version=base_info.metadata.model_version)
    base_manager.load_model(BASE_CHECKPOINT, base_model, device=torch.device("cpu"))
    trainer = FodciTrainer(
        base_model,
        train_examples,
        validation_examples,
        TrainingConfig(epochs=1, max_steps=args.max_steps, batch_size=2, learning_rate=args.learning_rate, weight_decay=0.01, max_grad_norm=1.0, device="cpu", seed=SEED, validation_interval=1, checkpoint_interval=0, output_dir=OUTPUT_CHECKPOINT.parent),
        model_version=MODEL_VERSION,
        checkpoint_run_metadata={"phase": "13.13", "language": "en", "stage": "instruction-tuning", "base_checkpoint": str(BASE_CHECKPOINT), "tokenizer_sha256": sha256(TOKENIZER_PATH)},
    )
    baseline_loss, baseline_steps, baseline_tokens = trainer.evaluate(validation_examples)
    before = [parameter.detach().clone() for parameter in base_model.parameters()]
    started = time.perf_counter()
    result = trainer.train()
    trained_loss, validation_steps, validation_tokens = trainer.evaluate(validation_examples)
    saved = trainer.save_checkpoint(OUTPUT_CHECKPOINT, run_metadata={"phase": "13.13", "language": "en", "stage": "instruction-tuning", "base_checkpoint": str(BASE_CHECKPOINT), "tokenizer_sha256": sha256(TOKENIZER_PATH)})
    reloaded_model = FodciModel(model_config)
    loaded = CheckpointManager(saved.parent, model_version=MODEL_VERSION).load_model(saved, reloaded_model, device=torch.device("cpu"))
    reloaded_trainer = FodciTrainer(reloaded_model, train_examples, validation_examples, TrainingConfig(epochs=1, max_steps=1, batch_size=2, device="cpu", seed=SEED, checkpoint_interval=0, output_dir=OUTPUT_CHECKPOINT.parent), model_version=MODEL_VERSION)
    reloaded_loss, _, _ = reloaded_trainer.evaluate(validation_examples)
    parameters_changed = any(not torch.equal(old, new) for old, new in zip(before, base_model.parameters(), strict=True))

    prompts = (
        "What is a unit test in Python?",
        "Explain what an API is in one short paragraph.",
        "Hello. Please introduce yourself in clear English.",
        "How should passwords be stored?",
    )
    engine = InferenceEngine(FodciModel(model_config), tokenizer, InferenceConfig(max_new_tokens=48, device="cpu", seed=SEED, model_version=MODEL_VERSION, checkpoint_path=saved))
    outputs = []
    for prompt in prompts:
        result_probe = engine.generate(f"### Instruction\nAnswer clearly in English.\n\n### Input\n{prompt}\n\n### Response\n")
        outputs.append({"prompt": prompt, "text": result_probe.generated_text, "non_empty": bool(result_probe.generated_text.strip()), "generated_tokens": result_probe.generated_token_count, "stopped_reason": result_probe.stopped_reason})

    report = {
        "format": "fodci.phase1313_english_instruction_tuning",
        "schema_version": "1.0",
        "phase": "13.13",
        "language": "en",
        "model_version": MODEL_VERSION,
        "parameter_count": base_model.num_parameters,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "checkpoint_path": str(saved),
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_sha256": sha256(TOKENIZER_PATH),
        "data": data_stats,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "global_step": result.global_step,
        "training_seconds": time.perf_counter() - started,
        "baseline_validation_loss": baseline_loss,
        "trained_validation_loss": trained_loss,
        "reloaded_validation_loss": reloaded_loss,
        "validation_improvement": baseline_loss - trained_loss,
        "checkpoint_exists": saved.is_file(),
        "checkpoint_reload": loaded.metadata.model_version == MODEL_VERSION,
        "finite_loss": all(torch.isfinite(torch.tensor(value)) for value in (baseline_loss, trained_loss, reloaded_loss)),
        "parameters_changed": parameters_changed,
        "non_empty_split": bool(train_examples and validation_examples),
        "response_probes": outputs,
        "all_response_probes_non_empty": all(item["non_empty"] for item in outputs),
        "all_gates_passed": all((saved.is_file(), loaded.metadata.model_version == MODEL_VERSION, parameters_changed, bool(train_examples and validation_examples), trained_loss < baseline_loss)),
    }
    if not report["all_gates_passed"]:
        raise RuntimeError(f"instruction tuning gates failed: {report}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 13.13 — English Instruction Tuning", "", "The 25M English foundation checkpoint was instruction-tuned with response-only loss masking on curated English records. The stable runtime remains unchanged until response quality is acceptable.", "", f"- Base checkpoint: `{BASE_CHECKPOINT.name}`", f"- Output checkpoint: `{saved.name}`", f"- Parameters: `{base_model.num_parameters:,}`", f"- Training steps: `{result.global_step}`", f"- Validation loss: `{baseline_loss:.6f}` → `{trained_loss:.6f}`", "", "## Response probes", ""]
    for item in outputs:
        lines.extend([f"### {item['prompt']}", "", f"```text\n{item['text']}\n```", ""])
    lines.extend(["The probes are diagnostic; a non-empty response is not sufficient for a quality claim. The checkpoint is not activated as the stable runtime in this stage.", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"model_version": MODEL_VERSION, "parameter_count": base_model.num_parameters, "global_step": result.global_step, "baseline_validation_loss": baseline_loss, "trained_validation_loss": trained_loss, "all_gates_passed": report["all_gates_passed"], "all_response_probes_non_empty": report["all_response_probes_non_empty"], "checkpoint": str(saved)}, indent=2))


if __name__ == "__main__":
    main()
