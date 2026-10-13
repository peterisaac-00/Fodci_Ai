"""English-only pretraining runner for the Phase 16.2 corpus.

The corpus is streamed from clean Gutenberg shards. Only bounded batches are
materialized, so a 20M-token corpus does not become a giant Python list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator

from backend_ai.checkpoint import CheckpointManager
from backend_ai.dataset.samples import TrainingExample
from backend_ai.english_only import ENGLISH_ONLY_POLICY_VERSION, validate_english_text
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import EOS_ID, FodciTokenizer
from backend_ai.training import FodciTrainer, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "artifacts" / "phase162_english_corpus"
DEFAULT_TOKENIZER = ROOT / "tokenizers" / "fodci-english-v4.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "checkpoints"
MODEL_VERSION = "fodci-english-pretrain-11m-v1"
SEED = 2026
CONTEXT_LENGTH = 256
CONFIG = ModelConfig(seed=SEED, context_length=CONTEXT_LENGTH)


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def iter_examples(tokenizer: FodciTokenizer, path: Path, *, context_length: int) -> Iterator[TrainingExample]:
    text = path.read_text(encoding="utf-8")
    validate_english_text(text, field=f"corpus:{path.name}")
    token_ids = tokenizer.encode(text) + [EOS_ID]
    for start in range(0, len(token_ids) - 1, context_length):
        window = token_ids[start : start + context_length + 1]
        if len(window) != context_length + 1:
            continue
        yield TrainingExample(
            tuple(window[:-1]),
            tuple(window[1:]),
            f"gutenberg:{path.name}",
            (True,) * context_length,
        )


def stream_files(tokenizer: FodciTokenizer, files: tuple[Path, ...]) -> Iterator[TrainingExample]:
    for path in files:
        yield from iter_examples(tokenizer, path, context_length=CONTEXT_LENGTH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded English-only Fodci pretraining over the Phase 16.2 corpus.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0:
        raise ValueError("max-steps and batch-size must be positive")
    manifest_path = args.corpus / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing corpus manifest: {manifest_path}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("language") != "en" or manifest.get("language_policy") != ENGLISH_ONLY_POLICY_VERSION:
        raise ValueError("corpus manifest is not an English-only Phase 16.2 manifest")
    if int(manifest.get("total_tokens", 0)) < 20_000_000:
        raise ValueError("corpus manifest is below the 20M-token minimum")
    tokenizer = FodciTokenizer.load(args.tokenizer)
    files = tuple(sorted(args.corpus.glob("clean/*.txt")))
    if len(files) < 2:
        raise ValueError("corpus must contain at least two clean text shards")
    validation_count = max(1, len(files) // 10)
    train_files = files[:-validation_count]
    validation_files = files[-validation_count:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{MODEL_VERSION}.pt"
    trainer = FodciTrainer(
        FodciModel(CONFIG),
        lambda: stream_files(tokenizer, train_files),
        lambda: stream_files(tokenizer, validation_files),
        TrainingConfig(
            epochs=1,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=3e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            device=args.device,
            seed=SEED,
            validation_interval=1,
            checkpoint_interval=0,
            output_dir=args.output_dir,
        ),
        model_version=MODEL_VERSION,
        checkpoint_run_metadata={
            "phase": "16.2",
            "purpose": "english_pretraining_only",
            "language_policy": ENGLISH_ONLY_POLICY_VERSION,
            "corpus_manifest_sha256": sha256(manifest_path),
        },
    )
    started = time.perf_counter()
    result = trainer.train()
    saved = trainer.save_checkpoint(checkpoint_path, run_metadata={"phase": "16.2", "language_policy": ENGLISH_ONLY_POLICY_VERSION})
    report = {
        "format": "fodci.phase162_english_pretraining_report",
        "schema_version": "1.0",
        "phase": "16.2",
        "purpose": "english_pretraining_only",
        "language": "en",
        "language_policy": ENGLISH_ONLY_POLICY_VERSION,
        "model_version": MODEL_VERSION,
        "parameter_count": trainer.model.num_parameters,
        "corpus_manifest": str(manifest_path),
        "corpus_manifest_sha256": sha256(manifest_path),
        "corpus_tokens": manifest["total_tokens"],
        "train_shards": len(train_files),
        "validation_shards": len(validation_files),
        "requested_device": args.device,
        "resolved_device": str(trainer.device),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "global_step": result.global_step,
        "checkpoint": str(saved),
        "checkpoint_sha256": sha256(saved),
        "elapsed_seconds": time.perf_counter() - started,
        "stable_runtime_replaced": False,
        "history": [item.to_dict() for item in result.history],
    }
    report_path = ROOT / "artifacts" / "evaluation" / "phase162_english_pretraining.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "checkpoint": str(saved), "global_step": result.global_step, "resolved_device": str(trainer.device)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
