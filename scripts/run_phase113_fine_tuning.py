#!/usr/bin/env python3
"""Run one explicit offline Phase 11.3 candidate fine-tuning experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.training import FineTuningConfig, FineTuningStatus, fine_tune  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Phase 11.3 fine-tuning; this does not start fodci Agent runtime.")
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-model-version", default="candidate-v1")
    parser.add_argument("--tokenizer", dest="tokenizer_path", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts/training_runs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = FineTuningConfig(
            run_id=args.run_id,
            candidate_model_version=args.candidate_model_version,
            epochs=args.epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            validation_interval=args.validation_interval,
            log_interval=args.log_interval,
            output_directory=args.output_directory,
        )
        result = fine_tune(
            base_checkpoint=args.base_checkpoint,
            dataset_directory=args.dataset_directory,
            config=config,
            tokenizer_path=args.tokenizer_path,
            resume_checkpoint=args.resume_checkpoint,
        )
    except Exception as exc:
        print(f"Phase 11.3 configuration/input failure: {exc}", file=sys.stderr)
        return 1
    print(result.to_json())
    return 0 if result.status is FineTuningStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
