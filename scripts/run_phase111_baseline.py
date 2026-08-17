"""Run the Phase 11.1 baseline against an existing local Fodci checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from backend_ai.evaluation.baseline import (
    BaselineEvaluationConfig,
    BaselineEvaluationRunner,
    BaselineEvaluationStore,
    create_current_model_runtime,
    load_evaluation_dataset,
    model_identity_from_checkpoint,
)
from backend_ai.tokenizer import TOKENIZER_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible Phase 11.1 Fodci baseline evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/fodci-tiny-v1.pt"))
    parser.add_argument("--evaluation-id", default="baseline-fodci-tiny-v1-2026-08-17-1")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--store", type=Path, default=Path("artifacts/evaluation/baseline_runs.json"))
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    model_version = "fodci-tiny-v1"
    config = BaselineEvaluationConfig(
        seed=2026,
        temperature=1.0,
        max_tokens=8,
        max_iterations=2,
        timeout_seconds=20.0,
        tool_configuration="ToolRegistry.default",
        store_path=args.store,
    )
    dataset = load_evaluation_dataset()
    identity = model_identity_from_checkpoint(checkpoint, model_version=model_version, tokenizer_version=TOKENIZER_VERSION)
    runtime = create_current_model_runtime(checkpoint, model_version=model_version, tokenizer_version=TOKENIZER_VERSION, config=config)
    run = BaselineEvaluationRunner(
        runtime=runtime,
        model_identity=identity,
        agent_version="0.1.0",
        config=config,
        store=BaselineEvaluationStore(args.store),
    ).run(
        dataset,
        evaluation_id=args.evaluation_id,
        project_root=project_root,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    print(json.dumps({"evaluation_id": run.evaluation_id, "status": run.status.value, "model_identity": run.model_identity.to_dict(), "dataset_version": run.dataset_version, "dataset_fingerprint": run.dataset_fingerprint, "aggregate": run.aggregate.to_dict(), "store": str(args.store)}, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
