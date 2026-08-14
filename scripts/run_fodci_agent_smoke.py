"""Real local smoke for the Phase 3.6 Agent API."""

from __future__ import annotations

from pathlib import Path

from backend_ai.agent import AgentConfig, AgentLoop
from backend_ai.inference import InferenceConfig, InferenceEngine
from backend_ai.model import FodciModel
from backend_ai.tokenizer import FodciTokenizer


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-tiny-v1.pt"


def main() -> None:
    if not CHECKPOINT.is_file():
        raise SystemExit(f"Local smoke checkpoint is unavailable: {CHECKPOINT}")
    engine = InferenceEngine(
        FodciModel(),
        FodciTokenizer(),
        InferenceConfig(
            max_new_tokens=4,
            device="cpu",
            checkpoint_path=CHECKPOINT,
        ),
    )
    result = AgentLoop(
        engine,
        config=AgentConfig(
            max_steps=2,
            max_tool_calls=2,
            max_context_tokens=256,
            reserve_response_tokens=32,
        ),
    ).run("Understand this project.", ROOT)
    print({
        "status": result.status.value,
        "stop_reason": result.stop_reason,
        "final_answer": result.final_answer,
        "steps": len(result.steps),
        "tool_calls": [call.name for call in result.tool_calls],
        "checkpoint": engine.checkpoint_identity,
        "project_type": result.project_context.project_type if result.project_context else None,
        "context_completeness": result.project_context.completeness if result.project_context else None,
    })
    assert result.project_context is not None
    assert result.project_context.root == ROOT.resolve()
    assert result.tool_calls[0].name == "project_context"
    assert result.status.value in {"completed", "context_limit", "inference_error", "invalid_action", "max_steps", "max_tool_calls"}


if __name__ == "__main__":
    main()
