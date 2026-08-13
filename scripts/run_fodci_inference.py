"""Run real local CPU inference smoke prompts for Phase 2.11."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.inference import InferenceConfig, InferenceEngine  # noqa: E402
from backend_ai.model import FodciModel  # noqa: E402
from backend_ai.tokenizer import FodciTokenizer  # noqa: E402

CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-tiny-v1.pt"
PROMPTS = (
    "Hi",
    "Write a Python function that adds two numbers.",
    "Create a simple backend API endpoint.",
)


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"The existing trained Fodci Tiny v1 checkpoint is required: {CHECKPOINT}"
        )
    engine = InferenceEngine(
        FodciModel(),
        FodciTokenizer(),
        InferenceConfig(
            max_new_tokens=8,
            temperature=1.0,
            do_sample=False,
            stop_on_eos=True,
            device="cpu",
            checkpoint_path=CHECKPOINT,
        ),
    )
    for prompt in PROMPTS:
        result = engine.generate(prompt)
        print(json.dumps({"prompt": prompt, **result.to_dict()}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
