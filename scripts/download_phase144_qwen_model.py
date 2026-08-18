#!/usr/bin/env python3
"""Download the explicitly selected Qwen model into an ignored local artifact directory."""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "artifacts" / "pretrained" / "qwen2.5-coder-0.5b-instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Qwen 0.5B for the explicit Phase 14.4 local experiment.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.gguf", "*.onnx", "*.msgpack", "*.h5"],
    )
    print({"model_id": args.model_id, "local_dir": str(args.local_dir), "snapshot": path})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
