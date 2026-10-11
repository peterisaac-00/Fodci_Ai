# Phase 16.1 — Fodci English-Only Foundation Policy

## Decision

Fodci is now an English-only training project. Training inputs are rejected when they contain Arabic-script text; the pipeline does not transliterate, silently remove, or send rejected content to a model provider.

The policy applies to training data and training artifacts. General filesystem, Unicode, and project-analysis tools remain Unicode-capable so they can safely inspect real user projects.

## Enforced rules

- Training corpus material must be English-only.
- Training instructions and system prompts must be English.
- Training and validation examples are checked recursively before they enter the trainer.
- Arabic-script records are rejected with `english_only_rejected` and remain traceable in the build report.
- The English tokenizer is explicit and recorded in the corpus manifest.
- No Qwen, hosted API, or external LLM is part of the training path.
- Training is offline; interactive use does not update model weights.
- The stable runtime is never replaced by this policy or by a training run.

## Device policy

Training supports `--device auto`, `--device cpu`, and `--device cuda` in the English Foundation runner. `auto` selects CUDA only when PyTorch reports CUDA availability; otherwise it selects CPU. Explicit `cuda` fails closed when CUDA is unavailable instead of silently falling back.

The user's AMD Ryzen 7 5800U system with integrated Radeon graphics should therefore use CPU unless a separately supported PyTorch/ROCm configuration is verified. CUDA is normally available on NVIDIA GPUs with a compatible PyTorch build, not on this machine by default.

Example:

```bash
PYTHONPATH=src python scripts/train_phase1313_english_foundation.py --device auto --max-steps 1024
```

## Acceptance rule

An English checkpoint remains experimental until it passes both structural training gates and held-out response-quality evaluation. Lower loss alone is insufficient. Promotion must continue to use the existing fail-closed shadow/promotion policy.
