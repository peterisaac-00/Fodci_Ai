# Phase 16.2 — 20M-token English Corpus and Pretraining

## Decision

This is the minimum experiment requested for Fodci. It is **English pretraining only**. Backend instruction tuning is intentionally deferred to a later phase.

The target is **20,000,000 tokens**, with a hard builder range of 20M–50M tokens. The corpus is generated from English `Text` records in Project Gutenberg's official CSV catalog. The catalog and source fingerprints are recorded in the manifest; the large raw/clean corpus remains under `artifacts/` and is not committed to Git.

## Why this is the correct experiment

- It tests whether the current tokenizer, model, and training loop learn broader English than the existing 3M-character corpus.
- It separates language learning from backend instruction behavior.
- It creates a measurable baseline for loss, perplexity, repetition, and held-out response probes.
- It avoids contaminating the result with Qwen, API calls, or online learning.

This will **not** make Fodci a good Backend Engineering assistant yet. It should improve basic English fluency and reduce degenerate output only if the model capacity and training budget are sufficient. Backend Q&A requires the later instruction-tuning phase.

## Build corpus

From the repository root:

```bash
PYTHONPATH=src python scripts/build_phase162_english_corpus.py \
  --target-tokens 20000000 \
  --max-books 500
```

The builder:

1. Downloads the official Gutenberg CSV catalog.
2. Selects English text records deterministically by book ID.
3. Downloads plain text books with provenance metadata.
4. Removes Gutenberg wrappers and normalizes whitespace.
5. Rejects Arabic-script content using the Fodci English-only policy.
6. Counts tokens with the exact Fodci tokenizer.
7. Stops only after the requested minimum is reached.
8. Writes `artifacts/phase162_english_corpus/manifest.json`.

The corpus must not be copied into Git. Only code, documentation, and small manifests belong in the repository.

## Run pretraining

After the manifest reaches at least 20M tokens:

```bash
PYTHONPATH=src python scripts/train_phase162_english_pretraining.py \
  --corpus artifacts/phase162_english_corpus \
  --max-steps 2000 \
  --batch-size 8 \
  --device auto
```

`--device auto` uses CUDA only when supported by the installed PyTorch build; otherwise it uses CPU. Explicit `--device cuda` fails if CUDA is unavailable.

The runner streams shards and materializes only bounded batches. It holds out the final 10% of shards for validation, records the corpus manifest hash, saves an experimental checkpoint, and writes an evaluation report. It never replaces `fodci-testing-qa-v1.pt`.

## Promotion gate

The checkpoint is not promoted merely because training finishes or loss decreases. Promotion requires:

- finite training and validation losses;
- no tokenizer/checkpoint mismatch;
- no regression in repetition and empty-response metrics;
- understandable English held-out probes;
- comparison against the current stable runtime;
- the existing fail-closed shadow/promotion policy.
