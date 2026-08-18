# Phase 13.13 — English Language Foundation

> This experiment establishes an English-only training path, but it does **not** claim human-readable generation. The existing `fodci-testing-qa-v1` runtime remains unchanged because the response probes are still repetitive or nonsensical.

## Dataset and tokenizer

The corpus uses five verified English Project Gutenberg UTF-8 sources with provenance recorded in `training_data/english_foundation/manifest.json`. Four books are used for training and Sherlock Holmes is held out for validation. The deterministic byte-BPE tokenizer at `tokenizers/fodci-english-v4.json` contains 512 learned merges and is loaded by the English checkpoints.

| Field | Value |
|---|---:|
| Train documents | 4 |
| Validation documents | 1 |
| Train characters | 2,502,566 |
| Validation characters | 562,678 |
| Training examples | 1,024 |
| Validation examples | 256 |
| Fixed context length | 256 |
| Tokenizer merges | 512 |
| Curated instruction chunks included | 32 |
| Training device | CPU only |
| Configured step budget per candidate | 1,024 |

## Matched model comparison

Both candidates were initialized and trained under the same English corpus, tokenizer, seed, CPU optimizer, and bounded schedule. The trainer completed 512 optimizer steps in the recorded run; the configured maximum was 1,024 because the one-epoch dataset budget was reached first.

| Model | Parameters | Baseline loss | Trained loss | Improvement | Structural gates |
|---|---:|---:|---:|---:|---:|
| english_11m | 11,424,400 | 9.275778 | 4.780644 | 4.495134 | `True` |
| english_25m | 25,985,488 | 9.273855 | 4.886634 | 4.387221 | `True` |

The structural gates verified finite losses, parameter changes, non-empty data splits, checkpoint existence, checkpoint reload, and lineage metadata. The 25M candidate is a real trained experimental checkpoint, not merely a parameter-count probe.

## Natural-response probes

The loss reductions are useful evidence that the model learned a statistical signal from the corpus, but the short generations remain unusable as conversational answers.

### english_11m

| Prompt | Output |
|---|---|
| What is a unit test in Python? | `Input` repeated eight times |
| Explain what an API is in one short paragraph. | `Input` repeated eight times |
| Hello. Please introduce yourself in clear English. | `Input` repeated eight times |

### english_25m

| Prompt | Output |
|---|---|
| What is a unit test in Python? | `Input` repeated with one `Inponse` token |
| Explain what an API is in one short paragraph. | `Inponse` repeated with `Resput` and `Input` |
| Hello. Please introduce yourself in clear English. | `Input` and `Inponse` repetitions |

These results fail the project quality bar for understandable English. The tokenizer is now correctly matched to the checkpoints, but a short CPU schedule from random initialization is insufficient to learn reliable grammar and response formatting.

## Decision

**Stable runtime preserved:** `fodci-testing-qa-v1` at 11,424,400 parameters.

Neither English foundation checkpoint is activated. The 25M artifact remains experimental and is not the default. A longer, staged English pretraining run, better corpus diversity, and more carefully designed instruction formatting are required before considering runtime replacement.
