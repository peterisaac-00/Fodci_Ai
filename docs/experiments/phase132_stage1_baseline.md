# Phase 13.2 — Stage 1 Baseline Evaluation

## Purpose

This document records the pre-training baseline for the default Fodci Tiny v1 model. The result is a diagnostic reference for comparing later checkpoints under the same fixed protocol; it is not a claim of language capability or production readiness.

## Reproducibility Contract

| Field | Value |
|---|---|
| Model | `fodci-tiny-v1` |
| Parameter count | `11,424,400` |
| Checkpoint | `artifacts/checkpoints/fodci-tiny-v1.pt` |
| Held-out dataset | `training_data/fundamentals/evaluation/stage_01.jsonl` |
| Dataset records | `24` |
| Seed | `2026` |
| Decoding | Greedy argmax |
| Maximum generated tokens | `32` |
| Prompt template | Instruction / Input / Response v1 |
| Device | CPU |

## Baseline Result

| Metric | Result |
|---|---:|
| Passed tasks | `0 / 24` |
| Keyword pass rate | `0.00%` |
| Non-empty response rate | `0.00%` |
| Average keyword coverage | `0.00%` |
| Average generated tokens | `32.00` |

| Category | Items | Pass rate | Non-empty rate | Keyword coverage |
|---|---:|---:|---:|---:|
| Architecture | 4 | `0.00%` | `0.00%` | `0.00%` |
| Backend concepts | 4 | `0.00%` | `0.00%` | `0.00%` |
| HTTP | 7 | `0.00%` | `0.00%` | `0.00%` |
| REST | 5 | `0.00%` | `0.00%` | `0.00%` |
| Security | 4 | `0.00%` | `0.00%` | `0.00%` |

## Interpretation

The checkpoint produced no decodable text responses under the fixed baseline protocol, so the deterministic keyword score is zero across all categories. This is a valid baseline diagnostic: it establishes that the current checkpoint should not be treated as a capable conversational model before the next training stage. The result should be compared only with future runs that preserve the same held-out dataset, prompt template, seed, decoding rule, and scoring thresholds.

The keyword score is a conservative proxy for Stage 1 concept coverage. It detects empty output and required-term coverage, but it does not measure semantic correctness, reasoning quality, code quality, or safety. Those dimensions should be added in later benchmark stages without changing this baseline dataset.

The complete machine-readable run is generated locally at `artifacts/evaluation/stage1_baseline.json`, and the executable runner is `scripts/benchmark_stage1.py`.
