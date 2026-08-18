# Phase 14.1 — Backend Response Benchmark

> This phase defines a held-out diagnostic benchmark for language-provider quality. It does not train a model, run tools, or claim semantic correctness from keyword matching.

## Scope

The benchmark contains 24 deterministic English questions across eight backend categories: Python backend, FastAPI, REST/HTTP, SQL/PostgreSQL, authentication/security, testing, debugging, and architecture. Each case has a stable ID, difficulty, expected concepts, optional forbidden concepts, and a bounded response length. The dataset is stored separately from all training data and is marked `benchmark_only`.

| Field | Value |
|---|---:|
| Benchmark version | `backend-response-v1` |
| Dataset version | `phase141-v1` |
| Cases | 24 |
| Categories | 8 |
| Easy / medium / hard | 8 / 9 / 7 |
| Code-required cases | 3 |
| Dataset fingerprint | `sha256:ca6ac06e3c665051689326637c4cb96ae6f2733f1485a2524082901de3b01095` |
| Training source paths | none |

## Scoring contract

The scorer records non-empty output, word count, expected-concept coverage, forbidden-concept hits, repeated-token rate, code presence, and an `understandable_heuristic` diagnostic. The heuristic is intentionally conservative and never replaces human review or execution-aware correctness tests. Every score retains `manual_review_required: true`.

## Phase gates

| Gate | Result |
|---|---|
| Benchmark is benchmark-only | `True` |
| Required categories present | `True` |
| Stable case IDs unique | `True` |
| Expected rubrics present | `True` |
| Training contamination checked | `True` |
| Model executed | `False` |
| Training performed | `False` |
| All Phase 14.1 gates | `True` |

The machine-readable report is `artifacts/evaluation/phase141_backend_benchmark.json`. The runner is:

```text
PYTHONPATH=src python scripts/run_phase141_backend_benchmark.py
```

Phase 14.2 will use this exact dataset to measure the stable Fodci 11M baseline. No model checkpoint or runtime was changed by Phase 14.1.
