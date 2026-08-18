# Phase 14.6 — Final Comparison and Decision

> The final decision compares the exact same held-out benchmark and keeps the stable Fodci runtime unchanged.

## Comparison

| Metric | Fodci 11M | Qwen 0.5B | Delta |
|---|---:|---:|---:|
| Non-empty rate | 1.0000 | 1.0000 | +0.0000 |
| Understandable heuristic rate | 0.0000 | 0.9167 | +0.9167 |
| Average keyword coverage | 0.0000 | 0.7188 | +0.7188 |
| Repeated-token rate | 0.3278 | 0.2366 | -0.0912 |

## Decision

The evidence supports using Qwen 0.5B as an **experimental language provider behind `BackendScopedProvider`**, not as a replacement for the stable Fodci runtime. The same 24-case benchmark was used for both models, and the stable release checkpoint hash matched `v13.12-stable`.

| Gate | Result |
|---|---|
| Same benchmark dataset | `True` |
| Baseline and candidate complete | `True` |
| Qwen readability improved | `True` |
| Backend scope policy passed | `True` |
| Stable release hash matches | `True` |
| Stable runtime replaced | `False` |
| Q4 quantization validated | `False` |
| Semantic correctness proven | `False` |
| All Phase 14.6 gates | `True` |

Manual review remains essential. Five likely technical issues were recorded in the Qwen report, including an inappropriate `jsonify` suggestion for FastAPI, an `aiohttp` example for a database question, imprecise password hashing terminology, and an irrelevant `pytest-django` recommendation. The readability improvement is real, but it is not sufficient evidence for unrestricted autonomous coding behavior.

The final recommendation is to keep `fodci-testing-qa-v1` as stable and expose Qwen only through an explicitly selected experimental provider wrapped by Backend policy and output guard. No 1.5B fallback is needed yet, and no Q4 result is claimed. Execution-aware correctness tests are required before any future runtime activation.
