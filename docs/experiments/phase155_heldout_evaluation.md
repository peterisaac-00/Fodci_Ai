# Phase 15.5 — Held-out Evaluation and Regression

> The distilled candidate and stable runtime were evaluated on the same fixed benchmark. The candidate was not promoted automatically.

The experimental Phase 15.4 checkpoint and the stable `fodci-testing-qa-v1` checkpoint were evaluated on all 24 Phase 14.1 Backend cases with identical CPU-only protocol and 32-token bound.

| Metric | Distilled Fodci | Stable Fodci | Delta |
|---|---:|---:|---:|
| Non-empty rate | 1.0000 | 1.0000 | +0.0000 |
| Understandable heuristic rate | 0.0000 | 0.0000 | +0.0000 |
| Average keyword coverage | 0.0000 | 0.0000 | +0.0000 |
| Repeated-token rate | 0.6970 | 0.3278 | +0.3692 |

The offline training gates passed, and both evaluations completed, but the distilled checkpoint did not improve held-out language quality. Its repeated-token rate was materially worse. This is an important negative result: the candidate remains experimental and was not promoted; the stable runtime remains unchanged.

| Gate | Result |
|---|---|
| Same benchmark dataset | `True` |
| Both models completed | `True` |
| Distilled checkpoint lineage present | `True` |
| Candidate promoted | `False` |
| Response quality accepted | `False` |
| Stable runtime replaced | `False` |
| All phase gates | `True` |

A future distillation run must improve the data volume, response formatting, teacher verification, and training schedule before a student promotion can be considered. A lower training loss alone is not sufficient.
