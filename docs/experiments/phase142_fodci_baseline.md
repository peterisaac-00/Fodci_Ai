# Phase 14.2 — Fodci 11M Backend Baseline

> This phase measures the stable Fodci language provider on the held-out Phase 14.1 backend-response benchmark. It is a diagnostic baseline, not a semantic acceptance test.

## Protocol

The stable `fodci-testing-qa-v1` checkpoint was evaluated on CPU with seed `2026`, the fixed Phase 14.1 dataset, provider-default greedy decoding, and `max_new_tokens=32`. The benchmark contains 24 cases across eight backend categories. The stable runtime was not replaced.

| Field | Value |
|---|---:|
| Model | `fodci-testing-qa-v1` |
| Parameters | 11,424,400 |
| Dataset | `phase141-v1` |
| Cases | 24 |
| Completed cases | 24 |
| Device | CPU |
| Maximum new tokens | 32 |
| Stable runtime replaced | `False` |
| Phase gates | `True` |

## Results

| Diagnostic | Result |
|---|---:|
| Non-empty rate | 1.0000 |
| Understandable heuristic rate | 0.0000 |
| Average keyword coverage | 0.0000 |
| Average repeated-token rate | 0.3278 |
| Forbidden-concept hit rate | 0.0000 |
| Manual review required | `True` |

All 24 inference calls completed without provider failures. The outputs were non-empty but remained gibberish or repetitive, for example the first case produced `e an the te the te an then then`. Therefore the phase validates the reproducible baseline protocol and confirms the previously observed language-capability limitation; it does not claim that the stable model can answer backend questions understandably.

The machine-readable report is `artifacts/evaluation/phase142_fodci_baseline.json`. The runner is:

```text
PYTHONPATH=src python scripts/run_phase142_fodci_baseline.py --max-new-tokens 32
```

Phase 14.3 may add an experimental pretrained provider, but the stable Fodci runtime remains unchanged until the same benchmark shows a clear improvement.
