# Phase 15.6 — Shadow Mode and Controlled Promotion

## Purpose

Phase 15.6 introduces a **shadow-only** execution path for comparing the experimental distilled checkpoint with the stable Fodci runtime. The candidate is executed beside the primary provider, but the user-facing response always comes from the stable primary provider.

> Shadow execution is observational. It does not replace the stable checkpoint and it does not train online.

## Protocol

The runner uses two CPU-only `FodciLocalProvider` instances: one loaded from `fodci-testing-qa-v1.pt` and one loaded from the Phase 15.4 distilled checkpoint. A single Backend prompt is sent to both providers. The primary response is returned; the candidate response is recorded only for comparison.

The promotion policy then consumes the immutable Phase 15.5 held-out report. It requires completed evaluation, accepted response quality, human approval, non-worsening readability and keyword coverage, and no increase in repeated-token rate.

## Result

The Phase 15.4 candidate was **rejected**. Phase 15.5 showed equal zero readability and keyword coverage for both models, while candidate repetition was worse (`0.6970` versus `0.3278`). Human approval was also absent, and response quality was explicitly not accepted.

| Safety gate | Result |
|---|---|
| Candidate ran alongside primary | Passed |
| User-facing response source | Stable primary |
| Candidate promotion | Rejected |
| Human approval | Not provided |
| Stable checkpoint replaced | No |
| Online training performed | No |
| Phase 15.6 gates | Passed |

The rejection is the intended behavior, not a failure of the phase. It prevents a lower-loss but visibly worse candidate from reaching the stable runtime. The stable `fodci-testing-qa-v1.pt` checkpoint remains the default.

## Reproduction

From the repository root, with the Phase 15.4 candidate checkpoint available locally:

```bash
PYTHONPATH=src:. python scripts/run_phase156_shadow_promotion.py
```

The command writes `artifacts/evaluation/phase156_shadow_promotion.json` and fails closed if the candidate becomes eligible or if any stable-runtime preservation gate fails.
