# Phase 13.10 — Model Scaling Analysis

> This is a bounded CPU experiment. The approximately 26M-parameter candidate is experimental only; the default Fodci runtime and all existing checkpoints remain on the 11.4M-parameter architecture.

## Objective

The experiment performs a bounded CPU training run on a larger configuration and measures whether the resulting evidence justifies replacing the default model. It does not claim that parameter count alone produces better language, code, security, or testing behavior.

## Configurations

| Field | Default 11.4M | Scaled candidate |
|---|---:|---:|
| Model version | `fodci-testing-qa-v1` | `fodci-scaling-25m-experimental-v1` |
| Parameters | 11,424,400 | 25,985,488 |
| Hidden size | 320 | 448 |
| Transformer layers | 4 | 7 |
| Attention heads | 5 | 7 |
| Feed-forward size | 1280 | 1792 |
| Context length | 256 | 256 |
| Scaled checkpoint | not applicable | `True` |
| Parameter multiplier | 1.00× | 2.27× |

## Resource and execution measurements

| Metric | Default 11.4M | Scaled candidate |
|---|---:|---:|
| Forward/backward loss | 4.826046 | 9.223847 |
| Forward/backward seconds | 0.0641 | 0.1438 |
| RSS after backward (MB) | 805.36 | 1060.59 |
| Short training steps | 4 | 4 |
| Short training seconds | 0.9835 | 2.1663 |
| Gradients/loss finite | `True` / `True` | `True` / `True` |

## Validation loss evidence

| Metric | Default 11.4M checkpoint | Scaled candidate after short run |
|---|---:|---:|
| Validation loss | 2.864019141 | 6.927106306 |
| Evaluation examples | 10 | 10 |
| Dataset version | `testing-qa-specialist-v1` | `testing-qa-specialist-v1` |

The scaled candidate starts from random initialization because the 11.4M checkpoint is not shape-compatible with the larger configuration. It then completes a bounded four-step CPU training run, saves an experimental checkpoint, and passes reload validation. The loss comparison is still not a fair capability comparison because the larger model has not received matched full-stage training. A valid quality comparison requires the same data, protocol, compute budget, and held-out tasks for both models.

## Decision

> **Decision: retain the 11.4M model as the default.**

The scaled candidate completed the bounded training and checkpoint-reload gates, but the current evidence does not demonstrate a semantic or benchmark advantage. The candidate checkpoint is saved only as an experimental artifact and is not wired into the normal runtime. A future scaling decision should require equivalent specialist training, repeatable benchmark gains, acceptable CPU/memory budgets, and execution-aware task improvements.

## Reproducibility

| Field | Value |
|---|---|
| Base checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-testing-qa-v1.pt` |
| Base checkpoint SHA-256 | `sha256:3af5d2b6009f5a0fd0ff98644d9666bd0c30f0dfe8994f91524ae6df11433bfa` |
| Dataset SHA-256 | `sha256:e35fd796050a8f2753240f8de5441d194ac5cead1888ed52b71dd78c308e2ecb` |
| Seed | `2026` |
| CPU threads | `1` |
| Scaled checkpoint saved | `True` |
| Validation reload delta | `0.000000000000` |
| All scaling gates passed | `True` |
