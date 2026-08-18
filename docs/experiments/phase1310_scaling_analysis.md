# Phase 13.10 — Model Scaling Analysis

> This is a bounded CPU experiment. The approximately 48M-parameter candidate is experimental only; the default Fodci runtime and all existing checkpoints remain on the 11.4M-parameter architecture.

## Objective

The experiment measures whether a larger configuration is technically feasible on CPU and whether the available evidence justifies replacing the default model. It does not claim that parameter count alone produces better language, code, security, or testing behavior.

## Configurations

| Field | Default 11.4M | Scaled candidate |
|---|---:|---:|
| Model version | `fodci-testing-qa-v1` | `fodci-scaling-48m-experimental-v1` |
| Parameters | 11,424,400 | 47,877,840 |
| Hidden size | 320 | 608 |
| Transformer layers | 4 | 8 |
| Attention heads | 5 | 8 |
| Feed-forward size | 1280 | 2432 |
| Context length | 256 | 256 |
| Parameter multiplier | 1.00× | 4.19× |

## Resource and execution measurements

| Metric | Default 11.4M | Scaled candidate |
|---|---:|---:|
| Forward/backward loss | 4.826046 | 9.229577 |
| Forward/backward seconds | 0.0715 | 0.2957 |
| RSS after backward (MB) | 804.90 | 1163.18 |
| Short training steps | 2 | 2 |
| Short training seconds | 0.7266 | 2.8480 |
| Gradients/loss finite | `True` / `True` | `True` / `True` |

## Validation loss evidence

| Metric | Default 11.4M checkpoint | Scaled candidate after short run |
|---|---:|---:|
| Validation loss | 2.864019141 | 6.863537566 |
| Evaluation examples | 10 | 10 |
| Dataset version | `testing-qa-specialist-v1` | `testing-qa-specialist-v1` |

The scaled candidate starts from random initialization because the 11.4M checkpoint is not shape-compatible with the larger configuration. Therefore the loss comparison is diagnostic, not a fair capability comparison. A valid quality comparison requires a larger-model training run with the same data, protocol, compute budget, and held-out tasks.

## Decision

> **Decision: retain the 11.4M model as the default.**

The scaled candidate is technically runnable, but the current evidence does not demonstrate a semantic or benchmark advantage. The candidate checkpoint is intentionally not saved or wired into the normal runtime. A future scaling decision should require equivalent specialist training, repeatable benchmark gains, acceptable CPU/memory budgets, and execution-aware task improvements.

## Reproducibility

| Field | Value |
|---|---|
| Base checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-testing-qa-v1.pt` |
| Base checkpoint SHA-256 | `sha256:3af5d2b6009f5a0fd0ff98644d9666bd0c30f0dfe8994f91524ae6df11433bfa` |
| Dataset SHA-256 | `sha256:e35fd796050a8f2753240f8de5441d194ac5cead1888ed52b71dd78c308e2ecb` |
| Seed | `2026` |
| CPU threads | `1` |
| Scaled checkpoint saved | `False` |
