# Phase 13.3 — Stage 1 Training & Pipeline Validation

> This is a bounded CPU training experiment. It validates the engineering pipeline and does not claim general language capability or production readiness.

## Reproducibility

| Field | Value |
|---|---|
| Model version | `fodci-stage1-v1` |
| Parameters | `11,424,400` |
| Dataset version | `stage1-fundamentals-v1` |
| Dataset SHA-256 | `sha256:aa0ca2a8308b6c75be671f750c4eaef78ee21483204f4d34ce349410a6bb449e` |
| Seed | `2026` |
| Device | `cpu` |
| Checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-stage1-v1.pt` |

## Dataset split

| Metric | Value |
|---|---:|
| Documents | 20 |
| Train documents | 16 |
| Validation documents | 4 |
| Train examples | 18 |
| Validation examples | 4 |

## Training configuration

| Field | Value |
|---|---:|
| Epochs | 1 |
| Maximum steps | 4 |
| Batch size | 2 |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Training time (seconds) | 0.7706 |
| Global step | 4 |

## Before/after validation loss

| Metric | Random baseline | After Stage 1 training |
|---|---:|---:|
| Validation loss | 9.418322181 | 7.215589296 |
| Response loss | 9.418322181 | 7.215589296 |
| Perplexity | 12311.907733 | 1360.475155 |
| Evaluation examples | 2 | 2 |
| Response tokens | 647 | 647 |

| Comparison | Value |
|---|---:|
| Loss improvement | 2.202732885 |
| Relative loss improvement | 23.3877% |
| Perplexity improvement | 10951.432578 |
| Relative perplexity improvement | 88.9499% |

## Validation gates

| Gate | Result |
|---|---|
| Dataset produced examples | `True` |
| Train/validation split non-empty | `True` |
| Finite training loss | `True` |
| Validation loss available | `True` |
| Checkpoint exists | `True` |
| Checkpoint reload succeeds | `True` |
| Parameters changed | `True` |
| Pipeline validation | `True` |

The workflow validates dataset loading, response-only masking, model forward pass, loss calculation, backpropagation, optimizer updates, checkpoint writing, checkpoint compatibility, and validation measurement. It deliberately does not run generation or modify the normal interactive agent runtime.
