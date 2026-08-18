# Phase 13.4 — Python for Backend Specialist Training

> This is a bounded specialization experiment. It validates transfer from the Stage 1 checkpoint to Python backend patterns; it does not claim general programming competence.

## Specialization scope

The dataset contains **32 training records** and **8 validation records** covering four balanced areas: Python type hints, asynchronous backend patterns, Pydantic validation, and robust error handling. The validation documents are separate from the training documents and are selected deterministically by sorted document identity.

## Reproducibility

| Field | Value |
|---|---|
| Base checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-stage1-v1.pt` |
| Base model version | `fodci-stage1-v1` |
| Specialist checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-python-backend-v1.pt` |
| Specialist model version | `fodci-python-backend-v1` |
| Dataset version | `python-backend-specialist-v1` |
| Dataset SHA-256 | `sha256:2818c765317c99cd21d916e3bb6789d92f3b6f221f0b57053a280030b5c0f9ba` |
| Seed | `2026` |
| Parameters | `11,424,400` |

## Training configuration

| Field | Value |
|---|---:|
| Device | `cpu` |
| Epochs | 1 |
| Maximum steps | 12 |
| Batch size | 2 |
| Learning rate | 0.0002 |
| Weight decay | 0.01 |
| Training seconds | 1.5716 |
| Global step | 12 |

## Before/after validation

| Metric | Stage 13.3 base | After Python specialization |
|---|---:|---:|
| Validation loss | 7.383549168 | 6.215177594 |
| Response loss | 7.383549168 | 6.215177594 |
| Perplexity | 1609.291290 | 500.284829 |
| Evaluation examples | 3 | 3 |
| Response tokens | 222 | 222 |

## Validation gates

| Gate | Result |
|---|---|
| Base checkpoint version is Stage 13.3 | `True` |
| Specialist dataset produced examples | `True` |
| Train/validation split non-empty | `True` |
| Base evaluation finite | `True` |
| Training loss finite | `True` |
| Specialist checkpoint exists | `True` |
| Specialist checkpoint reload succeeds | `True` |
| Parameters changed | `True` |
| All gates passed | `True` |

The objective loss result validates the specialization pipeline and must be read together with the held-out generation benchmark at `training_data/python_backend/evaluation/phase_134.jsonl`. Keyword coverage is a conservative proxy and is not a semantic judge.
