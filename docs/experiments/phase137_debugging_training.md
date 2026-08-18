# Phase 13.7 — Debugging & Root Cause Analysis Training

> This is a bounded specialization experiment. It validates transfer from the Phase 13.6 checkpoint to traceback analysis, root-cause reasoning, safe repair, and verification; it does not claim autonomous debugging competence.

## Specialization scope

The dataset contains **32 training records** and **8 validation records** covering traceback reading, root-cause isolation, minimal repairs, regression testing, safe error handling, and completion verification. The validation documents are separate from the training documents and are selected deterministically by sorted document identity.

## Reproducibility

| Field | Value |
|---|---|
| Base checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-rest-api-v1.pt` |
| Base model version | `fodci-rest-api-v1` |
| Specialist checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-debugging-v1.pt` |
| Specialist model version | `fodci-debugging-v1` |
| Dataset version | `debugging-root-cause-specialist-v1` |
| Dataset SHA-256 | `sha256:7dff8b06d89a5e2c345048a7ea3682350fbdc55cef82ca259287d8fcd5997db8` |
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
| Training seconds | 1.2049 |
| Global step | 12 |

## Before/after validation

| Metric | Phase 13.6 base | After Debugging specialization |
|---|---:|---:|
| Validation loss | 3.808188123 | 3.222222894 |
| Response loss | 3.808188123 | 3.222222894 |
| Perplexity | 45.068706 | 25.083817 |
| Evaluation examples | 8 | 8 |
| Response tokens | 773 | 773 |

## Validation gates

| Gate | Result |
|---|---|
| Base checkpoint version is Phase 13.6 | `True` |
| Specialist dataset produced examples | `True` |
| Train/validation split non-empty | `True` |
| Base evaluation finite | `True` |
| Training loss finite | `True` |
| Specialist checkpoint exists | `True` |
| Specialist checkpoint reload succeeds | `True` |
| Parameters changed | `True` |
| All gates passed | `True` |

The objective loss result validates the specialization pipeline and must be read together with the held-out generation benchmark at `training_data/debugging/evaluation/phase_137.jsonl`. Keyword coverage is a conservative proxy and is not a semantic judge.
