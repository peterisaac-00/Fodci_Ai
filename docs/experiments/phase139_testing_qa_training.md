# Phase 13.9 — Testing & Quality Assurance Training

> This is a bounded specialization experiment. It validates transfer from the Phase 13.8 checkpoint to Pytest unit and integration testing, fixtures, test doubles, and code coverage reasoning; it does not claim complete quality assurance or production verification.

## Specialization scope

The dataset contains **32 training records** and **8 validation records** covering unit tests, integration tests, Pytest fixtures and test doubles, and code coverage. The curriculum emphasizes deterministic tests, public contracts, isolated boundaries, reliable cleanup, meaningful assertions, and risk-aware coverage review. The validation documents are separate from the training documents and are selected deterministically by sorted document identity.

## Reproducibility

| Field | Value |
|---|---|
| Base checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-security-auth-v1.pt` |
| Base model version | `fodci-security-auth-v1` |
| Specialist checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-testing-qa-v1.pt` |
| Specialist model version | `fodci-testing-qa-v1` |
| Dataset version | `testing-qa-specialist-v1` |
| Dataset SHA-256 | `sha256:e35fd796050a8f2753240f8de5441d194ac5cead1888ed52b71dd78c308e2ecb` |
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
| Training seconds | 1.1697 |
| Global step | 12 |

## Before/after validation

| Metric | Phase 13.8 base | After Testing specialization |
|---|---:|---:|
| Validation loss | 3.069860184 | 2.864019273 |
| Response loss | 3.069860184 | 2.864019273 |
| Perplexity | 21.538891 | 17.531851 |
| Evaluation examples | 10 | 10 |
| Response tokens | 994 | 994 |

## Validation gates

| Gate | Result |
|---|---|
| Base checkpoint version is Phase 13.8 | `True` |
| Specialist dataset produced examples | `True` |
| Train/validation split non-empty | `True` |
| Base evaluation finite | `True` |
| Training loss finite | `True` |
| Specialist checkpoint exists | `True` |
| Specialist checkpoint reload succeeds | `True` |
| Parameters changed | `True` |
| All gates passed | `True` |

The objective loss result validates the specialization pipeline and must be read together with the held-out generation benchmark at `training_data/testing_qa/evaluation/phase_139.jsonl`. Keyword coverage is a conservative proxy and is not a semantic judge; testing quality still requires executable suites, review, and meaningful assertions.
