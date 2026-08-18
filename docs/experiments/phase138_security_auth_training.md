# Phase 13.8 — Security & Authentication Patterns Training

> This is a bounded specialization experiment. It validates transfer from the Phase 13.7 checkpoint to JWT, OAuth2, password hashing, and security middleware patterns; it does not claim production security certification or complete threat-model coverage.

## Specialization scope

The dataset contains **32 training records** and **8 validation records** covering JWT validation, OAuth2 flows, password hashing, and authentication middleware. The curriculum emphasizes fail-closed behavior, least privilege, redaction, bounded lifetimes, and tenant-aware authorization. The validation documents are separate from the training documents and are selected deterministically by sorted document identity.

## Reproducibility

| Field | Value |
|---|---|
| Base checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-debugging-v1.pt` |
| Base model version | `fodci-debugging-v1` |
| Specialist checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-security-auth-v1.pt` |
| Specialist model version | `fodci-security-auth-v1` |
| Dataset version | `security-auth-specialist-v1` |
| Dataset SHA-256 | `sha256:0544a60f07b7c02a75040a14c1a426ad303879945fb19b818e57c8511deb6552` |
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
| Training seconds | 1.2032 |
| Global step | 12 |

## Before/after validation

| Metric | Phase 13.7 base | After Security specialization |
|---|---:|---:|
| Validation loss | 3.282491785 | 3.011269321 |
| Response loss | 3.282491785 | 3.011269321 |
| Perplexity | 26.642076 | 20.313167 |
| Evaluation examples | 10 | 10 |
| Response tokens | 1013 | 1013 |

## Validation gates

| Gate | Result |
|---|---|
| Base checkpoint version is Phase 13.7 | `True` |
| Specialist dataset produced examples | `True` |
| Train/validation split non-empty | `True` |
| Base evaluation finite | `True` |
| Training loss finite | `True` |
| Specialist checkpoint exists | `True` |
| Specialist checkpoint reload succeeds | `True` |
| Parameters changed | `True` |
| All gates passed | `True` |

The objective loss result validates the specialization pipeline and must be read together with the held-out generation benchmark at `training_data/security_auth/evaluation/phase_138.jsonl`. Keyword coverage is a conservative proxy and is not a semantic judge; security correctness still requires threat modeling, implementation review, and execution-aware tests.
