# Phase 13.5 — SQL & Database Reasoning Training

> This is a bounded specialization experiment. It validates transfer from the Phase 13.4 checkpoint to SQL and database reasoning patterns; it does not claim general database competence.

## Specialization scope

The dataset contains **32 training records** and **8 validation records** covering SQL querying, schema design, indexes, transactions, joins, constraints, and migration reasoning. The validation documents are separate from the training documents and are selected deterministically by sorted document identity.

## Reproducibility

| Field | Value |
|---|---|
| Base checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-python-backend-v1.pt` |
| Base model version | `fodci-python-backend-v1` |
| Specialist checkpoint | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-sql-database-v1.pt` |
| Specialist model version | `fodci-sql-database-v1` |
| Dataset version | `sql-database-specialist-v1` |
| Dataset SHA-256 | `sha256:c457058b30b6c2a8352bbf99dc663c4e1acfa66e582acecba3a9bd3b2ddcdcd3` |
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
| Training seconds | 1.1300 |
| Global step | 12 |

## Before/after validation

| Metric | Phase 13.4 base | After SQL specialization |
|---|---:|---:|
| Validation loss | 6.141384743 | 4.879417161 |
| Response loss | 6.141384743 | 4.879417161 |
| Perplexity | 464.696611 | 131.553967 |
| Evaluation examples | 6 | 6 |
| Response tokens | 602 | 602 |

## Validation gates

| Gate | Result |
|---|---|
| Base checkpoint version is Phase 13.4 | `True` |
| Specialist dataset produced examples | `True` |
| Train/validation split non-empty | `True` |
| Base evaluation finite | `True` |
| Training loss finite | `True` |
| Specialist checkpoint exists | `True` |
| Specialist checkpoint reload succeeds | `True` |
| Parameters changed | `True` |
| All gates passed | `True` |

The objective loss result validates the specialization pipeline and must be read together with the held-out generation benchmark at `training_data/sql_database/evaluation/phase_135.jsonl`. Keyword coverage is a conservative proxy and is not a semantic judge.
