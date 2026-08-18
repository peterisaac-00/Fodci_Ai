# Fodci Stage 1 Baseline Evaluation

> This report is a pre-training baseline for the approximately 11M-parameter Fodci model. It must be preserved before comparing later checkpoints.

## Run Identity

- **Run ID:** `phase137-debugging-c11bf6e8d87bf7fc`
- **Model:** `fodci-debugging-v1`
- **Parameters:** `11,424,400`
- **Checkpoint:** `artifacts/checkpoints/fodci-debugging-v1.pt`
- **Dataset:** `training_data/debugging/evaluation/phase_137.jsonl`
- **Dataset records:** `8`
- **Seed:** `2026`
- **Maximum generated tokens:** `32`

## Aggregate Metrics

| Metric | Value |
|---|---:|
| Passed tasks | 0 / 8 |
| Keyword pass rate | 0.00% |
| Non-empty response rate | 100.00% |
| Average keyword coverage | 0.00% |
| Average generated tokens | 32.00 |

## Category Metrics

| Category | Items | Pass rate | Non-empty rate | Keyword coverage |
|---|---:|---:|---:|---:|
| repair | 2 | 0.00% | 100.00% | 0.00% |
| root_cause | 2 | 0.00% | 100.00% | 0.00% |
| traceback_reading | 2 | 0.00% | 100.00% | 0.00% |
| verification | 2 | 0.00% | 100.00% | 0.00% |

## Interpretation

The keyword score is a deterministic proxy for Stage 1 concept coverage; it is not a substitute for human review or a semantic judge. Future checkpoints must use the same dataset, prompt template, seed, decoding rule, and scoring thresholds for a valid comparison.
