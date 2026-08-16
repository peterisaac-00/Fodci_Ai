# Phase 8.4 Regression Checkpoint

## Comparison context

This checkpoint compares two completed Phase 8.3 evaluation results using the explicit API `compare_evaluations`. The baseline is **Version 0.1** and the candidate is **Version 0.2**. Both use benchmark definition `1.0`, evaluation version `8.3`, scoring policy `1.0`, and the same six-task benchmark.

| Task category | Baseline | Candidate | Delta | Classification |
|---|---:|---:|---:|---|
| API endpoint | 0.70 | 0.78 | +0.08 | IMPROVED |
| Authentication | 0.72 | 0.80 | +0.08 | IMPROVED |
| Database | 0.68 | 0.74 | +0.06 | IMPROVED |
| Bug-fixing | 0.75 | 0.70 | -0.05 | REGRESSED |
| Testing | 0.66 | 0.76 | +0.10 | IMPROVED |
| Docker | 0.71 | 0.79 | +0.08 | IMPROVED |

The baseline aggregate is approximately **0.7033** and the candidate aggregate is approximately **0.7617**, for an aggregate delta of approximately **+0.0583**. Because the bug-fixing task regressed, the final classification is **`IMPROVED_WITH_REGRESSIONS`**, not plain `IMPROVED`. The comparison retains task-level and dimension-level evidence references.

## Determinism check

The exact same immutable baseline and candidate results were compared twice with the same `ComparisonConfig(epsilon=0.01)`. The test asserts that both canonical JSON serializations are byte-for-byte identical. The checkpoint contains six task comparisons in stable task-ID order and bounded evidence references.

## Scope boundary

This report compares already-produced results only. It does not execute a benchmark, run tests, modify files, access the network, mutate Git, or begin Phase 9.
