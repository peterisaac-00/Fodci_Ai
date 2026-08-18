# Scripts

This directory is reserved for small, reviewed project-maintenance scripts. Phase 0 intentionally adds no operational automation because the project does not yet execute tools, manage environments, or run an agent loop.


## Phase 11.2 training dataset

`run_phase112_training_dataset.py` consumes only an existing local Experience Record store and writes the deterministic training artifact directory. It never creates source experiences, generates synthetic examples, loads a model, tokenizes data, trains, or changes weights.

```text
python scripts/run_phase112_training_dataset.py \
  --experience-store path/to/experience_records.json \
  --output artifacts/training/dataset-v1 \
  --version dataset-v1 \
  --seed 2026
```

The command reports the source/accepted/rejected/duplicate counts, train/validation/test counts, version, and final fingerprint. The output contains `manifest.json`, `metadata.json`, `train.json`, `validation.json`, and `test.json`.


## Phase 11.3 offline fine-tuning

`run_phase113_fine_tuning.py` is a developer-only workflow. It is intentionally separate from the normal `fodci` Agent runtime and consumes only an existing Phase 11.2 artifact plus a compatible local base checkpoint.

```text
python scripts/run_phase113_fine_tuning.py \
  --base-checkpoint artifacts/checkpoints/fodci-tiny-v1.pt \
  --dataset-directory artifacts/training/dataset-v1 \
  --run-id candidate-v1-smoke \
  --candidate-model-version candidate-v1 \
  --epochs 1 \
  --max-steps 1 \
  --batch-size 1 \
  --device cpu \
  --output-directory artifacts/training_runs
```

The run writes `run.json`, `metrics.json`, and run-linked `initial.pt`, intermediate, and `final.pt` checkpoints under `artifacts/training_runs/<run_id>/`. A compatible Phase 11.3 checkpoint can be resumed with `--resume-checkpoint`; resume requires a new run ID and preserves `resumed_from` lineage. The output is a candidate trained artifact only and is not production acceptance.


## Phase 11.5 benchmark comparison

`run_phase115_benchmark.py` is an explicit local developer workflow. It compares a real Base checkpoint with either a real Candidate checkpoint or a verified Phase 11.4 Model Artifact under one versioned benchmark dataset and one protocol. It never trains, modifies weights, changes benchmark tasks, touches the source repository, or accepts/promotes a model.

```text
python scripts/run_phase115_benchmark.py \
  --base-checkpoint artifacts/checkpoints/fodci-tiny-v1.pt \
  --candidate-artifact artifacts/models/candidate-v1 \
  --candidate-version candidate-v1 \
  --comparison-id candidate-v1-backend-v1 \
  --report artifacts/evaluation/candidate-v1-backend-v1.txt
```

The command writes immutable raw runs and comparison metadata to the configured JSON stores. It requires real local inputs; if a Candidate artifact is unavailable, the command fails rather than fabricating benchmark scores. The benchmark dataset loader performs task validation and training-contamination checks before any model runtime is created.


## Phase 11.6 regression and acceptance

`run_phase116_acceptance.py` consumes persisted Phase 11.5 benchmark runs and comparisons. It does not rerun inference or benchmarks.

```text
python scripts/run_phase116_acceptance.py \
  --evaluation-id candidate-v1-backend-v1 \
  --comparison-store artifacts/evaluation/benchmark_comparisons.json \
  --runs-store artifacts/evaluation/benchmark_runs.json \
  --candidate-artifact artifacts/models/candidate-v1 \
  --training-dataset-fingerprint sha256:<training-dataset-fingerprint> \
  --held-out-test \
  --acceptance-store artifacts/evaluation/acceptance_reports.json \
  --human-report artifacts/evaluation/candidate-v1-acceptance.txt \
  --json-report artifacts/evaluation/candidate-v1-acceptance.json
```

A policy JSON object can be supplied with `--policy` to override documented `AcceptancePolicy` fields. The command returns exit code `0` only for `ACCEPT`, `2` for a valid evidence set that is `REJECT` or `INVALID_EVALUATION`, and `1` for missing/corrupt input stores or other command errors. Every evaluated decision is written to the immutable acceptance store when the required input records are available.

The acceptance workflow requires real persisted evidence. It does not fabricate a Candidate artifact, training configuration, benchmark result, test-set identity, or fingerprint. Acceptance never promotes the candidate or changes the normal `fodci` runtime.


## Phase 12.1 planning-aware autonomous workflow

The existing `run_fodci_autonomous_tool_loop_smoke.py` now prints a concise validated plan and final step statuses. It remains a developer smoke workflow for the explicit opt-in autonomous loop; the normal interactive `fodci` session is not redesigned.

```text
python scripts/run_fodci_autonomous_tool_loop_smoke.py
```

The planning layer performs typed task analysis, creates dependency-aware steps, validates the plan before selection, records `pending`, `in_progress`, `completed`, `failed`, `blocked`, and `skipped` states, and can trigger a bounded replan when the existing recovery policy reports a meaningful recoverable failure. It never dispatches tools itself, executes commands, modifies files, runs parallel operations, or bypasses the read-only default registry.


## Phase 12.2 codebase understanding

Phase 12.2 does not add a new production script or change the normal `fodci` command. The reusable API is `CodebaseUnderstandingBuilder.build(task, project_root)` and the autonomous-loop integration constructs it only inside the explicit bounded loop workflow. A caller may also construct `AutonomousLoopRequest` with a prebuilt `codebase_understanding` record.

The builder is read-only and bounded. It reuses existing project discovery/context, targeted UTF-8 reads, and bounded search evidence; it never executes the target project, invokes a shell, installs packages, accesses the network, modifies files, mutates Git, or changes the default read-only registry. `update_from_tool_result()` is intended for later structured observations, while `compact_summary()` is the bounded planner-facing representation.

The focused validation is kept in `tests/unit/test_phase122_codebase_understanding.py` and `tests/integration/test_phase122_codebase_understanding_integration.py`. Phase 12.3 long-context behavior is not included.


## Phase 12.3 long context

Phase 12.3 does not add a new production script or change the normal `fodci` command. `ContextManager` is initialized by `AutonomousToolLoop` from the existing loop configuration and inference tokenizer, then assembles every action, selection-failure, and completion prompt. Developers can inspect `AutonomousLoopResult.context_assembly` and its serialized `ContextMetrics` during tests or explicit loop workflows.

The manager keeps the current task, active execution state, failed-test/error evidence, and required verification information ahead of lower-priority context. It compresses large file/tool observations deterministically, preserves diagnostic tails, deduplicates repeated context, and marks file-dependent items invalidated after successful repository changes. It never runs commands, calls external APIs, stores hidden reasoning, creates parallel tool calls, or replaces the existing memory system.

The focused coverage is in `tests/unit/test_phase123_context_manager.py` and `tests/integration/test_phase123_long_context_integration.py`. Phase 12.4 and later phases remain outside this workflow.
