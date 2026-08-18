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


## Phase 12.4 Parallel Tool Execution Workflows

Phase 12.4 does not introduce a standalone production CLI script; parallel tool execution and scheduling are integrated directly into `AutonomousToolLoop` and `ToolScheduler`.

The execution engine uses `ToolExecutionProfile` definitions and `ToolScheduler` to group independent read-only tool calls into parallel batches. When `parallel_execution_enabled=True` (the default) and multiple read-only tool calls are requested, the autonomous loop executes them concurrently via `ThreadPoolExecutor` (bounded by `max_parallel_tools`), tracking concurrency metrics in `ParallelMetrics`.

Developers can configure concurrency limits via `AutonomousLoopConfig(parallel_execution_enabled=True, max_parallel_tools=4)` and inspect `AutonomousLoopResult.parallel_metrics` after execution. When disabled or when mutating tools are invoked, the scheduler automatically forces safe sequential execution.

All safety boundaries are fully preserved: no shell execution, no network access, strict read-only tool registry preservation, immutable result records, and thread-safe memory and budget updates.


## Phase 12.5 Better Error Recovery Workflows

Phase 12.5 integrates diagnostic-driven error recovery into the autonomous tool loop. When any tool call, shell command, or test execution fails, `ErrorClassifier` normalizes the output into structured categories (such as `TEST_FAILURE`, `FILE_ERROR`, `DEPENDENCY_ERROR`, `RUNTIME_ERROR`, `TIMEOUT`), computes a stable error signature, and consults `RecoverabilityPolicy` to choose a non-blind recovery strategy (`INSPECT_FILE`, `REPLAN`, `VERIFY`, etc.).

Developers can configure and inspect recovery limits (`max_recovery_attempts`, `max_identical_failures`) and recovery history within autonomous loop configurations. Parallel tool execution failures are processed independently so that successful parallel results are preserved.


## Phase 12.6 Advanced Memory Workflows

Phase 12.6 integrates persistent, scoped, and ranked memory into the agent architecture. `AdvancedMemorySystem` manages memory records with rich metadata (provenance, confidence, scope, importance, status).

Developers can initialize the memory system with a persistent JSON store path (`AdvancedMemorySystem(store_path)`), add verified solutions or error resolutions, and retrieve context-aware candidate memories during agent tasks. All operations maintain local disk persistence via atomic temporary file replacement (`fsync` + `os.replace`), strict project isolation, and complete zero-dependency operation.


## Phase 12.7 Multi-Agent Architecture Workflows

Phase 12.7 integrates multi-agent orchestration into the backend engineering agent. `AgentOrchestrator` consumes a set of dependency-aware `SubTask` items, resolves specialized agents via `AgentRegistry`, maintains shared `TaskState`, and persists successful solution records into `AdvancedMemorySystem`.

Developers can instantiate `AgentOrchestrator(registry, memory_system)` and execute multi-agent workflows programmatically or integrate them with custom task definitions, ensuring robust local execution without external API dependencies.


## Phase 12.8 Advanced Autonomy & Control Workflows

Phase 12.8 finalizes the Fodci AI Backend Engineering Agent with the `AutonomyController`. It provides robust execution budgets, loop detection, explicit task lifecycles, checkpoints, and human control hooks (`pause`, `resume`, `cancel`).

Developers can instantiate `AutonomyController(orchestrator, memory_system, budget)` to execute end-to-end backend engineering tasks with guaranteed bounded termination, complete observability, and full regression test coverage (1025 tests).


## Phase 13 Training & Curriculum Workflows

Phase 13 introduces dedicated scripts for curriculum-based model specialization and benchmarking:
- `generate_stage1_data.py`: Generates high-quality instruction dataset records for Stage 1 (Backend Fundamentals).
- `benchmark_stage1.py`: Runs baseline evaluation and benchmarks model generation against stage-specific test questions.
- `train_stage1.py`: Executes end-to-end training using `FodciTrainer` and `InstructionDatasetPipeline`, producing validated checkpoints.


## Phase 13.2 — Benchmark Suite & Baseline Evaluation

`benchmark_stage1.py` runs the held-out Stage 1 benchmark against the current local checkpoint without training or changing model weights. The default protocol uses the approximately 11.4M-parameter `fodci-tiny-v1` checkpoint, CPU-only greedy decoding, seed `2026`, the fixed instruction/input/response prompt template, and 32 generated tokens per question.

```text
PYTHONPATH=src python scripts/benchmark_stage1.py
```

The benchmark dataset is `training_data/fundamentals/evaluation/stage_01.jsonl`. It is deliberately separate from the training split and contains 24 unique, versioned records across backend concepts, HTTP, REST, security, and architecture. The runner validates the JSONL schema and rejects duplicate IDs or questions before loading the model.

The command writes `artifacts/evaluation/stage1_baseline.json` and `artifacts/evaluation/stage1_baseline.md`. Reports include the run ID, model parameter count, checkpoint and dataset SHA-256 fingerprints, protocol identity, per-question outputs, aggregate metrics, and category-level metrics. The deterministic keyword-coverage score is a conservative proxy for concept coverage; it is not a semantic judge. A baseline with empty outputs or zero keyword coverage is a valid diagnostic result and should be preserved rather than hidden. Future checkpoints must be evaluated with the same dataset, prompt, seed, decoding rule, and thresholds.


## Phase 13.3 — Stage 1 Training & Pipeline Validation

`train_stage1.py` runs a bounded local CPU training experiment on `training_data/fundamentals`. It reserves the final 20 percent of sorted instruction documents for validation, evaluates a fresh baseline, trains for four maximum optimizer steps by default, saves `artifacts/checkpoints/fodci-stage1-v1.pt`, reloads that checkpoint, and compares before/after validation loss.

```text
PYTHONPATH=src python scripts/train_stage1.py
```

Optional limits can be supplied explicitly, for example `--epochs 1 --max-steps 4 --batch-size 2 --learning-rate 3e-4`. The workflow validates that examples exist, both partitions are non-empty, training loss is finite, validation loss is available, parameters changed, the checkpoint exists, and checkpoint reload reproduces the trained validation loss. It writes `artifacts/evaluation/stage1_training.json` and the tracked report `docs/experiments/phase133_stage1_training.md`.

This experiment validates dataset loading, response-only masking, forward/backward execution, optimizer updates, checkpoint compatibility, and objective evaluation. It does not perform generation, does not modify the normal interactive `fodci` command, and does not establish that the model is conversationally capable.


## Phase 13.4 — Python for Backend Specialist

`generate_phase134_data.py` creates the reviewed Python backend specialist corpus under `training_data/python_backend`, with separate `train`, `validation`, and held-out `evaluation/phase_134.jsonl` outputs. The curriculum is balanced across type hints, async patterns, Pydantic, and error handling.

```text
python scripts/generate_phase134_data.py
```

`train_phase134_python_backend.py` continues from the verified Phase 13.3 checkpoint and runs a bounded CPU specialization experiment. It defaults to 12 maximum optimizer steps, batch size two, learning rate `2e-4`, response-only masking, and a deterministic validation split.

```text
PYTHONPATH=src python scripts/train_phase134_python_backend.py
```

The workflow writes `artifacts/checkpoints/fodci-python-backend-v1.pt`, `artifacts/evaluation/phase134_python_backend_training.json`, and `docs/experiments/phase134_python_backend_training.md`. It verifies base-checkpoint lineage, non-empty partitions, finite loss, parameter changes, checkpoint compatibility, and reload consistency.

The held-out specialist benchmark uses the shared runner:

```text
PYTHONPATH=src python scripts/benchmark_stage1.py \
  --dataset training_data/python_backend/evaluation/phase_134.jsonl \
  --checkpoint artifacts/checkpoints/fodci-python-backend-v1.pt \
  --model-version fodci-python-backend-v1 \
  --run-prefix phase134-python-backend \
  --report artifacts/evaluation/phase134_python_backend_benchmark.json \
  --markdown docs/experiments/phase134_python_backend_benchmark.md
```

The benchmark remains a conservative keyword-coverage proxy. In the bounded run, the objective validation loss improved while the generation output remained empty, so the result validates the training path but does not establish conversational or semantic programming ability.


## Phase 13.5 — SQL & Database Reasoning

`generate_phase135_data.py` creates the SQL and database reasoning corpus under `training_data/sql_database`, with separate train, validation, and held-out evaluation data. The curriculum covers SQL querying, joins and aggregation, schema design, constraints, indexes, transactions, concurrency, and migrations.

```text
python scripts/generate_phase135_data.py
```

`train_phase135_sql_database.py` continues from `artifacts/checkpoints/fodci-python-backend-v1.pt` and writes the SQL specialist checkpoint `artifacts/checkpoints/fodci-sql-database-v1.pt`. The default bounded run uses CPU, one epoch, twelve maximum steps, batch size two, learning rate `2e-4`, and a 64-token training window chosen to retain every short SQL instruction record.

```text
PYTHONPATH=src python scripts/train_phase135_sql_database.py
```

The workflow writes `artifacts/evaluation/phase135_sql_database_training.json` and `docs/experiments/phase135_sql_database_training.md`. It validates checkpoint lineage, dataset coverage, finite objective loss, parameter changes, checkpoint existence, and reload consistency.

The held-out SQL benchmark is run with:

```text
PYTHONPATH=src python scripts/benchmark_stage1.py \
  --dataset training_data/sql_database/evaluation/phase_135.jsonl \
  --checkpoint artifacts/checkpoints/fodci-sql-database-v1.pt \
  --model-version fodci-sql-database-v1 \
  --run-prefix phase135-sql-database \
  --report artifacts/evaluation/phase135_sql_database_benchmark.json \
  --markdown docs/experiments/phase135_sql_database_benchmark.md
```

The benchmark is a deterministic keyword-coverage and non-empty-output diagnostic. It must not be interpreted as a semantic SQL judge; objective loss, held-out tasks, and later execution-aware evaluation remain separate evidence.


## Phase 13.6 — RESTful API Design & Implementation

`generate_phase136_data.py` creates the RESTful API specialist corpus under `training_data/rest_api`, with separate train, validation, and held-out evaluation data. The curriculum covers resource modeling, HTTP semantics, pagination, filtering, versioning, OpenAPI, errors, service boundaries, idempotency, and contract testing.

```text
python scripts/generate_phase136_data.py
```

`train_phase136_rest_api.py` continues from `artifacts/checkpoints/fodci-sql-database-v1.pt` and writes `artifacts/checkpoints/fodci-rest-api-v1.pt`. The default bounded run uses CPU, one epoch, twelve maximum steps, batch size two, learning rate `2e-4`, and the same short-record-safe training window used by the preceding specialist stages.

```text
PYTHONPATH=src python scripts/train_phase136_rest_api.py
```

The workflow writes `artifacts/evaluation/phase136_rest_api_training.json` and `docs/experiments/phase136_rest_api_training.md`. It validates checkpoint lineage, dataset coverage, finite objective loss, parameter changes, checkpoint existence, and reload consistency.

The held-out REST benchmark is run with:

```text
PYTHONPATH=src python scripts/benchmark_stage1.py \
  --dataset training_data/rest_api/evaluation/phase_136.jsonl \
  --checkpoint artifacts/checkpoints/fodci-rest-api-v1.pt \
  --model-version fodci-rest-api-v1 \
  --run-prefix phase136-rest-api \
  --report artifacts/evaluation/phase136_rest_api_benchmark.json \
  --markdown docs/experiments/phase136_rest_api_benchmark.md
```

The benchmark is a deterministic keyword-coverage and non-empty-output diagnostic. It is not a substitute for schema validation, contract tests, or execution-aware API evaluation.


## Phase 13.7 — Debugging & Root Cause Analysis

`generate_phase137_data.py` creates the debugging specialist corpus under `training_data/debugging`, with 32 training records, 8 validation records, and 8 held-out benchmark records. The balanced curriculum covers traceback reading, root-cause isolation, minimal repair, and verification.

```text
python scripts/generate_phase137_data.py
```

`train_phase137_debugging.py` continues from `artifacts/checkpoints/fodci-rest-api-v1.pt` and writes `artifacts/checkpoints/fodci-debugging-v1.pt`. The default CPU-only run is bounded to twelve optimizer steps, uses response-only loss masking and a short-record-safe context window, and validates lineage, non-empty splits, finite loss, parameter changes, checkpoint existence, and reload consistency.

```text
PYTHONPATH=src python scripts/train_phase137_debugging.py
```

The workflow writes `artifacts/evaluation/phase137_debugging_training.json` and the tracked report `docs/experiments/phase137_debugging_training.md`. The held-out debugging benchmark is run with:

```text
PYTHONPATH=src python scripts/benchmark_stage1.py \
  --dataset training_data/debugging/evaluation/phase_137.jsonl \
  --checkpoint artifacts/checkpoints/fodci-debugging-v1.pt \
  --model-version fodci-debugging-v1 \
  --run-prefix phase137-debugging \
  --report artifacts/evaluation/phase137_debugging_benchmark.json \
  --markdown docs/experiments/phase137_debugging_benchmark.md
```

The benchmark uses deterministic greedy decoding and keyword coverage as a conservative diagnostic. A non-empty rate of 1.0 with zero keyword coverage is valid evidence that the checkpoint emits text but has not yet demonstrated reliable debugging semantics; it must not be presented as successful autonomous repair.

## Phase 13.8 — Security & Authentication Patterns

`generate_phase138_data.py` creates the security specialist corpus under `training_data/security_auth`, with 32 training records, 8 validation records, and 8 held-out benchmark records. The balanced curriculum covers JWT validation, OAuth2 flows, password hashing, and authentication middleware.

```text
python scripts/generate_phase138_data.py
```

`train_phase138_security_auth.py` continues from `artifacts/checkpoints/fodci-debugging-v1.pt` and writes `artifacts/checkpoints/fodci-security-auth-v1.pt`. The default CPU-only run is bounded to twelve optimizer steps, uses response-only loss masking and a short-record-safe context window, and validates lineage, non-empty splits, finite loss, parameter changes, checkpoint existence, and reload consistency.

```text
PYTHONPATH=src python scripts/train_phase138_security_auth.py
```

The workflow writes `artifacts/evaluation/phase138_security_auth_training.json` and the tracked report `docs/experiments/phase138_security_auth_training.md`. The held-out security benchmark is run with:

```text
PYTHONPATH=src python scripts/benchmark_stage1.py \
  --dataset training_data/security_auth/evaluation/phase_138.jsonl \
  --checkpoint artifacts/checkpoints/fodci-security-auth-v1.pt \
  --model-version fodci-security-auth-v1 \
  --run-prefix phase138-security-auth \
  --report artifacts/evaluation/phase138_security_auth_benchmark.json \
  --markdown docs/experiments/phase138_security_auth_benchmark.md
```

The benchmark uses deterministic greedy decoding and keyword coverage as a conservative diagnostic. It does not establish that the model can implement secure authentication, and security claims require threat modeling, code review, and execution-aware tests.

## Phase 13.9 — Testing & Quality Assurance

`generate_phase139_data.py` creates the testing and quality assurance specialist corpus under `training_data/testing_qa`, with 32 training records, 8 validation records, and 8 held-out benchmark records. The balanced curriculum covers Pytest unit tests, integration tests, fixtures and test doubles, and code coverage.

```text
python scripts/generate_phase139_data.py
```

`train_phase139_testing_qa.py` continues from `artifacts/checkpoints/fodci-security-auth-v1.pt` and writes `artifacts/checkpoints/fodci-testing-qa-v1.pt`. The default CPU-only run is bounded to twelve optimizer steps, uses response-only loss masking and a short-record-safe context window, and validates lineage, non-empty splits, finite loss, parameter changes, checkpoint existence, and reload consistency.

```text
PYTHONPATH=src python scripts/train_phase139_testing_qa.py
```

The workflow writes `artifacts/evaluation/phase139_testing_qa_training.json` and the tracked report `docs/experiments/phase139_testing_qa_training.md`. The held-out QA benchmark is run with:

```text
PYTHONPATH=src python scripts/benchmark_stage1.py \
  --dataset training_data/testing_qa/evaluation/phase_139.jsonl \
  --checkpoint artifacts/checkpoints/fodci-testing-qa-v1.pt \
  --model-version fodci-testing-qa-v1 \
  --run-prefix phase139-testing-qa \
  --report artifacts/evaluation/phase139_testing_qa_benchmark.json \
  --markdown docs/experiments/phase139_testing_qa_benchmark.md
```

The benchmark uses deterministic greedy decoding and keyword coverage as a conservative diagnostic. It does not establish that the model can generate executable tests or guarantee meaningful coverage; those claims require running and reviewing actual test suites.
