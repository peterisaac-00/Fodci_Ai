# Backend Engineering Agent

> **Current status: Phase 7.5 — Bounded Self-Correction Loop; Phase 7.6+ not started.**

Backend Engineering Agent is the foundation for a future **local, terminal-based AI agent** focused on backend engineering work. The intended product will use an interchangeable local or open-weight language-model provider rather than depend on hosted OpenAI, Anthropic, or Gemini APIs.

This repository includes the complete Phase 1 CLI foundation, Phase 2.1's minimal typed LLM provider boundary, Phase 2.2's small decoder-only Transformer architecture, Phase 2.3's reversible byte-level tokenizer, Phase 2.4's local streaming dataset pipeline, Phase 2.5's CPU-friendly training engine, Phase 2.6's first real Fodci Tiny v1 training experiment, Phase 2.7's metadata-aware checkpoint manager, Phase 2.8's CPU-first evaluation pipeline, Phase 2.9's local backend-engineering coding corpus and manifest layer, Phase 2.10's local instruction-training dataset and response-masked training path, and Phase 2.11's local CPU inference API. Phase 2.12 connects that existing inference path to the official `fodci` terminal session through `FodciLocalProvider`. Phase 3.1 adds the first read-only Agent tool, `list_files`, for safe deterministic discovery of an explicitly selected project root. Phase 3.2 adds the second read-only tool, `read_file`, for bounded exact UTF-8 reading inside that root. Phase 3.3 adds the third standalone read-only tool, `search_code`, for bounded literal or explicitly enabled regex search across safe UTF-8 source files. Phase 3.4 adds `project_structure`, a bounded evidence-based structural detector for technologies, components, languages, configurations, tests, and likely entry points. Phase 3.5 adds the canonical immutable `ProjectContext` layer and builder that transforms structural facts into a compact deterministic context for future Agent reasoning. Phase 3.6 adds the first bounded read-only `AgentLoop`, a deterministic `ToolRegistry`, a strict ACTION/ARGS protocol, and structured execution results over the existing tools. Phase 4.1 adds `write_file`, a bounded atomic create-only tool that is available through an explicit opt-in registry but is not automatically used by `AgentLoop`. Phase 4.2 adds `edit_file`, a bounded atomic exact replacement tool for existing UTF-8 files, also available only through an explicit modification registry. Phase 4.3 adds `delete_file`, a regular-file-only deletion tool with no recursive behavior and explicit opt-in registry exposure. Phase 4.4 adds a reusable `SafeEditPolicy`/`SafeEditSession` infrastructure layer with immutable snapshots, bounded internal diffs, optional controlled backups, and post-mutation verification over the existing mutation tools. Phase 4.5 adds `git_diff`, a read-only bounded Git working-tree inspection tool exposed only through an explicit Git-inspection registry. Phase 4.6 adds `git_status`, a read-only bounded structured working-tree status tool reusing the same Git adapter. Phase 4.7 adds a read-only `ModificationVerifier`/`verify_modification` layer that validates explicit create/edit/delete postconditions and can compare a bounded baseline for unexpected changes. Phase 4.8 adds a conservative single-operation `ModificationTransaction`/recovery layer over `SafeEditSession`; it reuses existing atomic mutation and backup primitives, preserves user changes, and never claims unsafe multi-file rollback. Phase 5.1 adds an opt-in, low-level `run_command` foundation for explicit argv process execution with shell disabled, explicit root-contained working directories, bounded output, bounded timeouts, and structured results. Phase 5.2 adds a separate deny-by-default `CommandPolicy` layer above that executor. The policy evaluates argv, shell-bypass patterns, dangerous categories, argument paths, working directories, executable approval, environment variables, and read-only Git/package/network restrictions before any process can start. Phase 5.3 adds `ProcessManager` as a lifecycle layer for an already-approved `CommandRequest`; it handles explicit state transitions, drain-safe bounded output capture, timeout termination, reaping, cleanup, and structured process metadata without making security decisions. Phase 5.4 adds a bounded `ApplicationRunner` that resolves only evidence-backed Python/Node launch plans from existing `ProjectContext`/`ProjectStructure`, validates them through `CommandPolicy`, and executes them through `ProcessManager`. Phase 5.5 adds a bounded `TestRunner` that detects supported Python/Node test evidence, resolves deterministic test argv, validates explicit or automatic plans through the same policy, and executes them through the same process lifecycle manager. Phase 5.6 adds a read-only `TestResultParser` that consumes only the bounded raw `TestRunResult` and converts strong pytest/unittest/Jest/Vitest evidence into structured semantic outcomes. Phase 6.1 adds a deterministic, side-effect-free `Planner` that converts a user task and optionally supplied `ProjectContext` into a validated declarative `ExecutionPlan`; it does not inspect projects, select tools, execute actions, or activate the AgentLoop. Phase 6.2 adds a deterministic `ToolSelector` that maps each plan step to capabilities exposed by an explicitly supplied `ToolRegistry`; it produces declarative selection decisions only and never calls a tool. Phase 6.3 adds the first controlled, explicitly opt-in `AutonomousToolLoop` that validates strict model actions and dispatches selected tools only through the supplied `ToolRegistry`, with bounded context/history, structured observations, and a fixed emergency execution boundary. Phase 6.4 adds a deterministic, immutable `StopConditionEvaluator` that evaluates plan progress, tool results, modification/test verification evidence, blocked capabilities, safety denials, malformed actions, and the emergency bound into `DONE`, `CONTINUE`, `FAILED`, or `BLOCKED` without executing anything. Phase 6.5 adds a centralized immutable `ExecutionBudget`/`ExecutionUsage`/`ExecutionBudgetLedger` layer with configurable finite limits for iterations, tool calls, mutations, commands, tests, applications, wall time, accumulated tool/stdout/stderr bytes, context tokens, and action steps. Phase 6.6 adds an explicit-only, deterministic error-classification and recovery-decision layer over structured tool evidence. It distinguishes safety/policy failures from actionable failures, proposes only bounded `INSPECT`, `VERIFY`, or `REPLAN` actions, preserves original failures, records bounded recovery history, and never bypasses existing registries, policies, budgets, verification, or emergency bounds. Phase 6.7 adds a pure `TaskCompletionVerifier` that aggregates plan progress, tool results, verification evidence, test outcomes, recovery state, budgets, and explicit expected criteria. It independently evaluates `ACTION: FINAL`, prevents false-positive `DONE`, distinguishes complete/incomplete/blocked/failed/verification-unavailable/insufficient-evidence outcomes, and exposes bounded criterion items, evidence strength, confidence, remaining work, and blocking conditions. Phase 7.1 adds explicit-only `AutomaticTestOrchestrator` models and orchestration. It decides whether a bounded verification boundary warrants tests, reuses the existing `run_tests` capability through `ToolRegistry`, preserves raw `TestRunResult`, consumes the existing test budget before dispatch, and never diagnoses failures, edits files, retries, or reruns. Phase 7.2 adds a pure immutable `TestFailureAnalyzer` over existing `TestParseResult`/`TestRunResult` evidence. It produces bounded findings, failure locations, provenance-preserving evidence, conservative taxonomy classifications, confidence, diagnostic chains, related-failure groups, and explicitly labeled primary/derived inference. It redacts sensitive values and never executes commands, reads files, modifies the project, calls the LLM, retries, or proposes fixes. Phase 7.3 adds a pure immutable `RootCauseAnalyzer` above `TestFailureAnalysis`. It produces bounded evidence-backed root-cause hypotheses, mechanisms, supporting and contradicting evidence, alternatives, causal relations, primary/derived and cascading inferences, conservative locations, explicit confidence, and bounded causal depth. It never marks a hypothesis as confirmed, never reads the filesystem, executes commands, runs tests, calls the LLM, edits files, retries, or invokes a fix. Phase 7.4 adds an explicit-only `AutomaticFixPlanner` and `AutomaticFixOrchestrator` above the existing mutation infrastructure. A structured `FixPlan` must contain the target, exact location, change type, intended change, expected post-state, old/new UTF-8 content, evidence, risk, confidence, hypothesis ID, and affected failure IDs. Plans are rejected when RCA evidence is weak, locations are missing/ambiguous, paths are unsafe/sensitive/outside the project root, policy denies editing, scope is unsupported, or the mutation budget is exhausted. Accepted edits execute exactly once through `ModificationTransaction` and `SafeEditPolicy`, preserve user changes through existing snapshots/backups/recovery semantics, and become `FIX_VERIFIED` only after existing post-state verification succeeds. A successful mutation is never confused with test pass; tests are intentionally not rerun. Phase 7.5 adds explicit-only `BoundedSelfCorrectionLoop` orchestration over the existing test, parser, failure-analysis, RCA, automatic-fix, budget, stop, and recovery boundaries. It runs bounded `RUN_TESTS → PARSE_RESULT → ANALYZE_FAILURE → ROOT_CAUSE_ANALYSIS → APPLY_FIX → RETEST` transitions, stops immediately on PASS, permits at most one fix per attempt, uses host-configured finite `max_attempts`, fingerprints structured failure/action evidence without secrets, and stops on repeated failure, no progress, no actionable fix, policy/safety/recovery block, or budget exhaustion. It never creates a new ledger, bypasses `AutomaticTestOrchestrator`, retries blindly, or lets model/test output change limits. The model remains intentionally tiny at 11,424,400 parameters; no external LLM, pretrained component, RAG, memory, or unrestricted autonomous loop is present.

## Purpose and Long-Term Vision

The eventual agent is intended to understand a bounded software project, plan a change, read and modify code, execute commands, run tests, analyze failures, correct problems, and verify the result. Those capabilities are future work rather than promises of the current package.

```text
Understand project
        ↓
Plan
        ↓
Read and modify code
        ↓
Execute commands and tests
        ↓
Analyze and correct errors
        ↓
Verify result
```

## Architecture Direction

Phase 0 establishes a small set of independent boundaries. Future orchestration must depend on provider and subsystem interfaces, not a concrete model implementation.

```text
CLI
 ↓
Agent
 ↓
LLM Provider
 ↓
Tools
 ↓
Execution
 ↓
Evaluation
 ↓
Memory
```

The package exposes minimal typed contracts for `Agent`, `LLMProvider`, `Message`, `LLMRequest`, `LLMResponse`, `Tool`, `Memory`, and `Evaluator`. `ProviderBackedAgent` accepts an `LLMProvider` through dependency injection and delegates one request only. The isolated `backend_ai.model` package contains `FodciModel`, a configurable decoder-only Transformer with token embeddings, learned positional embeddings, causal multi-head attention, GELU feed-forward blocks, final normalization, and a language-modeling head. The official `fodci` console script composes `InteractiveSession` with `FodciLocalProvider`; the CLI itself does not import Transformer or PyTorch internals. See [the architecture notes](docs/architecture.md) for the dependency direction.

## Repository Layout

```text
.
├── src/backend_ai/
│   ├── agent/          # Agent protocol and provider-injected adapter
│   ├── application.py  # Application startup and session composition
│   ├── cli/            # Minimal console-entry boundary
│   ├── config/         # Small environment-backed settings abstraction
│   ├── core/           # Shared protocols, startup, and project context
│   ├── evaluation/     # CPU-first loss/perplexity evaluation and comparison
│   ├── inference/      # Local CPU autoregressive decoding API
│   ├── llm/            # Typed provider boundary and local Fodci adapter
│   ├── model/          # From-scratch Transformer architecture, no training
│   ├── tokenizer/      # Reversible byte-level tokenizer and tiny BPE training
│   ├── dataset/        # Local coding/instruction validation, samples, and manifests
│   ├── training/       # CPU training loop, metrics, and resumable checkpoints
│   ├── checkpoint/     # Atomic metadata-aware checkpoint management
│   ├── data/           # Small local backend-focused train/validation corpus
│   ├── memory/         # Future memory boundary
│   ├── commands/       # Command parsing and dispatch boundaries
│   ├── terminal/       # Session lifecycle, commands, and provider-backed input
│   └── tools/          # Read-only filesystem tools, project structure, and context
├── tests/
│   ├── unit/           # Foundation, CLI, model, tokenizer, dataset, and training tests
│   └── integration/    # CLI subprocess and cross-component tests
├── docs/               # Architecture, security, and experiment reports
├── scripts/            # Reviewed workflows for training, evaluation, manifests, and inference
├── .env.example
├── pyproject.toml
└── README.md
```

## Development

Use Python 3.11 or later. The base package and CLI have no runtime dependencies. The optional `model` extra adds PyTorch for the model architecture, Phase 2.5/2.6/2.10 training workflows, Phase 2.11 inference, and the Phase 2.12 local CLI provider. The official executable is `fodci`, mapped to the existing `backend_ai.cli.main:main` entry point.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "[dev,model]"
```

Run the test suite with:

```bash
pytest
```

Verify the official console entry point after installation with:

```bash
fodci
```

The command initializes the existing application configuration and logger, resolves the project root, loads `artifacts/checkpoints/fodci-tiny-v1.pt` once through `FodciLocalProvider`, enters the persistent session lifecycle, and reads normal text from stdin:

```text
You > hello
Fodci > ...
```

Input is preserved except for the line ending added by stdin. Empty input is retained and does not terminate the session. A command is recognized only when `/` is the first character; command names are case-insensitive, while arguments remain available as text. The available local commands are:

```text
/help
/exit
```

`/help` displays the registered command list and keeps the session active. `/exit` prints `Goodbye.` and requests a clean session stop. Arguments such as `/help now` and `/exit now` return usage text without triggering command behavior. An unregistered `/status` is reported as unknown without crashing. Normal text such as `Build /api/users` is passed through unchanged. EOF and Ctrl+C terminate the session cleanly.

## Tokenizer

`FodciTokenizer` uses UTF-8 bytes as a permanent lossless fallback, with optional deterministic byte-pair merges learned only from a caller-provided small corpus. Its default vocabulary is exactly 10,000 IDs, compatible with `ModelConfig.vocab_size`. Special IDs are stable: `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, and `<EOS>=3`; byte tokens start at ID 4. Encoding does not normalize, truncate, lowercase, or alter whitespace. `decode(encode(text))` reconstructs supported text exactly, including source code, indentation, URLs, JSON, and Unicode.

Tokenizer training is separate from LLM training. It accepts in-memory text only, performs no scraping or dataset collection, and produces a small versioned JSON definition through `save()` and `load()`. The Phase 2.4 dataset pipeline reads only caller-provided local files; it does not download, scrape, or silently normalize source text.

## Training engine

`FodciTrainer` is the first phase that permits the randomly initialized model to learn. It receives an existing Fodci model and re-iterable or callable train/validation sources of `TrainingExample` objects. It batches bounded streams, validates sequence lengths and vocabulary IDs, computes standard next-token categorical cross-entropy, performs backpropagation, applies optional gradient clipping, and updates parameters with AdamW. A `TrainingExample` may carry an optional boolean `loss_mask`; when present, only selected target positions contribute to the mean loss and effective token counts. The default device is CPU; `auto` selects CUDA only when available, while an explicit unavailable CUDA request fails clearly.

Each epoch reports training and validation loss, step counts, token counts, learning rate, elapsed time, and guarded perplexity. `TrainingConfig.max_steps` provides an optional hard budget in addition to epochs. Checkpoints contain only the model state, optimizer state, epoch, global step, training configuration, and relevant metrics. `resume()` restores those values and continues from the following epoch. Generated checkpoints are written under ignored artifact directories and are never part of the source repository.

The required tiny smoke run is an engineering validation of forward/backward execution, parameter updates, validation, checkpoint creation, checkpoint loading, and resume behavior. It is not evidence that Fodci has acquired useful language capability.

## Tiny Model Training experiment

Phase 2.6 runs `scripts/run_fodci_tiny_v1.py` as a Python workflow, not as a new CLI command. It uses the small, repository-local corpus under `data/fodci_tiny_v1/`, keeps train and validation directories separate, records deterministic SHA-256 dataset fingerprints, evaluates a fresh random model before optimization, and trains with the existing `FodciDatasetPipeline` and `FodciTrainer`. The official run uses CPU, a fixed seed, a conservative step budget, and the unchanged 11,424,400-parameter model.

The human-readable experiment record is [Fodci Tiny v1](docs/experiments/fodci-tiny-v1.md). Generated JSON reports and model checkpoints live under ignored `artifacts/` directories and are never committed. The experiment demonstrates an actual from-scratch parameter update on backend-focused local examples; it does not implement generation, inference, an agent, or a training CLI.

## Checkpoint management

`CheckpointManager` wraps the existing training engine with an atomic, metadata-aware storage boundary. Each checkpoint records model version, model configuration, tokenizer version, vocabulary size, context length, epoch, global step, training configuration, metrics, seed, format identifier, format version, and UTC creation time. `inspect()` reads this metadata without constructing another model; `load_model()` maps model tensors to the requested device and validates identity and structural compatibility without creating an optimizer; `load()` remains the training resume path that restores both model and optimizer state.

The manager supports `save()`, `load()`, `inspect()`, `exists()`, `list()`, `latest()`, and `best()`. Saving writes a temporary file, flushes it, and atomically replaces the destination, preventing an interrupted write from exposing a partial final checkpoint. `latest()` uses metadata progress rather than filenames, while `best()` selects the lowest recorded validation loss. Generated weights remain under ignored `artifacts/` or `checkpoints/` paths and are never committed to Git.

## Evaluation pipeline

`FodciEvaluator` measures the existing causal language-model objective without changing model parameters or optimizer state. It calls `model.eval()` and evaluates inside `torch.no_grad()`, reporting loss, perplexity, evaluation examples, evaluated tokens, and elapsed time. The evaluation source is explicit, so the random baseline and trained checkpoint can be measured on exactly the same validation split.

Phase 2.8 uses `scripts/run_fodci_evaluation.py` to compare a fresh random Fodci Tiny v1 against `artifacts/checkpoints/fodci-tiny-v1.pt`. The evaluator validates checkpoint compatibility through `CheckpointManager`, supports multiple checkpoints and best-checkpoint selection, computes loss/perplexity deltas and relative improvements, and writes the human-readable report to [Fodci Tiny v1 Evaluation](docs/experiments/fodci-tiny-v1-evaluation.md). The generated JSON report remains under ignored `artifacts/reports/`. This is an early small-scale objective evaluation, not generation, inference, intelligence, programming understanding, or production-readiness evidence.

## Phase 2.9 coding dataset

`data/fodci_coding/` is a small, repository-local corpus focused on backend engineering. It is split explicitly into `train/` and `validation/` with no shared exact content hashes. The corpus contains coherent Python backend examples, REST routing and validation, authentication and authorization, SQL and repository transactions, configuration and environment variables, background jobs, tests, backend architecture documentation, JSON/API validation, and a Dockerfile example.

`CodingDatasetManifestBuilder` composes the existing `FodciDatasetPipeline`; it does not create a second tokenizer or dataset loader. The builder records exact file paths, UTF-8 bytes, characters, tokens including EOS, training examples, language/file-type distribution, duplicate and rejected-file counts, split hashes, tokenizer version, vocabulary size, context length, and a deterministic dataset SHA-256. It also reports unsupported extensions and rejects train/validation exact-content leakage. The reproducible manifest is [fodci-coding-manifest.json](docs/datasets/fodci-coding-manifest.json), with a human-readable summary in [fodci-coding.md](docs/datasets/fodci-coding.md).

Phase 2.9 improves the corpus and its identity only. It does not start a new training run, change the Transformer, add inference or generation, integrate the CLI, or claim that Fodci has gained intelligence or useful coding ability.

## Phase 2.10 instruction training

`data/fodci_instructions/` contains a small, deterministic backend-engineering instruction dataset split into `train/` and `validation/`. Each text file uses ordinary delimiters—`### Instruction`, `### Input`, and `### Response`—because no new tokenizer special tokens were introduced. The parser rejects missing sections, empty input, empty responses, malformed ordering, unsupported files, invalid UTF-8, duplicate examples, and train/validation leakage.

`InstructionDatasetPipeline` reuses the existing tokenizer and produces regular `TrainingExample` objects with a response boundary mask. The instruction and input are conditioning context; response target tokens and their EOS boundary are the only positions included in the causal cross-entropy. This minimal response-only masking extension keeps the existing model architecture, optimizer, checkpoint manager, and evaluation infrastructure unchanged.

The reproducible instruction manifest is [fodci-instruction-manifest.json](docs/datasets/fodci-instruction-manifest.json), with a human-readable summary in [fodci-instructions.md](docs/datasets/fodci-instructions.md). The bounded CPU smoke workflow is [run_fodci_instruction_training.py](scripts/run_fodci_instruction_training.py), and its tracked experiment report is [Fodci Instruction Training](docs/experiments/fodci-instruction-training.md). The generated checkpoint and JSON report remain under ignored `artifacts/` directories.

Phase 2.10 validates data parsing, response masking, checkpoint compatibility, and before/after response-only objective metrics. It does not claim that Fodci can reliably follow arbitrary instructions, write production code, or perform generation.

## Phase 2.11 local inference

`InferenceEngine` is the first local generation boundary. It accepts an existing `FodciModel`, the existing `FodciTokenizer`, and an optional compatible checkpoint path. The engine validates checkpoint identity through `CheckpointManager`, calls `model.eval()` inside `torch.inference_mode()`, encodes the prompt without truncation, runs autoregressive next-token decoding, and decodes only the generated token IDs back to text.

The default is deterministic greedy decoding: `argmax(logits / temperature)` with `temperature=1.0`, `do_sample=False`, EOS stopping enabled, and a conservative `max_new_tokens` budget. Optional `temperature`, `top_k`, and seeded multinomial sampling are supported; invalid temperatures, invalid top-k values, vocabulary mismatches, empty prompts, and prompts exceeding model context produce clear errors. Generation stops on EOS, the new-token budget, or context-length capacity and never silently truncates the prompt.

`InferenceResult` exposes generated text, prompt and generated token counts, stopped reason, model version, checkpoint identity, and the effective configuration. The smoke workflow is [run_fodci_inference.py](scripts/run_fodci_inference.py). It uses the existing ignored `artifacts/checkpoints/fodci-tiny-v1.pt` on CPU and verifies completion for English and backend-oriented prompts. The resulting text is not evidence of intelligence or production readiness; it only validates the checkpoint → model → tokenizer → autoregressive decoding path.

## Phase 2.12 CLI integration

`FodciLocalProvider` adapts the existing `LLMRequest`/`LLMResponse` contract to `InferenceEngine.generate()`. `Application` resolves the project root, constructs the checkpoint path, creates the provider once, and injects it into `InteractiveSession`; `cli.main` remains unaware of `FodciModel`, PyTorch, tokenizer internals, checkpoint metadata, and sampling details. A session preserves system, user, and assistant messages in deterministic bounded history, never persists them to disk, and reports context-limit or inference failures explicitly without silently truncating user input.

The existing `/help` and `/exit` behavior, EOF handling, and Ctrl+C handling remain available. Normal input is rendered as `Fodci > ...`; an unavailable, malformed, or incompatible checkpoint produces a concise startup error and never falls back to random weights. The local model may produce whitespace, repetitive text, or weak responses because this is an extremely small from-scratch model trained on a tiny local corpus. Phase 2.12 success means only that `fodci` reaches the local inference pipeline.

The end-to-end test sends `Hi` followed by `/exit` to the real `fodci` subprocess and verifies provider-backed output, clean termination, and exit code `0`. No external LLM API, network access, tool invocation, file analysis, terminal command, project understanding, planning, RAG, memory, or agent loop is introduced.

Check that every package module compiles with:

```bash
python -m compileall -q src
```

For the minimal runtime package installation only, use:

```bash
python -m pip install -e .
```

## Phase 3.1 file discovery

`backend_ai.tools.list_files(project_root)` is the first concrete Agent tool. It recursively discovers regular files and directories below an explicit project root and returns a `FileDiscoveryResult` rather than a formatted string. Files expose root-relative POSIX paths, names, extensions, and byte sizes; directories are returned separately. Results include the normalized root, totals, and explicit truncation metadata.

The traversal is deterministic: entries and final result collections are ordered by normalized relative path using a platform-independent case-folded comparison with a stable tie-break. Default exclusions cover `.git`, `__pycache__`, `node_modules`, virtual environments, test/type/linter caches, `dist`, `build`, and `.eggs`. Hidden files are included by default when they are not excluded directories, so `.env.example`, `.gitignore`, and `.dockerignore` remain discoverable. Set `include_hidden=False` to exclude dot-prefixed entries; custom ignored directory names extend the defaults.

Discovery is read-only and bounded. `max_files`, `max_directories`, and `max_depth` have explicit defaults and are validated as non-negative integers. When a bound stops traversal, the result sets `truncated=True` and records `truncation_reason` as `max_files`, `max_directories`, or `max_depth`; files are never silently truncated without metadata. File contents are not read, and no file is modified, created, deleted, executed, or downloaded.

Every symbolic link is skipped, including symlinked files, directories, links outside the root, and recursive links. The tool normalizes the explicit root and rejects missing paths, non-directory roots, invalid arguments, permission failures, and filesystem errors with `ToolError` and stable `ToolErrorCode` values. Full `.gitignore` semantics are intentionally not implemented in this phase; the centralized default exclusion set is the documented policy and can be extended later without adding a dependency.

The tool layer remains separate from both the LLM and the Agent loop. `ListFilesTool` exposes stable metadata and an input schema through the existing `Tool` protocol, but `fodci` does not automatically invoke it. Phase 3.1 does not implement `read_file`, `search_code`, `ProjectContext` expansion, framework detection, project understanding, planning, tool calling, file modification, terminal execution, memory, RAG, or an autonomous Agent loop.

## Phase 3.2 read file

`backend_ai.tools.read_file(project_root, path)` reads one explicitly requested regular file through the same tool boundary. The project root is always explicit and the requested path is normally root-relative. The structured `ReadFileResult` contains `relative_path`, `file_name`, exact `content`, `encoding="utf-8"`, and `size_bytes`; the tool returns data to its caller and never prints file content or logs source bodies.

The tool preserves bytes after UTF-8 decoding exactly, including spaces, tabs, indentation, Unicode and Arabic text, punctuation, CRLF/LF line endings, and final-newline behavior. Invalid UTF-8 is rejected with `INVALID_UTF8`; no replacement or ignored decoding is used. UTF-8 BOM bytes are preserved as decoded content because no special BOM stripping is performed.

Reading is bounded by `max_bytes`, which defaults to 1 MiB and is checked before and during the binary read. A file at the limit is accepted; a file above it returns `FILE_TOO_LARGE` with the requested relative path and configured maximum. The tool reads only regular files, rejects directories and special filesystem entries with `NOT_A_FILE`, and reports missing paths, permissions, invalid arguments, filesystem failures, and paths outside the root through the shared `ToolError`/`ToolErrorCode` system.

Path normalization uses `pathlib` semantics rather than string-prefix checks. Relative `.` and `..` segments are normalized, absolute paths are allowed only when they remain inside the explicit root, Windows drive/UNC-looking paths cannot bypass the root, and mixed separators are normalized for the request. Every symlink component is rejected, including external file/directory links, internal links, broken links, and loops, matching Phase 3.1's safer skip policy.

Phase 3.2 is read-only. It does not implement `search_code`, grep/ripgrep/regex/AST search, ProjectContext expansion, framework detection, project understanding, file mutation, terminal execution, planning, memory, RAG, LLM tool-calling, or an Agent loop. The existing `fodci` interactive application is unchanged.

## Phase 3.3 code search

`backend_ai.tools.search_code(project_root, query)` is the third standalone Agent tool. It searches regular UTF-8 files under an explicit root or an optional root-relative `path` and returns an immutable `SearchCodeResult`. Each `SearchMatch` contains a normalized relative path, a 1-based line number, the exact source line without its line terminator, and 0-based Unicode column start/end positions. The result also reports matches returned, files searched, skipped-file count/reasons, and truncation metadata.

The default search is a literal substring search; regex interpretation is never implicit. Set `use_regex=True` to compile the query as a regular expression, and use `case_sensitive=False` for explicit case-insensitive matching. Invalid regex patterns return `INVALID_REGEX`. Literal mode escapes regex metacharacters, so a query such as `a.b` searches for those literal characters. Results are deterministic by normalized relative path, line number, and column position.

Search reuses Phase 3.1's centralized default exclusions and skips `.git`, `__pycache__`, `node_modules`, virtual environments, caches, `dist`, `build`, and `.eggs`, while hidden files such as `.env.example` remain searchable. Symlinks, special filesystem entries, and invalid UTF-8 files are not searched; invalid UTF-8 is reported through `skipped_reasons` rather than corrupting or aborting a project-wide search. Full `.gitignore` semantics are not claimed.

Search is bounded by `max_results` (default 100, maximum 10,000), `max_file_bytes` (default 1 MiB, maximum 16 MiB), and a bounded traversal depth/directory policy. Oversized files are skipped without partial results, and a result records `truncated=True` with a reason such as `max_results`, `max_file_bytes`, `max_depth`, or `max_directories`. No shell commands, subprocesses, network access, imports of project code, file mutation, or stdout output are used.

`SearchCodeTool` exposes the existing `Tool` protocol and metadata schema but is not connected to `fodci`, the LLM, or an Agent loop. Phase 3.3 does not implement ProjectContext expansion, framework detection, project understanding, planning, memory, RAG, file modification, terminal execution, or Phase 3.4+ behavior.

## Phase 3.4 project structure

`backend_ai.tools.project_structure(project_root)` is a standalone read-only structural detector. It reuses `list_files` for deterministic inventory and `read_file` only for a bounded set of known dependency/configuration/entry-point files. It does not read every source file, execute project code, import the target project, or use an LLM.

The immutable `ProjectStructureResult` reports project type, framework detections, language counts, package managers, databases, test frameworks, infrastructure, classified directories, important/config/dependency files, test/source directories, likely entry points, overall confidence, evidence, warnings, and truncation metadata. Individual technology detections carry their own `name`, `confidence` (`high`/`medium`), and sorted evidence items. Evidence strings correspond to observed paths or bounded content observations; ambiguous directory/file names alone do not claim a framework.

The detector covers generic Python and Node projects, Django, FastAPI, Flask, Express, React, JavaScript, TypeScript, PostgreSQL, MySQL, MariaDB, SQLite, MongoDB, pytest, unittest, Jest, Vitest, generic test structure, Docker, Docker Compose, and common CI configuration. It also classifies common source/test/documentation/database/configuration/scripts directories and detects common Python/Node package managers and entry-point conventions.

Structural inspection is bounded by the existing discovery limits plus `max_file_bytes` (default 64 KiB, maximum 1 MiB) and `max_inspected_files` (default 64, maximum 256). Sensitive files such as `.env`, credential/secret/private/password-named files, and key/certificate files are excluded from content inspection; their contents are never returned. If discovery or targeted inspection is incomplete, the result reports `truncated` or warnings rather than presenting an unqualified complete analysis.

The result is deterministic: files, directories, languages, detections, evidence, important files, and entry points are normalized and sorted. This is structural detection, not full project understanding. Phase 3.4 does not add `ProjectContext`, framework execution, dependency graphs, AST analysis, code summarization, planning, tool selection by the LLM, file modification, terminal execution, memory, RAG, or an Agent loop.

## Phase 3.5 canonical project context

`backend_ai.tools.project_context(project_root)` builds the canonical immutable `ProjectContext` consumed by future Agent reasoning. `ProjectContextBuilder` composes the existing `project_structure` tool rather than creating a second filesystem scanner. It preserves safe structural facts while separating them from derived context, evidence, confidence, warnings, and completeness.

The context includes the normalized root, project type, concise `stack_summary`, languages, frameworks, package managers, databases, test frameworks, infrastructure, source/test/documentation directories, configuration/dependency/important files, likely entry points, bounded project-file paths, confidence, evidence, warnings, `truncated`, `truncation_reason`, and `completeness` (`complete` or `partial`). All nested detections remain immutable and serializable through `to_dict()`.

The stack summary is derived only from detected evidence, for example `Python + FastAPI + PostgreSQL + pytest + Docker`. Empty or ambiguous projects do not receive invented framework claims. Confidence and evidence are inherited from the structural detector, while bounded discovery or targeted-inspection limits are promoted to partial context with explicit warnings.

`ProjectContextTool` exposes the existing `Tool` protocol and requires an explicit `project_root`; discovery limits remain configurable and bounded. The builder never executes or imports the target project, uses no LLM, does not read sensitive files, does not write inside the project, and does not access the network.

## Phase 3.6 first bounded Agent loop

`backend_ai.agent.AgentLoop` is the first orchestration layer. It starts with the explicit user task and `project_context`, invokes the existing inference engine, parses the model output, dispatches only through `ToolRegistry`, injects structured tool results into bounded history, and repeats until a final answer or an explicit limit/error stops execution.

The model-facing protocol is deliberately strict:

```text
FINAL: answer text
```

or:

```text
ACTION: search_code
ARGS: {"query":"FastAPI"}
```

Free-form JSON, arbitrary natural-language tool calls, malformed actions, unknown tools, and invalid arguments are never executed. They become structured `invalid_action`, `UNKNOWN_TOOL`, or tool-boundary errors. The registry owns only deterministic discovery/lookup/dispatch and registers `list_files`, `read_file`, `search_code`, `project_structure`, and `project_context`; the tools retain their own validation and safety logic.

`AgentConfig` bounds execution with `max_steps=8`, `max_tool_calls=8`, a 256-token model context budget with reserved response space, bounded tool-result characters, and bounded history. `ContextBudget` estimates tokens with the existing tokenizer, compacts optional project context and history deterministically, truncates oversized tool results with an explicit marker, and returns `context_limit` rather than silently cutting required task information.

`AgentResult` preserves final answer, status, immutable steps, tool calls/results, canonical project context, stop reason, usage counters, warnings, and errors. The loop uses the existing `InferenceEngine` and does not create another model runtime. It remains read-only: no file creation/edit/delete, shell or command execution, package installation, Git, network, memory, RAG, or external APIs.

The existing `fodci` CLI remains unchanged in this phase to preserve its provider-backed interactive behavior. The clean integration boundary is the public `AgentLoop` API; CLI wiring can be added only in a later explicitly scoped change.

## Phase 4.1 safe file creation

`backend_ai.tools.write_file(project_root, path, content)` creates exactly one new regular UTF-8 file inside an explicitly validated, existing project root. Missing parent directories may be created one component at a time when they remain safely inside the root; the tool never creates the root itself, overwrites an existing path, follows symbolic links, or accepts traversal/absolute paths outside the root. Content is bounded by `max_bytes` after UTF-8 encoding, with a default of 1 MiB, so Arabic and other Unicode text are handled by byte-accurate validation.

The write uses a private `0o600` temporary file, flushes and `fsync`s its complete content, then publishes it through an exclusive atomic hard-link operation. A concurrent target is rejected with the structured `FILE_EXISTS` error, temporary artifacts are removed after success or failure, and newly-created parent directories are cleaned up if the operation fails. Results are immutable `WriteFileResult` values with relative path, filename, `size_bytes`, encoding, and `created` status. Parent creation is bounded by `max_parent_directories`, defaulting to 32.

`WriteFileTool` implements the existing `Tool` protocol and is exported from `backend_ai.tools`. `ToolRegistry.default()` remains the original five-tool Phase 3 read-only registry. `ToolRegistry.with_write_file()` is an explicit Phase 4.1 opt-in registry; the existing `AgentLoop` does not automatically use it and no agent modification workflow is added. Phase 4.1 does not implement `edit_file`, `delete_file`, diffs, Git status, command/test execution, shell/subprocess access, package installation, network access, memory, RAG, or autonomous behavior.

## Phase 4.2 safe exact file editing

`backend_ai.tools.edit_file(project_root, path, old_content, new_content)` modifies an existing regular UTF-8 file only. It performs a literal, case-sensitive, byte-preserving-text replacement: `old_content` must occur exactly once, and the replacement is `original.replace(old_content, new_content, 1)`. Zero matches return `MATCH_NOT_FOUND`; multiple matches return `AMBIGUOUS_MATCH`; neither condition changes the file. Empty `old_content`, fuzzy matching, regular expressions, whitespace normalization, line-ending conversion, Unicode normalization, and whole-file replacement are not supported.

The target must exist inside the explicit root, be a readable and writable regular file, and contain valid UTF-8. Traversal, Windows/UNC paths, symlink targets or parents, broken links, directories, FIFOs, devices, invalid UTF-8, and bounded-size violations are rejected with structured `ToolError` values. The default maximum is 1 MiB for the existing file, old text, new text, and resulting file; each limit is configurable independently.

A no-op replacement (`old_content == new_content`) returns an immutable `EditFileResult` with `changed=False` and does not rewrite the file. A real edit writes the complete result to a private temporary file, preserves the original permission mode including the executable bit, flushes and `fsync`s the temporary content, then uses atomic `os.replace`. The original remains unchanged if validation, matching, encoding, size, temporary writing, or replacement fails. An optimistic snapshot checks device/inode/size/timestamps and content identity before replacement and returns `CONCURRENT_MODIFICATION` when the target changed during preparation; a filesystem race after the final check remains dependent on platform/filesystem behavior and is not claimed to be race-free.

`EditFileTool` is exported from `backend_ai.tools`. `ToolRegistry.with_file_modification()` is an explicit Phase 4.2 registry containing the read-only tools plus `write_file` and `edit_file`. `ToolRegistry.default()` and `ToolRegistry.with_write_file()` are unchanged, and `AgentLoop` does not automatically edit files.

## Phase 4.3 safe regular-file deletion

`backend_ai.tools.delete_file(project_root, path)` deletes exactly one existing regular file inside an explicitly validated project root. It does not read the file contents, create paths, delete parent directories, recurse, create backups, or print output. Missing targets return `FILE_NOT_FOUND`; directories, symlinks, broken symlinks, FIFOs, sockets, devices, and other special entries return structured errors without deletion.

The tool reuses the existing path and symlink protections, rejects traversal and paths outside the root, and opens the parent directory with no-follow flags where the platform supports them. It revalidates the target immediately before unlinking and compares device, inode, mode, size, and timestamps; a detected replacement returns `CONCURRENT_MODIFICATION`. This narrows TOCTOU risk but does not claim absolute race-free deletion on every filesystem/platform. Only the requested file entry is unlinked; parent and unrelated files remain untouched.

`DeleteFileResult` is immutable and serializable with relative path, filename, original `size_bytes`, and `deleted=True`. `DeleteFileTool` implements the existing `Tool` protocol. `ToolRegistry.with_file_modification()` now explicitly contains `write_file`, `edit_file`, and `delete_file`, while `ToolRegistry.default()` and `ToolRegistry.with_write_file()` remain unchanged. `AgentLoop` does not automatically receive or invoke deletion. Phase 4.3 does not implement backups, diffs, Git status, command/test execution, shell/subprocess access, network access, memory, RAG, or autonomous behavior.

## Phase 4.4 Safe Editing Infrastructure

`backend_ai.tools.safe_editing` provides reusable, policy-guarded infrastructure without replacing or weakening `write_file`, `edit_file`, or `delete_file`. `SafeEditPolicy` is conservative by default: all three mutation capabilities are disabled until an explicit `SafeEditPolicy.for_modification()` or equivalent policy is supplied, while explicit project roots, symlink rejection, atomic writes, metadata preservation, verification, and concurrency detection are required and cannot be disabled.

`SafeEditSession.snapshot()` returns an immutable `FileSnapshot` containing only root-relative metadata, file type, size, mode, device/inode identity, modification time, and a bounded SHA-256 hash for regular files. It does not expose file contents and rejects paths outside the root or through symlinks. `SafeEditSession` wraps the existing mutation APIs, so Phase 4.1–4.3 callers retain their public behavior and stronger existing safety logic.

The layer can generate deterministic internal unified diffs for create, edit, and delete operations. Diffs use relative `a/` and `b/` paths, never call Git or an external command, and obey `max_diff_bytes` and `max_diff_lines`; oversized output is marked with `[diff truncated]`. Diffs may contain source text when explicitly requested, but they are never printed automatically.

Optional backups are disabled by default. When enabled for a controlled edit or delete, the original bytes are copied with bounded size into a hashed, project-relative `.fodci/backups/` path using exclusive atomic creation and `0o600` permissions. Backups never overwrite unrelated files. They are removed after successful operations unless `retain_backup_on_success=True`; a failed mutation leaves the backup available for recovery inspection. This is a conservative snapshot mechanism, not transactional rollback.

After a successful wrapper operation, `SafeEditSession` snapshots the result and verifies the expected existence/hash state. It returns an immutable `SafeEditResult` with operation, relative path, success/change flags, sizes, hashes, bounded diff, backup metadata, and verification status. Optimistic snapshot identity checks use metadata and content hashes where applicable; a filesystem race after the final check is not claimed to be absolutely race-free.

`ToolRegistry.default()` remains read-only, and `ToolRegistry.with_file_modification()` remains the explicit registry for the three mutation tools. Safe editing infrastructure is not a Tool and is not automatically injected into `AgentLoop`; no planning, autonomous retries, self-directed modifications, Git features, command execution, memory, RAG, or network access is added in Phase 4.4.

## Phase 4.5 read-only Git diff inspection

`backend_ai.tools.git_diff(project_root)` inspects only the explicitly supplied repository root through `GitReadOnlyAdapter`, which whitelists `git rev-parse`, `git branch --show-current`, `git status --porcelain`, and read-only `git diff`/`git diff --cached`/numstat operations. It never accepts arbitrary command strings, never uses a shell, and never initializes or mutates a repository.

`GitDiffResult` is immutable and includes repository detection, branch/HEAD when available, deterministic repository-relative `GitChangedFile` records, separate staged and unstaged unified diffs, a bounded combined diff, insertion/deletion totals, truncation metadata, and warnings. It recognizes clean repositories, staged and unstaged changes, staged-plus-unstaged files, deleted/renamed/added paths, untracked files, and binary metadata without decoding binary bytes as UTF-8. Untracked files are identified structurally but their full contents are not read into a synthetic diff.

The tool applies limits for diff bytes, diff lines, changed files, command output bytes, and command timeout. Non-Git directories return a clean structured `is_git_repository=False` result; unavailable Git, command failures, invalid output, and timeout are reported through structured `ToolError` codes. All returned paths are normalized repository-relative POSIX paths and no absolute repository paths are embedded in change records.

`ToolRegistry.default()` remains unchanged and does not expose `git_diff` or `git_status`. `ToolRegistry.with_git_inspection()` is an explicit read-only registry containing the Phase 3 tools plus both Git inspection tools. `AgentLoop` does not automatically receive or invoke Git diff or Git status, and Phase 4.6 does not implement Git mutation, command execution, terminal access, network access, memory, RAG, or autonomous behavior.

## Phase 4.6 read-only Git status inspection

`backend_ai.tools.git_status(project_root)` reuses `GitReadOnlyAdapter` and the same explicit repository-root policy as `git_diff`. It reads `git status --porcelain=v1 -z --branch` with optional `--ignored`, parses NUL-delimited paths without whitespace splitting, and returns an immutable `GitStatusResult`.

The result distinguishes index/staged state from working-tree/unstaged state and classifies untracked, ignored, renamed, deleted, added, modified, and common conflict states. Each `GitStatusFile` reports repository-relative POSIX paths, index/worktree status codes, old/new paths where applicable, and explicit untracked/ignored/conflicted flags. Branch, HEAD, detached/unborn/normal head state, local upstream name, ahead/behind values when Git provides local metadata, clean state, warnings, and truncation metadata are represented without fabricated values.

Ignored files are excluded by default and can be requested explicitly with `include_ignored=True`; they are metadata only and their contents are never read. The operation is bounded by `max_files`, `max_output_bytes`, `max_path_length`, and timeout. Non-Git directories return structured `is_git_repository=False`, while Git unavailable/command failure/timeout conditions use the shared structured error model. The adapter remains whitelist-only, uses no shell, and performs no Git mutation or network operation.

`ToolRegistry.with_git_inspection()` now explicitly exposes both `git_diff` and `git_status`; `ToolRegistry.default()` remains read-only and unchanged, and `AgentLoop` is not modified to invoke Git status. Phase 4.7 is verification infrastructure only; Phase 5 and later work are not included.

## Phase 4.7 modification verification

`backend_ai.tools.verify_modification(project_root, expected_changes, ...)` is a strictly read-only verifier for explicit expected post-mutation states. `ExpectedModification.created`, `.modified`, `.deleted`, and `.unchanged` require the caller to state the expected path/state and may include strict UTF-8 expected content, expected byte size, expected SHA-256, and an optional pre-mutation `FileSnapshot`. The verifier never guesses intent and never returns file contents.

`ModificationVerificationResult` and `ModificationVerificationItem` are immutable and deterministic. Per-target records distinguish `present_regular_file`, `missing`, `symlink`, `directory`, `special_file`, `unreadable`, and `invalid_utf8`, with machine-readable statuses such as `VERIFIED`, `MISSING`, `CONTENT_MISMATCH`, `HASH_MISMATCH`, `TYPE_CHANGED`, `UNEXPECTED_MODIFICATION`, `UNEXPECTED_CREATION`, `UNEXPECTED_DELETION`, `VERIFICATION_ERROR`, and `VERIFICATION_UNAVAILABLE`. Results include success, complete/truncated state, warnings, errors, verified targets, and bounded unexpected changes.

The verifier reuses explicit-root and path normalization rules while lstat-ing the final entry without following symlinks. It rejects traversal and symlink parents, treats a final symlink/directory/FIFO/socket/device as an observed type rather than following it, uses strict UTF-8 decoding without replacement or ignored errors, hashes regular files with bounded reads, and performs no mutation, directory creation, subprocess execution, network access, or project-code execution.

When a baseline mapping of `FileSnapshot` records is supplied, the verifier uses the existing bounded `list_files` policy to identify unexpected modifications, creations, and deletions outside intended targets. The result is marked incomplete/truncated when discovery limits prevent a complete comparison. Without a baseline, explicit target verification remains valid but is clearly marked incomplete for project-wide claims.

`SafeEditSession.create/edit/delete` now attaches the read-only `ModificationVerificationResult` to `SafeEditResult.verification` after each successful mutation while preserving existing public fields and behavior. Direct callers of `write_file`, `edit_file`, and `delete_file` remain unchanged; `ToolRegistry.default()` and `AgentLoop` remain conservative and non-mutating.

## Phase 4.8 modification transaction and recovery

`ModificationTransaction(project_root, ModificationOperation, policy=...)` executes exactly one controlled create/edit/delete operation. The immutable lifecycle is represented through `ModificationOperation.status`: `planned`, `snapshotted`, `executing`, `verified`, `committed`, `failed`, `recovery_required`, `recovered`, or `recovery_unavailable`. `ModificationTransactionResult` reports committed/failed/recovered operations, verification, recovery status, warnings, errors, completeness, and conservative recoverability without source contents or random transaction IDs.

The transaction delegates mutation to the existing `SafeEditSession` and therefore reuses Phase 4.1–4.3 atomic publication, permissions, path safety, concurrency checks, Phase 4.4 snapshots/backups, and Phase 4.7 post-state verification. Create/edit targets are fully written and fsynced before atomic publication; failed pre-publication work cleans temporary artifacts. A controlled backup is created only when policy enables it, inside the existing `.fodci/backups/` policy, and is removed after a verified commit unless retention is explicitly requested.

Recovery is intentionally conservative. An edit can be restored only when the current file still exactly matches the transaction-generated post-state and the controlled backup is valid UTF-8 and safely rooted; restoration uses the existing exact atomic edit path and re-verifies the original snapshot. If the user changed the file, the transaction returns `user_change_preserved` and does not overwrite it. Delete recovery is reported unavailable because recreating a deleted file cannot be proven safe without a stronger transaction-generated-state guarantee. Multi-file rollback is not implemented or claimed.

`ModificationTransactionResult.to_dict()` is deterministic and excludes operation content. The layer performs no shell/subprocess/network/Git mutation, never edits outside the intended target, does not expose backup contents, and is not registered in `ToolRegistry` or automatically invoked by `AgentLoop`.

## Phase 5.1 command execution foundation

`backend_ai.tools.run_command(argv, project_root=..., working_directory=...)` and `CommandRequest` provide a low-level process primitive using an explicit tuple/list of argv strings. Shell parsing is deliberately absent: `shell=False` is used, and pipes, redirects, `&&`, `||`, globbing, command substitution, shell variables, PowerShell, `cmd.exe`, and `bash -c` are not interpreted.

The project root and working directory are required explicitly. The working directory must be a real directory inside the root, with traversal, absolute escapes, Windows drive/UNC bypasses, mixed separators, and symlink components rejected. Output is captured separately for stdout/stderr with independent byte limits; invalid UTF-8 is replacement-decoded and recorded in warnings. A bounded timeout terminates the process where supported and returns structured timeout metadata. The executor never returns environment values, uses `stdin=DEVNULL` rather than an interactive terminal, and applies an explicit environment overlay with optional parent-environment inheritance.

`CommandResult` distinguishes successful completion, non-zero exit, output-limit termination, timeout, executable-not-found, permission/start failure, and invalid working-directory/argument failures. It includes lifecycle, exit code, relative working directory, bounded output, truncation/decoding flags, termination, warnings, and structured error code/message. `RunCommandTool` is exposed only through `ToolRegistry.with_command_execution()`; `ToolRegistry.default()` and `AgentLoop` remain unchanged. Platform-specific process-tree termination behavior may differ, especially on Windows.

## Phase 5.2 command safety and policy

`CommandPolicy` is an independently testable, immutable, deterministic security boundary above `run_command`. Its conservative default is **deny-by-default**: only bounded version/info commands and read-only Git inspection are recognized without an explicit bounded rule. `CommandDecision` records allowed/denied state, risk level, normalized secret-safe argv, matched rule, reason, warnings, and structured error code.

The policy rejects shell interpreters and emulation (`bash -c`, `sh -c`, PowerShell, `cmd /c`), privilege escalation, destructive filesystem/system commands, package installation, network/download/remote-execution families, Git mutation, unknown executables, unsafe absolute/traversal/Windows/UNC argument paths, symlink-escaping working directories, and disallowed environment variables such as `PYTHONPATH`, `NODE_PATH`, loader hooks, and shell startup variables. Environment inheritance is disabled by default for policy-wrapped execution, and environment values never appear in decisions or errors.

`PolicyRunCommandTool` evaluates the request first and calls the existing `run_command` only after an allowed decision. A denied request raises a shared structured `ToolError` and spawns no process. Controlled exact-argv and approved-executable-path overrides are available, but there is no unrestricted allow-anything switch and core shell/path safety invariants cannot be bypassed. The wrapper is exposed only through `ToolRegistry.with_command_policy()`; `ToolRegistry.default()`, `ToolRegistry.with_command_execution()`, and `AgentLoop` retain their previous boundaries.

> **Command Safety Policy is a security boundary, not a guarantee that arbitrary developer commands are safe.**

Phase 5.2 does not implement shell execution, pipelines, command chaining, redirection, package installation, network capability, Git mutation, service/system administration, application running, test running, test-result parsing, project intelligence, or automatic AgentLoop execution.

## Phase 5.3 process management

`ProcessManager` manages exactly one already-approved `CommandRequest`. It does not decide whether a command is safe; that remains the responsibility of `CommandPolicy`. The policy-wrapped path is conceptually `PolicyRunCommandTool → CommandPolicy.evaluate() → ProcessManager → direct subprocess`.

The immutable `ProcessLifecycle` state machine records `REQUESTED`, `VALIDATING`, `STARTING`, `RUNNING`, `COMPLETED`, `FAILED_TO_START`, `TIMED_OUT`, `TERMINATING`, `TERMINATED`, `KILLED`, `OUTPUT_LIMIT_REACHED`, and `CLEANED_UP`. Invalid transitions raise a shared structured `PROCESS_INVALID_STATE` error. `CommandResult` remains backward-compatible while adding process state/history, termination-attempted/killed flags, and bounded captured stdout/stderr byte metadata.

The manager preserves `shell=False`, `stdin=DEVNULL`, explicit root-contained cwd validation, controlled environment construction, separate stdout/stderr, invalid-UTF-8 warnings, and bounded timeout/output behavior. Unlike the earlier low-level capture path, output-limit handling continues draining pipes while retaining only the configured bytes, preventing avoidable pipe backpressure while making truncation explicit. Timeout handling marks the process timed out, attempts process-group/session termination where supported, waits for a bounded grace period, escalates to kill when needed, reaps the direct process, and preserves safely collected partial output. Descendant cleanup cannot be guaranteed identically on every operating system, especially on Windows.

No background queue, daemon, retry, scheduler, test runner, result parser, shell, network, package installation, Git mutation, or AgentLoop automatic execution is added. `ProcessManager` is reusable infrastructure and is not registered in `ToolRegistry.default()`; `PolicyRunCommandTool` uses it only after an allowed policy decision.

## Phase 5.4 application runner

`ApplicationRunner` is a bounded launch layer rather than a generic command generator. It consumes the existing structural context and bounded project files. Automatic resolution supports only explicit evidence-backed patterns: Python `main.py`/`app.py`/`server.py` with a supported entry-point marker, Django `manage.py` only when Django evidence is present, and Node `package.json` `scripts.start`/`main` only when the target is an existing `.js`/`.mjs`/`.cjs`/`.ts` file and the command is an exact safe `node target` form. Mixed projects or multiple candidates return deterministic `AMBIGUOUS_ENTRYPOINT`; missing or unsupported evidence returns structured unresolved statuses rather than a guess.

Explicit argv mode preserves the caller’s argv and still evaluates it through `CommandPolicy`; command strings, shell interpreters, package/network/Git/system operations, and unsafe paths remain rejected. Resolved plans contain safe normalized argv, working directory, source/evidence, project type, confidence, and warnings. `ApplicationRunResult` carries plan, policy decision, ProcessManager `CommandResult`, status/failure classification, evidence, candidates, and warnings without environment values or sensitive file content.

The runner requires a bounded timeout default inherited from the command layer, does not leave long-running processes unmanaged, and returns `TIMED_OUT` after ProcessManager termination/cleanup. `RunApplicationTool` is exposed only through `ToolRegistry.with_application_execution()`; `ToolRegistry.default()` and `AgentLoop` remain unchanged.

## Phase 5.5 bounded Test Runner

`TestRunner` answers a narrower question than a generic command generator: given an explicit project root and existing structural evidence, what is the safest supported argv for one test execution? It composes `ProjectContextBuilder`/`ProjectStructure`, `CommandPolicy`, and `ProcessManager`; it does not create a second scanner, executor, policy, or lifecycle system. `RunTestsTool` is available only through `ToolRegistry.with_test_execution()`. `ToolRegistry.default()` remains the original read-only registry and `AgentLoop` does not gain test execution automatically.

The resolver supports Python `pytest` and `unittest`, plus Node/Javascript/TypeScript evidence for Jest, Vitest, and an explicitly declared `package.json` `scripts.test`. Existing project metadata, bounded configuration inspection, dependency evidence, actual test structure, and visible runner evidence are used conservatively. A package-defined `test` script is preferred over framework candidates. Strong framework candidates are selected only when their priority is unique; equally ranked candidates return `AMBIGUOUS_TEST_COMMAND`, and insufficient evidence returns `NO_TEST_COMMAND`. No package is installed, no manifest/configuration is changed, and commands are never invented from directory names alone.

Explicit argv mode accepts only a list/tuple of strings. It never becomes a shell string and is passed through `CommandPolicy` before `ProcessManager`; shell operators, interpreters, redirection/substitution, unsafe paths, sensitive-file paths, package/network/Git mutation, and unapproved executables are rejected before process creation. Optional test targets and bounded argv arguments are validated as safe path/module arguments rather than injected into arbitrary command strings. Environment inheritance remains disabled by default and any explicit environment still uses the existing policy allowlist; environment values are never included in results.

`TestRunResult` preserves raw bounded execution facts: resolved plan, framework/evidence, policy decision, exit code, stdout/stderr, byte counts, truncation and invalid-UTF-8 metadata, timeout/termination/cleanup information, lifecycle data, warnings, and technical failure codes. `COMPLETED` with a non-zero exit code remains an execution fact classified as `NONZERO_EXIT`; the runner does not decide semantic test PASS/FAIL/ERROR, count tests, identify failed names, or parse assertions. Those interpretations belong exclusively to the future Phase 5.6 `TestResultParser`, which is not implemented. The runner does not retry, rerun, schedule, daemonize, or clean arbitrary artifacts that a test framework may naturally create.

The manual smoke workflow is [run_fodci_test_runner_smoke.py](scripts/run_fodci_test_runner_smoke.py). It uses only temporary local fixtures and an explicit read-only `python --version` check against the actual repository; it does not install packages, access the network, mutate Git, or modify the real repository.

## Phase 5.6 deterministic Test Result Parser

`parse_test_result(test_result)` and the opt-in `TestResultParserTool` form a read-only semantic layer above Phase 5.5. The parser accepts an existing immutable `TestRunResult`; it never executes a command, invokes `ProcessManager`, reads project files, imports target code, accesses the network, modifies files, or calls `AgentLoop`. `ToolRegistry.with_test_result_parsing()` exposes the tool explicitly, while `ToolRegistry.default()` and `AgentLoop` remain unchanged.

The parser reports `PASS`, `FAIL`, `ERROR`, `NO_TESTS`, `TIMEOUT`, `OUTPUT_LIMIT`, `EXECUTION_ERROR`, or `UNKNOWN`. Technical execution metadata has first precedence: timeout remains `TIMEOUT`, output truncation remains `OUTPUT_LIMIT`, and start/policy/execution failures remain `EXECUTION_ERROR`. Framework-level collection/runtime errors outrank assertion failures; recognized assertion failures become `FAIL`; successful complete summaries with no failures become `PASS`; a valid zero-test run becomes `NO_TESTS`. A non-zero exit code without strong framework evidence is `UNKNOWN` rather than an automatic `FAIL`. Contradictory or ambiguous evidence produces warnings and bounded/unknown results instead of invented certainty.

Framework-aware parsing is intentionally conservative. It supports common bounded text summaries for pytest, unittest, Jest, Vitest, and package scripts whose output clearly matches one supported format. It extracts counts, duration when reliably present, bounded failed/error test names, bounded failure/error records, summaries, framework/format, confidence, warnings, and parse completeness. It does not diagnose root causes, modify files, rerun tests, retry, interpret arbitrary output, or claim universal framework compatibility.

`TestParseLimits` bounds input bytes, failure/error records, test-name length, message length, and raw excerpts. Parser input is truncated before analysis when necessary, sensitive key/token/password-like values in structured messages/excerpts are redacted, and truncated input yields `truncated=True` and `parse_completeness="partial"`. Repeated parsing of the same raw result produces identical `to_dict()` output with no generated timestamps or randomness. The smoke workflow is [run_fodci_test_result_parser_smoke.py](scripts/run_fodci_test_result_parser_smoke.py).

Phase 5.6 completes Phase 5. It does not add automatic debugging, self-correction, automatic retries/reruns, file mutation, planning, memory, RAG, package installation, network access, Git mutation, shell execution, background execution, autonomous AgentLoop behavior, or Phase 6 functionality.

## Phase 6.1 deterministic Planner

`backend_ai.agent.Planner` accepts a user task, an optional already-supplied `ProjectContext`, and bounded `PlannerConfig` budgets. It produces immutable `PlannerResult` and `ExecutionPlan` values containing the original and normalized task, goal, conservative task category, declarative `PlanStep` DAG, assumptions, constraints, bounded risks, expected change categories, verification strategy, confidence, warnings, and completeness. The convenience API is `create_plan(task, project_context=..., config=...)`; planner output is available through `plan.to_dict()` and is deterministic for identical inputs.

The Planner normalizes whitespace without inventing implementation details and conservatively classifies feature, bug-fix, refactor, test-addition, configuration, documentation, dependency, investigation, and unknown tasks. Ambiguous requests preserve ambiguity, lower confidence, add clarification warnings, and can produce `REQUIRES_CLARIFICATION`. Missing context is represented explicitly as low-confidence/partial planning; partial supplied context carries its warnings and does not become an unconfirmed fact. Project-aware constraints use only the fields already present in the supplied `ProjectContext`.

`PlanValidator` verifies required text, enum values, unique step IDs, valid dependency references, DAG acyclicity, bounded step/text counts, and the absence of executable tool calls, shell payloads, command fields, or mutation instructions disguised as prose. `PlannerConfig` bounds steps, assumptions, constraints, risks, warnings, task text, and plan text; truncation adds structured warnings and incomplete planning remains visible. `PlanStep` describes what should happen, not a specific tool invocation. Verification strategy is described but never run by the Planner.

The Planner is not a Tool Selector, Tool Executor, code generator, debugger, self-correction engine, filesystem scanner, test runner, command executor, or autonomous AgentLoop. It performs no file reads, filesystem discovery, subprocess calls, network access, package installation, environment/secrets access, Git operations, or background execution. Phase 6.2 is now the separate Tool Selection layer; Phase 6.3 Tool Loop, stop conditions, execution budgets, error recovery, task completion verification, memory, RAG, and all autonomous behavior remain not started.

## Phase 6.2 deterministic Tool Selection

`ToolSelector` consumes a validated `ExecutionPlan`, an explicitly supplied `ToolRegistry`, optional supplied `ProjectContext`, available-input evidence, and bounded `ToolSelectionConfig`. It discovers actual registered names and metadata through the registry without enabling additional tools. `ToolSelectionRequest`, `ToolCapability`, `ToolCandidate`, `ToolSelectionDecision`, and `ToolSelectionResult` are immutable and serialize through `to_dict()`.

The capability catalog maps existing registered tools to `READ_ONLY`, `MUTATING`, `EXECUTION`, or `DESTRUCTIVE` capabilities and records supported intents, required/optional inputs, expected output, safety notes, and prerequisites. Logical intents resolve to the actual registered name, including policy-wrapped `run_command_with_policy` and the existing `parse_test_result` name. Unknown registered capabilities are not selected automatically. The selector respects the supplied registry and returns `TOOL_UNAVAILABLE` rather than pretending an absent tool exists.

Selection is plan-driven and inspection-first. Project discovery maps to `project_structure` or `project_context`; locating unknown implementation maps to `search_code` with bounded `list_files` alternatives; known source inspection maps to `read_file`; explicit creation/modification/deletion maps only to the corresponding mutation capability; repository review maps to `git_status`/`git_diff`; explicit application/test/approved-command steps map to the supplied execution capability; existing test-result interpretation maps to `parse_test_result`. Inappropriate mutation or execution tools are listed as forbidden where relevant, and equal inspection candidates produce `AMBIGUOUS_SELECTION` instead of an arbitrary choice.

Every decision records selection status, selected tool, category, reason, confidence, required and optional inputs, prerequisites and missing prerequisites, expected output, alternatives, forbidden tools, risk level, warnings, and bounded candidates. Strict prerequisite mode can return `MISSING_PREREQUISITES`; missing registry capabilities return `TOOL_UNAVAILABLE`; unmappable steps return `NO_SUITABLE_TOOL`. Mutation decisions remain subject to `SafeEditSession`/`SafeEditPolicy`, and command/application/test decisions remain subject to `CommandPolicy`/`ProcessManager`; selection never grants execution permission.

`ToolSelectionValidator` checks plan-step IDs, duplicate selections, available tool names, alternatives, enum values, risk/confidence, mutation intent, execution intent, and malformed decision structures. Repeated selection with identical plan/registry/context/configuration is deterministic and bounded by candidate, alternative, step, prerequisite, and warning budgets. `ToolSelector` performs no filesystem access, subprocess/network operation, tool dispatch, environment/secrets access, Git mutation, or AgentLoop invocation. The smoke workflow is [run_fodci_tool_selection_smoke.py](scripts/run_fodci_tool_selection_smoke.py).

Phase 6.2 stops at `ExecutionPlan → Tool Selection`. Phase 6.3 adds the separate `AutonomousToolLoop` boundary:

```text
Task + explicit project root
             ↓
          Planner
             ↓
       ExecutionPlan
             ↓
       ToolSelector
             ↓
  strict model ACTION/ARGS validation
             ↓
 supplied ToolRegistry.dispatch()
             ↓
 structured ToolResult observation
             ↓
 bounded ContextBudget history → next action
```

`AutonomousLoopRequest`, `AutonomousLoopConfig`, `AutonomousLoopState`, `AutonomousLoopStep`, `AutonomousLoopResult`, `LoopAction`, and the lifecycle/status/failure enums are immutable or bounded structured models. The action protocol accepts only `ACTION: TOOL` with a JSON object containing `tool` and `arguments`, or `ACTION: FINAL` with a JSON `message`; prose, invalid JSON, unknown action shapes, shell-like payloads, and tool names not selected by the current `ToolSelector` are rejected before dispatch.

`AutonomousToolLoop` is separate from the existing Phase 3.6 `AgentLoop`; the original read-only loop and CLI behavior remain unchanged. The new loop is explicitly opt-in through direct construction or `create_autonomous_tool_loop()` and uses the caller-supplied registry without automatically enabling mutation, command, application, test, or Git capabilities. Every tool call goes through `ToolRegistry.dispatch()`, preserving the registered tool's path, SafeEdit, CommandPolicy, ProcessManager, ApplicationRunner, TestRunner, and parser boundaries.

The lifecycle is explicit: `CREATED → PLANNING → SELECTING_TOOL → VALIDATING_ACTION → EXECUTING_TOOL → OBSERVING_RESULT → UPDATING_CONTEXT → REQUESTING_NEXT_ACTION`, terminating only at `COMPLETED` for an explicit `ACTION: FINAL` or at a structured failure state. Tool results are normalized into bounded observations, sensitive key-like arguments are redacted in state/history serialization, output is truncated through the existing context budget, and history preserves task/plan/current-step/tool-observation sections with deterministic truncation metadata. Failed tools are recorded once and terminate the invocation; there are no automatic retries, argument changes, tool switching, debugging, or self-correction.

A private fixed emergency bound of eight tool executions per invocation prevents infinite development loops. It is deterministic and cannot be overridden by model output; it is not the configurable Phase 6.5 max-iterations feature. If a plan step has no suitable tool, the loop records a bounded skip and may request a final action; unavailable or ambiguous required capabilities return structured failure. Mutation still requires an explicitly mutation-enabled registry and remains subject to SafeEdit policy, while execution remains subject to CommandPolicy and ProcessManager.

Phase 6.3 is the first controlled autonomous execution phase. Phase 6.4 adds only semantic stop evaluation: a valid FINAL action can be DONE only when no required plan steps or verification obligations remain; otherwise the result is CONTINUE or BLOCKED. A successful mutation remains CONTINUE until structured verification passes, unavailable tools and policy denials are BLOCKED, fatal infrastructure errors are FAILED, and emergency-bound termination is non-DONE. The evaluator is pure and does not call the model, tools, filesystem, subprocesses, network, or Git. Configurable max iterations, error recovery, retries, self-correction, autonomous debugging, memory, RAG, network, package installation, Git mutation, shell execution, background agents, and automatic CLI autonomy remain deferred to Phase 6.5+.

## Configuration and Logging

Copy `.env.example` to `.env` only for local development. `.env` is ignored by Git. The settings abstraction recognizes `LOG_LEVEL` and `PROJECT_ROOT`, uses the current working directory when `PROJECT_ROOT` is omitted, and validates the log level. During application startup, the resolved project root must exist and be a directory; an explicitly invalid path fails clearly without falling back to the current working directory. Phase 1.7 does not inspect files inside the root. The example also documents reserved names for later stages without reading them yet.

Centralized logging is available through `backend_ai.config.configure_logging`. It applies a compact timestamped format, accepts a level, avoids global mutable application state, and does not log configuration values or secrets.

## Security Foundation

Future implementation must preserve explicit project boundaries, keep secrets out of source control and logs, treat shell execution as potentially dangerous, validate tool inputs, and avoid arbitrary filesystem access outside an explicitly authorized project root. The Phase 0 package does not yet implement a sandbox or execution system. See [security principles](docs/security.md).

## Roadmap

| Phase | Scope |
| --- | --- |
| 0 | Foundation |
| 1 | CLI |
| 2 | Local LLM |
| 3 | Project Understanding |
| 4.1 | Safe file creation (`write_file`) |
| 4.2 | Safe exact editing (`edit_file`) |
| 4.3 | Safe regular-file deletion (`delete_file`) |
| 4.4 | Safe Editing Infrastructure |
| 4.5 | Read-only Git diff inspection (`git_diff`) |
| 4.6 | Read-only Git status inspection (`git_status`) |
| 4.7 | Modification Verification |
| 4.8 | Modification Transaction / Recovery; final Phase 4 sub-phase |
| 5.1 | Command Execution Foundation (`run_command`) |
| 5.2 | Command Safety & Policy (`CommandPolicy`) |
| 5.3 | Process Management (`ProcessManager`) |
| 5.4 | Bounded Application Runner (`run_application`) |
| 5.5 | Bounded Test Runner (`run_tests`) |
| 5.6 | Deterministic Test Result Parser (`parse_test_result`); final Phase 5 sub-phase |
| 6.1 | Deterministic Planner (`Planner`, `create_plan`) |
| 6.2 | Deterministic Tool Selection (`ToolSelector`, `create_tool_selection`) |
| 6.3 | Explicitly opt-in bounded Autonomous Tool Loop (`AutonomousToolLoop`, `parse_loop_action`) |
| 6.4 | Deterministic Stop Conditions (`StopConditionEvaluator`, `evaluate_stop_condition`) |
| 6.5 | Centralized Execution Budgets (`ExecutionBudget`, `ExecutionUsage`, `ExecutionBudgetLedger`) |
| 6.6 | Bounded Error Recovery (`ErrorClassifier`, `RecoverabilityPolicy`, `RecoveryContext`, `RecoveryDecision`) |
| 6.7 | Task Completion Verification (`TaskCompletionVerifier`, `TaskCompletionResult`) |
| 7.1 | Automatic Test Execution (`AutomaticTestOrchestrator`, `AutomaticTestResult`) |
| 7.2 | Test Failure Analysis (`TestFailureAnalyzer`, `TestFailureAnalysis`) |
| 7.3 | Root Cause Analysis (`RootCauseAnalyzer`, `RootCauseAnalysis`) |
| 7.4 | Automatic Fix (`AutomaticFixPlanner`, `AutomaticFixOrchestrator`, `AutomaticFixResult`) |
| 7.5 | Bounded Self-Correction (`BoundedSelfCorrectionLoop`, `SelfCorrectionResult`); Phase 7.6+ not started |
| 5+ | Later product phases, not started |
| 5 | Terminal + Execution |
| 6 | Autonomous Agent Loop |
| 7 | Testing + Self-Correction |
| 8 | Evaluation |
| 9 | Memory |
| 10 | Experience Dataset |
| 11 | Model Improvement |
| 12 | Advanced Agent |

## Non-Goals for the Current Phase

Phase 7.3 adds only bounded root-cause hypothesis generation over existing `TestFailureAnalysis`. `RootCauseAnalyzer` explicitly separates the observed failure from an inferred root-cause hypothesis and from any confirmed cause; this phase never produces a confirmed cause. Each hypothesis contains a statement, classification, location, bounded failure mechanism, supporting evidence, contradicting evidence, affected finding IDs, confidence, evidence strength, and causal status. Alternatives are retained rather than collapsed into one answer. Primary/derived and cascading relationships are labeled as inference, and missing evidence returns `UNKNOWN`/`INCONCLUSIVE` semantics instead of invented certainty.

The analyzer uses only immutable structured `ProjectContext` metadata when supplied; it does not inspect files, read `.env` or secrets, execute application code, run tests, install packages, access the network, mutate Git, or modify the environment. Location candidates are conservative `TEST`, `IMPLEMENTATION`, `CONFIGURATION`, `DEPENDENCY`, `FIXTURE`, `ENVIRONMENT`, `DATABASE`, `EXTERNAL_SERVICE`, or `UNKNOWN` records, with inferred status and confidence. Causal chains have a validated `max_causal_depth`; when reached, `causal_chain_truncated` is explicit. Supporting/contradicting evidence, unknowns, warnings, evidence completeness, and analysis completeness are serialized for later phases. `AutonomousToolLoop.analyze_root_cause()` exposes this as explicit structured diagnostic context only. `TestFailureAnalyzer` accepts `TestFailureAnalysisRequest` and returns immutable `TestFailureAnalysis` with `ANALYZED`, `NO_FAILURE`, `INCOMPLETE`, `INSUFFICIENT_EVIDENCE`, `UNAVAILABLE`, or `INVALID` status. Its taxonomy distinguishes assertion, exception, import/module, type, syntax, configuration, dependency, database/connection, authentication/API, fixture, discovery, environment, timeout, output-limit, execution, and unknown failures. It reports observed failure evidence rather than a root cause: test location and any suspected implementation location are kept distinct, locations are hypotheses unless parser evidence supports them, and related failures/primary-vs-derived relationships are explicitly labeled as inference. Confidence is derived from parser confidence, test identity, exact file/line evidence, exception type, and structured message evidence; missing or truncated evidence lowers confidence and prevents false certainty.

The analyzer uses bounded input/finding/group/traceback/excerpt/path/message/chain limits, preserves truncation and parser completeness, and redacts passwords, tokens, API keys, credentials, authorization values, and private keys from diagnostic excerpts. It is deterministic and side-effect-free. `AutonomousToolLoop.analyze_test_failure()` exposes the analysis as an explicit structured observation helper; the loop does not modify files or act on the analysis. Phase 7.6 regression protection and Phase 7.7 final verification remain unimplemented. `AutomaticTestOrchestrator` decides `RUN`, `SKIP`, `BLOCKED`, `UNAVAILABLE`, `INVALID`, or `BUDGET_EXHAUSTED` from explicit task/plan/completion evidence and capability availability. It triggers only at bounded verification points such as an explicit user request, a plan test step, completion-required test evidence, or an implementation task reaching its verification boundary. Documentation and investigation tasks can skip tests; testing is not run after every file operation. The orchestrator accepts structured target and test-argument values, passes them to the existing `run_tests` tool, and preserves `TestRunResult` metadata. It never guesses commands, constructs shell strings, calls subprocess directly, or performs result diagnosis. It does not execute tools, inspect files, run tests, call the model, mutate files/Git, access secrets, or invent project requirements. `TaskCompletionVerifier` distinguishes completed actions from completed tasks: every required plan step must be accounted for, required verification must pass, required test evidence must be relevant and successful, recovery/budget/safety blockers must be resolved, and incomplete or truncated evidence prevents `DONE`. A final model message is only a claim, never completion proof. Explicit criteria can represent expected files or behavior, while existing modification verification and test parsing remain the sources of truth.

Phase 6.6 adds only bounded error recovery over structured evidence. It is not unrestricted self-correction or autonomous debugging. `ErrorClassifier` maps existing tool error codes/results to stable categories, severity, recoverability, safety boundary, and user-intervention requirements. `RecoverabilityPolicy` returns immutable decisions with conservative actions: safety/policy/root violations block, concurrent user changes require intervention, budget/context/internal/unknown failures stop, and actionable file/test/command/application/verification failures may continue only through a different bounded plan step or request inspection/replanning. The failed action is never blindly repeated. Recovery uses the existing Phase 6.5 budgets, preserves the original error, and remains separate from Phase 7.1 test orchestration. It is not unrestricted self-correction or autonomous debugging. `ErrorClassifier` maps existing tool error codes/results to stable categories, severity, recoverability, safety boundary, and user-intervention requirements. `RecoverabilityPolicy` returns immutable decisions with conservative actions: safety/policy/root violations block, concurrent user changes require intervention, budget/context/internal/unknown failures stop, and actionable file/test/command/application/verification failures may continue only through a different bounded plan step or request inspection/replanning. The failed action is never blindly repeated. Recovery uses the existing Phase 6.5 budgets, preserves the original error, records bounded history, and exposes the decision through `AutonomousLoopResult` and loop state. Recovery is orchestration over existing ToolRegistry and safety layers, not a second execution framework.

Phase 6.5 adds the centralized `ExecutionBudget`/`ExecutionBudgetLedger` enforcement layer. Conservative defaults are finite: 16 iterations, 16 tool calls, 4 mutations, 4 commands, 4 test executions, 2 application launches, 300 seconds, 131,072 accumulated tool-output bytes, 65,536 stdout bytes, 65,536 stderr bytes, 65,536 context tokens, and 16 action steps. Hosts may provide validated finite limits; zero explicitly disables that dimension. Attempts are counted before dispatch, successful/failed results are accounted once, policy-denied operations never create subprocesses, and remaining values are clamped at zero. Phase 6.4 Stop Conditions answer whether the agent should continue; Phase 6.5 budgets answer whether it is still allowed to continue. Budget exhaustion has precedence and returns structured `BUDGET_EXHAUSTED` with dimension, limit, usage, remaining budget, operation, and `operation_started=false`. The fixed Phase 6.3 emergency bound remains a final immutable backstop. Phase 7.4 adds only one explicit, bounded mutation attempt for one structured fix request. `AutomaticFixPlanner` validates RCA status, hypothesis/actionability, exact target location, relative path, sensitive-path policy, risk/confidence thresholds, evidence, expected post-state, bounded UTF-8 old/new content, and explicit edit policy. `AutomaticFixOrchestrator` checks action-step and mutation dimensions of `ExecutionBudgetLedger` before any mutation, then delegates the single edit to `ModificationTransaction`, which reuses `SafeEditSession`, atomic edit, snapshots/backups, concurrent-change protection, recovery semantics, and post-state verification. Results distinguish `PROPOSED`, `ACCEPTED`, `MUTATION_SUCCEEDED`, `FIX_VERIFIED`, `REJECTED`, `FAILED`, `BLOCKED`, `NO_SAFE_FIX`, and `RECOVERY_REQUIRED`; they expose whether a mutation was attempted, whether it was verified, the transaction result, budget decision, and explicit `tests_rerun=false`/`retries=0`. No test rerun occurs in this phase, and no loop or recursive self-correction exists. Dependency installation, environment/secret changes, Git mutation, network, shell, background processes, broad refactoring, and multiple unrelated files are excluded.

Phase 7.5 adds `SelfCorrectionConfig`, `SelfCorrectionRequest`, `SelfCorrectionAttempt`, `RetryDecision`, and `SelfCorrectionResult` around `BoundedSelfCorrectionLoop`. A host supplies finite `max_attempts`; model output, test output, failure messages, and fix proposals cannot alter it. Each attempt is bounded and traceable. PASS returns `PASSED` immediately without another fix or retry. FAIL is parsed and analyzed through existing components, then at most one validated fix is applied before a retest. Failure/action fingerprints are normalized and secret-redacted; an unchanged pair stops as `REPEATED_FAILURE`, an unverified/no-change fix stops as `NO_PROGRESS`, missing actionable RCA stops as `NO_ACTIONABLE_FIX`, and policy/recovery/budget boundaries stop as `BLOCKED` or `BUDGET_EXHAUSTED`. The same `ExecutionBudgetLedger` is passed through every test and mutation, and existing `StopConditionEvaluator` is consulted for terminal budget/safety decisions. No global counters, background workers, scheduled retries, or unbounded recursion exist. Phase 7.6 regression protection and Phase 7.7 final verification remain unimplemented.

## License

This project is distributed under the [MIT License](LICENSE).
