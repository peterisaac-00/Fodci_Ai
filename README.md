# Backend Engineering Agent

> **Current status: Phase 4.4 — Safe Editing Infrastructure; Phase 4.4 only.**

Backend Engineering Agent is the foundation for a future **local, terminal-based AI agent** focused on backend engineering work. The intended product will use an interchangeable local or open-weight language-model provider rather than depend on hosted OpenAI, Anthropic, or Gemini APIs.

This repository includes the complete Phase 1 CLI foundation, Phase 2.1's minimal typed LLM provider boundary, Phase 2.2's small decoder-only Transformer architecture, Phase 2.3's reversible byte-level tokenizer, Phase 2.4's local streaming dataset pipeline, Phase 2.5's CPU-friendly training engine, Phase 2.6's first real Fodci Tiny v1 training experiment, Phase 2.7's metadata-aware checkpoint manager, Phase 2.8's CPU-first evaluation pipeline, Phase 2.9's local backend-engineering coding corpus and manifest layer, Phase 2.10's local instruction-training dataset and response-masked training path, and Phase 2.11's local CPU inference API. Phase 2.12 connects that existing inference path to the official `fodci` terminal session through `FodciLocalProvider`. Phase 3.1 adds the first read-only Agent tool, `list_files`, for safe deterministic discovery of an explicitly selected project root. Phase 3.2 adds the second read-only tool, `read_file`, for bounded exact UTF-8 reading inside that root. Phase 3.3 adds the third standalone read-only tool, `search_code`, for bounded literal or explicitly enabled regex search across safe UTF-8 source files. Phase 3.4 adds `project_structure`, a bounded evidence-based structural detector for technologies, components, languages, configurations, tests, and likely entry points. Phase 3.5 adds the canonical immutable `ProjectContext` layer and builder that transforms structural facts into a compact deterministic context for future Agent reasoning. Phase 3.6 adds the first bounded read-only `AgentLoop`, a deterministic `ToolRegistry`, a strict ACTION/ARGS protocol, and structured execution results over the existing tools. Phase 4.1 adds `write_file`, a bounded atomic create-only tool that is available through an explicit opt-in registry but is not automatically used by `AgentLoop`. Phase 4.2 adds `edit_file`, a bounded atomic exact replacement tool for existing UTF-8 files, also available only through an explicit modification registry. Phase 4.3 adds `delete_file`, a regular-file-only deletion tool with no recursive behavior and explicit opt-in registry exposure. Phase 4.4 adds a reusable `SafeEditPolicy`/`SafeEditSession` infrastructure layer with immutable snapshots, bounded internal diffs, optional controlled backups, and post-mutation verification over the existing mutation tools. The model remains intentionally tiny at 11,424,400 parameters; no external LLM, pretrained component, terminal execution, RAG, memory, or autonomous loop is present.

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
| 4.5+ | Git/diff/status and later Phase 4 work |
| 5 | Terminal + Execution |
| 6 | Autonomous Agent Loop |
| 7 | Testing + Self-Correction |
| 8 | Evaluation |
| 9 | Memory |
| 10 | Experience Dataset |
| 11 | Model Improvement |
| 12 | Advanced Agent |

## Non-Goals for the Current Phase

Phase 4.4 adds only the reusable safe-edit policy/session, immutable snapshots, bounded internal diffs, optional controlled backups, post-mutation verification, tests, documentation, and no automatic AgentLoop integration. It does not add `git_diff`, `git_status`, Git operations, command or shell execution, package installation, network access, memory, RAG, embeddings, external APIs, planning, autonomous modification, or Phase 4.5+ functionality.

## License

This project is distributed under the [MIT License](LICENSE).
