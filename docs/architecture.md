# Architecture Direction

## Application startup boundary

Phase 1.2 through 2.3 add a thin application boundary between the console entry point and future agent orchestration:

```text
fodci
    ↓
CLI entry point
    ↓
Application.start()
    ↓
existing configuration + logging bootstrap
    ↓
ProjectContext(root)
    ↓
Application provider composition
    ↓
FodciLocalProvider (one loaded engine)
    ↓
InteractiveSession
    ↓
InputProvider
    ↓
CommandParser
    ↓
CommandDispatcher
    ├── /help → local help output
    ├── /exit → structured stop request
    └── normal input passthrough
```

`backend_ai.cli.main` is responsible only for process-facing output, clean startup errors, and status. `backend_ai.application` composes the currently available startup steps through `core.bootstrap`, resolves a minimal `ProjectContext`, creates the configured provider once, and then delegates session persistence and input reception to `InteractiveSession`. `ProjectContext` contains only an absolute, normalized, validated root path; resolution checks existence and directory type but never scans the root. `InputProvider` is injectable for deterministic tests and defaults to stdin in production. `CommandParser` recognizes only a leading `/`; `CommandDispatcher` routes registered names and reports unknown commands without executing them. Phase 1.6 registers `/help` and `/exit` through this same registry. `/help` derives its output from registered metadata, while `/exit` returns a structured stop request that the session handles. The CLI module itself does not import or initialize a concrete provider or model; the application boundary owns that composition.

## Phase 0 boundary model

The first phase defines dependency boundaries without implementing agent behavior. `core.contracts` owns small shared protocols. The domain packages re-export the protocol relevant to their boundary, while future concrete implementations must live behind those contracts.

```text
Agent orchestration ──────> LLMProvider protocol
        │
        ├─────────────────> Tool protocol
        ├─────────────────> Memory protocol
        └─────────────────> Evaluator protocol

Concrete local provider / tool / store / evaluator
        └───────────────> implements its protocol
```

The dependency direction is intentional: orchestration may depend on interfaces, but it must not import a particular local model implementation. This keeps a future model provider replaceable without rewriting the agent boundary.

## Phase 2.1 provider boundary

Phase 2.1 defines the minimal typed boundary:

```text
Agent
  ↓
LLMProvider
  ↓
Future Local Model
```

`Message`, `LLMRequest`, and `LLMResponse` contain only role/content messages and generated text. `LLMProviderError` provides one typed provider-level failure. `ProviderBackedAgent` accepts the provider through dependency injection and delegates one request only. Phase 2.2 adds an isolated `FodciModel` architecture with local random initialization and no provider or CLI integration.

## Phase 2.2 model architecture

The first Fodci model is a configurable decoder-only Transformer:

```text
Input token IDs
      ↓
Token + learned position embeddings
      ↓
TransformerBlock × N
  ├── pre-LayerNorm
  ├── causal multi-head self-attention
  ├── residual connection
  ├── pre-LayerNorm
  ├── GELU feed-forward network
  └── residual connection
      ↓
Final LayerNorm
      ↓
Language-modeling head
      ↓
Logits (batch, sequence, vocabulary)
```

The default configuration is intentionally extremely lightweight: 320 hidden dimensions, five heads of 64 dimensions each, four blocks, a 1,280-unit feed-forward layer, a 256-token context, and a 10,000-token synthetic vocabulary. The default is approximately 11.4 million trainable parameters, within the 5–15 million target and below the hard 20 million ceiling. All weights are initialized locally with a configurable normal-distribution standard deviation and optional seed. The learned positional embeddings were selected for simplicity and direct compatibility with a short fixed context; rotary or relative representations are deferred.

The concrete local model is intentionally not implemented as an LLM provider in Phase 2.2. Phase 2.3 adds a separate `FodciTokenizer` boundary; Phase 2.4 adds a separate local dataset boundary without adding a training loop, checkpoint, model download, inference runtime, or CLI integration.

## Phase 2.3 tokenizer boundary

The tokenizer converts text to IDs and back without normalization:

```text
Text
  ↓
FodciTokenizer
  ↓
Token IDs: 0 ... 9,999
  ↓
FodciModel
```

`FodciTokenizer` uses UTF-8 bytes as a permanent fallback and optionally learns deterministic byte-pair merges from a caller-provided in-memory corpus. The four special IDs are stable: `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, and `<EOS>=3`; raw byte tokens begin at ID 4. Because every input string is first encoded as UTF-8 bytes, arbitrary Unicode, Arabic, source code, whitespace, punctuation, and unseen symbols remain reversible. Tokenization never truncates to the model context; sequence truncation remains a later training responsibility.

Tokenizer training is deliberately separate from language-model training. `save()` and `load()` persist only a small versioned JSON vocabulary definition; no corpus, scraping, or generated artifact is committed.

## Phase 2.4 dataset pipeline

The dataset package consumes only a caller-provided local directory and keeps the transformation path deterministic and streaming-oriented:

```text
Raw local documents
        ↓
LocalDocumentLoader
        ↓
UTF-8, size, empty-content, and extension validation
        ↓
ExactDeduplicator (SHA-256, first file wins)
        ↓
FodciTokenizer
        ↓
EOS document boundary
        ↓
Context-length chunking
        ↓
TrainingExample(input_ids, target_ids)
```

`DatasetConfig` controls the input directory, supported extensions, maximum file size, context length, optional line-ending normalization, and EOS boundary behavior. `LocalDocumentLoader.iter_documents()` recursively discovers supported files in sorted relative-path order, validates each file, preserves source text by default, and yields one `Document` at a time. Invalid UTF-8, unreadable, oversized, empty, and whitespace-only files become structured `LoadIssue` records instead of crashing the complete scan.

`ExactDeduplicator` hashes decoded UTF-8 content with SHA-256 and retains the first occurrence in deterministic order. `TokenSequenceBuilder` tokenizes each document, optionally appends `<EOS>`, and emits fixed-size next-token examples without crossing document boundaries. `FodciDatasetPipeline.iter_samples()` composes these stages lazily; the batch `load_documents()` method exists only for explicit inspection and diagnostics. The package performs no downloads, scraping, training, optimization, model-weight loading, or autonomous behavior.

## Phase 2.5 training engine

Phase 2.5 introduces the first learning loop while preserving the existing model and dataset boundaries:

```text
FodciDatasetPipeline.iter_samples()
              ↓
bounded batch of TrainingExample
              ↓
input_ids / target_ids validation
              ↓
FodciModel(input_ids) → logits (B, T, V)
              ↓
flattened categorical cross-entropy
              ↓
backward → optional gradient clipping → AdamW step
              ↓
metrics and optional checkpoint
              ↓
validation: eval() + no_grad()
```

`TrainingConfig` keeps CPU as the default device and exposes epochs, an optional max-step budget, batch size, learning rate, weight decay, gradient norm, seed, logging and checkpoint intervals, validation interval, and output directory. `FodciTrainer` accepts a re-iterable or callable source so that streaming dataset pipelines can be recreated for each epoch without materializing the full dataset. It validates sequence length, equal input/target shapes, and vocabulary ranges before the model forward pass; it does not shift the target a second time.

The trainer uses standard PyTorch cross-entropy and AdamW only. Each checkpoint stores the model and optimizer state dictionaries, completed epoch, global step, serialized training configuration, and latest metrics. `resume()` restores these values and starts at the following epoch. Checkpoint output is directed to ignored artifact directories. The CPU smoke run validates engineering behavior—finite loss, gradients, parameter updates, validation, checkpoint loading, and resume—not useful language capability.

## Phase 2.6 Fodci Tiny v1 experiment

Phase 2.6 runs the first real training experiment on a small corpus authored locally for this repository:

```text
data/fodci_tiny_v1/train
        ↓
FodciDatasetPipeline + FodciTokenizer
        ↓
TrainingExample stream
        ↓
FodciTrainer (CPU, bounded max_steps)
        ↓
artifacts/checkpoints/fodci-tiny-v1

same pipeline, separate source
        ↓
validation baseline → final validation metrics
```

The train and validation directories are separate and are never merged by the experiment workflow. `scripts/run_fodci_tiny_v1.py` fingerprints each split from deterministic relative names and document content hashes, records exact file names and document/token/example counts, evaluates a fresh random model on the validation stream before optimization, runs a conservative CPU budget, records elapsed time and token counts, checks that parameters changed, and verifies loading the resulting checkpoint. `TrainingConfig.max_steps` provides an explicit upper bound in addition to the epoch count.

The model configuration remains the original 10,000-token vocabulary, 256-token context, 320 hidden dimensions, five attention heads, four blocks, and 1,280-unit feed-forward layer, totaling 11,424,400 parameters. The experiment writes a machine-readable JSON report and a tracked human-readable Markdown report; the checkpoint and JSON artifact are ignored by Git. This is a from-scratch engineering experiment, not a generation or capability evaluation.

## Phase 2.7 checkpoint management

Phase 2.7 places a dedicated `CheckpointManager` around the existing training state without creating duplicate model instances:

```text
model + optimizer + training metadata
                ↓
CheckpointMetadata identity/schema validation
                ↓
temporary .tmp file → fsync → atomic os.replace()
                ↓
ignored .pt checkpoint

inspect() → metadata only
load_model() → compatibility validation → model weights only
load() → compatibility validation → model + optimizer restoration
list() → latest()/best() from metadata progress/loss
```

The checkpoint envelope contains `format`, `format_version`, `model_version`, the full model configuration, tokenizer version, vocabulary size, context length, epoch, global step, serialized training configuration, metrics, seed, and UTC creation time, alongside model and optimizer state dictionaries. `inspect()` parses the metadata and required fields without constructing a model. `load()` maps tensors to the requested device and validates model version, tokenizer version, vocabulary, context length, hidden size, layer count, attention heads, feed-forward size, and activation before loading state into the supplied objects.

Saving never writes directly to the final path: it writes a unique temporary file, flushes it, and atomically replaces the destination. `list()` ignores malformed or incompatible files rather than presenting them as usable checkpoints, `latest()` orders valid entries by metadata global step/epoch/time, and `best()` chooses the lowest recorded `validation_loss` with a latest-step tie-break. All generated weights remain under ignored artifact directories and are excluded from Git.

## Phase 2.8 evaluation pipeline

Phase 2.8 adds an objective evaluation boundary without generation or inference:

```text
explicit validation source
          ↓
FodciEvaluator
  ├── model.eval()
  ├── torch.no_grad()
  ├── existing cross-entropy path
  └── loss + perplexity + examples + tokens + elapsed time
          ↓
EvaluationResult
          ↓
EvaluationComparison(random baseline, trained checkpoint)
```

`FodciEvaluator` evaluates a supplied re-iterable or callable `TrainingExample` source through the existing `FodciTrainer.evaluate()` implementation, so batching and loss mathematics are not duplicated. A fresh random model is evaluated first; a second model loads the trained checkpoint through `CheckpointManager`, which validates model/tokenizer compatibility before evaluation. Neither evaluation path calls `backward()` or an optimizer step, and tests snapshot both model parameters and optimizer state.

`EvaluationResult` records the explicit dataset path and split, optional document count and dataset hash, checkpoint identity/path, model and tokenizer versions, device, loss, guarded perplexity, evaluation example/token counts, elapsed time, epoch, and global step. `EvaluationComparison` reports trained-minus-baseline deltas and relative improvements. The evaluator can also score multiple checkpoint paths or select `CheckpointManager.best()` while keeping the validation source unchanged. JSON output stays under ignored `artifacts/reports/`; the human-readable report is tracked under `docs/experiments/`.

## Phase 8.1 evaluation task model

Phase 8.1 adds a declarative benchmark-definition layer without adding a benchmark runner or an evaluation runtime. Its boundary is:

```text
Stable task definition
        ↓
EvaluationTask
  ├── identity/category/difficulty
  ├── project definition
  ├── requirements and expected behavior
  ├── allowed scope and expected areas
  ├── declarative tests and success criteria
  ├── forbidden changes and constraints
  └── ground truth and metadata
        ↓
EvaluationTaskValidator
        ↓
immutable validation result + canonical JSON
```

`EvaluationTask` is an immutable frozen dataclass containing stable `EVAL-*` identity, semantic version, task description, user intent, `ProjectDefinition`, `Requirement`, `ExpectedBehavior`, `AllowedScope`, `ExpectedArea`, `TestDefinition`, `SuccessCriterion`, `ForbiddenChange`, `EvaluationConstraint`, `GroundTruth`, and deterministic metadata. Tuple snapshots and a read-only metadata mapping prevent callers from changing a definition after construction. No runtime state such as `EvaluationRun`, score, metrics, process, or result is stored in the task.

The model uses explicit enums for task category, difficulty, change type, expected-area type, test type, success-criterion type, and forbidden-change type. Ground truth preserves multiple valid implementation alternatives and records correctness through behavior, outcomes, interfaces, and invariants rather than one prescribed file or line. Evaluation constraints are declarative requirements and prohibitions; they are separate from `ExecutionBudget` and do not enforce anything in this phase.

`EvaluationTaskValidator` is pure and deterministic. It returns immutable `EvaluationTaskValidationResult` and `ValidationIssue` records and checks stable ID/version/text shape, enum values, project-definition fields, duplicate requirement/test/criterion/behavior IDs, references between definitions, path traversal and absolute paths, allowed/forbidden contradictions, expected areas, forbidden changes, constraints, ground-truth references, and metadata. It never repairs input, inspects a project, executes setup/tests/commands, invokes tools, installs packages, accesses the network, mutates Git, scores a task, or calls an LLM.

`EvaluationTask.to_json()` and `serialize_evaluation_task()` produce canonical UTF-8-preserving JSON with sorted keys, compact separators, deterministic collection order, and enum values. `create_evaluation_task()` constructs a declarative definition; `validate_evaluation_task()` is the explicit validator entry point. This phase is independent from future success-criteria evaluation, scoring, efficiency metrics, reports, version comparison, regression evaluation, and LLM judging. **Phase 8.1 is complete; Phase 8.2 consumes these definitions without changing them.**

## Phase 8.2 benchmark runner

Phase 8.2 adds bounded sequential execution and evidence collection over the Phase 8.1 task definitions. Its boundary is:

```text
EvaluationTask collection
        ↓
BenchmarkRequest + finite BenchmarkConfig
        ↓
validate task
        ↓
private temporary workspace
        ↓
explicit existing Fodci runtime adapter
        ↓
bounded state/evidence snapshots
        ↓
BenchmarkTaskRun collection
        ↓
raw BenchmarkRunSummary + BenchmarkResult
```

`BenchmarkRunner` is an orchestration layer, not a second agent implementation. It requires an explicitly supplied `BenchmarkRuntime` adapter and never changes `AgentLoop`, `ToolRegistry.default()`, CLI behavior, command policy, mutation policy, or the existing execution budget ledger. Tasks run in stable request order, one at a time. A request may supply a project-root template and fixture provider; each task gets a private workspace under a temporary benchmark directory, with `.git`, `.venv`, and cache directories excluded from template copying. The user’s real project is never used as a mutation target.

`BenchmarkConfig` validates finite host-controlled limits for task count, total and per-task wall time, artifact/evidence/log bytes, deterministic mode, cleanup, fail-fast, and continuation policy. Runtime adapters receive the per-task wall-time bound and remain responsible for enforcing their own existing `ExecutionBudgetLedger`, `CommandPolicy`, `ProcessManager`, `AutomaticTestOrchestrator`, `BoundedSelfCorrectionLoop`, and `FinalVerification` limits. The runner does not create or reset a budget ledger, retry, spawn background work, or run hidden subprocesses.

`BenchmarkTaskRun` records task status, lifecycle timing, task validation, workspace/project identity, cleanup status, changed/expected/unexpected/forbidden paths, mutation/test/completion/final-verification/stop evidence, failure/recovery/budget/policy evidence, bounded logs, bounded artifacts, and warnings. Fixture materialization is separated from agent mutation evidence, and runtime cache files are ignored. Passwords, tokens, API keys, secrets, authorization values, credentials, and private-key blocks are redacted before bounded logs/failures are stored. Skipped tasks after fail-fast or wall-time termination remain explicit records.

`BenchmarkRunSummary` contains raw counts only: total, passed, failed, blocked, timed out, skipped, unavailable, infrastructure-failure, and incomplete-evidence tasks. `BenchmarkResult` distinguishes `COMPLETED`, `FAILED`, `PARTIAL`, `BLOCKED`, and `TIMED_OUT` benchmark states and records termination reason, fail-fast trigger, deterministic task order, and the explicit `scoring: NOT_IMPLEMENTED_IN_PHASE_8_2` metadata. Execution completion is never interpreted as quality or success score. Phase 8.2 deliberately does not implement scoring, pass percentages, weighted metrics, model comparison, regression comparison, leaderboard, or LLM judging; those remain Phase 8.3/8.4 work.

## Phase 2.9 coding dataset

Phase 2.9 adds a higher-quality, small backend-engineering corpus while keeping the Transformer, tokenizer, training engine, checkpoint manager, and evaluation pipeline unchanged:

```text
data/fodci_coding/
├── train/       → coherent backend examples, services, repositories, tests, docs
└── validation/  → separate API, SQL, Docker, and test examples
        ↓
CodingDatasetManifestBuilder
        ↓
existing FodciDatasetPipeline
        ↓
manifest + split statistics + exact leakage identity
```

`CodingDatasetManifestBuilder` composes `FodciDatasetPipeline` twice, once for each explicit split. It preserves deterministic relative path ordering, UTF-8 source text, exact content hashes, EOS-aware token counts, and full-chunk training-example counts. The manifest records file-level bytes, characters, tokens, language/file-type distribution, duplicate and rejected-file counts, split SHA-256 values, tokenizer version, vocabulary size, context length, and a combined dataset SHA-256.

The loader now records unsupported extensions as structured issues instead of silently ignoring them. Strict manifest mode rejects missing or malformed split directories, rejected or duplicate files, tokenizer vocabulary mismatch, and any exact content-hash intersection between train and validation. The generated JSON manifest is tracked under `docs/datasets/`, while no model weights, checkpoints, training run, or generated artifacts are part of this phase.

## Phase 2.10 instruction training

Phase 2.10 adds instruction structure and response-only masking without changing the Transformer:

```text
### Instruction
      ↓
### Input
      ↓
### Response
      ↓
InstructionExample.parse
      ↓
InstructionDatasetPipeline
      ├── serialized context tokens
      ├── response boundary
      └── boolean target loss mask
      ↓
FodciTrainer causal cross-entropy
      ├── context tokens: conditioning only
      └── response + EOS: loss positions
      ↓
CheckpointManager + FodciEvaluator
```

The format uses ordinary textual headers rather than new special tokenizer tokens. `InstructionDatasetLoader` validates one example per local file, rejects malformed sections and duplicates, and keeps train/validation exact instruction hashes separate. `InstructionDatasetManifestBuilder` records the dataset version, tokenizer compatibility, serialized token count, response-token count, training-example count, split identity, and combined dataset SHA-256.

`TrainingExample.loss_mask` is optional for backward compatibility. Generic corpus examples retain an all-token loss, while instruction examples set false for instruction/input target positions and true for response target positions, including the response EOS boundary. `FodciTrainer` reduces cross-entropy over active mask positions and reports effective response-token counts; no separate optimizer or model path is introduced. `FodciEvaluator` labels these metrics as `response_only` and exposes `response_loss` while retaining the existing no-grad/eval safety guarantees.

The Phase 2.10 workflow is a bounded CPU smoke run from random initialization. It records model/dataset/tokenizer versions, dataset SHA-256, seed, optimizer settings, steps, device, time, checkpoint identity, and before/after response-only validation metrics. It deliberately does not add generation, inference, chat, CLI integration, Agent behavior, or Phase 3 functionality.

## Phase 2.11 local inference

Phase 2.11 adds a local inference boundary without changing the model or tokenizer architecture:

```text
prompt
  ↓
FodciTokenizer.encode()
  ↓
InferenceEngine
  ├── prompt context validation
  ├── FodciModel.eval()
  ├── torch.inference_mode()
  ├── final-position logits
  └── greedy or optional temperature/top-k selection
  ↓
EOS / max_new_tokens / context stop
  ↓
FodciTokenizer.decode(generated IDs)
  ↓
InferenceResult
```

`InferenceEngine` receives an existing `FodciModel` and `FodciTokenizer`. If a checkpoint is configured, it uses `CheckpointManager.load_model()` to validate metadata and restore only model weights; inference never creates an optimizer, steps it, or exposes optimizer state. Model version, tokenizer version, vocabulary size, context length, and structural model fields are validated before loading.

The default is CPU greedy decoding with `temperature=1.0`, `do_sample=False`, EOS stopping, and a bounded `max_new_tokens`. Optional seeded multinomial sampling supports positive finite temperature and optional positive `top_k`; `top_k` is filtered before sampling. Prompts are encoded without truncation, and an empty or over-context prompt fails clearly. Generation stops when EOS is selected, the new-token budget is exhausted, or the context window is full. `InferenceResult` exposes only generated text, counts, stop reason, model version, checkpoint identity, and effective configuration.

The real smoke workflow uses the existing ignored Tiny v1 checkpoint on CPU with short English, Python, and backend prompts. It validates the checkpoint → model → tokenizer → autoregressive decoding path only; generated text is not evidence of intelligence or production readiness. No CLI integration, Agent loop, tools, memory, file operations, or Phase 2.12/3 functionality is included.

## Phase 2.12 CLI integration

Phase 2.12 connects the existing local inference path to the existing terminal application without changing the model architecture or introducing a second provider abstraction:

```text
User input
    ↓
fodci CLI entry point
    ↓
Application.start()
    ├── bootstrap settings + logging
    ├── resolve ProjectContext(root)
    └── FodciLocalProvider.from_checkpoint(root/artifacts/checkpoints/fodci-tiny-v1.pt)
            ↓ one construction per application
    FodciModel + FodciTokenizer + InferenceEngine
            ↓
InteractiveSession
    ├── /help and /exit through CommandDispatcher
    └── normal input → LLMRequest → FodciLocalProvider → LLMResponse
            ↓
Fodci > generated text
```

`FodciLocalProvider` is structurally an `LLMProvider`: it receives the existing typed message request, formats a deterministic instruction-style prompt, delegates to the existing `InferenceEngine.generate()`, and returns the existing `LLMResponse`. The minimal system prompt identifies Fodci as a local backend-engineering model, asks for concise and honest behavior, and forbids claiming tools, file inspection, or command execution. It does not add tool instructions or project-analysis behavior.

`InteractiveSession` keeps system, user, and assistant messages only for the active process. History is bounded to a deterministic even number of messages, old complete user/assistant turns are removed from the left when the bound is reached, and nothing is persisted to disk. User prompts are never silently truncated; the inference engine returns a clear context-limit failure when the formatted prompt cannot fit. Assistant whitespace is retained as valid history because the tiny checkpoint may generate whitespace.

Provider construction fails clearly for missing, malformed, or incompatible checkpoints. The CLI catches normal startup failures and Ctrl+C without exposing a traceback, never falls back to random weights, and never downloads or creates a replacement checkpoint. Session inference failures are rendered as `Fodci error: ...`, while `/help`, `/exit`, EOF, and the standard-library terminal UX remain unchanged.

The integration suite verifies provider protocol behavior, one-time construction, multi-turn history, model/optimizer safety, checkpoint errors, no-network scope, and the real subprocess flow `Hi` → `Fodci > ...` → `/exit`. This phase intentionally does not implement project understanding, tools, file operations, terminal execution, code search, planning, RAG, memory, tool calling, or autonomous loops.

## Phase 3.1 file discovery tool

Phase 3.1 introduces the first concrete Agent tool without connecting it to the LLM, `ProviderBackedAgent`, or the `fodci` interactive loop:

```text
Explicit project_root
        ↓
ListFilesTool.run(arguments)
        ↓ validation + ToolError boundary
list_files(project_root)
        ↓
read-only deterministic traversal
        ├── regular files → DiscoveredFile metadata
        ├── directories   → DiscoveredDirectory metadata
        ├── default/custom ignore policy
        ├── hidden-file policy
        ├── symlink skip policy
        └── max_files/max_directories/max_depth bounds
        ↓
FileDiscoveryResult
```

`backend_ai.tools` reuses the existing core `Tool` protocol and adds package-owned `ToolMetadata`, `ToolError`, and stable `ToolErrorCode` values. `ListFilesTool` exposes the name `list_files`, a description, and an explicit input schema requiring `project_root`. The direct `list_files()` function accepts the same root plus bounded options and returns immutable dataclass records rather than a formatted string.

The root is expanded and normalized explicitly; missing roots, file-shaped roots, invalid arguments, permission failures, and filesystem failures become structured errors. Results use root-relative POSIX paths, separate files from directories, include cheap file metadata only, and use case-folded relative-path ordering with a stable tie-break so Windows and Linux have equivalent ordering as reasonably as possible. The tool never reads complete file contents.

The default ignore set is centralized and deliberately small: `.git`, `__pycache__`, `node_modules`, `.venv`, `venv`, `env`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.coverage`, `dist`, `build`, and `.eggs`. Hidden files are included by default except for ignored entries; `include_hidden=False` excludes dot-prefixed entries. Custom ignored directory names extend the defaults. Full `.gitignore` semantics are intentionally not implemented or claimed in this phase.

All symbolic links are skipped, including symlinked files, directories outside the root, and recursive links. Special filesystem entries such as sockets and devices are ignored rather than treated as regular files. The result reports `truncated=True` and a reason when `max_files`, `max_directories`, or `max_depth` stops traversal, so bounded discovery never silently presents a complete-looking result.

The tool layer owns filesystem access. The LLM does not receive filesystem APIs, the CLI does not invoke the tool automatically, and no Agent loop or tool-calling protocol is added. `ProjectContext` remains a validated root-only dataclass and is not expanded with file lists or project analysis. Later phases may add `search_code` and orchestration, but they are intentionally absent here.

## Phase 3.2 read file tool

Phase 3.2 adds the second concrete filesystem tool while reusing the Phase 3.1 `Tool`, `ToolError`, `ToolErrorCode`, root validation, and path-safety conventions:

```text
Explicit project_root + relative path
              ↓
ReadFileTool.run(arguments)
              ↓ validation + shared ToolError boundary
read_file(project_root, path, max_bytes)
              ├── normalize root-relative request
              ├── reject traversal/absolute escape
              ├── reject every symlink component
              ├── validate regular file
              ├── check bounded byte size
              ├── read binary once
              └── decode strict UTF-8
              ↓
ReadFileResult(relative_path, file_name, content, encoding, size_bytes)
```

`ReadFileTool` remains structurally compatible with the existing core `Tool` protocol and exposes explicit metadata requiring `project_root` and `path`, with an optional non-negative `max_bytes`. The direct `read_file()` function returns an immutable `ReadFileResult`; it does not print content or log source bodies. The default maximum is 1 MiB. Files exactly at the limit are accepted, while larger files fail with `FILE_TOO_LARGE` before partial content can be returned; a second bounded read check handles growth between stat and open.

The tool reads regular files as bytes and decodes strictly as UTF-8. It preserves spaces, tabs, indentation, Unicode/Arabic text, CRLF/LF line endings, and final-newline behavior. Invalid byte sequences return `INVALID_UTF8` rather than using replacement or ignore modes. No BOM stripping is performed, so BOM behavior remains predictable through normal UTF-8 decoding. Directories, FIFOs, sockets, devices, and other non-regular entries return `NOT_A_FILE`.

Path handling uses normalized `pathlib` semantics rather than string-prefix checks. Relative `.`/`..` segments and mixed separators are normalized; absolute paths are allowed only when they remain within the explicit root, and Windows drive/UNC-looking paths cannot bypass it. Every symlink component is rejected, including internal links, external links, broken links, and loops, matching Phase 3.1's safer skip policy. The tool never mutates, executes, downloads, or accesses the network.

Phase 3.2 does not add `search_code`, grep/ripgrep/regex/AST search, ProjectContext expansion, framework detection, planning, LLM tool-calling, memory, RAG, terminal execution, file modification, or an Agent loop. The LLM, inference engine, tokenizer, checkpoint manager, and existing `fodci` terminal application remain independent from filesystem access.

## Phase 3.3 code search tool

Phase 3.3 adds a third standalone tool over the same filesystem boundary. It does not connect the tools to the LLM or introduce orchestration:

```text
Explicit project_root + query + optional scope
                    ↓
SearchCodeTool.run(arguments)
                    ↓ validation + shared ToolError boundary
search_code(project_root, query, path, options)
                    ↓
Deterministic scope traversal
                    ├── default generated/dependency exclusions
                    ├── symlink and special-entry skip policy
                    ├── bounded UTF-8 binary reads
                    ├── literal or explicit regex line matching
                    └── max results / bytes / depth / directories
                    ↓
SearchCodeResult
    └── SearchMatch(path, line, line_number, column_start, column_end)
```

`SearchCodeTool` reuses the core `Tool` protocol, package-owned `ToolMetadata`, `ToolError`, and `ToolErrorCode`. Its required inputs are `project_root` and `query`; optional inputs are `path`, `max_results`, `max_file_bytes`, `case_sensitive`, and `use_regex`. The default mode is literal substring search. Regex is compiled only when `use_regex=True`, and malformed expressions return `INVALID_REGEX` rather than escaping or crashing. Literal mode escapes regex metacharacters. Columns are 0-based Unicode code-point offsets on the returned source line; line numbers are 1-based.

Search scope uses the explicit root or an explicit root-relative file/directory path. Root/path normalization rejects traversal, absolute escape, Windows drive/UNC bypass, mixed-separator escape, and symlink components. Directory traversal uses the centralized Phase 3.1 exclusions and deterministic normalized relative-path order. Explicitly selecting an excluded directory still yields no searched files, preserving the default exclusion policy. Symlinks and special entries are never followed or searched.

The implementation reads regular files in bounded binary chunks, checks `max_file_bytes` before and during reading, and decodes strict UTF-8. Invalid UTF-8 files are skipped with `skipped_reasons=("invalid_utf8",)`; no replacement or ignore decoding is used. Oversized files are skipped without partial matches and mark the result truncated with `max_file_bytes`. A result distinguishes no matches from bounded/truncated search using `truncated` and `truncation_reason`; other reasons include `max_results`, `max_depth`, and `max_directories`. Query length and maximum result/file-byte options are themselves bounded to prevent untrusted input from requesting unbounded work.

The tool returns only matching source lines and match coordinates, never complete file content, and never prints or logs source. It does not invoke grep, ripgrep, subprocesses, shell commands, network APIs, project imports, mutation, or execution. `ProjectContext` remains root-only, and the existing `fodci` application is unchanged. Later phases may add orchestration, but Phase 3.3 intentionally does not implement search selection by the LLM, project understanding, file modification, terminal execution, memory, RAG, or an Agent loop.

## Phase 3.4 project structure tool

Phase 3.4 adds structural detection as a fourth standalone tool. It reuses `list_files` for the bounded deterministic inventory and `read_file` for a small allowlist of known dependency/configuration/entry-point files:

```text
Explicit project_root
        ↓
ProjectStructureTool.run(arguments)
        ↓ validation + shared ToolError boundary
project_structure(project_root, bounded options)
        ├── list_files inventory
        ├── classify files/directories/languages
        ├── select known non-sensitive inspection candidates
        ├── read bounded UTF-8 evidence only
        ├── run modular evidence detectors
        └── sort all logical output
        ↓
ProjectStructureResult
```

`ProjectStructureResult` is immutable and contains project type, framework detections, language counts, package managers, databases, test frameworks, infrastructure, classified directories, important/config/dependency files, test/source directories, entry points, overall confidence, evidence, warnings, and truncation metadata. Individual `Detection` records carry a name, confidence (`high`/`medium`), and sorted evidence strings. Evidence is tied to observed paths or bounded content matches; directory names such as `django/` or files such as `flask.py` do not claim frameworks by themselves.

The detector covers generic Python/Node/JavaScript/TypeScript projects, Django, FastAPI, Flask, Express, React, PostgreSQL, MySQL, MariaDB, SQLite, MongoDB, pytest, unittest, Jest, Vitest, generic test structure, Docker, Docker Compose, common CI, common package managers, language counts, major directory categories, important files, and likely entry points. It is heuristic and structural, not a static analyzer or a certainty claim.

The scan is bounded by Phase 3.1 discovery limits plus `max_file_bytes` (default 64 KiB, maximum 1 MiB) and `max_inspected_files` (default 64, maximum 256). Sensitive names such as `.env`, credential/secret/private/password files, and key/certificate files are never passed to `read_file`; their contents are never returned. The tool preserves the shared root/path/symlink safety model and never executes, imports, mutates, downloads, logs, or accesses the network.

Every result is deterministic: inventory paths, classifications, languages, detections, evidence, important files, and entry points are sorted with normalized relative paths. Discovery truncation and bounded evidence warnings are explicit. The tool is not connected to the LLM, `InferenceEngine`, model, tokenizer, checkpoint manager, `ProjectContext`, or Agent loop. It does not implement deep project understanding, AST/dependency graphs, planning, file modification, terminal execution, memory, RAG, or later Phase 3 behavior.

## Phase 3.5 canonical project context

Phase 3.5 adds a canonical context layer over the structural tool output without adding orchestration:

```text
Explicit project_root
        ↓
ProjectContextTool.run(arguments)
        ↓ validation + shared ToolError boundary
ProjectContextBuilder.build(...)
        ↓
project_structure(...)
        ↓
ProjectStructureResult
        ↓ safe structural projection
ProjectContext
```

`ProjectContextBuilder` composes `project_structure`; it does not implement a second filesystem scanner and does not execute `list_files`, `read_file`, or source code independently. The structure result now also exposes normalized `project_files`, allowing the context to preserve a bounded project inventory without raw contents or absolute file paths.

`ProjectContext` is an immutable dataclass. It separates structural facts (root, project type, languages, detections, directories, files, entry points) from derived context (`stack_summary`), traceability (`evidence`), quality (`confidence`), limitations (`warnings`), and completeness (`truncated`, `truncation_reason`, `completeness`). The stack summary is a deterministic join of evidence-backed language/runtime, framework, database, test, and infrastructure names; when no meaningful stack evidence exists it returns an explicit insufficient-evidence label.

Context inherits the structure tool's root normalization, symlink/path safety, default exclusions, sensitive-file protection, and bounded discovery. Targeted inspection limits from Phase 3.4 are promoted to partial context when warnings indicate `max_inspected_files` or byte limits. No source bodies, credentials, API keys, `.env` content, private keys, or certificates enter the context. Serialization through `to_dict()` is deterministic and JSON-compatible.

`ProjectContextTool` is a standalone implementation of the existing `Tool` protocol with explicit `project_root` and the same bounded discovery/inspection options. It is not connected to the model, tokenizer, inference engine, checkpoint manager, CLI, LLM tool-calling, planning, memory, RAG, terminal execution, file mutation, or Agent loop. Phase 3.5 establishes a canonical data layer only; later phases may consume it.

## Phase 3.6 first bounded Agent loop

Phase 3.6 adds the first real orchestration layer without claiming mature autonomous reasoning:

```text
User task + explicit project_root
        ↓
AgentLoop
        ├── project_context (initial read-only tool)
        ├── compact context budget using existing tokenizer
        ├── InferenceEngine.generate()
        ├── strict FINAL/ACTION parser
        ├── ToolRegistry lookup and existing Tool validation
        ├── bounded ToolResult injection
        └── repeat until final answer or explicit stop
```

The model-facing action protocol is intentionally deterministic:

```text
FINAL: answer text
```

or:

```text
ACTION: search_code
ARGS: {"query":"FastAPI"}
```

A response without `ACTION:` is a final answer; `ACTION:` must be followed by one valid identifier-like tool name and one JSON object on `ARGS:`. Free-form JSON, arbitrary natural-language calls, unknown tools, malformed JSON, missing arguments, and invalid tool arguments are never executed. Parser failures become structured Agent errors, while tool failures become `ToolResult(success=False, error_code, message)` and may be observed by the next model step.

`ToolRegistry` owns only deterministic discovery, metadata, lookup, and dispatch. Its default order is `list_files`, `project_context`, `project_structure`, `read_file`, `search_code`; it does not duplicate or bypass any tool implementation. `AgentLoop` always establishes context through `project_context`, rewrites/validates every later call against the explicit root, and never dispatches a call outside that boundary.

Agent state is immutable and serializable through `AgentMessage`, `AgentTask`, `AgentStep`, `ToolCall`, `ToolResult`, `AgentUsage`, and `AgentResult`. `AgentResult` records the final answer, status, complete bounded history, project context, stop reason, usage, warnings, and errors.

`AgentConfig` sets `max_steps=8`, `max_tool_calls=8`, a 256-token maximum context with reserved response space, bounded tool-result characters, and bounded history. `ContextBudget` uses the existing tokenizer to estimate prompt tokens. It shrinks optional project fields, drops oldest history deterministically, bounds tool-result text with `[tool_result_truncated]`, and returns `context_limit` if the task itself cannot fit; it never silently truncates required task text.

The implementation reuses the existing `InferenceEngine` and does not create a second model runtime. The current tiny model may return empty or non-protocol text; that is represented as a completed empty final answer or a structured invalid-action/inference result rather than fabricated tool use. The existing `fodci` CLI remains unchanged in Phase 3.6; the public `AgentLoop` API is the clean integration boundary and preserves the Phase 2 provider-backed CLI behavior.

The loop is strictly read-only. It cannot create, edit, or delete files; execute shell/commands/tests; install packages; change Git; access the network; call external LLMs; use memory/RAG; or start background/autonomous loops. Phase 3.6 closes the first Agent foundation only; it is not a full autonomous coding agent.

## Phase 4.1 create-only write tool

Phase 4.1 adds one narrowly scoped mutating primitive without changing the AgentLoop:

```text
write_file(project_root, relative_path, UTF-8 content)
        ↓
validate root, path, parent, symlink components, content bytes, max_bytes
        ↓
write + flush + fsync private temporary file
        ↓
exclusive atomic publish to the absent target
        ↓
WriteFileResult or structured ToolError
```

`WriteFileTool` implements the existing `Tool` protocol. It requires an explicit existing root, refuses traversal/absolute-outside paths and symlink components, accepts only string content encodable as UTF-8, and applies a bounded byte limit. Missing parent directories may be created one component at a time, only inside the root and only up to the bounded depth; the root itself is never created. It never overwrites files, directories, FIFOs, or other existing paths. The implementation writes to a private `0o600` temporary file, flushes and synchronizes it, and publishes it with an exclusive hard link; temporary artifacts and newly-created empty parents are cleaned on handled failures.

`WriteFileResult` is immutable and serializable, containing the root-relative path, filename, `size_bytes`, UTF-8 encoding, and creation status. The default maximum content size is 1 MiB and the default maximum newly-created parent depth is 32. `FILE_EXISTS`, `PATH_NOT_FOUND`, `NOT_DIRECTORY`, `PATH_OUTSIDE_ROOT`, `INVALID_UTF8`, `FILE_TOO_LARGE`, `PERMISSION_DENIED`, `FILESYSTEM_ERROR`, and `INVALID_ARGUMENT` remain machine-readable through the shared `ToolError` boundary.

`ToolRegistry.default()` intentionally remains the five-tool Phase 3 read-only registry. `ToolRegistry.with_write_file()` is an explicit opt-in registry for direct Phase 4.1 consumers. `AgentLoop` still calls `ToolRegistry.default()` when no registry is injected, so no model-generated action can write a file merely because Phase 4.1 is installed. No edit/delete/diff/Git/command/test/execution or autonomous modification workflow is included.

## Phase 4.2 exact edit tool

Phase 4.2 adds a targeted existing-file edit primitive rather than an ambiguous whole-file write:

```text
edit_file(root, path, old_content, new_content)
        ↓
validate existing regular UTF-8 target and bounded inputs
        ↓
count exact literal matches
        ├── 0 → MATCH_NOT_FOUND, unchanged
        ├── >1 → AMBIGUOUS_MATCH, unchanged
        └── 1 → construct exact replacement in memory
        ↓
optimistic snapshot verification
        ↓
write + fsync private temporary replacement
        ↓
permission-preserving atomic os.replace
        ↓
EditFileResult or structured ToolError
```

`EditFileTool` reuses the Phase 3/4.1 root and symlink validation conventions. The target must be an existing readable and writable regular file inside the explicit root. Matching is literal, case-sensitive, exact decoded UTF-8 text; whitespace, line endings, indentation, Unicode, and final-newline state are not normalized. Empty old text, fuzzy/regex matching, and whole-file replacement are rejected. A no-op where `old_content == new_content` returns `changed=False` without rewriting the inode.

The default 1 MiB bounds independently cover the existing file, old text, new text, and resulting file. The immutable `EditFileResult` reports relative path, filename, original/new byte sizes, signed byte delta, match count, selected occurrence, and changed status. Errors use the shared `ToolError` system with `FILE_NOT_FOUND`, `MATCH_NOT_FOUND`, `AMBIGUOUS_MATCH`, `INVALID_UTF8`, `FILE_TOO_LARGE`, `PATH_OUTSIDE_ROOT`, `NOT_A_FILE`, `PERMISSION_DENIED`, `CONCURRENT_MODIFICATION`, `INVALID_ARGUMENT`, and `FILESYSTEM_ERROR` as applicable.

Real edits preserve the original permission mode, including executable bits, by writing a complete private temporary file, flushing and synchronizing it, then atomically replacing the target. The original remains unchanged for validation, matching, encoding, size, temporary-write, or replacement failures; temporary files are cleaned. An optimistic check compares device/inode/size/mtime/ctime fingerprints and SHA-256 content identity before replacement. This prevents overwriting changes observed during preparation; a filesystem race after the final check remains platform-dependent and is explicitly not claimed to be absolutely race-free.

`ToolRegistry.with_file_modification()` is an explicit Phase 4.2 registry containing the read-only tools plus `write_file` and `edit_file`. `ToolRegistry.default()` and `ToolRegistry.with_write_file()` remain unchanged. `AgentLoop` does not automatically receive or invoke `edit_file`; no planning, self-correction, deletion, Git, command execution, or autonomous modification workflow is included.

## Phase 4.3 regular-file deletion

Phase 4.3 adds one narrowly bounded deletion primitive:

```text
delete_file(root, relative_path)
        ↓
validate root, path, existing target, and symlink components
        ↓
lstat target: regular file only
        ↓
open parent directory with no-follow flags where supported
        ↓
revalidate target identity immediately before unlink
        ↓
unlink target entry only
        ↓
DeleteFileResult or structured ToolError
```

`DeleteFileTool` reuses the existing `Tool` protocol, root/path validation, and shared `ToolError` system. It requires an explicit existing root and existing target, never reads file contents, never creates directories, never deletes directories or parents, never recurses, and never follows or deletes symlinks. Directories, FIFOs, sockets, devices, broken links, traversal paths, and outside-root paths are rejected. `DeleteFileResult` is immutable and reports only the relative path, filename, original metadata size, and `deleted` status.

The implementation uses `lstat` and `stat(..., follow_symlinks=False)` and compares device, inode, mode, size, and modification time before `unlink`. Where supported, the parent directory is opened with `O_DIRECTORY | O_NOFOLLOW` and unlink is performed relative to that descriptor. If the entry changes during the checked operation, `CONCURRENT_MODIFICATION` is returned and the target is not intentionally deleted. Filesystem races after the final check cannot be made absolutely race-free across every platform, so no stronger guarantee is claimed. Parent directories and unrelated files are never removed by this tool.

`ToolRegistry.with_file_modification()` is extended additively to contain the read-only tools plus `write_file`, `edit_file`, and `delete_file`. `ToolRegistry.default()` and `ToolRegistry.with_write_file()` remain unchanged, and `AgentLoop` does not automatically receive or invoke deletion. Phase 4.3 does not include backups, diffs, Git, command/test execution, shell/subprocess access, network, memory, RAG, planning, or autonomous modification loops.

## Phase 4.4 Safe Editing Infrastructure

Phase 4.4 introduces `backend_ai.tools.safe_editing` as a reusable layer above the three existing mutation tools, not as a second tool framework:

```text
Explicit caller + SafeEditPolicy
             ↓
       SafeEditSession
       ├── FileSnapshot / identity + bounded SHA-256
       ├── bounded internal DiffResult
       ├── optional controlled BackupResult
       ├── existing write_file / edit_file / delete_file
       └── post-operation verification
             ↓
       immutable SafeEditResult
```

`SafeEditPolicy` is conservative by default. Create, edit, and delete capabilities are disabled until explicitly allowed; roots remain explicit, symlinks remain rejected, atomic writes and metadata preservation remain required, and verification/concurrency detection cannot be disabled. Resource limits bound file/content hashing and diff bytes/lines. The policy is configuration, not permission to bypass the underlying tools.

`FileSnapshot` contains root-relative path, existence, size, mtime, device/inode identity, file type, mode, and an optional SHA-256 hash. It is immutable, deterministic, and does not expose file contents. `SafeEditSession` compares snapshots before mutation and verifies expected state afterward. A changed identity or hash fails with `CONCURRENT_MODIFICATION`; this is optimistic protection and is not claimed to be race-free after the last filesystem check.

`DiffResult` is an internal deterministic unified diff for create/edit/delete. It uses only relative `a/` and `b/` paths and never invokes Git, subprocesses, or external commands. It is bounded by byte and line limits and reports `truncated=True` with an explicit marker when necessary. Diffs are returned only when explicitly requested through the session and are never logged or printed automatically.

When enabled, backups are created only for existing edit/delete targets, only inside project-relative `.fodci/backups/`, with hashed deterministic names, bounded reads, exclusive atomic creation, and restrictive permissions. A successful operation removes the backup unless `retain_backup_on_success=True`; a failed mutation leaves the backup for inspection. This is a controlled snapshot mechanism, not transactional rollback or a claim of multi-operation atomicity.

`SafeEditSession.create/edit/delete` delegates the actual mutation to the existing `write_file`, `edit_file`, and `delete_file` implementations, then returns immutable `SafeEditResult` metadata including operation, path, success/change flags, sizes, hashes, optional bounded diff, backup metadata, and verification status. The layer is not registered as a Tool, `ToolRegistry.default()` remains read-only, `ToolRegistry.with_file_modification()` remains explicit, and `AgentLoop` is not modified to mutate files.

## Phase 4.5 read-only Git diff

Phase 4.5 adds `backend_ai.tools.git_diff` as a read-only Tool-protocol integration:

```text
git_diff(explicit_project_root)
          ↓
  GitReadOnlyAdapter
  ├── repo-root / HEAD / branch inspection
  ├── porcelain status parsing
  ├── staged and unstaged diff reads
  └── staged and unstaged numstat reads
          ↓
  bounded deterministic GitDiffResult
```

`GitReadOnlyAdapter` accepts only fixed argv tuples for read-only Git operations. It uses `subprocess.Popen` with `shell=False`, `stdin=DEVNULL`, explicit repository cwd, `GIT_OPTIONAL_LOCKS=0`, C locale, byte/time output bounds, and a hard timeout. It does not accept arbitrary command strings, invoke a general shell, initialize repositories, follow project configuration, or expose Git mutation commands.

`GitDiffResult` is immutable and contains repository detection, optional branch/HEAD, sorted `GitChangedFile` records, separate staged/unstaged/combined unified diff text, insertions/deletions, truncation state/reason, and warnings. Status parsing handles staged, unstaged, staged-plus-unstaged, untracked, added, deleted, renamed/copy metadata, and binary numstat markers. Untracked files are listed structurally without being read into synthetic full diffs, and binary content is never decoded into the result.

The explicit root must itself be the repository returned by `git rev-parse --show-toplevel`; a non-Git root returns `is_git_repository=False` rather than selecting an unrelated parent repository. All returned paths are repository-relative POSIX paths. Diff bytes, diff lines, changed-file count, command output, and timeout are independently bounded. `ToolRegistry.with_git_inspection()` is opt-in, `ToolRegistry.default()` remains read-only, and `AgentLoop` is not given automatic Git access.

## Phase 4.6 read-only Git status

Phase 4.6 reuses the same `GitReadOnlyAdapter` and adds only two whitelisted read operations: porcelain status with branch headers, and the same command with explicit ignored entries. `GitStatusTool` requires an explicit root and returns immutable `GitStatusResult` metadata.

The parser consumes NUL-delimited `--porcelain=v1 --branch` records. It does not split filenames on whitespace, so spaces, tabs, Unicode, quotes, and punctuation remain part of the repository-relative path. Each `GitStatusFile` preserves the two-character index/worktree status codes and derives stable classifications for staged, unstaged, untracked, ignored, renamed, deleted, added, modified, and common unmerged conflict states. Rename records keep old and new paths.

Branch/HEAD semantics are explicit: normal committed branches are `head_state="branch"`, detached commits are `"detached"`, and an unborn branch is `"unborn"` with `head=None`. Local upstream and ahead/behind information is reported only when Git exposes it in local status metadata; no network is contacted and unavailable values remain `None`. Ignored entries are excluded by default and included only with `include_ignored=True`; their contents are never read.

Status records are bounded by maximum files, command output bytes, path length, and timeout. Truncation is marked and explained. Non-Git roots return structured `is_git_repository=False`, while Git availability, command failure, and timeout use shared `ToolError` codes. `ToolRegistry.with_git_inspection()` exposes both read-only Git tools, `ToolRegistry.default()` remains unchanged, and `AgentLoop` receives no automatic Git Status capability.

## Phase 4.7 read-only modification verification

Phase 4.7 adds an additive verifier around the existing Safe Editing layer:

```text
ExpectedModification records + optional FileSnapshot baseline
                         ↓
          ModificationVerifier / verify_modification
                         ↓
      strict root/path/lstat/bounded hash + UTF-8 checks
                         ↓
     immutable ModificationVerificationResult
                         ↓
       SafeEditResult.verification metadata (optional caller view)
```

`ExpectedModification` makes the expected intent explicit as `created`, `modified`, `deleted`, or `unchanged`. It can carry expected UTF-8 content, expected byte size, expected SHA-256, and a pre-mutation `FileSnapshot`; expected content is used only for strict comparison and is never returned in result serialization. `ModificationVerificationItem` reports repository-relative path, expected/actual state, status, sizes, hashes, type, and a bounded diagnostic message without source content.

The verifier resolves an explicit project root and relative path using existing path normalization. Parent symlinks, traversal, absolute escapes, Windows/UNC bypasses, and NUL paths are rejected. The final entry is inspected with `lstat` without following symlinks, allowing the verifier to classify a replaced symlink, directory, FIFO, socket, device, or other special entry as a type change rather than reading through it. Regular files are hashed with bounded reads and decoded with strict UTF-8 only; replacement and ignored decoding errors are forbidden.

Target statuses are deterministic: successful expected states are `VERIFIED`; missing targets, content/hash mismatches, type changes, unexpected modifications/creations/deletions, invalid UTF-8, and unavailable verification each receive an explicit machine-readable status. A baseline mapping enables bounded comparison of non-target files through the existing `list_files` discovery limits, detecting unexpected modifications, creations, and deletions outside intended targets. Without a baseline, target verification can still succeed when `detect_unexpected=False`, but project-wide completeness is explicitly reported as unavailable/incomplete rather than inferred.

`SafeEditSession.create/edit/delete` continues to delegate all mutation to the Phase 4.1–4.3 tools. After successful mutation, it attaches a read-only `ModificationVerificationResult` to `SafeEditResult.verification` while preserving prior result fields and mutation behavior. The verifier itself performs no mutation, directory creation, subprocess execution, network access, project-code execution, Git access, or AgentLoop integration; `ToolRegistry.default()` remains conservative.

## Phase 4.8 modification transaction and recovery

Phase 4.8 adds a single-operation transaction boundary above `SafeEditSession`:

```text
ModificationOperation
        ↓ planned → snapshotted → executing
SafeEditSession mutation + existing atomic publication/backups
        ↓
ModificationVerifier post-state check
        ↓
committed  OR  failed/recovery_required
        ↓
conservative RecoveryResult when provably safe
```

`ModificationOperation` records only one explicit create/edit/delete plan and its immutable lifecycle metadata. `ModificationTransactionResult` reports operation records, committed/failed/recovered paths, verification, recovery state, warnings, errors, completeness, and recoverability. It intentionally does not support multi-file rollback or claim filesystem-wide transactional guarantees.

The transaction reuses `SafeEditSession`, including its existing atomic same-directory temporary publication, fsync, permissions, `FileSnapshot`, backup, concurrency, and Phase 4.7 verification mechanisms. It does not duplicate an atomic writer or backup store. Successful mutations are verified before backup cleanup. A failure before publication leaves the original target unchanged and cleans the controlled backup when safe. If a mutation appears to have published the expected state but finalization fails, the transaction enters `recovery_required` rather than silently discarding evidence.

Edit recovery is allowed only when the current target still has the exact transaction-generated hash/size and the controlled backup remains safely rooted and valid strict UTF-8. The existing exact `edit_file` path then restores the pre-mutation bytes atomically and the pre-snapshot is reverified. If the current target differs, the result is `user_change_preserved` and no overwrite is attempted. Delete recovery is explicitly `recovery_unavailable` because recreating a deleted file cannot be proven safe under this architecture. Temporary paths and backups are bounded, project-relative, permission-restricted, and cleaned after successful finalization; cleanup failures remain visible in structured results.

No transaction/recovery class is registered as an Agent Tool, `ToolRegistry.default()` remains unchanged, `AgentLoop` receives no automatic mutation capability, and Git inspection remains read-only and independent from recovery.

## Phase 5.1 command execution foundation

Phase 5.1 adds a low-level process primitive behind an explicit opt-in boundary:

```text
CommandRequest / explicit argv
          ↓
root + working-directory validation
          ↓
subprocess.Popen(argv, shell=False, stdin=DEVNULL)
          ↓
bounded stdout/stderr capture + timeout
          ↓
immutable CommandResult
```

`CommandRequest` accepts only an argv sequence, explicit `project_root`, explicit `working_directory`, optional environment overlay/inheritance, timeout, and independent stdout/stderr byte limits. A shell command string is rejected. No pipes, redirects, `&&`, `||`, glob expansion, command substitution, shell variables, `bash -c`, `sh -c`, `cmd.exe`, or PowerShell interpretation is implemented.

Working-directory validation reuses the existing root and path conventions. The root must exist; the working directory must be a real directory inside it. Traversal, absolute escapes, Windows drive/UNC bypasses, mixed-separator bypasses, NUL paths, and symlink components/final entries are rejected. Result fields expose only a project-relative working-directory representation; environment values are never serialized or logged.

`CommandResult` distinguishes start failure, executable-not-found, permission failure, invalid argument/working directory, non-zero exit, timeout, and output-limit termination. stdout and stderr remain separate, each is independently bounded, and invalid UTF-8 is replacement-decoded with an explicit validity flag and warning. Timeout handling kills and waits for the direct process where supported; process-tree termination is platform-dependent and not claimed to be perfect. stdin is `DEVNULL`, so execution never waits for an interactive terminal.

`RunCommandTool` is exposed only through `ToolRegistry.with_command_execution()`. `ToolRegistry.default()` and `AgentLoop` remain unchanged.

## Phase 5.2 command safety and policy

Phase 5.2 adds a deterministic policy layer above the existing executor without modifying its `shell=False`, argv, cwd, output, timeout, or result guarantees:

```text
CommandRequest
      ↓
CommandPolicy.evaluate()
      ├── denied → structured ToolError; no Popen
      └── allowed → existing run_command(request)
                         ↓
                    CommandResult
```

`CommandPolicy` is immutable and independently testable. `CommandDecision` contains allowed/denied state, `CommandRiskLevel`, normalized secret-safe argv, matched rule/category, reason, warnings, and a shared `ToolErrorCode`. The default policy is deny-by-default. It recognizes only bounded Python/Node version commands and read-only Git inspection; explicit exact-argv or approved executable-path rules remain bounded and cannot override shell/path safety invariants.

Evaluation precedence rejects malformed argv, shell interpreters and emulation patterns, dangerous executable families, destructive/privileged/package/network/system/Git mutation categories, suspicious arguments, unsafe absolute/traversal/Windows/UNC paths, symlink-escaping working directories, unknown executable paths, and disallowed environment variables before any process is started. Environment inheritance is disabled by default in policy-wrapped requests; explicit environment variables use an allowlist, and values never enter decisions or error messages.

`PolicyRunCommandTool` is an opt-in wrapper that calls the process-management layer only after an allowed decision. Denials raise shared structured errors such as `COMMAND_NOT_ALLOWED`, `SHELL_BYPASS_ATTEMPT`, `UNSAFE_ARGUMENT`, `UNSAFE_WORKING_DIRECTORY`, `UNSAFE_EXECUTABLE`, `ENVIRONMENT_NOT_ALLOWED`, `GIT_MUTATION_DENIED`, `NETWORK_COMMAND_DENIED`, or `PACKAGE_OPERATION_DENIED`. The policy never invokes a shell, parses shell syntax, calls the AgentLoop, mutates files/Git, accesses the network, or installs packages. `ToolRegistry.with_command_policy()` is explicit; `ToolRegistry.default()` and the low-level `with_command_execution()` registry remain unchanged.

> **Command Safety Policy is a security boundary, not a guarantee that arbitrary developer commands are safe.**

## Phase 5.3 process management

Phase 5.3 introduces a reusable `ProcessManager` for one already-approved `CommandRequest`. It deliberately does not make security decisions:

```text
CommandRequest
      ↓
CommandPolicy.evaluate()
      ↓ allowed
ProcessManager
      ↓
Popen(shell=False, stdin=DEVNULL, explicit cwd)
      ↓
ProcessLifecycle + bounded capture + termination/reaping
      ↓
CommandResult
```

`ProcessLifecycle` is an immutable state-history object. Valid transitions include `REQUESTED → VALIDATING → STARTING → RUNNING → COMPLETED → CLEANED_UP`, `RUNNING → TIMED_OUT → TERMINATING → TERMINATED → CLEANED_UP`, and output-limit paths through `OUTPUT_LIMIT_REACHED`. Start failures pass through `FAILED_TO_START` and still reach cleanup. Invalid transitions raise `PROCESS_INVALID_STATE`; no singleton or global active-process registry is used.

`ProcessManager` validates the technical request and builds the controlled environment, but it does not duplicate or override `CommandPolicy`. It starts a direct process with shell disabled, stdin detached, explicit root-contained cwd, and POSIX process-group/session isolation where supported. stdout/stderr are collected separately through bounded selector reads. When a stream exceeds its limit, excess bytes are drained and discarded while the retained prefix and truncation flag remain bounded and deterministic. This avoids leaving a writing child blocked on a full pipe.

On timeout, the manager records `TIMED_OUT`, attempts graceful process-group/session termination where supported, waits for a bounded grace period, escalates to forced kill if necessary, waits for reaping, then closes the pipes. The result distinguishes `started`, `completed`, `succeeded`, `exit_code`, `timed_out`, `termination_attempted`, `killed`, lifecycle state/history, retained output byte counts, and structured process failure codes. Direct-child cleanup is bounded; descendant cleanup and signal semantics differ across POSIX and Windows and are not claimed to be identical.

`PolicyRunCommandTool` uses `ProcessManager` only after `CommandPolicy.evaluate()` allows the request. `ProcessManager` itself is not registered in `ToolRegistry.default()` and does not enable AgentLoop execution.

## Phase 5.4 application runner

Phase 5.4 adds one bounded application-launch layer without creating a second command, policy, process, or project-scanning system:

```text
ApplicationRunRequest
          ↓
ProjectContextBuilder / bounded ProjectStructure evidence
          ↓
ApplicationCommandResolver
          ↓ ApplicationRunPlan or unresolved/ambiguous result
CommandPolicy.evaluate()
          ↓ allowed
ProcessManager
          ↓
ApplicationRunResult
```

`ApplicationCommandResolver` uses the existing `ProjectContext` and its underlying `ProjectStructure` evidence. It does not import target projects, execute project code to discover launch behavior, inspect `.env`/credential/private-key content, install dependencies, or access the network. Automatic resolution is deliberately small: Python `main.py`/`app.py`/`server.py` only with a supported entry-point marker, Django `manage.py` only with detected Django evidence, and Node `package.json` `scripts.start`/`main` only for an existing exact `node <target>` script. Existing target paths must be safe, bounded, and supported. Mixed projects or multiple candidates produce deterministic `AMBIGUOUS_ENTRYPOINT`; insufficient evidence produces `NO_APPLICATION_ENTRYPOINT`, `UNSUPPORTED_PROJECT`, or `RESOLUTION_FAILED` rather than an invented command.

Explicit argv mode does not reinterpret or modify the caller’s argv. It still constructs a `CommandRequest` and passes through `CommandPolicy`; shell strings, shell wrappers, chaining, redirects, package/network/Git/system operations, unsafe paths, and unknown executable paths remain rejected before ProcessManager. Plans expose safe normalized argv, relative working directory, evidence/source, project type, confidence, explicit-vs-automatic mode, and warnings. Results preserve safe ProcessManager output/lifecycle metadata and add application status/failure classification without environment values or sensitive file contents.

Long-running applications are bounded by the request timeout. The runner does not detach, daemonize, queue, retry, schedule, or register background processes. When the timeout expires, ProcessManager terminates/reaps/cleans the process and the runner returns `TIMED_OUT`; no unmanaged application is intentionally left alive. `RunApplicationTool` is available only through `ToolRegistry.with_application_execution()`. `ToolRegistry.default()` and `AgentLoop` remain unchanged.

## Phase 5.5 test runner

Phase 5.5 adds a bounded test-execution layer without duplicating project scanning, command execution, policy, or process lifecycle:

```text
TestRunRequest
      ↓
ProjectContextBuilder / bounded ProjectStructure evidence
      ↓
TestCommandResolver
      ↓ TestRunPlan or NO_TEST_COMMAND/AMBIGUOUS_TEST_COMMAND
CommandPolicy.evaluate()
      ↓ allowed
ProcessManager
      ↓
TestRunResult (raw execution facts)
```

`TestCommandResolver` consumes the canonical `ProjectContext`, including test-framework detections, package/dependency evidence, configuration paths, test directories, and bounded project-file paths. It adds only bounded `package.json` test-script inspection through the existing safe file-reading boundary; it does not execute or import project code and does not inspect sensitive files. Supported evidence includes Python `pytest` and `unittest`, Node/Javascript/TypeScript Jest and Vitest evidence, and an explicit `package.json` `scripts.test`. A package test script has the highest automatic priority. Framework candidates are ranked deterministically; equally ranked candidates return `AMBIGUOUS_TEST_COMMAND`, while missing safe evidence returns `NO_TEST_COMMAND`. Jest/Vitest direct execution is not invented merely from a dependency or config name; a visible runner script is required.

Explicit argv mode accepts a sequence only and never constructs a shell string. Targets and optional test arguments are bounded argv values; traversal, absolute/Windows paths, symlink components, sensitive paths, shell operators/substitution, shell interpreters, package/network/Git operations, and unapproved executables remain subject to denial. All accepted plans first pass through `CommandPolicy`, then through `ProcessManager`; policy denial returns before process creation. Environment inheritance remains disabled by default and environment values are never copied into plans, decisions, results, warnings, or logs.

`TestRunResult` deliberately preserves raw facts for the future parser: plan, framework/evidence, policy decision, `CommandResult`, exit code, bounded stdout/stderr, byte counts, UTF-8/truncation flags, timeout/termination/cleanup, lifecycle history, warnings, and technical failure code. `COMPLETED` with a non-zero exit code is an execution-level state with `NONZERO_EXIT`; the runner does not classify semantic PASS/FAIL/ERROR, parse test counts or names, extract assertion summaries, retry/rerun tests, or debug/fix files. Test frameworks may naturally create caches during execution, but the runner does not intentionally mutate project files or implement generalized cleanup. Phase 5.6 `TestResultParser` is not implemented.

`RunTestsTool` is exposed only through `ToolRegistry.with_test_execution()`. The default registry and `AgentLoop` remain read-only and do not gain automatic test execution, automatic tool calling, or autonomous test/fix loops.

## Phase 5.6 test-result parser

Phase 5.6 is the final Phase 5 layer. It consumes one existing raw `TestRunResult` and performs no execution or project access:

```text
TestRunner
      ↓ raw bounded TestRunResult
TestResultParser
      ↓ bounded framework-aware interpretation
TestParseResult
      ↓
PASS / FAIL / ERROR / NO_TESTS / TIMEOUT / OUTPUT_LIMIT /
EXECUTION_ERROR / UNKNOWN
```

`TestResultParser` never calls `ProcessManager`, `TestRunner`, `ApplicationRunner`, `subprocess`, the network, a target-project import, or a filesystem scanner. `parse_test_result()` and `TestResultParserTool` accept only the already-captured result; `ToolRegistry.with_test_result_parsing()` is opt-in. `ToolRegistry.default()` and `AgentLoop` remain unchanged, and parser execution cannot cause retries, reruns, debugging, file modification, or corrective actions.

The parser applies a documented precedence order. Technical execution facts come first: timeout maps to `TIMEOUT`, captured-output truncation maps to `OUTPUT_LIMIT`, and process/policy/start/resolution errors map to `EXECUTION_ERROR`. Strong framework collection/runtime errors then map to `ERROR`; recognized assertion/test failures map to `FAIL`; complete successful summaries with no failures/errors map to `PASS`; an explicit zero-test result maps to `NO_TESTS`. A non-zero exit code without strong semantic evidence is `UNKNOWN`, not an automatic `FAIL`. Contradictory or multi-framework evidence adds warnings and returns `UNKNOWN`/partial rather than trusting a weak token such as `PASS`, `FAIL`, `TEST`, or `ERROR` by itself.

Phase 5.6 supports bounded text formats for pytest, unittest, Jest, Vitest, and package scripts when one framework format is strongly identifiable. `TestParseResult` preserves execution status and exit code and adds counts, bounded failure/error records, failed/error test names, bounded stdout/stderr summaries, framework/format, confidence, warnings, truncation, parse completeness, and reliable duration when present. `TestParseLimits` bounds input bytes, record counts, names, messages, and excerpts. Sensitive key/token/password-like values in extracted structured text are redacted; no unbounded output is copied into a result. The parser is deterministic, uses no timestamps or randomness, and treats output as untrusted data.

Phase 5.6 is semantic interpretation only. It does not diagnose root causes and does not implement a debugger, self-correction, autonomous AgentLoop, planning, memory, RAG, shell, network, package installation, Git mutation, background processes, retries, or Phase 6.

## Phase 6.1 planner

Phase 6.1 introduces the first planning boundary above the existing read-only AgentLoop without activating autonomous execution:

```text
User task + optional supplied ProjectContext + PlannerConfig
                         ↓
                       Planner
                         ↓
              PlanValidator / safe immutable plan
                         ↓
                    ExecutionPlan
                         ↓
             STOP — future Phase 6.2 only
```

`Planner` accepts only caller-supplied task text, an optional existing `ProjectContext`, and explicit bounded configuration. It does not build context, scan files, read paths, import target code, select tools, call tools, execute commands/tests, run the parser, modify files/Git, access the network/environment/secrets, or invoke `AgentLoop`. The existing `AgentLoop` and all `ToolRegistry` defaults remain unchanged.

The immutable schema consists of `PlannerRequest`, `PlannerConfig`, `PlanStep`, `PlanRisk`, `ExecutionPlan`, `PlanValidationResult`, and `PlannerResult`. `PlanStep` is declarative: it describes an objective, rationale, expected result, dependencies, risk, verification requirement, and status, but contains no executable tool call or command field. `ExecutionPlan` carries original/normalized task, goal, conservative task type, step DAG, assumptions, constraints, risks, expected change categories, verification strategy, confidence, warnings, and completeness. Missing or partial context is represented explicitly through lower confidence, assumptions, warnings, and partial completeness rather than silently treated as project facts.

Task normalization collapses whitespace and preserves unsupported or ambiguous requirements. Conservative categories include feature, bug fix, refactor, test addition, configuration, documentation, dependency, investigation, and `UNKNOWN`. Short or underspecified tasks can remain `REQUIRES_CLARIFICATION`; the planner does not turn “Fix authentication” into an invented JWT design. Expected changes use bounded categories rather than fabricated filenames. Verification strategy describes future checks but is never executed by the Planner.

`PlanValidator` checks required text and enum values, unique step IDs, valid dependency references, DAG acyclicity, configured step/collection/text budgets, and forbidden execution or mutation payloads disguised as prose. `PlannerConfig` makes truncation visible through warnings and incomplete/clarification status. Identical task/context/config inputs produce deterministic serialized plans; no time, randomness, network, filesystem discovery, hidden state, or external model is used.

Phase 6.1 is planning infrastructure only. The Planner is not a Tool Selector, Tool Executor, code generator, debugger, self-correction engine, test runner, command executor, or autonomous AgentLoop. Phase 6.2 is now a separate selection layer; Phase 6.3 Tool Loop, stop conditions, execution budgets, error recovery, task completion verification, memory, RAG, and all later autonomous behavior are intentionally absent.

## Phase 6.2 tool selection

Phase 6.2 consumes the declarative Phase 6.1 plan and maps each selected plan step to an existing registered capability without executing it:

```text
ExecutionPlan + supplied ToolRegistry + optional ProjectContext + inputs
                              ↓
                         ToolSelector
                              ↓
                    ToolSelectionResult
                              ↓
              STOP — Phase 6.3 owns eventual execution
```

`ToolSelector` discovers actual tool names and metadata through the supplied `ToolRegistry`; it never creates or enables tools. The capability model is extensible and classifies known tools as `READ_ONLY`, `MUTATING`, `EXECUTION`, or `DESTRUCTIVE`. It records supported intent, required/optional inputs, safety notes, and expected output. Policy-wrapped `run_command_with_policy` is represented as the execution capability selected for logical approved-command intent, and the existing `parse_test_result` name is used for test-result interpretation.

The immutable API consists of `ToolSelectionRequest`, `ToolSelectionConfig`, `ToolCapability`, `ToolCandidate`, `ToolSelectionDecision`, and `ToolSelectionResult`. A decision contains plan-step ID, status, actual selected tool, category, reason, confidence, required/optional inputs, prerequisites, missing prerequisites, expected output, alternatives, forbidden tools, risk, warnings, and bounded candidates. Statuses include `SELECTED`, `TOOL_UNAVAILABLE`, `NO_SUITABLE_TOOL`, `AMBIGUOUS_SELECTION`, `MISSING_PREREQUISITES`, `INVALID_REQUEST`, and `INCOMPLETE`.

Selection is conservative and plan-driven. Discovery/context steps prefer `project_structure`/`project_context`; unknown locations prefer `search_code` with `list_files` as a bounded alternative; known file contents map to `read_file`; explicit create/edit/delete intent maps to the corresponding mutation capability; repository review maps to `git_status`/`git_diff`; explicit command/application/test intent maps only to the supplied execution capability; and interpretation of an existing test result maps to `parse_test_result`. Mutation or execution capabilities are not selected merely because they are registered. Equal inspection candidates return `AMBIGUOUS_SELECTION`, and missing capabilities return `TOOL_UNAVAILABLE` without pretending the tool exists.

Prerequisites are declarative selection facts, such as a known target path for `read_file`, confirmed mutation intent for `edit_file`, an approved plan and `CommandPolicy` permission for execution, or an existing `TestRunResult` for parsing. Strict prerequisite mode can return `MISSING_PREREQUISITES`; normal selection may record missing prerequisites as warnings for a later phase. Risk and confidence expose selection uncertainty but never grant permission: mutation remains behind `SafeEditSession`/`SafeEditPolicy`, while command/application/test execution remains behind `CommandPolicy`/`ProcessManager`.

`ToolSelectionValidator` rejects unknown plan-step IDs, unavailable selected tools, duplicate selections, duplicate/unavailable alternatives, invalid enums, mutation without mutation intent, execution without execution intent, and malformed decision structures. All candidate/alternative/step/warning collections are bounded and deterministically ordered. `ToolSelector` has no filesystem, subprocess, network, environment/secrets, Git mutation, tool dispatch, or AgentLoop calls.

## Phase 6.3 autonomous tool loop

Phase 6.3 introduces the first controlled autonomous execution boundary, explicitly separate from the original read-only `AgentLoop`:

```text
Task + explicit project root
             ↓
          Planner
             ↓
       ExecutionPlan
             ↓
       ToolSelector
             ↓
  strict ACTION/ARGS validation
             ↓
 supplied ToolRegistry.dispatch()
             ↓
 structured ToolResult observation
             ↓
 bounded ContextBudget history → next action
```

`AutonomousToolLoop` accepts an `AutonomousLoopRequest` and an explicitly supplied `ToolRegistry`. The loop may construct an initial `ProjectContext` only by selecting and dispatching the existing registered `project_context` tool when no context is supplied. It then calls the existing `Planner`, selects one current plan step with `ToolSelector`, asks the injected inference engine for one bounded action, validates that action against the current selection, and dispatches only through `ToolRegistry`. It never instantiates arbitrary tools, contains an alternative command executor, or enables mutation/execution capabilities automatically. The CLI and Phase 3.6 `AgentLoop` remain unchanged.

The immutable/bounded API consists of `AutonomousLoopRequest`, `AutonomousLoopConfig`, `AutonomousLoopState`, `AutonomousLoopStep`, `AutonomousLoopResult`, `LoopAction`, and lifecycle/action/status/failure enums. The strict model contract is either `ACTION: TOOL` followed by one JSON object containing `tool` and `arguments`, or `ACTION: FINAL` followed by one JSON object containing `message`. Natural-language prose, malformed JSON, unknown action shapes, unknown tools, selection mismatches, shell-like payloads, and project-root escapes fail before tool dispatch. A final action is the only successful terminal signal in this phase.

The explicit state machine uses `CREATED`, `PLANNING`, `SELECTING_TOOL`, `VALIDATING_ACTION`, `EXECUTING_TOOL`, `OBSERVING_RESULT`, `UPDATING_CONTEXT`, `REQUESTING_NEXT_ACTION`, `COMPLETED`, and `FAILED`. Invalid transitions raise structured `LoopStateError`; failed tools are recorded once and stop the invocation. There is no generic retry, rerun, argument rewriting, or safety bypass. Phase 6.6 may classify a failed structured result and request one different bounded plan action through the existing loop, but it never blindly repeats a failed action or performs autonomous debugging.

After every successful tool execution, the loop stores a bounded structured observation in `AgentMessage` history and re-renders the next model prompt through the existing `ContextBudget`. Truncation remains visible through `context_truncated`, `truncation_reason`, `preserved_sections`, warnings, and usage metadata. History and action arguments are sanitized for secret-like keys and bounded text; raw credentials, private keys, environment dumps, and unrestricted file/output contents are not copied into loop state.

A private fixed bound of eight tool executions per invocation prevents infinite development loops. This emergency bound is explicit, deterministic, and not model-overridable, but it is not the configurable Phase 6.5 max-iterations feature. Plan steps with `NO_SUITABLE_TOOL` may be recorded as bounded non-executable skips; unavailable or ambiguous required capabilities produce structured failure. Mutation requires an explicitly supplied mutation registry and remains behind `SafeEditSession`/`SafeEditPolicy`; command/application/test actions remain behind `CommandPolicy`/`ProcessManager`.

Phase 6.3 is the first autonomous execution phase. Phase 6.4 adds semantic stop evaluation without adding another planner, registry, executor, or recovery mechanism.

## Phase 6.4 stop conditions

`StopConditionEvaluator` is a pure deterministic layer over explicitly supplied `ExecutionPlan`, completed/skipped/blocked step IDs, `ToolResult`, verification evidence, action validity, capability availability, context completeness, and emergency-bound state. It does not call the LLM, inspect the filesystem, dispatch tools, execute subprocesses, use the network, or modify files/Git.

The immutable public model is:

```text
StopConditionRequest
        ↓
StopConditionEvaluator.evaluate()
        ↓
StopEvaluation
  ├── DONE
  ├── CONTINUE
  ├── FAILED
  └── BLOCKED
```

`StopReason` distinguishes final response, task completion, verification passed, remaining plan work, follow-up evidence, verification required/failed, tool failure, invalid action, missing capability, safety/policy block, emergency bound, incomplete context, unresolved state, and internal error. `VerificationEvidence` records `NOT_REQUIRED`, `REQUIRED`, `PENDING`, `PASSED`, `FAILED`, `UNAVAILABLE`, or `INCOMPLETE` without serializing source contents or secrets.

A valid `ACTION: FINAL` is not automatically completion proof. It is `DONE` only if the current plan has no remaining required steps, context is sufficient, and no verification obligation is pending. If required steps remain, the evaluator returns `CONTINUE`; if the context/capability/safety boundary prevents progress, it returns `BLOCKED`. A successful mutation sets verification to `PENDING`, while `verify_modification` success with complete explicit evidence or `parse_test_result` with semantic `PASS` can produce `VerificationEvidence.PASSED` and allow `DONE` when no required work remains. Test failures remain non-DONE; they produce structured verification failure/continuation evidence rather than automatic recovery.

Phase 6.6 adds `ErrorClassifier` and `RecoverabilityPolicy` as pure evidence-driven layers. They classify existing structured error codes into immutable categories such as validation, policy/safety/root violation, file/match failure, concurrent modification, verification failure, command/process/application failure, test failure/error, budget/context limit, internal, or unknown. Safety and policy failures are never recoverable automatically; concurrent user changes require `USER_INTERVENTION_REQUIRED`; exhausted budgets and emergency bounds stop further work; actionable failures may request `INSPECT`, `VERIFY`, or `REPLAN` only when a different bounded plan step exists. The original `ToolResult` remains preserved, the failed action signature is recorded, and repeated identical actions are stopped deterministically. Recovery cannot inspect the filesystem directly, call tools directly, alter budgets, bypass `ToolRegistry`, weaken `CommandPolicy`, or overwrite user changes. The fixed eight-tool emergency bound remains a safety backstop and yields non-DONE `BLOCKED` evaluation with `EMERGENCY_BOUND_REACHED`; it is not configurable max iterations.

`AutonomousLoopResult.stop_evaluation` carries the structured decision, reason, bounded evidence, blocking conditions, remaining required steps, verification state, tool state, confidence, warnings, and emergency-bound metadata. The existing legacy `LoopStatus` values remain available for backward compatibility, while Phase 6.4 semantic stop state is authoritative for completion decisions. The autonomous loop remains explicit opt-in and the default CLI/`AgentLoop` behavior remains unchanged.

Phase 6.4 intentionally does not implement Phase 6.5 configurable iteration limits. Phase 6.6 error recovery is bounded orchestration only; it does not implement generic retries, self-correction, autonomous debugging, memory, RAG, network, package installation, Git mutation, shell execution, background agents, scheduling, daemon processes, unrestricted command/file modification, or automatic CLI autonomy.

## Phase 6.5 execution budgets

`ExecutionBudget` is the host-configured immutable limit set, `ExecutionUsage` is the bounded immutable usage snapshot, `BudgetDecision` is the pre-execution allow/deny record, and `ExecutionBudgetLedger` is the single authoritative accounting mechanism for one autonomous invocation. The ledger uses a monotonic clock for duration accounting and never permits negative remaining values.

The conservative defaults are finite: 16 iterations, 16 tool-call attempts, 4 mutation operations, 4 command executions, 4 test executions, 2 application launches, 300 seconds of autonomous wall time, 131,072 accumulated tool-result bytes, 65,536 stdout bytes, 65,536 stderr bytes, 65,536 context tokens, and 16 action steps. A host may pass a smaller validated `ExecutionBudget`; zero explicitly blocks that dimension, negative values are rejected, and safety ceilings prevent absurdly large values. Model output and task prose cannot alter the limits.

Each iteration and action step is checked before model work. Each tool is checked before dispatch, including the general tool-call limit and the applicable mutation/command/test/application dimension. Tool attempts are counted before dispatch, including denied attempts; completed results are counted after return; output and context bytes/tokens are accumulated centrally and are never reset by individual tools. Existing `CommandPolicy`, `ProcessManager`, `ApplicationRunner`, `TestRunner`, `SafeEditPolicy`, and `ContextBudget` remain the execution and safety boundaries rather than being duplicated.

Budget exhaustion is represented by `BudgetExhaustion` and takes precedence over a model request to continue. The loop does not execute a blocked operation and returns `LoopStatus.BLOCKED` with `StopDecision.BUDGET_EXHAUSTED`, the exhaustion dimension, configured limit, usage, remaining budget, operation name, and `operation_started=false`. The result exposes a bounded `ExecutionBudgetSnapshot` containing limits, usage, remaining values, exhausted dimensions, warnings, and usage completeness. The Phase 6.3 fixed emergency bound of eight tool executions remains an immutable final backstop; the stricter effective limit wins. Stop Conditions answer whether the agent should continue, while Execution Budgets answer whether it is still allowed to continue.

Phase 6.6 does not implement generic retries, automatic code fixes, self-correction, autonomous debugging, memory, RAG, network, package installation, Git mutation, shell execution, background agents, scheduling, daemon processes, unrestricted command/file modification, or automatic CLI autonomy. Phase 7 remains unimplemented.

## Phase 6.7 task completion verification

`TaskCompletionVerifier` is a pure, deterministic evaluator over an explicit `TaskCompletionRequest`. It does not call the model, execute tools, inspect paths, read secrets, run tests, mutate files/Git, use subprocesses, or access the network. It aggregates existing `ExecutionPlan`, completed/skipped steps, bounded `ToolResult` history, `VerificationEvidence`, `RecoveryResult`, `ExecutionBudgetSnapshot`, explicit criteria/evidence, and surfaced unexpected modifications.

The immutable completion model is:

```text
TaskCompletionRequest
        ↓
TaskCompletionVerifier.verify()
        ↓
TaskCompletionResult
  ├── COMPLETE
  ├── INCOMPLETE
  ├── BLOCKED
  ├── FAILED
  ├── VERIFICATION_UNAVAILABLE
  └── INSUFFICIENT_EVIDENCE
```

A successful action is not task completion, a passing unrelated test is not sufficient proof, and `ACTION: FINAL` is only a model claim. Required plan steps, explicit criteria, relevant verification/test evidence, recovery state, budget state, and safety boundaries are evaluated independently. Pending or unavailable required verification cannot silently become `DONE`; critical unexpected modifications block completion, while non-critical unexpected changes remain explicitly unverified. Investigation/documentation plans are not forced to have mutation or test criteria that the plan does not require.

`AutonomousToolLoop` runs the verifier when it receives `ACTION: FINAL` and exposes `TaskCompletionResult` in both `AutonomousLoopResult` and `AutonomousLoopState`. A non-complete completion result is passed into the existing `StopConditionEvaluator` through `completion_decision`; the stop evaluator remains authoritative for `DONE`, `CONTINUE`, `BLOCKED`, and `FAILED`. Budget exhaustion and safety/policy blocks retain higher precedence. Recovery consumes completion evidence but remains responsible for recovery decisions; the completion verifier never performs recovery itself.

Evidence and history are bounded by criteria, evidence, tool-result, text, plan-step, and unexpected-modification limits. Completion confidence is `HIGH`, `MEDIUM`, `LOW`, or `UNKNOWN`, and each criterion reports satisfied, unsatisfied, blocked, unverified, or not-applicable status plus expected/observed evidence. The layer is intentionally conservative: insufficient evidence produces continuation or verification-unavailable semantics instead of a false positive.

Phase 6.7 does not implement Phase 7, memory, RAG, dataset collection, model training, fine-tuning, network access, package installation, Git mutation, shell bypasses, background agents, unrestricted autonomy, or automatic CLI autonomy.

## Phase 7.1 automatic test execution

`AutomaticTestOrchestrator` is an explicit, bounded orchestration layer. It does not duplicate `TestCommandResolver`, `TestRunner`, `RunTestsTool`, `CommandPolicy`, `ProcessManager`, `ExecutionBudget`, `StopConditionEvaluator`, or `TaskCompletionVerifier`. Its only execution path is:

```text
AutomaticTestOrchestrator
        ↓
ToolRegistry.run_tests
        ↓
TestRunner / TestCommandResolver
        ↓
CommandPolicy
        ↓
ProcessManager
        ↓
TestRunResult
```

The immutable API includes `AutomaticTestConfig`, `AutomaticTestRequest`, `AutomaticTestDecision`, `AutomaticTestExecution`, and `AutomaticTestResult`. Decisions are `RUN`, `SKIP`, `BLOCKED`, `UNAVAILABLE`, `INVALID`, or `BUDGET_EXHAUSTED`. A run is eligible only at a bounded evidence-based boundary: an explicit user request, a plan test/verification step, completion-required test evidence, or an implementation/bug-fix/refactor task that has changed implementation state and reached verification. Documentation and investigation tasks do not trigger tests by default, and the layer never runs tests after every file operation.

When `RUN` is selected, the orchestrator checks `ExecutionBudgetLedger.check_tool_operation("run_tests")` before dispatch, consumes the normal tool-call and test-execution dimensions, and passes only structured project root, target, test arguments, working directory, timeout, and output limits to the existing `run_tests` tool. If the budget denies execution, no test process starts and the result is `BUDGET_EXHAUSTED`. If the capability is absent, the result is `BLOCKED`; if the existing resolver cannot find an evidence-backed command, the raw result is preserved as `UNAVAILABLE`. No guessed `pytest`, `npm test`, Jest, or Vitest command is constructed.

The raw `TestRunResult` remains authoritative and retains command lifecycle metadata, exit code, stdout/stderr, timeout/output-limit state, policy decision, framework evidence, and warnings. Phase 7.1 does not interpret failures, diagnose root causes, edit files, retry, rerun, or self-correct. `TestResultParser` may be invoked separately through its existing explicit capability; Phase 7.2+ owns analysis. `StopConditionEvaluator` and `TaskCompletionVerifier` can consume the resulting structured evidence, but the automatic-test layer does not replace either evaluator. `ToolRegistry.default()` and the base `AgentLoop` remain unchanged and do not gain execution permissions.

Phase 7.1 does not implement diagnosis, root-cause analysis, automatic fixing, retries, regression protection, memory, RAG, network access, package installation, Git mutation, shell bypasses, background agents, unrestricted autonomy, or Phase 7.2–7.7 functionality. The later Phase 8.1 task-definition layer remains independent from this execution boundary; Phase 8.2 is implemented as a separate explicit benchmark orchestration layer, while Phase 8.3 remains unimplemented.

## Phase 7.2 test failure analysis

`TestFailureAnalyzer` is a pure, deterministic layer over an existing `TestRunResult` and `TestParseResult`. It does not run commands, call `TestRunner`, invoke `ProcessManager`, read the project filesystem, modify files, call the LLM, retry, rerun, or select a fix. Its boundary is:

```text
TestRunResult
        ↓
TestResultParser / TestParseResult
        ↓
TestFailureAnalyzer
        ↓
TestFailureAnalysis
```

The immutable API includes `FailureAnalysisConfig`, `TestFailureAnalysisRequest`, `FailureLocation`, `FailureEvidence`, `FailureFinding`, `FailureGroup`, and `TestFailureAnalysis`. The analyzer returns `ANALYZED`, `NO_FAILURE`, `INCOMPLETE`, `INSUFFICIENT_EVIDENCE`, `UNAVAILABLE`, or `INVALID`, and uses explicit enums for classification, severity, confidence, and location kind. Its conservative taxonomy includes assertion, exception, import/module, type, syntax, configuration, dependency, database/connection, authentication/API, fixture, test-discovery, environment, timeout, output-limit, execution, and unknown failure categories.

Findings preserve provenance through source-tagged evidence: parser records, test names, parser-provided file/line locations, bounded output excerpts, exception types, messages, exit status, and parser confidence/completeness. A test location is distinct from a suspected implementation location; this phase does not claim an implementation cause without evidence. Diagnostic chains are bounded observed sequences such as test identifier, exception type, and message. The analyzer may group identical normalized observed failures and mark primary/derived relationships, but `causal_inference=true` explicitly signals inference rather than confirmed root cause.

Confidence is conservative. High confidence requires strong parser confidence together with an exact test/file/line record; medium confidence uses partial structured identity or exception evidence; low confidence uses weak messages only; unknown remains when evidence is insufficient. Timeout, output-limit, and execution states use technical parser precedence rather than inventing framework semantics. Truncated parser output produces incomplete analysis metadata and bounded warnings instead of pretending the result is complete.

Input, finding, group, related-failure, chain, path, message, traceback, and excerpt limits are validated and enforced. Diagnostic text is redacted for passwords, tokens, API keys, secrets, credentials, authorization values, and private keys. `AutonomousToolLoop.analyze_test_failure()` exposes analysis as an explicit structured observation helper only; it does not act on findings. Phase 7.4 automatic fixing, Phase 7.5 retries, Phase 7.6 regression protection, and Phase 7.7 final verification remain unimplemented.

## Phase 7.3 root cause analysis

`RootCauseAnalyzer` is a pure, deterministic layer above `TestFailureAnalysis`. It consumes only immutable failure findings, groups, evidence, parser provenance, and optional immutable `ProjectContext` metadata. It never invokes `TestRunner`, `ProcessManager`, commands, tests, filesystem inspection, secrets, network, package installation, Git mutation, environment mutation, an LLM, a retry, or a fix.

```text
TestFailureAnalysis
        ↓
RootCauseAnalyzer
        ↓
RootCauseAnalysis
  ├── hypotheses
  ├── alternatives
  ├── causal relations
  ├── supporting/contradicting evidence
  └── explicit unknowns
```

The immutable API contains `RootCauseAnalysisRequest`, `RootCauseAnalysisConfig`, `RootCauseHypothesis`, `RootCauseEvidence`, `RootCauseLocation`, `AlternativeCause`, `CausalRelation`, and `RootCauseAnalysis`. Statuses are `ANALYZED`, `NO_FAILURE`, `INSUFFICIENT_EVIDENCE`, `INCONCLUSIVE`, `BLOCKED`, `UNAVAILABLE`, and `INVALID`. A hypothesis is never marked confirmed: observed failure, inferred hypothesis, and confirmed root cause remain separate concepts, and `confirmed` is always false in this phase.

Candidate generation is conservative and classification-driven. Import/module and dependency failures produce dependency or module hypotheses; authentication/API failures produce implementation or configuration candidates with fixture/token alternatives; database and connection failures preserve database or external-service candidates; assertion/type/syntax failures preserve implementation candidates without asserting a faulty line; fixture/configuration/environment/timeout cases remain at their appropriate boundary. The mechanism is a bounded sequence of observed-to-inferred steps, not an unbounded causal proof.

Every hypothesis carries supporting evidence and can carry contradicting evidence, affected finding IDs, evidence strength, confidence, location, causal status, and a bounded causal chain. Alternatives carry their own statement, location, evidence, confidence, and why they remain possible. Shared primary/derived and cascading relations are explicitly marked `inferred`; correlated failures are not presented as confirmed common causality. If evidence is insufficient, the result is `INCONCLUSIVE` or `INSUFFICIENT_EVIDENCE` with explicit unknowns rather than invented diagnosis.

`max_causal_depth`, hypothesis/alternative/evidence limits, message limits, context limits, and chain limits are validated. Reaching the causal depth sets `causal_chain_truncated`; analysis and evidence completeness are serialized. The optional `ProjectContext` contributes only existing structured project type, stack, dependency/configuration/database metadata and never triggers a new scan. `AutonomousToolLoop.analyze_root_cause()` exposes the result as explicit diagnostic context only. Phase 7.4 automatic fixing and Phase 7.5 bounded self-correction are explicit helpers; Phase 7.6 regression protection and Phase 7.7 final verification remain unimplemented.

## Phase 7.4 automatic fix

`AutomaticFixPlanner` and `AutomaticFixOrchestrator` are explicit, bounded orchestration layers above the Phase 4 mutation infrastructure. Their boundary is:

```text
RootCauseAnalysis
        ↓
AutomaticFixPlanner
        ↓
FixDecision / FixPlan
        ↓
ModificationTransaction
        ↓
SafeEditSession / SafeEditPolicy
        ↓
ModificationVerifier
        ↓
AutomaticFixResult
```

A `FixPlan` is structured data, not free-form authorization. It must identify one project-relative target file, an exact evidence-backed location, one supported `FixChangeType`, intended change, expected post-state, old/new UTF-8 content, evidence, risk, confidence, selected hypothesis ID, and affected failure IDs. Planner validation rejects absent/non-`ANALYZED` RCA, unknown or non-actionable hypotheses, low confidence, missing or ambiguous locations, unsupported risk, malformed content, empty evidence, sensitive paths, traversal/absolute paths, and policy-denied edit capability. The default `ToolRegistry` and normal CLI remain unchanged; the helper is explicit opt-in.

`AutomaticFixOrchestrator` checks `ExecutionBudgetLedger` action-step and mutation dimensions before mutation. A denied budget returns `BLOCKED` with `operation_started=false`. An accepted plan performs exactly one edit through the existing `ModificationTransaction` and `SafeEditPolicy`; the fix layer contains no raw filesystem mutation. Existing snapshots, backups, atomic edit, path/symlink protections, concurrent-change detection, recovery semantics, and `ModificationVerifier` remain authoritative. User changes are preserved rather than forcibly overwritten.

Results distinguish proposed/accepted from attempted/succeeded/verified/rejected/failed/blocked/recovery-required states. `FIX_VERIFIED` means the explicit target post-state and transaction verification succeeded; it never means tests passed. `AutomaticFixResult` exposes the transaction, change summary, budget decision/snapshot, `tests_rerun=false`, and `retries=0`. Phase 7.4 intentionally performs no test rerun, no retry, no recursive fix loop, no diagnosis, no package installation, no network, no Git mutation, no secrets or `.env` access, no shell/background execution, and no broad refactoring.

## Phase 7.5 bounded self-correction

`BoundedSelfCorrectionLoop` is an explicit orchestration layer over the existing Phase 7.1–7.4 components:

```text
AutomaticTestOrchestrator
        ↓
TestResultParser
        ↓
TestFailureAnalyzer
        ↓
RootCauseAnalyzer
        ↓
AutomaticFixPlanner / AutomaticFixOrchestrator
        ↓
AutomaticTestOrchestrator (retest)
```

`SelfCorrectionConfig` holds host-controlled finite bounds, including `max_attempts`; model output, test output, failure messages, and fix plans cannot change them. `SelfCorrectionRequest` carries one shared `ExecutionBudgetLedger`, explicit test request, explicit fix-plan provider, safe edit policy, and bounded fix configuration. Every `SelfCorrectionAttempt` records ordered lifecycle steps, test/parse/failure/RCA/fix states, failure/action fingerprints, mutation verification, and next action. History is immutable and bounded.

The loop stops immediately with `PASSED` when parsed tests pass. For a failure, it requires new structured parsing and analysis, derives RCA through the existing analyzer, and permits at most one already-validated automatic fix before a retest. When `require_regression_protection=true`, a passed targeted test is not terminal: the loop must compare an explicit baseline against one conservative regression scope using the same ledger and test layers. It stops as `EXHAUSTED` when the host attempt bound is reached, `REPEATED_FAILURE` when the same redacted failure/action pair recurs without progress, `NO_PROGRESS` when the single fix is not verified, and `NO_ACTIONABLE_FIX` when no evidence-backed plan is available. Regression outcomes are exposed as `REGRESSION_FREE`, `PRE_EXISTING_FAILURES_ONLY`, `REGRESSION_DETECTED`, `REGRESSION_INCOMPLETE`, `REGRESSION_BLOCKED`, or `BUDGET_EXHAUSTED`; incomplete, blocked, missing-baseline, truncated, or newly failing evidence never becomes `DONE`. Test capability, policy, safety, recovery, or budget boundaries return structured blocked/exhausted outcomes. Existing `StopConditionEvaluator` is consulted for terminal budget/safety decisions; the loop never creates a replacement ledger or bypasses `AutomaticTestOrchestrator`, `ModificationTransaction`, `SafeEditPolicy`, `CommandPolicy`, or `ProcessManager`. No global retry state, background worker, scheduled retry, blind retry, or unbounded recursion exists. Phase 7.7 final verification is the final pure evidence gate described below.

## Phase 7.6 regression protection

`RegressionProtection` is an explicit additive layer after a verified targeted fix. It does not create another test runner, parser, retry loop, RCA system, fix system, or recovery mechanism. Its execution boundary is:

```text
Explicit baseline evidence
        ↓
AutomaticTestOrchestrator — conservative regression scope
        ↓
TestRunner / CommandPolicy / ProcessManager
        ↓
TestResultParser
        ↓
Deterministic baseline/post-fix comparison
```

`RegressionBaseline` captures only bounded structured evidence: execution and parser status, framework, counts, failure identities, redacted normalized fingerprints, parser completeness, truncation, lifecycle metadata, and warnings. No unlimited stdout/stderr is stored. Baselines are never fabricated; no baseline or an incomplete baseline yields `INSUFFICIENT_EVIDENCE` or `VERIFICATION_INCOMPLETE` rather than regression-free.

`RegressionTestScope` preserves the narrowest reliable evidence-backed boundary: affected test, affected module, related module, or project suite. The layer delegates command resolution and execution to the existing `AutomaticTestOrchestrator`, which reuses `ToolRegistry`, `TestRunner`, `CommandPolicy`, and `ProcessManager`. A shared `ExecutionBudgetLedger` is mandatory for execution; the layer never creates or resets a ledger. A denied or exhausted budget has `operation_started=false` and produces `BUDGET_EXHAUSTED` or a blocked result.

`compare_regression()` compares identities and bounded fingerprints rather than aggregate counts alone. Each finding is explicitly classified as `PRE_EXISTING`, `RESOLVED`, `PERSISTENT`, `NEW`, `CHANGED`, or `UNKNOWN`. `REGRESSION_FREE` requires completed post-fix execution, complete comparable evidence, and no new or materially changed failure. A persistent baseline failure without new failure is `PRE_EXISTING_FAILURES_ONLY`; a new identity or changed fingerprint is `REGRESSION_DETECTED`; blocked, timed-out, output-limited, unavailable, truncated, or insufficient evidence remains incomplete/blocked/failed.

When required by `SelfCorrectionConfig`, `BoundedSelfCorrectionLoop` runs regression protection only after targeted PASS and exposes the full `RegressionProtectionResult` as completion evidence. Regression-free requires both targeted success and regression success; targeted PASS alone cannot authorize completion. Documentation and investigation tasks remain able to opt out when the plan does not require regression verification.

## Phase 7.7 final verification

`FinalVerification` is a pure, deterministic, bounded evidence gate at the end of Phase 7. It consumes structured evidence already produced by the plan, mutation verification, test parser/orchestrator, regression protection, self-correction, recovery, budget, completion, and stop-condition layers. It performs no filesystem inspection, test execution, command execution, mutation, network access, model call, retry, or second completion authority.

Its request contains the task, optional `ExecutionPlan`, completed/skipped/blocked step IDs, bounded `ToolResult` records, `VerificationEvidence`, optional modification verification, optional regression result and requirement flag, optional self-correction result, recovery result, shared `ExecutionBudgetSnapshot`, unexpected/critical modification records, capability and safety/policy evidence, completeness markers, final claim text, and optional existing authority results. The request is immutable and validates bounded collections; model prose is retained only as an indirect claim and never treated as proof.

The verifier evaluates these criteria in deterministic order: plan completeness and dependency order; mutation/post-state verification when inferred or explicitly required; relevant targeted PASS evidence when the plan requires tests; regression-free evidence when required; clean recovery state; non-exhausted shared budget; safety, policy, capability, and project-boundary state; unexpected modifications; execution failures; evidence completeness/truncation; structured observations; and optional failure-analysis→RCA→fix→retest chain and authority-agreement checks. It returns exactly one status: `VERIFIED`, `NOT_VERIFIED`, `INCOMPLETE`, `BLOCKED`, `FAILED`, `INSUFFICIENT_EVIDENCE`, or `BUDGET_EXHAUSTED`. Missing, conflicting, truncated, timeout, output-limit, unavailable, unknown, pre-existing-unaccepted, or newly failing evidence cannot become `VERIFIED`.

The existing `TaskCompletionVerifier` remains the completion authority. When `final_verification_required=true`, it adds one additive criterion and requires a `VERIFIED` Final Verification result; any other status prevents `COMPLETE`. `AutonomousToolLoop` computes Final Verification only at its existing `ACTION: FINAL` boundary, passes the result into `TaskCompletionVerifier`, and then passes that completion decision to the existing `StopConditionEvaluator`. `StopConditionEvaluator` remains the only terminal `DONE` authority; Final Verification never bypasses it. Investigation and documentation plans can remain test-free when structured plan evidence says tests are not required, while implementation and bug-fix plans require their existing mutation/test boundaries.

Final Verification exposes immutable serializable criteria, bounded evidence, missing and conflicting evidence, truncation sources, warnings, confidence, and a final message. Its positive and negative real-project checkpoints cover targeted PASS plus `REGRESSION_FREE` and targeted PASS plus `REGRESSION_DETECTED`; both use the existing test, parser, RCA, fix, transaction, and shared-budget layers. Phase 7 ends at Final Verification; the independent Phase 8.1 declarative task model and Phase 8.2 benchmark execution/evidence layer are implemented, while Phase 8.3 remains unimplemented.

## Present implementation

The repository implements only these foundation pieces:

| Area | Phase 0 responsibility | Intentionally absent |
| --- | --- | --- |
| Configuration | Resolve a configured root path and validate a log level | Agent-specific settings, secret loading, provider configuration |
| LLM provider | Define typed messages, request/response, provider protocol, one provider error, and the local Fodci adapter | External APIs, network access, fallback models, tool calling |
| Tool layer | Reuse the `Tool` protocol for read-only Phase 3 tools plus create-only `WriteFileTool`/`write_file`, exact existing-file `EditFileTool`/`edit_file`, regular-file-only `DeleteFileTool`/`delete_file`, additive `safe_editing` policy/session, read-only `GitDiffTool`/`git_diff` plus `GitStatusTool`/`git_status`, read-only `ModificationVerifier`/`verify_modification`, additive `ModificationTransaction`/recovery models, opt-in `RunCommandTool`/`run_command`, opt-in `PolicyRunCommandTool`/`CommandPolicy`, reusable `ProcessManager`, opt-in `RunApplicationTool`/`ApplicationRunner`, opt-in `RunTestsTool`/`TestRunner`, opt-in read-only `TestResultParserTool`/`parse_test_result`, public side-effect-free `Planner`/`PlanValidator` models, public side-effect-free `ToolSelector`/`ToolSelectionValidator` models, explicit opt-in `AutonomousToolLoop` models, pure `StopConditionEvaluator`/`StopEvaluation` models, pure `ErrorClassifier`/`RecoverabilityPolicy` models, pure `TaskCompletionVerifier`/`TaskCompletionResult` models, explicit `AutomaticTestOrchestrator` models over `run_tests`, pure `TestFailureAnalyzer` models over `TestParseResult`, pure `RootCauseAnalyzer` models over `TestFailureAnalysis`, explicit `AutomaticFixPlanner`/`AutomaticFixOrchestrator` models over `ModificationTransaction`, explicit `BoundedSelfCorrectionLoop` models over the existing test/analyze/fix layers, pure `FinalVerification`/`FinalVerificationResult` models over existing structured evidence, and pure declarative `EvaluationTask`/`EvaluationTaskValidator` models and explicit `BenchmarkRunner`/`BenchmarkResult` orchestration models | Git mutation, terminal execution beyond explicit policy/process/application/test boundaries, unrestricted autonomy, LLM tool-calling by default, benchmark scoring/comparison |
| Agent adapter | Keep `ProviderBackedAgent` compatibility and bounded read-only `AgentLoop` orchestration over the default registry; expose Planner, ToolSelector, explicitly constructed `AutonomousToolLoop`, pure stop-condition APIs, execution budgets, explicit recovery APIs, explicit task-completion verification APIs, explicit automatic-test orchestration, explicit failure-analysis observation, explicit root-cause analysis observation, explicit automatic-fix helper, explicit bounded self-correction helper, explicit regression-protection helper, explicit Final Verification helper, and the independent declarative Evaluation Task Model without changing CLI/default AgentLoop behavior or auto-enabling mutation/execution capabilities | Automatic CLI autonomy, Git mutation, memory, RAG, autonomous/background loops, benchmark execution |
| Model architecture | Implement a small decoder-only Transformer with local random weights and forward logits | Dataset, training, checkpoints, provider/CLI integration |
| Training engine | Train the existing model with CPU batching, next-token cross-entropy, optional response-only masks, AdamW, clipping, validation, metrics, deterministic seeding, and resumable checkpoints | Architecture redesign, pretrained weights, downloads, generation, inference, CLI or Agent integration |
| Tiny v1 experiment | Run a bounded from-scratch CPU experiment on a local backend corpus, record baseline/results, and verify an ignored checkpoint | External datasets, scraping, pretrained components, generation, inference, Agent or CLI integration |
| Checkpoint management | Atomically save/load metadata-aware Fodci state, validate compatibility, inspect/list, and select latest/best checkpoints | Committing weights, distributed checkpointing, generation, inference, CLI or Agent integration |
| Evaluation pipeline | Measure fixed validation objective with no-grad, compare random/trained states, label response-only loss, and emit lightweight reports | Inference server, CLI or Agent integration |
| Local inference and CLI integration | Load a compatible checkpoint without an optimizer, validate prompts, decode autoregressively on CPU, adapt requests through `FodciLocalProvider`, preserve bounded active-session history, and render responses in `fodci` | Project understanding, tool invocation, memory, RAG, planning, file/terminal operations, autonomous loops, later Phase 3 behavior |
| Tokenizer | Implement reversible byte fallback, deterministic small-corpus merges, and versioned save/load | Dataset collection, scraping, LLM training, generation, inference |
| Dataset pipeline | Load local text, validate, report unsupported/rejected files, exact-deduplicate, tokenize, append EOS boundaries, and stream fixed next-token chunks | Internet downloads, scraping, training loop, optimizer, checkpoints, model weights, inference |
| Coding dataset manifest | Build deterministic train/validation statistics, file identities, language distribution, and leakage checks over the existing pipeline | New tokenizer, new dataset system, training run, generation, inference, CLI or Agent integration |
| Instruction dataset | Parse deterministic Instruction/Input/Response files, build response-masked samples, manifest exact identities, and prevent split leakage | New special tokens, architecture changes, pretrained components, generation, inference, CLI or Agent integration |
| Logging | Configure the project logger safely | Runtime telemetry, log shipping, event tracing |
| Core contracts | Define typed, runtime-checkable boundaries | Concrete agents, models, tools, stores, or evaluators |
| Package layout | Reserve cohesive packages for later work | Empty placeholder implementations |
| Application startup | Compose configuration, logging, project context, one local provider, and the terminal session behind testable boundaries | Agent startup, project analysis, tool initialization |
| Interactive session | Keep the process alive, preserve bounded active-session messages, delegate normal text to an injected provider, and retain commands | Persistent memory, planning, tools, file operations, terminal execution |
| Input provider | Read one unprocessed line from stdin or an injected test source | Command parsing, dispatch, LLM or Agent calls |
| Command parser | Recognize leading-slash syntax, normalize names, preserve arguments | Command behavior, execution, Agent or LLM calls |
| Command dispatcher | Route registered handlers and report unknown commands | `/status` or future command behavior |
| Built-in commands | Provide deterministic local `/help` and `/exit` handlers | LLM, Agent, external API, or process-level `sys.exit()` behavior |
| Project context | Hold one validated absolute project root in core; expose an immutable structural `ProjectContext` and bounded `AgentLoop` consumption through the tools/agent layers | File mutation, Git/model metadata, planning, memory, RAG, autonomous execution |

No package imports another component's future concrete implementation. Any future dependency that would create a cycle should be inverted through a contract in `core` or a deliberately owned boundary module.

## Phase 8.3 — Success Criteria and Evaluation Scoring

`BenchmarkRunner` remains the execution and evidence-collection boundary. Phase 8.3 adds `BenchmarkScorer`, which consumes `BenchmarkResult.task_runs` and the declarative `EvaluationTask.success_criteria`; it does not execute tests, mutate files, override `FinalVerification`, or alter stop-condition authority.

The scorer produces four normalized dimensions: **task success (50%)**, **tests (30%)**, **code quality (10%)**, and **efficiency (10%)**. The immutable `EvaluationWeights` model validates finite, non-negative weights whose total is exactly `1.0`, and the host may provide another valid weighting. The model or agent output cannot change those weights. `ScoringPolicy` records the explicit evaluation version (`8.3`) and scoring-policy version (`1.0`).

Criterion evaluation is driven by declared `SuccessCriterion` types and structured evidence. Required final-verification, completion, regression, test, file-scope, and forbidden-change signals are represented as `CriterionEvaluation` records with status, satisfaction, score, evidence identifiers, evidence strength, and an explanation. Missing required evidence becomes `INSUFFICIENT_EVIDENCE` or `UNAVAILABLE`; it is never treated as a pass. Full task-success credit requires authoritative verification and completion evidence and is gated by required-criterion failures, regressions, forbidden changes, and blocking task statuses.

Every dimension includes evidence identifiers and an explanation. Test execution is distinguished from absence of observed failures, while quality uses only bounded objective signals such as unexpected modifications, forbidden changes, safety blocks, and regressions. Efficiency uses measurable budget data and is gated by correctness, so a fast failure cannot receive efficient-success credit.

Benchmark aggregation uses the arithmetic mean across all evaluated task scores in canonical task-id order. Failed, blocked, incomplete, and unavailable tasks remain in the denominator and are counted explicitly in `BenchmarkScore`; they cannot disappear from the result. Empty task collections produce a zero aggregate and zero evidence completeness. Serialization is canonical JSON with sorted keys and deterministic ordering.

This phase intentionally excludes version-to-version comparison, historical regression analysis, trends, leaderboards, and model comparisons. Those capabilities are reserved for Phase 8.4.

## Phase 8.4 — Evaluation Regression and Version Comparison

Phase 8.4 is a read-only comparison boundary over completed evaluation artifacts:

```text
BenchmarkRunner → BenchmarkResult → BenchmarkScorer → EvaluationResult
                                                        ↓
                                             compare_evaluations()
                                                        ↓
                                            EvaluationComparisonResult
```

`EvaluationVersion` identifies the baseline or candidate explicitly through agent version, evaluation version, scoring-policy version, benchmark-definition version, optional commit SHA, and immutable metadata. `EvaluationSnapshot` binds that identity to one existing `EvaluationResult`; there is no implicit latest version and baseline/candidate are never swapped.

`ComparisonConfig` provides a host-controlled epsilon, bounded evidence-reference count, and complete-evidence requirement. Compatibility is checked before scoring deltas: benchmark identity, evaluation version, scoring policy, benchmark-definition version, task-ID set, and scoring dimensions must match. Incompatibility returns `INCOMPARABLE` with reasons and no improvement claim. Missing or incomplete evidence returns `INCONCLUSIVE`; missing values are not converted to zero.

Each common task receives overall and per-dimension comparisons for task success, tests, code quality, and efficiency. Deltas are classified as improved, regressed, unchanged, or inconclusive using epsilon. Explicit status transitions have priority over numeric deltas: PASS to FAIL, BLOCKED, INCOMPLETE, or UNAVAILABLE is a regression, while FAIL/BLOCKED/INCOMPLETE/UNAVAILABLE to PASS is an improvement. Aggregate score, task counts, and dimension scores are also compared, but aggregate improvement cannot hide task regressions. Such a result is `IMPROVED_WITH_REGRESSIONS`; a high-severity pass-to-failure transition remains `REGRESSED`.

Regression severity is explicit and deterministic: `NONE`, `LOW`, `MEDIUM`, `HIGH`, and `CRITICAL`. A previous passing task that fails is `HIGH`; multiple or integrity-threatening findings can be escalated by the result policy. `REGRESSION_FREE` is returned only when compatible, complete evidence shows no task regression and no significant dimension regression. Every finding contains values, delta, classification, severity, bounded evidence IDs, and an explanation. `EvaluationComparisonResult.to_json()` emits canonical sorted-key JSON with stable task and finding order.

The public API is explicit: callers invoke `compare_evaluations(baseline, candidate, config)` and may pass `EvaluationSnapshot` objects or completed `EvaluationResult` objects with explicit version identities. The normal CLI and agent loop do not automatically run two benchmarks. This phase adds no subprocess, network, package installation, Git mutation, background worker, mutable global state, duplicate scorer, or automatic benchmark execution. Phase 9 remains out of scope.
