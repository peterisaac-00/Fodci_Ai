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

## Present implementation

The repository implements only these foundation pieces:

| Area | Phase 0 responsibility | Intentionally absent |
| --- | --- | --- |
| Configuration | Resolve a configured root path and validate a log level | Agent-specific settings, secret loading, provider configuration |
| LLM provider | Define typed messages, request/response, provider protocol, one provider error, and the local Fodci adapter | External APIs, network access, fallback models, tool calling |
| Tool layer | Reuse the `Tool` protocol for read-only Phase 3 tools plus create-only `WriteFileTool`/`write_file`, exact existing-file `EditFileTool`/`edit_file`, regular-file-only `DeleteFileTool`/`delete_file`, additive `safe_editing` policy/session, read-only `GitDiffTool`/`git_diff` plus `GitStatusTool`/`git_status`, read-only `ModificationVerifier`/`verify_modification`, and additive `ModificationTransaction`/recovery models with structured results, snapshots, bounded internal diffs, optional backups, verification, deterministic boundaries, symlink safety, revalidation, and opt-in mutation/inspection | Git mutation, terminal execution, LLM tool-calling |
| Agent adapter | Keep `ProviderBackedAgent` compatibility and bounded `AgentLoop` orchestration over the default read-only registry; allow explicit external registry injection without enabling create/edit/delete mutation or Git inspection by default; do not inject SafeEditSession, GitDiffTool, GitStatusTool, ModificationVerifier, or ModificationTransaction | Agent modification loops, command execution, Git mutation, memory, RAG, autonomous/background loops |
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
