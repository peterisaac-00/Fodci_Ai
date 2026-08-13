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

`backend_ai.cli.main` is responsible only for process-facing output and status. `backend_ai.application` composes the currently available startup steps through `core.bootstrap`, resolves a minimal `ProjectContext`, and then delegates session persistence and input reception to `InteractiveSession`. `ProjectContext` contains only an absolute, normalized, validated root path; resolution checks existence and directory type but never scans the root. `InputProvider` is injectable for deterministic tests and defaults to stdin in production. `CommandParser` recognizes only a leading `/`; `CommandDispatcher` routes registered names and reports unknown commands without executing them. Phase 1.6 registers `/help` and `/exit` through this same registry. `/help` derives its output from registered metadata, while `/exit` returns a structured stop request that the session handles. The CLI does not import or initialize a concrete provider or model.

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

## Present implementation

The repository implements only these foundation pieces:

| Area | Phase 0 responsibility | Intentionally absent |
| --- | --- | --- |
| Configuration | Resolve a configured root path and validate a log level | Agent-specific settings, secret loading, provider configuration |
| LLM provider | Define typed messages, request/response, provider protocol, and one provider error | Concrete model, runtime, loading, inference, network access |
| Agent adapter | Accept an injected provider and delegate one request | Planning, tools, memory, execution, autonomous loop |
| Model architecture | Implement a small decoder-only Transformer with local random weights and forward logits | Dataset, training, checkpoints, provider/CLI integration |
| Tokenizer | Implement reversible byte fallback, deterministic small-corpus merges, and versioned save/load | Dataset collection, scraping, LLM training, generation, inference |
| Dataset pipeline | Load local text, validate, exact-deduplicate, tokenize, append EOS boundaries, and stream fixed next-token chunks | Internet downloads, scraping, training loop, optimizer, checkpoints, model weights, inference |
| Logging | Configure the project logger safely | Runtime telemetry, log shipping, event tracing |
| Core contracts | Define typed, runtime-checkable boundaries | Concrete agents, models, tools, stores, or evaluators |
| Package layout | Reserve cohesive packages for later work | Empty placeholder implementations |
| Application startup | Compose existing configuration and logging behind a testable boundary | Agent startup, command handling |
| Interactive session | Keep the process alive and receive normal text behind stoppable lifecycle boundaries | Command handling, prompts beyond the minimal input prompt, agent requests |
| Input provider | Read one unprocessed line from stdin or an injected test source | Command parsing, dispatch, LLM or Agent calls |
| Command parser | Recognize leading-slash syntax, normalize names, preserve arguments | Command behavior, execution, Agent or LLM calls |
| Command dispatcher | Route registered handlers and report unknown commands | `/status` or future command behavior |
| Built-in commands | Provide deterministic local `/help` and `/exit` handlers | LLM, Agent, external API, or process-level `sys.exit()` behavior |
| Project context | Hold one validated absolute project root | File lists, framework detection, Git or model metadata, project scanning |

No package imports another component's future concrete implementation. Any future dependency that would create a cycle should be inverted through a contract in `core` or a deliberately owned boundary module.
