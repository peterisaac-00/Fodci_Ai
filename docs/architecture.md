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
load() → compatibility validation → state restoration
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

`InferenceEngine` receives an existing `FodciModel` and `FodciTokenizer`. If a checkpoint is configured, it uses `CheckpointManager` to load the existing model and optimizer payload into a temporary optimizer solely for complete checkpoint validation; the optimizer is never stepped or exposed. Model version, tokenizer version, vocabulary size, context length, and structural model fields are validated before loading.

The default is CPU greedy decoding with `temperature=1.0`, `do_sample=False`, EOS stopping, and a bounded `max_new_tokens`. Optional seeded multinomial sampling supports positive finite temperature and optional positive `top_k`; `top_k` is filtered before sampling. Prompts are encoded without truncation, and an empty or over-context prompt fails clearly. Generation stops when EOS is selected, the new-token budget is exhausted, or the context window is full. `InferenceResult` exposes only generated text, counts, stop reason, model version, checkpoint identity, and effective configuration.

The real smoke workflow uses the existing ignored Tiny v1 checkpoint on CPU with short English, Python, and backend prompts. It validates the checkpoint → model → tokenizer → autoregressive decoding path only; generated text is not evidence of intelligence or production readiness. No CLI command, chatbot UI, Agent loop, tools, memory, file operations, or Phase 2.12/3 functionality is included.

## Present implementation

The repository implements only these foundation pieces:

| Area | Phase 0 responsibility | Intentionally absent |
| --- | --- | --- |
| Configuration | Resolve a configured root path and validate a log level | Agent-specific settings, secret loading, provider configuration |
| LLM provider | Define typed messages, request/response, provider protocol, and one provider error | Concrete model, runtime, loading, inference, network access |
| Agent adapter | Accept an injected provider and delegate one request | Planning, tools, memory, execution, autonomous loop |
| Model architecture | Implement a small decoder-only Transformer with local random weights and forward logits | Dataset, training, checkpoints, provider/CLI integration |
| Training engine | Train the existing model with CPU batching, next-token cross-entropy, optional response-only masks, AdamW, clipping, validation, metrics, deterministic seeding, and resumable checkpoints | Architecture redesign, pretrained weights, downloads, generation, inference, CLI or Agent integration |
| Tiny v1 experiment | Run a bounded from-scratch CPU experiment on a local backend corpus, record baseline/results, and verify an ignored checkpoint | External datasets, scraping, pretrained components, generation, inference, Agent or CLI integration |
| Checkpoint management | Atomically save/load metadata-aware Fodci state, validate compatibility, inspect/list, and select latest/best checkpoints | Committing weights, distributed checkpointing, generation, inference, CLI or Agent integration |
| Evaluation pipeline | Measure fixed validation objective with no-grad, compare random/trained states, label response-only loss, and emit lightweight reports | Inference server, CLI or Agent integration |
| Local inference | Load a compatible checkpoint, validate prompts, decode autoregressively on CPU, stop on EOS/budget/context, and return typed result metadata | Chat UI, CLI command, Agent loop, tools, memory, file/terminal operations, Phase 2.12/3 |
| Tokenizer | Implement reversible byte fallback, deterministic small-corpus merges, and versioned save/load | Dataset collection, scraping, LLM training, generation, inference |
| Dataset pipeline | Load local text, validate, report unsupported/rejected files, exact-deduplicate, tokenize, append EOS boundaries, and stream fixed next-token chunks | Internet downloads, scraping, training loop, optimizer, checkpoints, model weights, inference |
| Coding dataset manifest | Build deterministic train/validation statistics, file identities, language distribution, and leakage checks over the existing pipeline | New tokenizer, new dataset system, training run, generation, inference, CLI or Agent integration |
| Instruction dataset | Parse deterministic Instruction/Input/Response files, build response-masked samples, manifest exact identities, and prevent split leakage | New special tokens, architecture changes, pretrained components, generation, inference, CLI or Agent integration |
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
