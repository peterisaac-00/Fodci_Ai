# Backend Engineering Agent

> **Current status: Phase 2.4 — Fodci Dataset Pipeline.**

Backend Engineering Agent is the foundation for a future **local, terminal-based AI agent** focused on backend engineering work. The intended product will use an interchangeable local or open-weight language-model provider rather than depend on hosted OpenAI, Anthropic, or Gemini APIs.

This repository includes the complete Phase 1 CLI foundation, Phase 2.1's minimal typed LLM provider boundary, Phase 2.2's small decoder-only Transformer architecture, Phase 2.3's reversible byte-level tokenizer, and Phase 2.4's local streaming dataset pipeline. The model remains randomly initialized and untrained; the tokenizer and dataset pipeline are isolated from the CLI and from any training runtime.

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

The package exposes minimal typed contracts for `Agent`, `LLMProvider`, `Message`, `LLMRequest`, `LLMResponse`, `Tool`, `Memory`, and `Evaluator`. `ProviderBackedAgent` accepts an `LLMProvider` through dependency injection and delegates one request only. The isolated `backend_ai.model` package contains `FodciModel`, a configurable decoder-only Transformer with token embeddings, learned positional embeddings, causal multi-head attention, GELU feed-forward blocks, final normalization, and a language-modeling head. The official `fodci` console script remains disconnected from the model. See [the architecture notes](docs/architecture.md) for the intended dependency direction.

## Repository Layout

```text
.
├── src/backend_ai/
│   ├── agent/          # Agent protocol and provider-injected adapter
│   ├── application.py  # Application startup and session composition
│   ├── cli/            # Minimal console-entry boundary
│   ├── config/         # Small environment-backed settings abstraction
│   ├── core/           # Shared protocols, startup, and project context
│   ├── evaluation/     # Future evaluator boundary
│   ├── llm/            # Typed provider boundary, no model integration
│   ├── model/          # From-scratch Transformer architecture, no training
│   ├── tokenizer/      # Reversible byte-level tokenizer and tiny BPE training
│   ├── dataset/        # Local validation, exact deduplication, and streaming chunks
│   ├── memory/         # Future memory boundary
│   ├── commands/       # Command parsing and dispatch boundaries
│   ├── terminal/       # Session lifecycle and normal text-input boundary
│   └── tools/          # Future tool boundary
├── tests/
│   ├── unit/           # Foundation, CLI, model, tokenizer, and dataset tests
│   └── integration/    # Reserved for cross-component tests
├── docs/               # Architecture and security foundation
├── scripts/            # Reserved for reviewed project-maintenance scripts
├── .env.example
├── pyproject.toml
└── README.md
```

## Development

Use Python 3.11 or later. The base package and CLI have no runtime dependencies. The optional `model` extra adds PyTorch only for the isolated architecture layer. The official executable is `fodci`, mapped to the existing `backend_ai.cli.main:main` entry point.

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

The command initializes the existing application configuration and logger, resolves the project root, enters the persistent session lifecycle, and reads normal text from stdin:

```text
You > hello
Received: hello
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

Check that every package module compiles with:

```bash
python -m compileall -q src
```

For the minimal runtime package installation only, use:

```bash
python -m pip install -e .
```

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
| 4 | File Modification |
| 5 | Terminal + Execution |
| 6 | Autonomous Agent Loop |
| 7 | Testing + Self-Correction |
| 8 | Evaluation |
| 9 | Memory |
| 10 | Experience Dataset |
| 11 | Model Improvement |
| 12 | Advanced Agent |

## Non-Goals for the Current Phase

Phase 2.4 adds only a local, deterministic, streaming dataset pipeline: supported-file discovery, UTF-8 and size validation, exact SHA-256 deduplication, tokenizer application, EOS document boundaries, and next-token chunks. It does not add internet downloads, scraping, code collection, a training loop, optimizer, loss, backpropagation, checkpointing, model weights, generation, inference runtime, agent integration, tools, memory, evaluation, external APIs, or self-learning.

## License

This project is distributed under the [MIT License](LICENSE).
