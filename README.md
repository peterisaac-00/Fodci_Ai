# Backend Engineering Agent

> **Current status: Phase 1.4 — User Input.**

Backend Engineering Agent is the foundation for a future **local, terminal-based AI agent** focused on backend engineering work. The intended product will use an interchangeable local or open-weight language-model provider rather than depend on hosted OpenAI, Anthropic, or Gemini APIs.

This repository includes the Phase 0 foundation, the Phase 1.1 executable entry point, the Phase 1.2 application startup flow, the Phase 1.3 interactive-session lifecycle, and the Phase 1.4 normal text-input layer. Installing the package provides `backend-ai`, which initializes the existing configuration and logging boundaries, enters a persistent session, and receives ordinary text from stdin. Phase 1.4 does **not** yet implement command parsing, `/help`, `/exit`, LLM inference, tool calling, filesystem access, terminal execution, planning, memory, evaluation behavior, or autonomous behavior.

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

The package currently exposes minimal protocols for `Agent`, `LLMProvider`, `Tool`, `Memory`, and `Evaluator`. They establish contracts only; they do not perform backend-agent work. The `backend-ai` console script delegates to `backend_ai.cli.main:main`, which delegates startup and session control to `backend_ai.application`. The application reuses the Phase 0 configuration and logging infrastructure, then hands control to `InteractiveSession` through the `InputProvider` boundary. The default provider reads stdin; input is recorded and acknowledged but not interpreted as commands. No agent or provider is initialized. See [the architecture notes](docs/architecture.md) for the intended dependency direction.

## Repository Layout

```text
.
├── src/backend_ai/
│   ├── agent/          # Future agent orchestration boundary
│   ├── application.py  # Application startup and session composition
│   ├── cli/            # Minimal console-entry boundary
│   ├── config/         # Small environment-backed settings abstraction
│   ├── core/           # Shared protocols and startup utilities
│   ├── evaluation/     # Future evaluator boundary
│   ├── llm/            # Provider contract, no model integration
│   ├── memory/         # Future memory boundary
│   ├── terminal/       # Session lifecycle and normal text-input boundary
│   └── tools/          # Future tool boundary
├── tests/
│   ├── unit/           # Meaningful Phase 0, 1.1, and 1.2 unit tests
│   └── integration/    # Reserved for cross-component tests
├── docs/               # Architecture and security foundation
├── scripts/            # Reserved for reviewed project-maintenance scripts
├── .env.example
├── pyproject.toml
└── README.md
```

## Development

Use Python 3.11 or later. The project has no runtime dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the test suite with:

```bash
pytest
```

Verify the Phase 1.1 console entry point after installation with:

```bash
backend-ai
```

The command initializes the existing application configuration and logger, enters the persistent session lifecycle, and reads normal text from stdin:

```text
You > hello
Received: hello
```

Input is preserved except for the line ending added by stdin. Empty input is retained and does not terminate the session. EOF and Ctrl+C terminate the session cleanly. No input is parsed as a command in the current phase.

Check that every package module compiles with:

```bash
python -m compileall -q src
```

For the minimal runtime package installation only, use:

```bash
python -m pip install -e .
```

## Configuration and Logging

Copy `.env.example` to `.env` only for local development. `.env` is ignored by Git. The Phase 0 settings abstraction currently recognizes `LOG_LEVEL` and `PROJECT_ROOT`, uses safe defaults, and validates the log level. The example also documents reserved names for later stages without reading them yet.

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

Phase 1.4 adds only normal text input through `InputProvider` and stdin. It does not add command handling, `/help`, `/exit`, special commands, project-path handling, model loading, hosted LLM SDK, chat interface, tool implementation, file-editing feature, terminal runner, git integration, planner, database, vector store, RAG system, training pipeline, multi-agent system, web interface, authentication, or deployment service.

## License

This project is distributed under the [MIT License](LICENSE).
