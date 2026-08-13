# Architecture Direction

## Application startup boundary

Phase 1.2 through 2.1 add a thin application boundary between the console entry point and future agent orchestration:

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

`Message`, `LLMRequest`, and `LLMResponse` contain only role/content messages and generated text. `LLMProviderError` provides one typed provider-level failure. `ProviderBackedAgent` accepts the provider through dependency injection and delegates one request only. The concrete local model is intentionally not implemented in Phase 2.1; no model runtime, weights, network access, streaming, tools, embeddings, planning, or autonomous loop exists.

## Present implementation

The repository implements only these foundation pieces:

| Area | Phase 0 responsibility | Intentionally absent |
| --- | --- | --- |
| Configuration | Resolve a configured root path and validate a log level | Agent-specific settings, secret loading, provider configuration |
| LLM provider | Define typed messages, request/response, provider protocol, and one provider error | Concrete model, runtime, loading, inference, network access |
| Agent adapter | Accept an injected provider and delegate one request | Planning, tools, memory, execution, autonomous loop |
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
