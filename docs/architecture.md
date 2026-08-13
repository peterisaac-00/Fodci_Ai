# Architecture Direction

## Application startup boundary

Phase 1.2 through 1.5 add a thin application boundary between the console entry point and future agent orchestration:

```text
backend-ai
    ↓
CLI entry point
    ↓
Application.start()
    ↓
existing configuration + logging bootstrap
    ↓
InteractiveSession
    ↓
InputProvider
    ↓
CommandParser
    ↓
CommandDispatcher ──→ registered handler
    ↓
normal input passthrough
```

`backend_ai.cli.main` is responsible only for process-facing output and status. `backend_ai.application` composes the currently available startup steps through `core.bootstrap`, then delegates session persistence and input reception to `InteractiveSession`. `InputProvider` is injectable for deterministic tests and defaults to stdin in production. `CommandParser` recognizes only a leading `/`; `CommandDispatcher` routes registered names and reports unknown commands without executing them. Normal input remains unchanged. No agent, provider, tool, memory, or execution component is initialized.

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

## Present implementation

The repository implements only these foundation pieces:

| Area | Phase 0 responsibility | Intentionally absent |
| --- | --- | --- |
| Configuration | Resolve a project root and validate a log level | Agent-specific settings, secret loading, provider configuration |
| Logging | Configure the project logger safely | Runtime telemetry, log shipping, event tracing |
| Core contracts | Define typed, runtime-checkable boundaries | Concrete agents, models, tools, stores, or evaluators |
| Package layout | Reserve cohesive packages for later work | Empty placeholder implementations |
| Application startup | Compose existing configuration and logging behind a testable boundary | Agent startup, command handling |
| Interactive session | Keep the process alive and receive normal text behind stoppable lifecycle boundaries | Command handling, prompts beyond the minimal input prompt, agent requests |
| Input provider | Read one unprocessed line from stdin or an injected test source | Command parsing, dispatch, LLM or Agent calls |
| Command parser | Recognize leading-slash syntax, normalize names, preserve arguments | Command behavior, execution, Agent or LLM calls |
| Command dispatcher | Route registered handlers and report unknown commands | Built-in `/help`, `/exit`, `/status`, or other command behavior |

No package imports another component's future concrete implementation. Any future dependency that would create a cycle should be inverted through a contract in `core` or a deliberately owned boundary module.
