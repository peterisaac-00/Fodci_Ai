# Phase 14.5 — Backend Domain Policy and Output Guard

> This phase constrains an experimental language provider to backend engineering at runtime. It does not erase knowledge from pretrained model weights.

## Runtime design

`BackendDomainPolicy` applies a deterministic conservative allowlist for Python backend, FastAPI, REST/HTTP, SQL/PostgreSQL, authentication, testing, debugging, and backend architecture. Explicitly external topics such as Unity, Android, frontend-only work, and general machine-learning training are blocked before reaching the inner provider.

`BackendOutputGuard` rejects empty responses, responses outside a bounded word range, excessively repetitive text, and responses with no backend signal. `BackendScopedProvider` composes both controls around any existing `LLMProvider`, including the experimental Qwen provider. The default Fodci application remains unchanged.

| Gate | Result |
|---|---|
| Probes | 5/5 passed |
| Backend questions accepted | 2/2 |
| Explicit out-of-scope questions blocked | 2/2 |
| Repetitive output rejected | `True` |
| Out-of-scope calls blocked before provider | `True` |
| Inner provider calls | 3 |
| Stable runtime replaced | `False` |
| All gates | `True` |

This design provides operational specialization: Qwen may still contain broad programming knowledge internally, but Fodci decides which requests reach it and which outputs are accepted. The policy is not a security boundary by itself; future production use should add stronger intent classification, audit logging, and task-specific correctness tests.
