# Security Foundation

The Phase 0 code does not execute shell commands, access project files on behalf of a user, call a model provider, or persist memory. Future phases must preserve the following principles before adding those abilities.

| Principle | Foundation requirement |
| --- | --- |
| Secret handling | Keep API keys and credentials out of source, Git history, tests, and logs. `.env` is ignored and `.env.example` contains no secrets. |
| Project boundary | Operate inside an explicit, resolved project root. Access outside that root requires explicit authorization. |
| Shell safety | Treat every command as potentially dangerous; validate intent, arguments, working directory, and execution policy. |
| Tool inputs | Validate types, paths, and command parameters before use. Do not trust model-generated input. |
| Logging | Record useful operational context without emitting credentials, tokens, or sensitive file content. |
| Least privilege | Give each provider, tool, and persistence component only the access it requires. |

These are design constraints, not yet a complete sandbox or authorization mechanism. Concrete enforcement belongs to later implementation phases and must be reviewed before execution features are introduced.
