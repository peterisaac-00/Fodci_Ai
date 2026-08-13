# Integration Tests

Phase 1.9 validates the public CLI through the installed `fodci` executable. The subprocess tests exercise the complete startup path in a temporary working directory and cover successful `/help` and `/exit`, normal-input passthrough, unknown commands, clean EOF, and invalid `PROJECT_ROOT` error handling.

These tests intentionally verify existing behavior only. They do not initialize an LLM or Agent, inspect project contents, execute project commands, or add new product functionality.
