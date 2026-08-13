"""Provider-injected agent wiring for Phase 2.1."""

from __future__ import annotations

from backend_ai.core.contracts import Agent, LLMProvider, LLMRequest


class ProviderBackedAgent:
    """Minimal Agent adapter that delegates one request to an injected provider.

    This is dependency wiring only. It does not plan, call tools, maintain
    memory, execute commands, or implement an autonomous loop.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMProvider:
        """Return the injected provider without constructing a concrete one."""

        return self._llm

    def run(self, task: str) -> str:
        """Delegate one task-shaped prompt and return provider text."""

        response = self._llm.generate(LLMRequest.from_prompt(task))
        return response.text


__all__ = ["Agent", "ProviderBackedAgent"]
