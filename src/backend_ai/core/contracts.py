"""Minimal contracts that preserve future subsystem independence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Agent(Protocol):
    """Future orchestration boundary; no implementation is supplied in Phase 0."""

    def run(self, task: str) -> str:
        """Process a requested backend-engineering task."""


@runtime_checkable
class LLMProvider(Protocol):
    """Model-provider boundary independent of any concrete local model."""

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Generate a text response for a prompt."""


@runtime_checkable
class Tool(Protocol):
    """Future tool boundary; concrete tools own their own input validation."""

    name: str
    description: str

    def run(self, arguments: Mapping[str, Any]) -> Any:
        """Run the tool with validated arguments."""


@runtime_checkable
class Memory(Protocol):
    """Future persistence boundary without committing to a storage mechanism."""

    def retrieve(self, query: str) -> list[str]:
        """Retrieve relevant stored entries."""

    def store(self, content: str) -> None:
        """Store an entry for possible later retrieval."""


@runtime_checkable
class Evaluator(Protocol):
    """Future evaluation boundary for structured assessment of an outcome."""

    def evaluate(self, subject: object) -> Mapping[str, float]:
        """Return named evaluation metrics for a subject."""
