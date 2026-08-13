"""Minimal typed contracts for future backend-agent subsystems."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

MessageRole = Literal["system", "user", "assistant"]
_ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True, slots=True)
class Message:
    """A minimal role/content message sent to a provider."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in _ALLOWED_MESSAGE_ROLES:
            raise ValueError(f"Unsupported message role: {self.role!r}.")


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """The smallest provider request: an ordered tuple of messages."""

    messages: tuple[Message, ...]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("LLMRequest requires at least one message.")

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> "LLMRequest":
        """Build a request from an optional system prompt and user prompt."""

        messages: list[Message] = []
        if system_prompt is not None:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return cls(messages=tuple(messages))


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Minimal provider output containing generated text only."""

    text: str


class LLMProviderError(RuntimeError):
    """A provider-level failure distinct from normal application behavior."""


@runtime_checkable
class Agent(Protocol):
    """Future orchestration boundary; no concrete agent is required yet."""

    def run(self, task: str) -> str:
        """Process a requested backend-engineering task."""


@runtime_checkable
class LLMProvider(Protocol):
    """Provider boundary independent of any concrete local model."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for a typed request."""


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
