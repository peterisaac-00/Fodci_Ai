from __future__ import annotations

import pytest

from backend_ai.agent import ProviderBackedAgent
from backend_ai.llm import (
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    Message,
)


class FakeLLMProvider:
    """Deterministic test double; it is not an AI implementation."""

    def __init__(self, response_text: str = "test response") -> None:
        self.response_text = response_text
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=self.response_text)


def test_fake_provider_satisfies_runtime_provider_contract() -> None:
    assert isinstance(FakeLLMProvider(), LLMProvider)


def test_request_contains_minimal_messages_and_preserves_order() -> None:
    request = LLMRequest.from_prompt("Fix the API", system_prompt="Backend only")

    assert request.messages == (
        Message(role="system", content="Backend only"),
        Message(role="user", content="Fix the API"),
    )


def test_response_contains_only_minimal_output_text() -> None:
    response = LLMResponse(text="deterministic output")

    assert response.text == "deterministic output"


def test_provider_backed_agent_uses_injected_provider() -> None:
    provider = FakeLLMProvider(response_text="injected response")
    agent = ProviderBackedAgent(provider)

    result = agent.run("inspect this backend task")

    assert agent.llm is provider
    assert result == "injected response"
    assert provider.requests == [
        LLMRequest.from_prompt("inspect this backend task"),
    ]


def test_provider_failure_uses_typed_provider_error() -> None:
    class FailingProvider:
        def generate(self, request: LLMRequest) -> LLMResponse:
            raise LLMProviderError("provider unavailable")

    agent = ProviderBackedAgent(FailingProvider())

    with pytest.raises(LLMProviderError, match="provider unavailable"):
        agent.run("task")


def test_empty_requests_and_unknown_roles_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LLMRequest(messages=())

    with pytest.raises(ValueError, match="Unsupported message role"):
        Message(role="developer", content="not supported")  # type: ignore[arg-type]
