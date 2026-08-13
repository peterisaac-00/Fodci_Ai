from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend_ai.llm import (
    FodciLocalProvider,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    Message,
)


class RecordingEngine:
    def __init__(self, response: str = "local answer") -> None:
        self.prompts: list[str] = []
        self.response = response

    def generate(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(generated_text=self.response)


def test_fodci_provider_implements_llm_provider_and_returns_response() -> None:
    engine = RecordingEngine()
    provider = FodciLocalProvider(engine)

    response = provider.generate(LLMRequest.from_prompt("Fix the API"))

    assert isinstance(provider, LLMProvider)
    assert isinstance(response, LLMResponse)
    assert response.text == "local answer"
    assert engine.prompts[0].startswith("### Instruction\n")
    assert "Fix the API" in engine.prompts[0]
    assert engine.prompts[0].endswith("### Response\n")


def test_fodci_provider_preserves_ordered_conversation_messages() -> None:
    engine = RecordingEngine(response="next")
    provider = FodciLocalProvider(engine)
    request = LLMRequest(
        messages=(
            Message(role="system", content="Backend only"),
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Second question"),
        )
    )

    assert provider.generate(request).text == "next"
    prompt = engine.prompts[0]
    assert prompt.index("Backend only") < prompt.index("First question")
    assert prompt.index("First question") < prompt.index("First answer")
    assert prompt.index("First answer") < prompt.index("Second question")
    assert "User:\nFirst question" in prompt
    assert "Fodci:\nFirst answer" in prompt


def test_fodci_provider_rejects_non_request_objects() -> None:
    provider = FodciLocalProvider(RecordingEngine())

    with pytest.raises(LLMProviderError, match="at least one message"):
        provider.generate(object())  # type: ignore[arg-type]


def test_fodci_provider_rejects_requests_without_final_user_turn() -> None:
    provider = FodciLocalProvider(RecordingEngine())
    request = LLMRequest(
        messages=(Message(role="assistant", content="not a user turn"),)
    )

    with pytest.raises(LLMProviderError, match="final message must be a user"):
        provider.generate(request)


def test_fodci_provider_wraps_inference_failures() -> None:
    class FailingEngine:
        def generate(self, prompt: str) -> object:
            raise ValueError("context length exceeded")

    provider = FodciLocalProvider(FailingEngine())

    with pytest.raises(LLMProviderError, match="context length exceeded"):
        provider.generate(LLMRequest.from_prompt("prompt"))


def test_missing_checkpoint_is_a_typed_startup_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"

    with pytest.raises(LLMProviderError, match="checkpoint is unavailable"):
        FodciLocalProvider.from_checkpoint(missing)
