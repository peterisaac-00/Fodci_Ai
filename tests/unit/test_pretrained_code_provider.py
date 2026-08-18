from __future__ import annotations

import pytest

from backend_ai.core.contracts import LLMRequest, LLMProviderError, Message
from backend_ai.llm.pretrained_code_provider import (
    PretrainedCodeProvider,
    PretrainedProviderConfig,
)


class _FakeTokenizer:
    eos_token_id = 0

    def __call__(self, prompt: str, **kwargs):
        self.prompt = prompt
        assert kwargs == {"return_tensors": "pt"}
        return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    def decode(self, sequence, *, skip_special_tokens: bool):
        assert skip_special_tokens is True
        assert sequence == [1, 2, 3]
        return self.prompt + "Use a parameterized SQL query."


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, input_ids, **kwargs):
        self.calls.append({"input_ids": input_ids, **kwargs})
        return [[1, 2, 3]]


def _provider() -> tuple[PretrainedCodeProvider, _FakeModel]:
    model = _FakeModel()
    provider = PretrainedCodeProvider(
        _FakeTokenizer(),
        model,
        config=PretrainedProviderConfig(model_id="local-test-model", max_new_tokens=24),
    )
    return provider, model


def test_pretrained_provider_generates_through_typed_boundary() -> None:
    provider, model = _provider()
    response = provider.generate(LLMRequest.from_prompt("How do I avoid SQL injection?"))

    assert response.text == "Use a parameterized SQL query."
    assert len(model.calls) == 1
    assert model.calls[0]["max_new_tokens"] == 24
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["temperature"] == 0.2


def test_pretrained_provider_accepts_explicit_system_policy() -> None:
    provider, _ = _provider()
    provider.system_prompt = "Backend only."
    formatted = provider._format_request(LLMRequest.from_prompt("Explain REST."))

    assert formatted.startswith("System:\nBackend only.")
    assert "User:\nExplain REST." in formatted
    assert formatted.endswith("Assistant:\n")


def test_pretrained_provider_rejects_invalid_conversations() -> None:
    provider, _ = _provider()

    with pytest.raises(LLMProviderError, match="final message"):
        provider.generate(LLMRequest(messages=(Message(role="user", content="ok"), Message(role="assistant", content="no"))))


def test_pretrained_provider_config_is_bounded() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        PretrainedProviderConfig(model_id="local", max_new_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        PretrainedProviderConfig(model_id="local", temperature=0.0)


def test_pretrained_provider_does_not_require_optional_runtime_at_import_time() -> None:
    config = PretrainedProviderConfig(model_id="local-cache-model")
    assert config.device == "cpu"
    assert config.trust_remote_code is False
