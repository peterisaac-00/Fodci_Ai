from __future__ import annotations

import pytest

from backend_ai.core.contracts import LLMRequest, LLMResponse
from backend_ai.llm.backend_scope import (
    BackendDomainPolicy,
    BackendOutputGuard,
    BackendScopedProvider,
    OUT_OF_SCOPE_RESPONSE,
)


class _FakeProvider:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.text)


def test_backend_policy_allows_backend_questions() -> None:
    policy = BackendDomainPolicy()

    assert policy.decide("How do I validate a JWT in FastAPI?").allowed is True
    assert policy.decide("How can I optimize a PostgreSQL query?").allowed is True
    assert policy.decide("How should pytest mock a backend database?").allowed is True


def test_backend_policy_rejects_explicitly_external_questions() -> None:
    policy = BackendDomainPolicy()

    decision = policy.decide("How do I create a Unity game?")
    assert decision.allowed is False
    assert "unity" in decision.matched_terms
    assert policy.decide("How do I build an Android app?").allowed is False
    assert policy.decide("Tell me a general story.").allowed is False


def test_backend_scoped_provider_does_not_call_inner_provider_outside_scope() -> None:
    inner = _FakeProvider("A backend API uses HTTP and validates the request.")
    provider = BackendScopedProvider(inner)

    response = provider.generate(LLMRequest.from_prompt("How do I build a Unity game?"))

    assert response.text == OUT_OF_SCOPE_RESPONSE
    assert inner.calls == 0


def test_backend_scoped_provider_accepts_good_backend_output() -> None:
    inner = _FakeProvider("Use FastAPI with Pydantic validation for the request and return a JSON response.")
    provider = BackendScopedProvider(inner)

    response = provider.generate(LLMRequest.from_prompt("How should FastAPI validate request data?"))

    assert response.text == inner.text
    assert inner.calls == 1


def test_backend_scoped_provider_replaces_empty_or_gibberish_output() -> None:
    inner = _FakeProvider("the the the the the the the")
    provider = BackendScopedProvider(inner)

    response = provider.generate(LLMRequest.from_prompt("How do I debug a Python traceback?"))

    assert response.text.startswith("I could not produce")
    assert inner.calls == 1


def test_backend_output_guard_bounds_response_length() -> None:
    guard = BackendOutputGuard(min_words=3, max_words=5)

    assert guard.inspect("Use SQL parameters to prevent injection.").accepted is False
    assert guard.inspect("FastAPI SQL test").accepted is True
    assert guard.inspect("FastAPI SQL test with many extra words").accepted is False


def test_backend_scoped_provider_rejects_invalid_request() -> None:
    provider = BackendScopedProvider(_FakeProvider("FastAPI handles HTTP requests."))

    with pytest.raises(Exception):
        provider.generate(LLMRequest(messages=()))
