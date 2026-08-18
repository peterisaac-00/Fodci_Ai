"""Typed LLM boundary and the local Fodci provider adapter."""

from backend_ai.llm.errors import LLMProviderError
from backend_ai.llm.fodci_provider import DEFAULT_FODCI_SYSTEM_PROMPT, FodciLocalProvider
from backend_ai.llm.backend_scope import (
    BackendDomainPolicy,
    BackendOutputGuard,
    BackendScopedProvider,
    DomainDecision,
    GuardDecision,
    OUT_OF_SCOPE_RESPONSE,
)
from backend_ai.llm.pretrained_code_provider import (
    DEFAULT_PRETRAINED_SYSTEM_PROMPT,
    PretrainedCodeProvider,
    PretrainedProviderConfig,
)
from backend_ai.llm.models import LLMRequest, LLMResponse, Message, MessageRole
from backend_ai.llm.provider import LLMProvider

__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "DEFAULT_FODCI_SYSTEM_PROMPT",
    "FodciLocalProvider",
    "BackendDomainPolicy",
    "BackendOutputGuard",
    "BackendScopedProvider",
    "DomainDecision",
    "GuardDecision",
    "OUT_OF_SCOPE_RESPONSE",
    "DEFAULT_PRETRAINED_SYSTEM_PROMPT",
    "PretrainedCodeProvider",
    "PretrainedProviderConfig",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "MessageRole",
]
