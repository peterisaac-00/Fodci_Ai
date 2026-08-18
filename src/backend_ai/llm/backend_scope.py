"""Runtime scope policy and output guard for backend-only language providers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

from backend_ai.core.contracts import LLMProvider, LLMProviderError, LLMRequest, LLMResponse


OUT_OF_SCOPE_RESPONSE: Final[str] = (
    "I specialize in backend engineering, so I cannot help with that topic. "
    "Please ask about Python backend services, FastAPI, REST, SQL, authentication, testing, debugging, or backend architecture."
)


@dataclass(frozen=True, slots=True)
class DomainDecision:
    allowed: bool
    reason: str
    matched_terms: tuple[str, ...]


class BackendDomainPolicy:
    """Conservative deterministic allowlist for the initial backend scope."""

    ALLOWED_TERMS: Final[tuple[str, ...]] = (
        "backend", "api", "rest", "http", "fastapi", "django", "flask", "python",
        "sql", "postgres", "postgresql", "mysql", "redis", "database", "schema",
        "query", "jwt", "oauth", "authentication", "authorization", "password",
        "security", "pytest", "test", "testing", "debug", "traceback", "exception",
        "error", "logging", "latency", "endpoint", "microservice", "repository",
        "service layer", "docker", "deployment", "webhook", "json", "pydantic",
    )
    EXCLUDED_TERMS: Final[tuple[str, ...]] = (
        "unity", "unreal", "android", "ios", "swiftui", "react native", "frontend",
        "css", "html page", "photoshop", "blender", "game development", "machine learning",
        "deep learning", "neural network training", "cryptocurrency trading",
    )

    def decide(self, text: str) -> DomainDecision:
        if not isinstance(text, str) or not text.strip():
            return DomainDecision(False, "empty request", ())
        normalized = _normalize(text)
        excluded = tuple(term for term in self.EXCLUDED_TERMS if _contains_term(normalized, term))
        if excluded:
            return DomainDecision(False, "explicitly excluded topic", excluded)
        matched = tuple(term for term in self.ALLOWED_TERMS if _contains_term(normalized, term))
        if matched:
            return DomainDecision(True, "backend term matched", matched)
        return DomainDecision(False, "no backend scope term matched", ())


@dataclass(frozen=True, slots=True)
class GuardDecision:
    accepted: bool
    reason: str
    normalized_word_count: int


class BackendOutputGuard:
    """Reject empty, repetitive, or obviously out-of-domain provider output."""

    def __init__(self, policy: BackendDomainPolicy | None = None, *, min_words: int = 3, max_words: int = 512) -> None:
        if not 1 <= min_words <= max_words:
            raise ValueError("word bounds are invalid")
        self.policy = policy or BackendDomainPolicy()
        self.min_words = min_words
        self.max_words = max_words

    def inspect(self, text: str) -> GuardDecision:
        if not isinstance(text, str) or not text.strip():
            return GuardDecision(False, "empty response", 0)
        words = re.findall(r"[A-Za-z][A-Za-z0-9_+#.-]*", text)
        if len(words) < self.min_words:
            return GuardDecision(False, "response is too short", len(words))
        if len(words) > self.max_words:
            return GuardDecision(False, "response exceeds output bound", len(words))
        normalized = [word.lower() for word in words]
        repeated = sum(count - 1 for count in {word: normalized.count(word) for word in set(normalized)}.values() if count > 1)
        if repeated / len(normalized) > 0.65:
            return GuardDecision(False, "response is excessively repetitive", len(words))
        if not any(_contains_term(" ".join(normalized), term) for term in BackendDomainPolicy.ALLOWED_TERMS):
            return GuardDecision(False, "response contains no backend signal", len(words))
        return GuardDecision(True, "response passed bounded backend guard", len(words))


class BackendScopedProvider:
    """Enforce backend-only input/output policy around an existing provider."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        policy: BackendDomainPolicy | None = None,
        guard: BackendOutputGuard | None = None,
        out_of_scope_response: str = OUT_OF_SCOPE_RESPONSE,
    ) -> None:
        if not callable(getattr(provider, "generate", None)):
            raise LLMProviderError("backend scoped provider requires an LLMProvider")
        if not isinstance(out_of_scope_response, str) or not out_of_scope_response.strip():
            raise ValueError("out_of_scope_response must contain text")
        self.provider = provider
        self.policy = policy or BackendDomainPolicy()
        self.guard = guard or BackendOutputGuard(self.policy)
        self.out_of_scope_response = out_of_scope_response

    def generate(self, request: LLMRequest) -> LLMResponse:
        user_text = _last_user_text(request)
        decision = self.policy.decide(user_text)
        if not decision.allowed:
            return LLMResponse(text=self.out_of_scope_response)
        response = self.provider.generate(request)
        guard_decision = self.guard.inspect(response.text)
        if not guard_decision.accepted:
            return LLMResponse(text="I could not produce a reliable backend answer for that request.")
        return response


def _last_user_text(request: LLMRequest) -> str:
    if not isinstance(request, LLMRequest) or not request.messages:
        raise LLMProviderError("Invalid LLM request: at least one message is required")
    user_messages = [message.content for message in request.messages if message.role == "user"]
    if not user_messages or not user_messages[-1].strip():
        raise LLMProviderError("Invalid LLM request: a non-empty user message is required")
    return user_messages[-1]


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9+#.\-]+", text.lower()))


def _contains_term(normalized: str, term: str) -> bool:
    candidate = _normalize(term)
    if not candidate:
        return False
    return candidate in normalized


__all__ = [
    "BackendDomainPolicy",
    "BackendOutputGuard",
    "BackendScopedProvider",
    "DomainDecision",
    "GuardDecision",
    "OUT_OF_SCOPE_RESPONSE",
]
