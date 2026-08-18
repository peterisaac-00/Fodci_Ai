"""Experimental local provider for a Hugging Face causal code model.

The provider is optional and lazy: importing the base project does not import
Transformers or download a model. Phase 14.3 only establishes the provider
boundary; Phase 14.4 supplies and evaluates a concrete local model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend_ai.core.contracts import LLMProviderError, LLMRequest, LLMResponse, Message


DEFAULT_PRETRAINED_SYSTEM_PROMPT = (
    "You are an experimental backend engineering assistant. "
    "Answer in clear English, stay within Python backend, FastAPI, REST, HTTP, "
    "SQL, authentication, testing, debugging, and backend architecture. "
    "If a request is outside backend engineering, say that it is outside scope. "
    "Do not claim to have run code or inspected files unless tools provide that evidence."
)


@dataclass(frozen=True, slots=True)
class PretrainedProviderConfig:
    """Bounded local generation settings for the experimental provider."""

    model_id: str
    device: str = "cpu"
    max_new_tokens: int = 128
    temperature: float = 0.2
    do_sample: bool = False
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must contain text")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        if not 1 <= self.max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be greater than zero and at most two")
        if not isinstance(self.do_sample, bool) or not isinstance(self.trust_remote_code, bool):
            raise ValueError("boolean provider settings are invalid")


class PretrainedCodeProvider:
    """Adapt a lazily loaded Transformers causal model to LLMProvider.

    Construction accepts injected tokenizer/model objects for deterministic unit
    tests. The normal `from_pretrained` constructor imports the optional runtime
    only when explicitly requested by a caller.
    """

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        config: PretrainedProviderConfig,
        system_prompt: str = DEFAULT_PRETRAINED_SYSTEM_PROMPT,
    ) -> None:
        if not callable(getattr(tokenizer, "__call__", None)):
            raise LLMProviderError("pretrained provider requires a callable tokenizer")
        if not callable(getattr(model, "generate", None)):
            raise LLMProviderError("pretrained provider requires a model with generate")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must contain text")
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.system_prompt = system_prompt

    @classmethod
    def from_pretrained(
        cls,
        config: PretrainedProviderConfig,
        *,
        system_prompt: str = DEFAULT_PRETRAINED_SYSTEM_PROMPT,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> "PretrainedCodeProvider":
        """Load an explicitly selected local/cache model; never auto-download."""

        if model is None or tokenizer is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise LLMProviderError(
                    "The optional pretrained provider requires the 'transformers' package; "
                    "the default Fodci runtime does not."
                ) from exc
            try:
                tokenizer = tokenizer or AutoTokenizer.from_pretrained(
                    config.model_id,
                    trust_remote_code=config.trust_remote_code,
                    local_files_only=True,
                )
                model = model or AutoModelForCausalLM.from_pretrained(
                    config.model_id,
                    trust_remote_code=config.trust_remote_code,
                    local_files_only=True,
                )
            except Exception as exc:
                raise LLMProviderError(
                    f"Unable to load pretrained model from local/cache path {config.model_id!r}: {exc}"
                ) from exc
        if config.device != "auto" and callable(getattr(model, "to", None)):
            try:
                model = model.to(config.device)
            except Exception as exc:
                raise LLMProviderError(f"Unable to move pretrained model to {config.device}: {exc}") from exc
        if callable(getattr(model, "eval", None)):
            model.eval()
        return cls(tokenizer, model, config=config, system_prompt=system_prompt)

    def generate(self, request: LLMRequest) -> LLMResponse:
        prompt = self._format_request(request)
        try:
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else getattr(encoded, "input_ids")
            attention_mask = encoded.get("attention_mask") if isinstance(encoded, dict) else getattr(encoded, "attention_mask", None)
            kwargs: dict[str, Any] = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.do_sample,
                "temperature": self.config.temperature,
                "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
            }
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            generated = self.model.generate(input_ids, **kwargs)
            sequence = generated[0] if hasattr(generated, "__getitem__") else generated
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]
            return LLMResponse(text=text.strip())
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Pretrained local inference failed: {exc}") from exc

    def _format_request(self, request: LLMRequest) -> str:
        if not isinstance(request, LLMRequest) or not request.messages:
            raise LLMProviderError("Invalid LLM request: at least one message is required")
        system_messages = [message for message in request.messages if message.role == "system"]
        if len(system_messages) > 1:
            raise LLMProviderError("Invalid LLM request: only one system message is supported")
        system_prompt = system_messages[0].content if system_messages else self.system_prompt
        conversation = [message for message in request.messages if message.role != "system"]
        if not conversation or conversation[-1].role != "user":
            raise LLMProviderError("Invalid LLM request: final message must be a user message")
        if any(not isinstance(message.content, str) or not message.content.strip() for message in conversation):
            raise LLMProviderError("Invalid LLM request: conversation messages must contain text")
        turns = "\n\n".join(f"{_role_label(message)}:\n{message.content}" for message in conversation)
        return f"System:\n{system_prompt}\n\n{turns}\n\nAssistant:\n"


def _role_label(message: Message) -> str:
    return "User" if message.role == "user" else "Assistant"


__all__ = [
    "DEFAULT_PRETRAINED_SYSTEM_PROMPT",
    "PretrainedCodeProvider",
    "PretrainedProviderConfig",
]
