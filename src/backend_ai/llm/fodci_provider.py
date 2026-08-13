"""Concrete local Fodci provider for the Phase 2 terminal integration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

from backend_ai.core.contracts import (
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    Message,
)

if TYPE_CHECKING:
    from backend_ai.inference import InferenceConfig, InferenceEngine


DEFAULT_FODCI_SYSTEM_PROMPT = (
    "Fodci: local backend-engineering model. Be concise and honest. "
    "Do not claim tools, files, or commands."
)


class FodciLocalProvider:
    """Adapt typed LLM requests to the existing local inference engine."""

    def __init__(
        self,
        engine: Any,
        *,
        system_prompt: str = DEFAULT_FODCI_SYSTEM_PROMPT,
    ) -> None:
        if not callable(getattr(engine, "generate", None)):
            raise LLMProviderError("Fodci local provider requires an inference engine.")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must contain non-whitespace text.")
        self.engine = engine
        self.system_prompt = system_prompt

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        system_prompt: str = DEFAULT_FODCI_SYSTEM_PROMPT,
        inference_config: InferenceConfig | None = None,
    ) -> "FodciLocalProvider":
        """Construct one CPU provider from one existing local checkpoint."""

        path = Path(checkpoint_path)
        if not path.is_file():
            raise LLMProviderError(
                f"Fodci checkpoint is unavailable: {path}. "
                "The existing local checkpoint is required; no fallback model will be created."
            )

        try:
            from backend_ai.inference import InferenceConfig, InferenceEngine
            from backend_ai.model import FodciModel
            from backend_ai.tokenizer import FodciTokenizer

            config = inference_config or InferenceConfig(
                device="cpu",
                checkpoint_path=path,
            )
            if config.device != "cpu":
                config = replace(config, device="cpu")
            if config.checkpoint_path != path:
                config = replace(config, checkpoint_path=path)
            engine = InferenceEngine(FodciModel(), FodciTokenizer(), config)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"Unable to load the local Fodci checkpoint: {path}: {exc}"
            ) from exc
        return cls(engine, system_prompt=system_prompt)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response without external APIs or model reloading."""

        try:
            prompt = self._format_request(request)
            result = self.engine.generate(prompt)
            text = getattr(result, "generated_text", None)
            if not isinstance(text, str):
                raise TypeError("inference result does not contain generated text")
            return LLMResponse(text=text)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Local Fodci inference failed: {exc}") from exc

    def _format_request(self, request: LLMRequest) -> str:
        if not isinstance(request, LLMRequest) or not request.messages:
            raise LLMProviderError("Invalid LLM request: at least one message is required.")

        system_messages = [message for message in request.messages if message.role == "system"]
        if len(system_messages) > 1:
            raise LLMProviderError("Invalid LLM request: only one system message is supported.")
        system_prompt = system_messages[0].content if system_messages else self.system_prompt
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise LLMProviderError("Invalid LLM request: system message must not be empty.")

        conversation = [message for message in request.messages if message.role != "system"]
        if not conversation or conversation[-1].role != "user":
            raise LLMProviderError("Invalid LLM request: the final message must be a user message.")
        for message in conversation:
            if not isinstance(message.content, str):
                raise LLMProviderError("Invalid LLM request: message content must be text.")
            if message.role == "user" and not message.content.strip():
                raise LLMProviderError("Invalid LLM request: user message content must not be empty.")

        turns = "\n\n".join(
            f"{_role_label(message)}:\n{message.content}" for message in conversation
        )
        return (
            "### Instruction\n"
            f"{system_prompt}\n\n"
            "### Input\n"
            f"{turns}\n\n"
            "### Response\n"
        )


def _role_label(message: Message) -> str:
    return "User" if message.role == "user" else "Fodci"


__all__ = ["DEFAULT_FODCI_SYSTEM_PROMPT", "FodciLocalProvider"]
