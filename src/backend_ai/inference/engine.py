"""Local autoregressive inference for the from-scratch Fodci model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn

from backend_ai.checkpoint import CheckpointManager
from backend_ai.inference.config import InferenceConfig
from backend_ai.tokenizer import EOS_ID, FodciTokenizer


class InferenceError(RuntimeError):
    """Base error for local inference failures."""


class PromptValidationError(ValueError):
    """Raised when a prompt cannot fit the model context."""


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Generated text plus testable public inference metadata."""

    generated_text: str
    prompt_token_count: int
    generated_token_count: int
    stopped_reason: str
    model_version: str
    checkpoint_identity: str
    configuration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InferenceEngine:
    """Run local, no-gradient autoregressive decoding on an existing Fodci model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: FodciTokenizer,
        config: InferenceConfig | None = None,
    ) -> None:
        self.config = config or InferenceConfig()
        self.device = _resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self._checkpoint_manager = CheckpointManager(
            Path(self.config.checkpoint_path).parent if self.config.checkpoint_path else Path("."),
            model_version=self.config.model_version,
            tokenizer_version=self.config.tokenizer_version,
        )
        self._checkpoint_identity = "random-initialization"
        self._validate_contract()
        if self.config.checkpoint_path is not None:
            self.load_checkpoint(self.config.checkpoint_path)

    @property
    def checkpoint_identity(self) -> str:
        """Return the loaded checkpoint path or random-initialization marker."""

        return self._checkpoint_identity

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Load one compatible checkpoint without downloading or creating weights."""

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Inference checkpoint does not exist: {path}")
        self._checkpoint_manager = CheckpointManager(
            path.parent,
            model_version=self.config.model_version,
            tokenizer_version=self.config.tokenizer_version,
        )
        # CheckpointManager requires an optimizer to restore its complete payload.
        # The temporary optimizer is not exposed and is never stepped during inference.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0)
        self._checkpoint_manager.load(path, self.model, optimizer, device=self.device)
        self._checkpoint_identity = str(path)

    def generate(self, prompt: str) -> InferenceResult:
        """Generate tokens one at a time from a validated prompt."""

        prompt_ids = self._encode_prompt(prompt)
        current_ids = list(prompt_ids)
        generated_ids: list[int] = []
        generator = torch.Generator(device=self.device.type)
        generator.manual_seed(self.config.seed)
        stopped_reason = "max_new_tokens"
        self.model.eval()

        with torch.inference_mode():
            for _ in range(self.config.max_new_tokens):
                if len(current_ids) >= self._context_length:
                    stopped_reason = "context_length"
                    break
                input_tensor = torch.tensor(
                    [current_ids],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = self.model(input_tensor)[:, -1, :].squeeze(0)
                next_token = self._select_next_token(logits, generator)
                generated_ids.append(next_token)
                current_ids.append(next_token)
                if self.config.stop_on_eos and next_token == EOS_ID:
                    stopped_reason = "eos"
                    break
            else:
                stopped_reason = "max_new_tokens"

        return InferenceResult(
            generated_text=self.tokenizer.decode(generated_ids),
            prompt_token_count=len(prompt_ids),
            generated_token_count=len(generated_ids),
            stopped_reason=stopped_reason,
            model_version=self.config.model_version,
            checkpoint_identity=self._checkpoint_identity,
            configuration=self.config.to_dict(),
        )

    def _encode_prompt(self, prompt: str) -> list[int]:
        if not isinstance(prompt, str):
            raise PromptValidationError("prompt must be a string.")
        if not prompt.strip():
            raise PromptValidationError("prompt must not be empty or whitespace-only.")
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise PromptValidationError("prompt produced no tokens.")
        if len(prompt_ids) > self._context_length:
            raise PromptValidationError(
                f"prompt has {len(prompt_ids)} tokens but context length is {self._context_length}; "
                "the prompt will not be truncated."
            )
        return prompt_ids

    def _select_next_token(self, logits: torch.Tensor, generator: torch.Generator) -> int:
        if logits.ndim != 1 or logits.shape[0] != self._vocab_size:
            raise InferenceError("Model final-position logits have an incompatible shape.")
        scaled = logits.float() / float(self.config.temperature)
        if self.config.top_k is not None:
            values, indices = torch.topk(scaled, self.config.top_k)
            filtered = torch.full_like(scaled, float("-inf"))
            filtered.scatter_(0, indices, values)
            scaled = filtered
        if not self.config.do_sample:
            return int(torch.argmax(scaled).item())
        probabilities = torch.softmax(scaled, dim=-1)
        if not bool(torch.isfinite(probabilities).all().item()) or float(probabilities.sum().item()) <= 0:
            raise InferenceError("Unable to construct a finite next-token distribution.")
        return int(torch.multinomial(probabilities, 1, generator=generator).item())

    def _validate_contract(self) -> None:
        model_config = getattr(self.model, "config", None)
        if model_config is None or not hasattr(model_config, "context_length") or not hasattr(model_config, "vocab_size"):
            raise TypeError("InferenceEngine requires a model with context_length and vocab_size.")
        self._context_length = int(model_config.context_length)
        self._vocab_size = int(model_config.vocab_size)
        if self.tokenizer.vocab_size != self._vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary size {self.tokenizer.vocab_size} does not match model vocabulary size {self._vocab_size}."
            )
        if self.config.top_k is not None and self.config.top_k > self._vocab_size:
            raise ValueError(
                f"top_k {self.config.top_k} cannot exceed vocabulary size {self._vocab_size}."
            )
        if self._context_length < 1:
            raise ValueError("model context_length must be positive.")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for inference.")
    return torch.device(value)
