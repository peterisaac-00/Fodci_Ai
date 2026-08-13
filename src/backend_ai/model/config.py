"""Configuration for the first from-scratch Fodci Transformer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Small decoder-only Transformer configuration.

    The default is intentionally extremely lightweight and CPU-testable:
    10,000 vocabulary entries, 320 hidden dimensions, four blocks, five
    attention heads, and a 1,280-unit feed-forward layer. This is approximately
    11.4 million parameters; each head has a 64-dimensional representation.
    """

    vocab_size: int = 10_000
    context_length: int = 256
    hidden_size: int = 320
    num_layers: int = 4
    num_attention_heads: int = 5
    feed_forward_size: int = 1_280
    dropout: float = 0.0
    activation: str = "gelu"
    initialization_std: float = 0.02
    seed: int | None = None

    def __post_init__(self) -> None:
        positive_fields = (
            "vocab_size",
            "context_length",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "feed_forward_size",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.activation != "gelu":
            raise ValueError("Phase 2.2 supports only the GELU activation.")
        if self.initialization_std <= 0.0:
            raise ValueError("initialization_std must be positive.")

    @property
    def head_dimension(self) -> int:
        """Return the hidden width assigned to each attention head."""

        return self.hidden_size // self.num_attention_heads
