"""Decoder-only Transformer block composition."""

from __future__ import annotations

import torch
from torch import nn

from backend_ai.model.attention import CausalSelfAttention
from backend_ai.model.config import ModelConfig
from backend_ai.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Pre-normalized attention and feed-forward residual block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply attention and feed-forward residual transformations."""

        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        hidden_states = hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))
        return hidden_states
