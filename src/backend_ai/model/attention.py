"""Readable multi-head causal self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn

from backend_ai.model.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Self-attention whose position i can attend only to positions <= i."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dimension = config.head_dimension
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute causal attention and optionally return raw attention weights."""

        batch_size, sequence_length, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dimension)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dimension)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = self.attention_dropout(weights) @ value
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            hidden_size,
        )
        output = self.output_dropout(self.output(attended))
        if return_attention:
            return output, weights
        return output
