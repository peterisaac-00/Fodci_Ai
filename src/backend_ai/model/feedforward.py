"""Feed-forward sublayer for a Transformer block."""

from __future__ import annotations

import torch
from torch import nn

from backend_ai.model.config import ModelConfig


class FeedForward(nn.Module):
    """Linear → GELU → Linear feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.hidden_size, config.feed_forward_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feed_forward_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform hidden states without changing their shape."""

        return self.network(hidden_states)
