"""Token and learned positional embeddings for the decoder-only model."""

from __future__ import annotations

import torch
from torch import nn

from backend_ai.model.config import ModelConfig


class TokenPositionEmbedding(nn.Module):
    """Add token embeddings to learned position embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position = nn.Embedding(config.context_length, config.hidden_size)
        self.context_length = config.context_length

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return embeddings for integer IDs shaped ``(batch, sequence)``."""

        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )
        position_embeddings = self.position(positions).unsqueeze(0).expand(batch_size, -1, -1)
        return self.token(input_ids) + position_embeddings
