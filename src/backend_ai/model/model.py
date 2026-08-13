"""The first from-scratch Fodci decoder-only Transformer."""

from __future__ import annotations

import torch
from torch import nn

from backend_ai.model.config import ModelConfig
from backend_ai.model.embeddings import TokenPositionEmbedding
from backend_ai.model.initialization import initialize_module
from backend_ai.model.transformer import TransformerBlock


class FodciModel(nn.Module):
    """Small configurable decoder-only language-model architecture."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.embeddings = TokenPositionEmbedding(self.config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        initialize_module(self, self.config)

    @property
    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits shaped ``(batch, sequence, vocabulary)``."""

        _validate_input_ids(input_ids, self.config)
        hidden_states = self.embeddings(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)


def _validate_input_ids(input_ids: torch.Tensor, config: ModelConfig) -> None:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch_size, sequence_length).")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must contain integer token IDs.")
    if input_ids.shape[1] > config.context_length:
        raise ValueError("sequence_length cannot exceed context_length.")
    if input_ids.numel() == 0:
        raise ValueError("input_ids cannot be empty.")
    if input_ids.min().item() < 0 or input_ids.max().item() >= config.vocab_size:
        raise ValueError("input_ids contain a token outside the vocabulary.")
