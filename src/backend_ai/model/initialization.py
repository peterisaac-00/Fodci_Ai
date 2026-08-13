"""Local random initialization for the Fodci Transformer."""

from __future__ import annotations

import torch
from torch import nn

from backend_ai.model.config import ModelConfig


def initialize_module(module: nn.Module, config: ModelConfig) -> None:
    """Initialize trainable modules with a small normal distribution locally."""

    if config.seed is not None:
        torch.manual_seed(config.seed)

    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.Embedding)):
            nn.init.normal_(child.weight, mean=0.0, std=config.initialization_std)
            if isinstance(child, nn.Linear) and child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)
