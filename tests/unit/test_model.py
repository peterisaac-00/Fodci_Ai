from __future__ import annotations

import torch
import pytest

from backend_ai.model import CausalSelfAttention, FodciModel, ModelConfig


def _small_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "context_length": 8,
        "hidden_size": 16,
        "num_layers": 2,
        "num_attention_heads": 4,
        "feed_forward_size": 32,
        "dropout": 0.0,
        "seed": 7,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_default_model_is_configurable_and_in_requested_small_parameter_range() -> None:
    model = FodciModel(ModelConfig(seed=11))

    assert 5_000_000 <= model.num_parameters <= 15_000_000
    assert model.num_parameters < 20_000_000
    assert len(model.blocks) == 4
    assert model.config.hidden_size % model.config.num_attention_heads == 0


def test_forward_returns_batch_sequence_vocabulary_logits() -> None:
    config = _small_config()
    model = FodciModel(config).eval()
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)

    logits = model(input_ids)

    assert logits.shape == (2, 3, config.vocab_size)
    assert logits.dtype == torch.float32


def test_seeded_initialization_is_reproducible() -> None:
    first = FodciModel(_small_config(seed=123))
    second = FodciModel(_small_config(seed=123))
    third = FodciModel(_small_config(seed=124))

    assert torch.equal(first.embeddings.token.weight, second.embeddings.token.weight)
    assert not torch.equal(first.embeddings.token.weight, third.embeddings.token.weight)


def test_causal_attention_masks_all_future_positions() -> None:
    config = _small_config(context_length=5)
    attention = CausalSelfAttention(config).eval()
    hidden_states = torch.randn(1, 5, config.hidden_size)

    _, weights = attention(hidden_states, return_attention=True)

    assert weights.shape == (1, config.num_attention_heads, 5, 5)
    assert torch.count_nonzero(weights.triu(diagonal=1)) == 0


def test_future_tokens_do_not_change_earlier_attention_outputs() -> None:
    config = _small_config(context_length=5)
    attention = CausalSelfAttention(config).eval()
    prefix = torch.randn(1, 2, config.hidden_size)
    first_future = torch.randn(1, 3, config.hidden_size)
    second_future = torch.randn(1, 3, config.hidden_size)

    first_output = attention(torch.cat((prefix, first_future), dim=1))
    second_output = attention(torch.cat((prefix, second_future), dim=1))

    assert torch.allclose(first_output[:, :2], second_output[:, :2])


def test_model_rejects_invalid_token_inputs() -> None:
    model = FodciModel(_small_config())

    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match="outside the vocabulary"):
        model(torch.tensor([[32]], dtype=torch.long))
    with pytest.raises(ValueError, match="context_length"):
        model(torch.zeros(1, 9, dtype=torch.long))
    with pytest.raises(ValueError, match="integer"):
        model(torch.zeros(1, 2, dtype=torch.float32))


def test_model_config_rejects_inconsistent_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(hidden_size=10, num_attention_heads=3)
