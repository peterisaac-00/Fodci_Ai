from __future__ import annotations

from pathlib import Path

import pytest
import torch

from backend_ai.inference import InferenceConfig, InferenceEngine, PromptValidationError
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import FodciTokenizer


def _small_model(vocab_size: int = 260, context_length: int = 16) -> FodciModel:
    return FodciModel(
        ModelConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            hidden_size=16,
            num_layers=2,
            num_attention_heads=4,
            feed_forward_size=32,
            dropout=0.0,
            seed=7,
        )
    )


def test_greedy_generation_is_deterministic_and_respects_max_tokens() -> None:
    first = InferenceEngine(
        _small_model(),
        FodciTokenizer(vocab_size=260),
        InferenceConfig(max_new_tokens=3, seed=2026),
    ).generate("Hi")
    second = InferenceEngine(
        _small_model(),
        FodciTokenizer(vocab_size=260),
        InferenceConfig(max_new_tokens=3, seed=2026),
    ).generate("Hi")

    assert first.generated_text == second.generated_text
    assert isinstance(first.generated_text, str)
    assert first.generated_token_count <= 3
    assert first.stopped_reason in {"max_new_tokens", "eos", "context_length"}
    assert first.prompt_token_count == 2


def test_sampling_with_temperature_and_top_k_is_deterministic_with_seed() -> None:
    config = InferenceConfig(max_new_tokens=4, temperature=0.7, top_k=5, do_sample=True, seed=99)
    first = InferenceEngine(_small_model(), FodciTokenizer(vocab_size=260), config).generate("API")
    second = InferenceEngine(_small_model(), FodciTokenizer(vocab_size=260), config).generate("API")
    assert first.generated_text == second.generated_text
    assert first.configuration["top_k"] == 5
    assert first.configuration["do_sample"] is True


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        InferenceConfig(temperature=temperature)


def test_invalid_top_k_and_vocab_mismatch_are_rejected() -> None:
    with pytest.raises(ValueError, match="top_k"):
        InferenceConfig(top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        InferenceEngine(
            _small_model(),
            FodciTokenizer(vocab_size=260),
            InferenceConfig(top_k=261),
        )
    with pytest.raises(ValueError, match="vocabulary size"):
        InferenceEngine(
            _small_model(),
            FodciTokenizer(vocab_size=300),
            InferenceConfig(),
        )


def test_prompt_validation_rejects_empty_unicode_safe_and_overlong_prompts() -> None:
    engine = InferenceEngine(
        _small_model(context_length=16),
        FodciTokenizer(vocab_size=260),
        InferenceConfig(max_new_tokens=1),
    )
    with pytest.raises(PromptValidationError, match="empty"):
        engine.generate("   ")
    with pytest.raises(PromptValidationError, match="context length"):
        engine.generate("x" * 17)

    arabic_result = engine.generate("مرحبا")
    assert arabic_result.prompt_token_count == len("مرحبا".encode("utf-8"))


def test_eos_stopping_can_be_forced_with_a_deterministic_fake_model() -> None:
    class EosModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = ModelConfig(
                vocab_size=260,
                context_length=8,
                hidden_size=16,
                num_layers=2,
                num_attention_heads=4,
                feed_forward_size=32,
                dropout=0.0,
                seed=1,
            )
            self.marker = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            logits = torch.full(
                (input_ids.shape[0], input_ids.shape[1], 260),
                -100.0,
                device=input_ids.device,
            )
            logits[:, -1, 3] = 100.0
            return logits

    result = InferenceEngine(
        EosModel(),
        FodciTokenizer(vocab_size=260),
        InferenceConfig(max_new_tokens=8, stop_on_eos=True),
    ).generate("x")
    assert result.generated_token_count == 1
    assert result.generated_text == ""
    assert result.stopped_reason == "eos"


def test_inference_does_not_create_gradients_or_change_parameters() -> None:
    model = _small_model()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    engine = InferenceEngine(model, FodciTokenizer(vocab_size=260), InferenceConfig(max_new_tokens=2))

    result = engine.generate("SQL")

    assert result.generated_token_count <= 2
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters())


def test_real_fodci_checkpoint_inference_on_cpu() -> None:
    checkpoint = Path("artifacts/checkpoints/fodci-tiny-v1.pt")
    if not checkpoint.is_file():
        pytest.skip("existing local Fodci Tiny v1 checkpoint is unavailable")
    engine = InferenceEngine(
        FodciModel(),
        FodciTokenizer(),
        InferenceConfig(max_new_tokens=2, device="cpu", checkpoint_path=checkpoint),
    )

    result = engine.generate("Hi")

    assert result.checkpoint_identity == str(checkpoint)
    assert result.model_version == "fodci-tiny-v1"
    assert result.prompt_token_count == 2
    assert result.generated_token_count <= 2
    assert isinstance(result.generated_text, str)


def test_checkpoint_inference_does_not_initialize_an_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = Path("artifacts/checkpoints/fodci-tiny-v1.pt")
    if not checkpoint.is_file():
        pytest.skip("existing local Fodci Tiny v1 checkpoint is unavailable")

    def fail_optimizer(*args: object, **kwargs: object) -> None:
        raise AssertionError("inference must not initialize an optimizer")

    monkeypatch.setattr(torch.optim, "AdamW", fail_optimizer)
    engine = InferenceEngine(
        FodciModel(),
        FodciTokenizer(),
        InferenceConfig(max_new_tokens=1, device="cpu", checkpoint_path=checkpoint),
    )

    assert engine.checkpoint_identity == str(checkpoint)
