from __future__ import annotations

import pytest

from backend_ai.english_only import (
    ENGLISH_ONLY_POLICY_VERSION,
    EnglishOnlyViolation,
    contains_arabic,
    validate_english_payload,
    validate_english_text,
)
from backend_ai.agent.training_dataset import TrainingDatasetError, TrainingExample


def test_policy_accepts_english_backend_training_payload() -> None:
    validate_english_payload(
        {
            "instruction": "Explain how to validate a JWT in FastAPI.",
            "response": "Validate the signature, issuer, audience, and expiration before using claims.",
            "metadata": {"language": "en", "domain": "backend"},
        }
    )
    assert ENGLISH_ONLY_POLICY_VERSION == "fodci-english-only-v1"
    assert contains_arabic("FastAPI and PostgreSQL") is False


def test_policy_rejects_arabic_text_in_nested_training_payload() -> None:
    with pytest.raises(EnglishOnlyViolation, match="Arabic-script"):
        validate_english_payload(
            {"instruction": "اشرح JWT", "response": "Explain JWT safely."},
            field="record",
        )


def test_policy_does_not_reject_non_language_unicode() -> None:
    validate_english_text("Use UTF-8: café — こんにちは", field="text")
    assert contains_arabic("123 + 456") is False


def test_policy_rejects_non_text_values_only_when_nested_as_unsupported() -> None:
    with pytest.raises(EnglishOnlyViolation, match="unsupported value type"):
        validate_english_payload({"bad": object()})


def test_training_example_rejects_arabic_even_when_constructed_directly() -> None:
    with pytest.raises(TrainingDatasetError, match="Arabic-script"):
        TrainingExample(
            "fodci.training_example",
            "11.2",
            "tex-example",
            "record-example",
            "record-example",
            "اشرح JWT",
            {},
            "Task:\nاشرح JWT",
            None,
            "Explain JWT safely.",
            {},
        )
