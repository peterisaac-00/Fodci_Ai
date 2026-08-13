from __future__ import annotations

from pathlib import Path

import pytest

from backend_ai.model.config import ModelConfig
from backend_ai.tokenizer import (
    BASE_VOCAB_SIZE,
    BOS_ID,
    EOS_ID,
    FodciTokenizer,
    PAD_ID,
    TOKENIZER_FORMAT,
    TOKENIZER_VERSION,
    UNK_ID,
)


@pytest.fixture
def tokenizer() -> FodciTokenizer:
    return FodciTokenizer()


def test_default_vocabulary_and_special_tokens_match_model_contract(
    tokenizer: FodciTokenizer,
) -> None:
    assert tokenizer.vocab_size == ModelConfig().vocab_size == 10_000
    assert tokenizer.special_tokens == {
        "<PAD>": PAD_ID,
        "<UNK>": UNK_ID,
        "<BOS>": BOS_ID,
        "<EOS>": EOS_ID,
    }
    assert BASE_VOCAB_SIZE == 260


def test_roundtrip_preserves_backend_text_and_source_code_exactly(
    tokenizer: FodciTokenizer,
) -> None:
    examples = (
        "Hello world",
        "REST API authentication middleware",
        'def add(a, b):\n    return a + b',
        "SELECT id FROM users WHERE id = 10;",
        '{"name": "Peter", "age": 19}',
        "https://example.com/api/users",
        "src/backend_ai/model/model.py",
        "\tindent\n    spaces\r\nArabic: مرحبًا — Fodci ∑",
        "",
    )

    for text in examples:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_fallback_handles_unseen_unicode_without_unknown_loss(
    tokenizer: FodciTokenizer,
) -> None:
    text = "🧪 \u0000 unusual symbols: § λ \ufeff"

    token_ids = tokenizer.encode(text)

    assert max(token_ids, default=0) < tokenizer.vocab_size
    assert tokenizer.decode(token_ids) == text


def test_encoding_is_deterministic_and_does_not_normalize_input(
    tokenizer: FodciTokenizer,
) -> None:
    text = "  def f():\n\treturn 1  "

    assert tokenizer.encode(text) == tokenizer.encode(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_explicit_bos_and_eos_are_stable_and_optional(
    tokenizer: FodciTokenizer,
) -> None:
    token_ids = tokenizer.encode("hello", add_bos=True, add_eos=True)

    assert token_ids[0] == BOS_ID
    assert token_ids[-1] == EOS_ID
    assert tokenizer.decode(token_ids) == "hello"
    assert tokenizer.decode(token_ids, skip_special_tokens=False).startswith("<BOS>")


def test_training_adds_deterministic_bpe_merges_without_removing_byte_fallback() -> None:
    corpus = (
        "def get_user(user_id):\n    return db.query(User)",
        "def get_user(user_id):\n    return db.query(User)",
    )
    first = FodciTokenizer().train(corpus, max_merges=12)
    second = FodciTokenizer().train(corpus, max_merges=12)

    assert first.merges == second.merges
    assert first.encode(corpus[0]) == second.encode(corpus[0])
    assert len(first.encode(corpus[0])) < len(FodciTokenizer().encode(corpus[0]))
    assert max(first.encode("unseen λ text")) < first.vocab_size
    assert first.decode(first.encode("unseen λ text")) == "unseen λ text"


def test_save_and_load_preserve_behavior_and_versioned_format(
    tmp_path: Path,
) -> None:
    original = FodciTokenizer().train(("SELECT id FROM users;",) * 2, max_merges=8)
    artifact = tmp_path / "tokenizer.json"

    original.save(artifact)
    loaded = FodciTokenizer.load(artifact)

    assert artifact.read_text(encoding="utf-8").startswith("{\n")
    assert TOKENIZER_FORMAT in artifact.read_text(encoding="utf-8")
    assert str(TOKENIZER_VERSION) in artifact.read_text(encoding="utf-8")
    assert loaded.vocab_size == original.vocab_size
    assert loaded.merges == original.merges
    assert loaded.encode("SELECT id FROM users;") == original.encode("SELECT id FROM users;")
    assert loaded.decode(loaded.encode("SELECT id FROM users;")) == "SELECT id FROM users;"


def test_invalid_vocab_and_ids_are_rejected(tokenizer: FodciTokenizer) -> None:
    with pytest.raises(ValueError, match="at least 260"):
        FodciTokenizer(vocab_size=BASE_VOCAB_SIZE - 1)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tokenizer.decode([tokenizer.vocab_size])
    with pytest.raises(TypeError, match="string"):
        tokenizer.encode(123)  # type: ignore[arg-type]
