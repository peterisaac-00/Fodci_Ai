"""A small byte-level tokenizer built from scratch for Fodci."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable


TOKENIZER_FORMAT = "fodci-byte-bpe"
TOKENIZER_VERSION = 1
DEFAULT_VOCAB_SIZE = 10_000

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
BYTE_OFFSET = 4
BASE_VOCAB_SIZE = BYTE_OFFSET + 256


class FodciTokenizer:
    """Deterministic UTF-8 byte tokenizer with optional learned BPE merges.

    Bytes provide a lossless fallback for every Unicode string and source-code
    symbol. Training only adds repeated adjacent byte/token pairs; it never
    removes the byte fallback, so unseen text remains representable.
    """

    def __init__(self, *, vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be at least {BASE_VOCAB_SIZE}.")
        self._vocab_size = vocab_size
        self._merge_rules: list[tuple[int, int, int]] = []
        self._token_bytes: dict[int, bytes] = {
            BYTE_OFFSET + value: bytes((value,)) for value in range(256)
        }

    @property
    def vocab_size(self) -> int:
        """Return the fixed ID range available to this tokenizer."""

        return self._vocab_size

    @property
    def special_tokens(self) -> dict[str, int]:
        """Return the stable special-token mapping."""

        return {
            "<PAD>": PAD_ID,
            "<UNK>": UNK_ID,
            "<BOS>": BOS_ID,
            "<EOS>": EOS_ID,
        }

    @property
    def merges(self) -> tuple[tuple[int, int, int], ...]:
        """Return learned merge rules in deterministic application order."""

        return tuple(self._merge_rules)

    def train(self, corpus: Iterable[str], *, max_merges: int | None = None) -> "FodciTokenizer":
        """Learn repeated adjacent pairs from a small caller-provided corpus.

        This is tokenizer training only. It accepts an iterable of text strings,
        does not download data, and does not train the language model.
        """

        if isinstance(corpus, str):
            corpus = (corpus,)
        texts = tuple(corpus)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("corpus must contain only strings.")
        if max_merges is not None and max_merges < 0:
            raise ValueError("max_merges must be non-negative.")

        self._merge_rules.clear()
        self._token_bytes = {
            BYTE_OFFSET + value: bytes((value,)) for value in range(256)
        }
        sequences = [self._base_tokens(text) for text in texts]
        merge_limit = self._vocab_size - BASE_VOCAB_SIZE
        if max_merges is not None:
            merge_limit = min(merge_limit, max_merges)

        while len(self._merge_rules) < merge_limit:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, frequency = max(
                counts.items(),
                key=lambda item: (item[1], -item[0][0], -item[0][1]),
            )
            if frequency < 2:
                break

            token_id = BASE_VOCAB_SIZE + len(self._merge_rules)
            self._merge_rules.append((pair[0], pair[1], token_id))
            self._token_bytes[token_id] = self._token_bytes[pair[0]] + self._token_bytes[pair[1]]
            sequences = [self._merge_sequence(sequence, pair, token_id) for sequence in sequences]

        return self

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to IDs without normalization or truncation."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        tokens = self._base_tokens(text)
        for left, right, token_id in self._merge_rules:
            tokens = self._merge_sequence(tokens, (left, right), token_id)

        if add_bos:
            tokens.insert(0, BOS_ID)
        if add_eos:
            tokens.append(EOS_ID)
        self._validate_ids(tokens)
        return tokens

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        """Decode IDs to text, preserving valid UTF-8 input exactly."""

        output: list[str] = []
        byte_buffer = bytearray()
        special_names = {value: name for name, value in self.special_tokens.items()}

        def flush_bytes() -> None:
            if byte_buffer:
                output.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("token_ids must contain integers.")
            if token_id < 0 or token_id >= self._vocab_size:
                raise ValueError(f"token ID {token_id} is outside the vocabulary.")
            if token_id in special_names:
                if skip_special_tokens:
                    continue
                flush_bytes()
                output.append(special_names[token_id])
                continue
            token_bytes = self._token_bytes.get(token_id)
            if token_bytes is None:
                raise ValueError(f"token ID {token_id} is not defined by this tokenizer.")
            byte_buffer.extend(token_bytes)
        flush_bytes()
        return "".join(output)

    def save(self, path: str | Path) -> None:
        """Save a small versioned tokenizer definition as JSON."""

        target = Path(path)
        payload = {
            "format": TOKENIZER_FORMAT,
            "version": TOKENIZER_VERSION,
            "vocab_size": self._vocab_size,
            "special_tokens": self.special_tokens,
            "merges": [
                {"left": left, "right": right, "token_id": token_id}
                for left, right, token_id in self._merge_rules
            ],
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FodciTokenizer":
        """Load a saved tokenizer without retraining it."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != TOKENIZER_FORMAT:
            raise ValueError("Unsupported tokenizer format.")
        if payload.get("version") != TOKENIZER_VERSION:
            raise ValueError("Unsupported tokenizer version.")
        tokenizer = cls(vocab_size=int(payload["vocab_size"]))
        expected_special_tokens = tokenizer.special_tokens
        if payload.get("special_tokens") != expected_special_tokens:
            raise ValueError("Tokenizer special-token mapping is incompatible.")
        for merge in payload.get("merges", []):
            tokenizer._append_merge(
                int(merge["left"]),
                int(merge["right"]),
                int(merge["token_id"]),
            )
        return tokenizer

    def _append_merge(self, left: int, right: int, token_id: int) -> None:
        if token_id != BASE_VOCAB_SIZE + len(self._merge_rules):
            raise ValueError("Tokenizer merge IDs must be contiguous and deterministic.")
        if left not in self._token_bytes or right not in self._token_bytes:
            raise ValueError("Tokenizer merge references an undefined token.")
        if token_id >= self._vocab_size:
            raise ValueError("Tokenizer merge exceeds vocab_size.")
        self._merge_rules.append((left, right, token_id))
        self._token_bytes[token_id] = self._token_bytes[left] + self._token_bytes[right]

    @staticmethod
    def _base_tokens(text: str) -> list[int]:
        return [BYTE_OFFSET + value for value in text.encode("utf-8")]

    @staticmethod
    def _merge_sequence(
        sequence: list[int],
        pair: tuple[int, int],
        token_id: int,
    ) -> list[int]:
        merged: list[int] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                merged.append(token_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged

    def _validate_ids(self, token_ids: Iterable[int]) -> None:
        for token_id in token_ids:
            if token_id < 0 or token_id >= self._vocab_size:
                raise ValueError(f"Generated token ID {token_id} is outside the vocabulary.")
