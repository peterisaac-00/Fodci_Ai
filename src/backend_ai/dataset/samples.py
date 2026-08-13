"""Streaming tokenization and next-token sample generation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.documents import Document
from backend_ai.tokenizer import EOS_ID, FodciTokenizer


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One autoregressive training sample without model-training behavior."""

    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    document_id: str
    loss_mask: tuple[bool, ...] | None = None


class TokenSequenceBuilder:
    """Tokenize documents and yield contiguous fixed-length next-token samples."""

    def __init__(self, tokenizer: FodciTokenizer, config: DatasetConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

    def iter_samples(self, documents: Iterable[Document]) -> Iterator[TrainingExample]:
        """Yield deterministic samples while keeping only one document stream in memory."""

        for document in documents:
            token_ids = self.tokenizer.encode(document.text)
            if self.config.use_eos_document_boundaries:
                token_ids.append(EOS_ID)
            yield from self._chunk(token_ids, document.document_id)

    def _chunk(self, token_ids: list[int], document_id: str) -> Iterator[TrainingExample]:
        width = self.config.context_length
        if len(token_ids) <= 1:
            return
        for start in range(0, len(token_ids) - 1, width):
            window = token_ids[start : start + width + 1]
            if len(window) < 2:
                continue
            if len(window) != width + 1:
                continue
            yield TrainingExample(
                input_ids=tuple(window[:-1]),
                target_ids=tuple(window[1:]),
                document_id=document_id,
            )
