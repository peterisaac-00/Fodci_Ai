"""Composable local dataset pipeline for Phase 2.4."""

from __future__ import annotations

from collections.abc import Iterator

from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.deduplicate import ExactDeduplicator
from backend_ai.dataset.documents import DocumentLoadResult, LoadIssue
from backend_ai.dataset.loader import LocalDocumentLoader
from backend_ai.dataset.samples import TokenSequenceBuilder, TrainingExample
from backend_ai.tokenizer import FodciTokenizer


class FodciDatasetPipeline:
    """Load, validate, deduplicate, tokenize, and chunk local documents."""

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: FodciTokenizer,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.loader = LocalDocumentLoader(config)
        self.deduplicator = ExactDeduplicator()
        self._last_load_result: DocumentLoadResult | None = None
        self._last_issues: tuple[LoadIssue, ...] = ()

    @property
    def last_issues(self) -> tuple[LoadIssue, ...]:
        """Return issues recorded by the most recent dataset iteration."""

        return self._last_issues

    def iter_samples(self) -> Iterator[TrainingExample]:
        """Stream samples in deterministic file and document order."""

        duplicate_issues: list[LoadIssue] = []
        unique_documents = self.deduplicator.iter_unique(
            self.loader.iter_documents(),
            duplicate_issues,
        )
        builder = TokenSequenceBuilder(self.tokenizer, self.config)
        yield from builder.iter_samples(unique_documents)
        self._last_issues = self.loader.issues + tuple(duplicate_issues)

    def load_documents(self) -> DocumentLoadResult:
        """Materialize and deduplicate documents for inspection or diagnostics."""

        loaded = self.loader.load()
        documents, duplicate_issues = self.deduplicator.apply(loaded.documents)
        result = DocumentLoadResult(
            documents=documents,
            issues=loaded.issues + duplicate_issues,
        )
        self._last_load_result = result
        self._last_issues = result.issues
        return result
