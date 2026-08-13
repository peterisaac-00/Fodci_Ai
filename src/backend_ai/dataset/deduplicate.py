"""Exact deterministic document deduplication."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from backend_ai.dataset.documents import Document, LoadIssue


class ExactDeduplicator:
    """Keep the first document for each exact content hash."""

    def iter_unique(
        self,
        documents: Iterable[Document],
        issues: list[LoadIssue] | None = None,
    ) -> Iterator[Document]:
        """Yield the first document for each content hash, lazily."""

        issue_sink = issues if issues is not None else []
        seen_hashes: set[str] = set()
        for document in documents:
            if document.content_hash in seen_hashes:
                issue_sink.append(LoadIssue(document.source_path, "duplicate_content"))
                continue
            seen_hashes.add(document.content_hash)
            yield document

    def apply(
        self,
        documents: Iterable[Document],
    ) -> tuple[tuple[Document, ...], tuple[LoadIssue, ...]]:
        """Materialize unique documents for the explicit batch API."""

        issues: list[LoadIssue] = []
        unique = tuple(self.iter_unique(documents, issues))
        return unique, tuple(issues)
