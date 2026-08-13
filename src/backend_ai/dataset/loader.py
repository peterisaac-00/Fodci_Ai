"""Deterministic local filesystem document loader."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.documents import Document, DocumentLoadResult, LoadIssue


class LocalDocumentLoader:
    """Discover and validate supported local text files recursively."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._issues: list[LoadIssue] = []

    @property
    def issues(self) -> tuple[LoadIssue, ...]:
        """Return issues observed during the current or most recent iteration."""

        return tuple(self._issues)

    def load(self) -> DocumentLoadResult:
        """Materialize documents for callers that explicitly request that API."""

        documents = tuple(self.iter_documents())
        return DocumentLoadResult(documents=documents, issues=self.issues)

    def iter_documents(self) -> Iterator[Document]:
        """Yield supported documents one at a time in path order."""

        root = self.config.input_dir.expanduser().resolve()
        self._issues = []
        if not root.exists():
            raise FileNotFoundError(f"Dataset input directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Dataset input path is not a directory: {root}")

        candidates = sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in self.config.supported_extensions
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        max_bytes = int(self.config.max_file_size_mb * 1024 * 1024)

        for path in candidates:
            try:
                if path.stat().st_size > max_bytes:
                    self._record_issue(path, "file_too_large")
                    continue
                raw_text = path.read_bytes().decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                self._record_issue(path, "invalid_utf8")
                continue
            except OSError:
                self._record_issue(path, "unreadable")
                continue

            text = self._normalize(raw_text)
            if text == "":
                self._record_issue(path, "empty")
                continue
            if text.strip() == "":
                self._record_issue(path, "whitespace_only")
                continue

            relative_path = path.relative_to(root).as_posix()
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            document_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
            yield Document(
                document_id=document_id,
                source_path=path,
                text=text,
                language=path.suffix.lower().lstrip("."),
                content_hash=content_hash,
            )

    def _record_issue(self, source_path: Path, reason: str) -> None:
        self._issues.append(LoadIssue(source_path, reason))

    def _normalize(self, text: str) -> str:
        if not self.config.normalize_line_endings:
            return text
        return text.replace("\r\n", "\n").replace("\r", "\n")
