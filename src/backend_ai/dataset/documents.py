"""Lightweight document and loader-result representations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    """A validated local text document with minimal metadata."""

    document_id: str
    source_path: Path
    text: str
    language: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class LoadIssue:
    """A non-fatal reason why one candidate file was skipped."""

    source_path: Path
    reason: str


@dataclass(frozen=True, slots=True)
class DocumentLoadResult:
    """Deterministic loaded documents and structured non-fatal issues."""

    documents: tuple[Document, ...]
    issues: tuple[LoadIssue, ...]
