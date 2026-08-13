"""Deterministic manifests and statistics for local Fodci coding datasets."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.documents import Document, LoadIssue
from backend_ai.dataset.pipeline import FodciDatasetPipeline
from backend_ai.tokenizer import DEFAULT_VOCAB_SIZE, FodciTokenizer, TOKENIZER_VERSION

MANIFEST_FORMAT = "fodci-dataset-manifest"
MANIFEST_VERSION = 1


class DatasetManifestError(ValueError):
    """Raised when a dataset cannot produce a trustworthy manifest."""


@dataclass(frozen=True, slots=True)
class FileManifestEntry:
    """Exact identity and lightweight statistics for one accepted document."""

    relative_path: str
    language: str
    bytes: int
    characters: int
    tokens_including_eos: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class SplitStatistics:
    """Deterministic statistics for one train or validation split."""

    name: str
    path: str
    document_count: int
    total_bytes: int
    total_characters: int
    total_tokens: int
    training_example_count: int
    language_distribution: dict[str, int]
    duplicate_count: int
    rejected_file_count: int
    issues: tuple[dict[str, str], ...]
    files: tuple[FileManifestEntry, ...]
    split_sha256: str

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["files"] = [asdict(entry) for entry in self.files]
        values["issues"] = list(self.issues)
        return values


@dataclass(frozen=True, slots=True)
class CodingDatasetManifest:
    """Complete reproducible manifest for train and validation corpus."""

    format: str
    version: int
    dataset_name: str
    tokenizer_version: int
    vocabulary_size: int
    context_length: int
    use_eos_document_boundaries: bool
    train: SplitStatistics
    validation: SplitStatistics
    dataset_sha256: str
    train_validation_leakage_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "dataset_name": self.dataset_name,
            "tokenizer_version": self.tokenizer_version,
            "vocabulary_size": self.vocabulary_size,
            "context_length": self.context_length,
            "use_eos_document_boundaries": self.use_eos_document_boundaries,
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "dataset_sha256": self.dataset_sha256,
            "train_validation_leakage_count": self.train_validation_leakage_count,
        }


class CodingDatasetManifestBuilder:
    """Build train/validation statistics by composing existing pipelines."""

    def __init__(
        self,
        root: Path | str,
        *,
        tokenizer: FodciTokenizer | None = None,
        context_length: int = 256,
        max_file_size_mb: float = 2.0,
        strict: bool = True,
        dataset_name: str = "fodci-coding",
    ) -> None:
        self.root = Path(root)
        self.tokenizer = tokenizer or FodciTokenizer()
        self.context_length = context_length
        self.max_file_size_mb = max_file_size_mb
        self.strict = strict
        self.dataset_name = dataset_name
        if self.tokenizer.vocab_size != DEFAULT_VOCAB_SIZE:
            raise DatasetManifestError(
                f"Tokenizer vocabulary size {self.tokenizer.vocab_size} is incompatible "
                f"with Fodci model vocabulary size {DEFAULT_VOCAB_SIZE}."
            )

    def build(self) -> CodingDatasetManifest:
        train = self._build_split("train")
        validation = self._build_split("validation")
        train_hashes = {entry.content_sha256 for entry in train.files}
        validation_hashes = {entry.content_sha256 for entry in validation.files}
        leakage = train_hashes & validation_hashes
        if leakage:
            raise DatasetManifestError(
                "Train/validation content leakage detected for "
                f"{len(leakage)} exact content hash(es)."
            )
        if self.strict:
            issues = train.issues + validation.issues
            if issues:
                summary = ", ".join(issue["reason"] for issue in issues)
                raise DatasetManifestError(f"Dataset contains rejected or duplicate files: {summary}")
        dataset_sha256 = _dataset_digest(train, validation, self.dataset_name)
        return CodingDatasetManifest(
            format=MANIFEST_FORMAT,
            version=MANIFEST_VERSION,
            dataset_name=self.dataset_name,
            tokenizer_version=TOKENIZER_VERSION,
            vocabulary_size=self.tokenizer.vocab_size,
            context_length=self.context_length,
            use_eos_document_boundaries=True,
            train=train,
            validation=validation,
            dataset_sha256=dataset_sha256,
            train_validation_leakage_count=0,
        )

    def _build_split(self, split_name: str) -> SplitStatistics:
        split_root = self.root / split_name
        if not split_root.exists():
            raise DatasetManifestError(f"Dataset split directory does not exist: {split_root}")
        if not split_root.is_dir():
            raise DatasetManifestError(f"Dataset split path is not a directory: {split_root}")
        config = DatasetConfig(
            split_root,
            max_file_size_mb=self.max_file_size_mb,
            context_length=self.context_length,
            use_eos_document_boundaries=True,
        )
        pipeline = FodciDatasetPipeline(config, self.tokenizer)
        loaded = pipeline.load_documents()
        split_documents = tuple(sorted(loaded.documents, key=lambda doc: _relative_path(doc, split_root)))
        entries = tuple(self._file_entry(document, split_root) for document in split_documents)
        example_count = sum(1 for _ in pipeline.iter_samples())
        issues = tuple(
            {"relative_path": _relative_issue_path(issue, split_root), "reason": issue.reason}
            for issue in loaded.issues
        )
        duplicate_count = sum(issue["reason"] == "duplicate_content" for issue in issues)
        rejected_count = len(issues) - duplicate_count
        language_distribution = dict(sorted(Counter(entry.language for entry in entries).items()))
        split_sha256 = _split_digest(split_name, entries, issues)
        return SplitStatistics(
            name=split_name,
            path=str(split_root),
            document_count=len(entries),
            total_bytes=sum(entry.bytes for entry in entries),
            total_characters=sum(entry.characters for entry in entries),
            total_tokens=sum(entry.tokens_including_eos for entry in entries),
            training_example_count=example_count,
            language_distribution=language_distribution,
            duplicate_count=duplicate_count,
            rejected_file_count=rejected_count,
            issues=issues,
            files=entries,
            split_sha256=split_sha256,
        )

    def _file_entry(self, document: Document, split_root: Path) -> FileManifestEntry:
        return FileManifestEntry(
            relative_path=_relative_path(document, split_root),
            language=document.language,
            bytes=len(document.text.encode("utf-8")),
            characters=len(document.text),
            tokens_including_eos=len(self.tokenizer.encode(document.text)) + 1,
            content_sha256=document.content_hash,
        )


def _relative_path(document: Document, root: Path) -> str:
    return document.source_path.resolve().relative_to(root.resolve()).as_posix()


def _relative_issue_path(issue: LoadIssue, root: Path) -> str:
    try:
        return issue.source_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return issue.source_path.name


def _split_digest(
    split_name: str,
    entries: tuple[FileManifestEntry, ...],
    issues: tuple[dict[str, str], ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(split_name.encode("utf-8"))
    for entry in entries:
        digest.update(entry.relative_path.encode("utf-8"))
        digest.update(entry.content_sha256.encode("ascii"))
        digest.update(str(entry.tokens_including_eos).encode("ascii"))
    for issue in issues:
        digest.update(issue["relative_path"].encode("utf-8"))
        digest.update(issue["reason"].encode("utf-8"))
    return digest.hexdigest()


def _dataset_digest(train: SplitStatistics, validation: SplitStatistics, dataset_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(dataset_name.encode("utf-8"))
    digest.update(train.split_sha256.encode("ascii"))
    digest.update(validation.split_sha256.encode("ascii"))
    return digest.hexdigest()
