"""Deterministic manifest and statistics for instruction-training data."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.documents import LoadIssue
from backend_ai.dataset.instructions import (
    INSTRUCTION_FORMAT_VERSION,
    InstructionDatasetLoader,
    InstructionDatasetPipeline,
)
from backend_ai.tokenizer import FodciTokenizer, TOKENIZER_VERSION

INSTRUCTION_MANIFEST_FORMAT = "fodci-instruction-manifest"
INSTRUCTION_MANIFEST_VERSION = 1


class InstructionManifestError(ValueError):
    """Raised when an instruction dataset is not valid or reproducible."""


@dataclass(frozen=True, slots=True)
class InstructionFileEntry:
    relative_path: str
    bytes: int
    characters: int
    content_sha256: str
    instruction_sha256: str
    serialized_tokens: int


@dataclass(frozen=True, slots=True)
class InstructionSplitStatistics:
    name: str
    path: str
    instruction_count: int
    total_bytes: int
    total_characters: int
    total_tokens: int
    response_tokens: int
    training_example_count: int
    duplicate_count: int
    rejected_file_count: int
    issues: tuple[dict[str, str], ...]
    files: tuple[InstructionFileEntry, ...]
    split_sha256: str

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["files"] = [asdict(entry) for entry in self.files]
        values["issues"] = list(self.issues)
        return values


@dataclass(frozen=True, slots=True)
class InstructionDatasetManifest:
    format: str
    version: int
    dataset_name: str
    instruction_format_version: int
    tokenizer_version: int
    vocabulary_size: int
    context_length: int
    train: InstructionSplitStatistics
    validation: InstructionSplitStatistics
    dataset_sha256: str
    train_validation_leakage_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "dataset_name": self.dataset_name,
            "instruction_format_version": self.instruction_format_version,
            "tokenizer_version": self.tokenizer_version,
            "vocabulary_size": self.vocabulary_size,
            "context_length": self.context_length,
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "dataset_sha256": self.dataset_sha256,
            "train_validation_leakage_count": self.train_validation_leakage_count,
        }


class InstructionDatasetManifestBuilder:
    """Build exact split statistics from the instruction pipeline."""

    def __init__(
        self,
        root: Path | str,
        *,
        tokenizer: FodciTokenizer | None = None,
        context_length: int = 256,
        max_file_size_mb: float = 2.0,
        strict: bool = True,
        dataset_name: str = "fodci-instructions",
    ) -> None:
        self.root = Path(root)
        self.tokenizer = tokenizer or FodciTokenizer()
        self.context_length = context_length
        self.max_file_size_mb = max_file_size_mb
        self.strict = strict
        self.dataset_name = dataset_name
        if self.tokenizer.vocab_size != 10_000:
            raise InstructionManifestError("Instruction dataset requires vocabulary size 10,000.")

    def build(self) -> InstructionDatasetManifest:
        train = self._build_split("train")
        validation = self._build_split("validation")
        train_hashes = {entry.instruction_sha256 for entry in train.files}
        validation_hashes = {entry.instruction_sha256 for entry in validation.files}
        leakage = train_hashes & validation_hashes
        if leakage:
            raise InstructionManifestError(
                f"Train/validation instruction leakage detected for {len(leakage)} example(s)."
            )
        issues = train.issues + validation.issues
        if self.strict and issues:
            reasons = ", ".join(issue["reason"] for issue in issues)
            raise InstructionManifestError(f"Instruction dataset contains invalid files: {reasons}")
        dataset_sha256 = _dataset_digest(self.dataset_name, train, validation)
        return InstructionDatasetManifest(
            format=INSTRUCTION_MANIFEST_FORMAT,
            version=INSTRUCTION_MANIFEST_VERSION,
            dataset_name=self.dataset_name,
            instruction_format_version=INSTRUCTION_FORMAT_VERSION,
            tokenizer_version=TOKENIZER_VERSION,
            vocabulary_size=self.tokenizer.vocab_size,
            context_length=self.context_length,
            train=train,
            validation=validation,
            dataset_sha256=dataset_sha256,
            train_validation_leakage_count=0,
        )

    def _build_split(self, split_name: str) -> InstructionSplitStatistics:
        split_root = self.root / split_name
        if not split_root.exists():
            raise InstructionManifestError(f"Instruction split does not exist: {split_root}")
        if not split_root.is_dir():
            raise InstructionManifestError(f"Instruction split is not a directory: {split_root}")
        config = DatasetConfig(
            split_root,
            supported_extensions=frozenset({".txt"}),
            max_file_size_mb=self.max_file_size_mb,
            context_length=self.context_length,
            use_eos_document_boundaries=False,
        )
        loader = InstructionDatasetLoader(config)
        loaded = loader.load()
        files: list[InstructionFileEntry] = []
        for example in sorted(
            loaded.examples,
            key=lambda item: item.source_path.resolve().relative_to(split_root.resolve()).as_posix(),
        ):
            serialized = example.serialize()
            files.append(
                InstructionFileEntry(
                    relative_path=example.source_path.resolve().relative_to(split_root.resolve()).as_posix(),
                    bytes=len(serialized.encode("utf-8")),
                    characters=len(serialized),
                    content_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                    instruction_sha256=example.content_sha256,
                    serialized_tokens=len(self.tokenizer.encode(serialized)),
                )
            )
        pipeline = InstructionDatasetPipeline(config, self.tokenizer)
        samples = tuple(pipeline.iter_training_examples())
        response_tokens = sum(sum(example.loss_mask or ()) for example in samples)
        issues = tuple(
            {"relative_path": _issue_path(issue, split_root), "reason": issue.reason}
            for issue in loaded.issues
        )
        duplicate_count = sum(
            issue["reason"] in {"duplicate_content", "duplicate_example"}
            for issue in issues
        )
        rejected_count = len(issues) - duplicate_count
        entries = tuple(files)
        split_digest = hashlib.sha256()
        split_digest.update(split_name.encode("utf-8"))
        for entry in entries:
            split_digest.update(entry.relative_path.encode("utf-8"))
            split_digest.update(entry.instruction_sha256.encode("ascii"))
        for issue in issues:
            split_digest.update(issue["relative_path"].encode("utf-8"))
            split_digest.update(issue["reason"].encode("utf-8"))
        return InstructionSplitStatistics(
            name=split_name,
            path=str(split_root),
            instruction_count=len(entries),
            total_bytes=sum(entry.bytes for entry in entries),
            total_characters=sum(entry.characters for entry in entries),
            total_tokens=sum(entry.serialized_tokens for entry in entries),
            response_tokens=response_tokens,
            training_example_count=len(samples),
            duplicate_count=duplicate_count,
            rejected_file_count=rejected_count,
            issues=issues,
            files=entries,
            split_sha256=split_digest.hexdigest(),
        )


def _issue_path(issue: LoadIssue, root: Path) -> str:
    try:
        return issue.source_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return issue.source_path.name


def _dataset_digest(
    name: str,
    train: InstructionSplitStatistics,
    validation: InstructionSplitStatistics,
) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(train.split_sha256.encode("ascii"))
    digest.update(validation.split_sha256.encode("ascii"))
    return digest.hexdigest()
