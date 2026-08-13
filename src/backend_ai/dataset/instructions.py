"""Local instruction examples and response-masked causal samples."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from backend_ai.dataset.config import DatasetConfig
from backend_ai.dataset.documents import LoadIssue
from backend_ai.dataset.loader import LocalDocumentLoader
from backend_ai.dataset.samples import TrainingExample
from backend_ai.tokenizer import EOS_ID, FodciTokenizer, TOKENIZER_VERSION

INSTRUCTION_HEADER = "### Instruction"
INPUT_HEADER = "### Input"
RESPONSE_HEADER = "### Response"
INSTRUCTION_FORMAT_VERSION = 1


class InstructionFormatError(ValueError):
    """Raised when a local instruction document is malformed."""


@dataclass(frozen=True, slots=True)
class InstructionExample:
    """One deterministic instruction/context/response record."""

    instruction: str
    input_text: str
    response: str
    source_path: Path
    example_id: str

    def serialize(self) -> str:
        return (
            f"{INSTRUCTION_HEADER}\n{self.instruction}\n\n"
            f"{INPUT_HEADER}\n{self.input_text}\n\n"
            f"{RESPONSE_HEADER}\n{self.response}\n"
        )

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()

    @classmethod
    def parse(cls, text: str, source_path: Path = Path("<memory>")) -> "InstructionExample":
        if not isinstance(text, str):
            raise InstructionFormatError("instruction document must be text")
        if not text.strip():
            raise InstructionFormatError("instruction document is empty")
        if not text.endswith("\n"):
            text = f"{text}\n"
        lines = text.splitlines()
        missing_headers = [
            header
            for header in (INSTRUCTION_HEADER, INPUT_HEADER, RESPONSE_HEADER)
            if header not in lines
        ]
        if missing_headers:
            missing_names = "; ".join(
                f"missing {header.removeprefix('### ').lower()}"
                for header in missing_headers
            )
            raise InstructionFormatError(missing_names)
        instruction_index = lines.index(INSTRUCTION_HEADER)
        input_index = lines.index(INPUT_HEADER)
        response_index = lines.index(RESPONSE_HEADER)
        if not instruction_index < input_index < response_index:
            raise InstructionFormatError("instruction headers must appear in the expected order")
        instruction = "\n".join(lines[instruction_index + 1 : input_index]).strip("\n")
        input_text = "\n".join(lines[input_index + 1 : response_index]).strip("\n")
        response = "\n".join(lines[response_index + 1 :]).strip("\n")
        if not instruction.strip():
            raise InstructionFormatError("missing instruction")
        if not input_text.strip():
            raise InstructionFormatError("empty input")
        if not response.strip():
            raise InstructionFormatError("missing response")
        example_id = hashlib.sha256(
            f"{instruction}\0{input_text}\0{response}".encode("utf-8")
        ).hexdigest()
        return cls(instruction, input_text, response, source_path, example_id)


@dataclass(frozen=True, slots=True)
class InstructionLoadResult:
    """Parsed examples plus structured malformed/duplicate issues."""

    examples: tuple[InstructionExample, ...]
    issues: tuple[LoadIssue, ...]


class InstructionDatasetLoader:
    """Load supported local files, then parse one instruction example per file."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._loader = LocalDocumentLoader(config)

    def load(self) -> InstructionLoadResult:
        examples: list[InstructionExample] = []
        issues: list[LoadIssue] = list()
        seen_content: set[str] = set()
        seen_example_ids: set[str] = set()
        for document in self._loader.iter_documents():
            try:
                example = InstructionExample.parse(document.text, document.source_path)
            except InstructionFormatError as exc:
                issues.append(LoadIssue(document.source_path, f"malformed_instruction:{exc}"))
                continue
            if example.content_sha256 in seen_content:
                issues.append(LoadIssue(document.source_path, "duplicate_content"))
                continue
            if example.example_id in seen_example_ids:
                issues.append(LoadIssue(document.source_path, "duplicate_example"))
                continue
            seen_content.add(example.content_sha256)
            seen_example_ids.add(example.example_id)
            examples.append(example)
        issues.extend(self._loader.issues)
        return InstructionLoadResult(tuple(examples), tuple(issues))


class InstructionDatasetPipeline:
    """Serialize and tokenize instruction records with response-only target masks."""

    def __init__(self, config: DatasetConfig, tokenizer: FodciTokenizer) -> None:
        if tokenizer.vocab_size != 10_000:
            raise ValueError("Instruction dataset requires the Fodci vocabulary size of 10,000.")
        self.config = config
        self.tokenizer = tokenizer
        self.loader = InstructionDatasetLoader(config)
        self._last_result: InstructionLoadResult | None = None

    def load_examples(self) -> InstructionLoadResult:
        result = self.loader.load()
        self._last_result = result
        return result

    def iter_training_examples(self) -> Iterator[TrainingExample]:
        result = self.load_examples()
        for example in result.examples:
            yield from self._tokenize_example(example)

    def _tokenize_example(self, example: InstructionExample) -> Iterator[TrainingExample]:
        context = (
            f"{INSTRUCTION_HEADER}\n{example.instruction}\n\n"
            f"{INPUT_HEADER}\n{example.input_text}\n\n"
            f"{RESPONSE_HEADER}\n"
        )
        context_ids = self.tokenizer.encode(context)
        response_ids = self.tokenizer.encode(example.response) + [EOS_ID]
        token_ids = context_ids + response_ids
        response_start = len(context_ids)
        width = self.config.context_length
        if len(token_ids) <= 1:
            return
        for start in range(0, len(token_ids) - 1, width):
            window = token_ids[start : start + width + 1]
            if len(window) != width + 1:
                continue
            target_start = start + 1
            target_end = target_start + width
            mask = tuple(response_start <= index < len(token_ids) for index in range(target_start, target_end))
            if not any(mask):
                continue
            yield TrainingExample(
                input_ids=tuple(window[:-1]),
                target_ids=tuple(window[1:]),
                document_id=example.example_id,
                loss_mask=mask,
            )


__all__ = [
    "INSTRUCTION_FORMAT_VERSION",
    "INSTRUCTION_HEADER",
    "INPUT_HEADER",
    "RESPONSE_HEADER",
    "InstructionDatasetLoader",
    "InstructionDatasetPipeline",
    "InstructionExample",
    "InstructionFormatError",
    "InstructionLoadResult",
]
