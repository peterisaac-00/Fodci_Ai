from __future__ import annotations

from pathlib import Path

import pytest

from backend_ai.dataset import DatasetConfig, FodciDatasetPipeline, LocalDocumentLoader
from backend_ai.tokenizer import EOS_ID, FodciTokenizer


def test_loader_discovers_supported_files_in_deterministic_order_and_preserves_text(
    tmp_path: Path,
) -> None:
    (tmp_path / "z.py").write_text("def z():\n\treturn 1\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.md").write_text("# title\n\ntext", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"binary")

    result = LocalDocumentLoader(DatasetConfig(tmp_path)).load()

    assert [document.source_path.relative_to(tmp_path).as_posix() for document in result.documents] == [
        "nested/a.md",
        "z.py",
    ]
    assert result.documents[1].text == "def z():\n\treturn 1\n"
    assert result.documents[0].language == "md"
    assert result.issues == ()


def test_loader_records_invalid_empty_whitespace_and_oversized_files_without_crashing(
    tmp_path: Path,
) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "spaces.txt").write_text(" \n\t", encoding="utf-8")
    (tmp_path / "invalid.txt").write_bytes(b"\xff\xfe")
    (tmp_path / "large.txt").write_text("0123456789", encoding="utf-8")

    result = LocalDocumentLoader(
        DatasetConfig(tmp_path, max_file_size_mb=0.000008),
    ).load()

    assert result.documents == ()
    assert {issue.reason for issue in result.issues} == {
        "empty",
        "whitespace_only",
        "invalid_utf8",
        "file_too_large",
    }


def test_loader_can_normalize_line_endings_only_when_explicitly_requested(tmp_path: Path) -> None:
    source = tmp_path / "lines.txt"
    source.write_bytes(b"one\r\ntwo\rthree")

    preserved = LocalDocumentLoader(DatasetConfig(tmp_path)).load()
    normalized = LocalDocumentLoader(
        DatasetConfig(tmp_path, normalize_line_endings=True),
    ).load()

    assert preserved.documents[0].text == "one\r\ntwo\rthree"
    assert normalized.documents[0].text == "one\ntwo\nthree"


def test_pipeline_deduplicates_exact_content_and_records_issue(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("same content", encoding="utf-8")
    (tmp_path / "second.md").write_text("same content", encoding="utf-8")
    pipeline = FodciDatasetPipeline(
        DatasetConfig(tmp_path, context_length=8),
        FodciTokenizer(),
    )

    result = pipeline.load_documents()

    assert len(result.documents) == 1
    assert [issue.reason for issue in result.issues] == ["duplicate_content"]
    assert result.documents[0].source_path.name == "first.txt"


def test_pipeline_streams_next_token_examples_with_document_boundaries(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("abcdefghij" * 8, encoding="utf-8")
    (tmp_path / "b.txt").write_text("klmnopqrst" * 8, encoding="utf-8")
    pipeline = FodciDatasetPipeline(
        DatasetConfig(tmp_path, context_length=8, use_eos_document_boundaries=True),
        FodciTokenizer(),
    )

    samples = list(pipeline.iter_samples())

    assert samples
    assert all(len(sample.input_ids) == 8 for sample in samples)
    assert all(len(sample.target_ids) == 8 for sample in samples)
    assert all(
        sample.target_ids == sample.input_ids[1:] + (sample.target_ids[-1],)
        for sample in samples
    )
    assert {sample.document_id for sample in samples}.__len__() == 2
    assert pipeline.last_issues == ()
    assert any(
        EOS_ID in sample.input_ids or EOS_ID in sample.target_ids
        for sample in samples
    )


def test_pipeline_rejects_missing_or_file_input_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LocalDocumentLoader(DatasetConfig(tmp_path / "missing")).load()
    source = tmp_path / "source.txt"
    source.write_text("text", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        LocalDocumentLoader(DatasetConfig(source)).load()
