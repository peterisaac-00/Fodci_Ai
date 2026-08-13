from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend_ai.dataset import (
    CodingDatasetManifestBuilder,
    DatasetManifestError,
)
from backend_ai.tokenizer import FodciTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CODING_ROOT = REPOSITORY_ROOT / "data" / "fodci_coding"


def test_coding_manifest_has_deterministic_identity_and_statistics() -> None:
    first = CodingDatasetManifestBuilder(CODING_ROOT).build()
    second = CodingDatasetManifestBuilder(CODING_ROOT).build()

    assert first.to_dict() == second.to_dict()
    assert first.format == "fodci-dataset-manifest"
    assert first.version == 1
    assert first.dataset_sha256 == "42c1f12f53dc553c8bfa93c9eb7cd48ff9c86102f568b5b0a4614b84c025c954"
    assert first.train.document_count == 7
    assert first.validation.document_count == 4
    assert first.train.language_distribution == {"md": 1, "py": 6}
    assert first.validation.language_distribution == {"dockerfile": 1, "py": 2, "sql": 1}
    assert first.train.duplicate_count == 0
    assert first.validation.duplicate_count == 0
    assert first.train.rejected_file_count == 0
    assert first.validation.rejected_file_count == 0
    assert first.train_validation_leakage_count == 0
    assert first.tokenizer_version == 1
    assert first.vocabulary_size == 10_000
    assert first.context_length == 256


def test_manifest_preserves_exact_file_identity_and_counts() -> None:
    manifest = CodingDatasetManifestBuilder(CODING_ROOT).build()
    train_paths = [entry.relative_path for entry in manifest.train.files]
    validation_paths = [entry.relative_path for entry in manifest.validation.files]

    assert train_paths == sorted(train_paths)
    assert validation_paths == sorted(validation_paths)
    assert "api/routes.py" in train_paths
    assert "db/schema.sql" in validation_paths
    assert all(entry.bytes == entry.characters for entry in manifest.train.files)
    assert all(entry.tokens_including_eos == entry.characters + 1 for entry in manifest.train.files)


def test_duplicate_content_is_reported_and_strict_mode_rejects_it(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    content = "same backend content\n"
    (tmp_path / "train" / "a.py").write_text(content, encoding="utf-8")
    (tmp_path / "train" / "b.py").write_text(content, encoding="utf-8")
    (tmp_path / "validation" / "v.py").write_text("different validation\n", encoding="utf-8")

    diagnostic = CodingDatasetManifestBuilder(tmp_path, strict=False).build()
    assert diagnostic.train.duplicate_count == 1
    assert diagnostic.train.document_count == 1
    assert diagnostic.train.issues[0]["reason"] == "duplicate_content"

    with pytest.raises(DatasetManifestError, match="duplicate"):
        CodingDatasetManifestBuilder(tmp_path, strict=True).build()


def test_train_validation_content_leakage_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    content = "shared exact content\n"
    (tmp_path / "train" / "train.py").write_text(content, encoding="utf-8")
    (tmp_path / "validation" / "validation.py").write_text(content, encoding="utf-8")

    with pytest.raises(DatasetManifestError, match="leakage"):
        CodingDatasetManifestBuilder(tmp_path, strict=False).build()


def test_rejected_files_and_malformed_structure_are_explicit(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    (tmp_path / "train" / "empty.py").write_text("", encoding="utf-8")
    (tmp_path / "train" / "spaces.py").write_text(" \n\t", encoding="utf-8")
    (tmp_path / "train" / "broken.py").write_bytes(b"\xff\xfe")
    (tmp_path / "train" / "valid.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "train" / "unsupported.bin").write_bytes(b"binary")
    (tmp_path / "validation" / "valid.py").write_text("print('validation')\n", encoding="utf-8")

    diagnostic = CodingDatasetManifestBuilder(tmp_path, strict=False).build()
    reasons = {issue["reason"] for issue in diagnostic.train.issues}
    assert reasons == {"empty", "whitespace_only", "invalid_utf8", "unsupported_extension"}
    assert diagnostic.train.rejected_file_count == 4

    with pytest.raises(DatasetManifestError, match="rejected"):
        CodingDatasetManifestBuilder(tmp_path, strict=True).build()

    (tmp_path / "validation" / "valid.py").unlink()
    (tmp_path / "validation").rmdir()
    with pytest.raises(DatasetManifestError, match="does not exist"):
        CodingDatasetManifestBuilder(tmp_path, strict=False).build()


def test_tokenizer_compatibility_is_required(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    (tmp_path / "train" / "a.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "validation" / "b.py").write_text("print('validation')\n", encoding="utf-8")

    with pytest.raises(DatasetManifestError, match="vocabulary size"):
        CodingDatasetManifestBuilder(tmp_path, tokenizer=FodciTokenizer(vocab_size=300)).build()


def test_manifest_json_is_serializable() -> None:
    manifest = CodingDatasetManifestBuilder(CODING_ROOT).build()
    encoded = json.dumps(manifest.to_dict(), sort_keys=True)
    assert "fodci-dataset-manifest" in encoded
    assert manifest.dataset_sha256 in encoded
