from __future__ import annotations

from pathlib import Path

from backend_ai.application import (
    DEFAULT_CHECKPOINT_RELATIVE_PATH,
    LEGACY_CHECKPOINT_RELATIVE_PATH,
    resolve_checkpoint_path,
)


def test_checkpoint_selection_prefers_stable_specialist(tmp_path: Path) -> None:
    stable = tmp_path / DEFAULT_CHECKPOINT_RELATIVE_PATH
    tiny = tmp_path / LEGACY_CHECKPOINT_RELATIVE_PATH
    stable.parent.mkdir(parents=True)
    stable.write_bytes(b"stable")
    tiny.write_bytes(b"tiny")

    assert resolve_checkpoint_path(tmp_path) == stable


def test_checkpoint_selection_falls_back_to_tiny(tmp_path: Path) -> None:
    tiny = tmp_path / LEGACY_CHECKPOINT_RELATIVE_PATH
    tiny.parent.mkdir(parents=True)
    tiny.write_bytes(b"tiny")

    assert resolve_checkpoint_path(tmp_path) == tiny


def test_checkpoint_selection_returns_stable_target_when_missing(tmp_path: Path) -> None:
    assert resolve_checkpoint_path(tmp_path) == tmp_path / DEFAULT_CHECKPOINT_RELATIVE_PATH
