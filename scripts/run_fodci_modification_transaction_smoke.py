"""Manual local smoke for Phase 4.8 transaction/recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from backend_ai.tools import (
    ModificationOperation,
    ModificationTransaction,
    SafeEditPolicy,
    SafeEditSession,
)


def _policy() -> SafeEditPolicy:
    return SafeEditPolicy.for_modification(backup_enabled=True, retain_backup_on_success=False)


def _run_project_fixture(root: Path, relative: str, content: str) -> None:
    created = ModificationTransaction(root, ModificationOperation.create(relative, content), policy=_policy()).execute()
    assert created.status == "committed"
    edited = ModificationTransaction(root, ModificationOperation.edit(relative, content, content.replace("old", "new")), policy=_policy()).execute()
    assert edited.status == "committed"
    deleted = ModificationTransaction(root, ModificationOperation.delete(relative), policy=_policy()).execute()
    assert deleted.status == "committed"
    assert not (root / relative).exists()
    assert not (root / ".fodci" / "backups").exists() or not list((root / ".fodci" / "backups").glob("*.bak"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-current-repo", action="store_true")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="fodci-phase48-smoke-") as directory:
        root = Path(directory)
        python_project = root / "python-project"
        python_project.mkdir()
        _run_project_fixture(python_project, "src/app.py", "value = 'old'\n")

        node_project = root / "node-project"
        node_project.mkdir()
        _run_project_fixture(node_project, "server.js", "const value = 'old';\n")
        (node_project / "package.json").write_text('{"name":"smoke"}\n', encoding="utf-8")
        assert (node_project / "package.json").exists()

        target = python_project / "recover.py"
        target.write_text("original", encoding="utf-8")
        transaction = ModificationTransaction(python_project, ModificationOperation.edit("recover.py", "original", "changed"), policy=_policy())
        original_cleanup = transaction._cleanup_backup
        transaction._cleanup_backup = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup failure"))  # type: ignore[method-assign]
        failed = transaction.execute()
        assert failed.status == "recovery_required"
        target.write_text("user change", encoding="utf-8")
        transaction._cleanup_backup = original_cleanup  # type: ignore[method-assign]
        preserved = transaction.recover()
        assert preserved.recovery is not None and preserved.recovery.status == "user_change_preserved"
        assert target.read_text(encoding="utf-8") == "user change"

    if args.inspect_current_repo:
        session = SafeEditSession(SafeEditPolicy())
        snapshot = session.snapshot(Path.cwd(), "README.md")
        assert snapshot.exists and snapshot.content_hash
        print("Actual Fodci repository read-only snapshot passed")

    print("Phase 4.8 modification transaction smoke passed")


if __name__ == "__main__":
    main()
