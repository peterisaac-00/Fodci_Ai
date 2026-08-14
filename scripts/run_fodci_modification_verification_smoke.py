"""Manual local smoke for Phase 4.7 modification verification."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from backend_ai.tools import ExpectedModification, SafeEditPolicy, SafeEditSession, verify_modification


def main() -> None:
    with TemporaryDirectory(prefix="fodci-phase47-verification-") as directory:
        root = Path(directory) / "project"
        root.mkdir()
        session = SafeEditSession(SafeEditPolicy.for_modification())

        created = session.create(root, "app.py", "value = 'old'\n")
        assert created.verification is not None and created.verification.success is True

        edited = session.edit(root, "app.py", "'old'", "'new'")
        assert edited.verification is not None and edited.verification.success is True
        assert edited.verification.verified_targets[0].status == "VERIFIED"

        mismatch = verify_modification(
            root,
            [ExpectedModification.modified("app.py", expected_content="wrong", before_snapshot=None)],
            detect_unexpected=False,
        )
        assert mismatch.success is False
        assert mismatch.verified_targets[0].status == "CONTENT_MISMATCH"

        deleted = session.delete(root, "app.py")
        assert deleted.verification is not None and deleted.verification.success is True
        assert deleted.verification.verified_targets[0].status == "VERIFIED"
        assert not (root / "app.py").exists()

    print("Phase 4.7 modification verification smoke passed")


if __name__ == "__main__":
    main()
