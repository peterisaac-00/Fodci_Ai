# Phase 15.1 — Teacher–Student Data Contract

> Interaction data is captured as provenance-rich records, but no record is training-eligible until verification and redaction gates pass.

The `TeacherStudentExample` contract stores the user prompt, teacher response, backend domain, teacher model identity and fingerprint, source, quality status, verification status, split, redaction status, user approval, execution evidence, metadata, timestamp, and a deterministic record ID. Accepted records must be positively verified, secret-reviewed, and assigned to a non-buffer split.

| Gate | Result |
|---|---|
| Stable record identity | `True` |
| JSON round-trip | `True` |
| Backend domain enforced | `True` |
| Provenance present | `True` |
| Accepted record training-eligible | `True` |
| Raw record training-eligible | `False` |
| Unverified acceptance rejected | `True` |
| Automatic training performed | `False` |
| All phase gates | `True` |

This contract deliberately separates capture from training. Raw Qwen output cannot become a Fodci training example merely because it exists; later phases must capture it, filter it, verify it, and train offline under a bounded run.
