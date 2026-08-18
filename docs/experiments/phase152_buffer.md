# Phase 15.2 — Interaction Capture and Training Buffer

> Interactions are persisted locally as bounded JSONL records. Capture never starts model training.

`InteractionBuffer` provides append-only local persistence for `TeacherStudentExample` records. It rejects symlink paths, malformed UTF-8/JSONL, duplicate record IDs, and capacity overflow. It exposes pending and training-eligible views but has no training side effect.

| Gate | Result |
|---|---|
| Records stored in contract runner | 2 |
| Pending records preserved | 2 |
| Duplicate rejected | `True` |
| Capacity overflow rejected | `True` |
| Training automatically started | `False` |
| External API used | `False` |
| Local-only storage | `True` |
| All phase gates | `True` |

The default runtime buffer belongs on the user’s local machine and is ignored by Git; only schema, code, tests, and aggregate reports are committed. Phase 15.3 determines which pending records can be verified and assigned to training or evaluation splits.
