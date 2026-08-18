# Phase 15.3 — Quality Filter and Verification Gate

> Quality heuristics decide eligibility; they do not claim semantic correctness.

`QualityFilter` checks response length, repeated-token rate, backend signal, code presence, and secret-like patterns. `VerificationGate` keeps clean but unevidenced records pending, rejects secret or repetitive records, and promotes only records with execution evidence or explicit human approval.

| Case | Result |
|---|---|
| Clean without evidence | `pending/unverified`; eligible `False` |
| Clean with execution evidence | `accepted/execution_pass`; eligible `True` |
| Secret-like response | `rejected`; redaction `blocked` |
| Repetitive response | `rejected` |
| Automatic training performed | `False` |
| All phase gates | `True` |

This phase does not certify that an answer is semantically correct. It prevents obvious contamination and requires a positive verification path before a record can enter a training split. Rejected records never enter Fodci training.
