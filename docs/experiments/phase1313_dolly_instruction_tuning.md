# Phase 13.13 — Dolly English Instruction Tuning

> This is an English-only response-generation experiment using the Databricks Dolly 15K dataset under **CC-BY-SA-3.0**. It remains experimental and does not replace the stable runtime.

| Field | Result |
|---|---|
| Base checkpoint | `fodci-english-25m-v1.pt` |
| Output checkpoint | `fodci-english-25m-dolly-v1.pt` |
| Tokenizer | `fodci-english-v4.json` |
| Dataset records read | 1,200 |
| Training examples | 1,131 |
| Validation examples | 69 |
| Training steps | 512 |
| Device | CPU only |
| Structural gates | `True` |

## Held-out loss

The response-only training run reduced held-out loss from **5.395007** to **4.673299**, an improvement of **0.721708**. Checkpoint reload, finite loss, parameter change, and non-empty split gates all passed.

## Response probes

| Prompt | Output |
|---|---|
| What is a unit test in Python? | `The Scarkeale is s.` |
| Explain what an API is in one short paragraph. | `The s.` |
| Hello. Please introduce yourself in clear English. | `The Spers, Mans.` |
| How should passwords be stored? | `The s.` |
| What does HTTP 201 mean? | `The Scarkek, Scix Scix Scix Sk, Scix Scix Scark, Scark, Sc` |

The outputs are non-empty but **not understandable English**. The lower validation loss therefore must not be interpreted as conversational capability. The 25M Dolly-tuned checkpoint remains an experimental artifact, and the stable `fodci-testing-qa-v1` checkpoint remains the default.
