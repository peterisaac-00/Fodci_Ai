# Fodci Coding Dataset

> **Phase 2.9 only improves the local training corpus. It does not claim that Fodci has gained intelligence or useful coding ability.**

## Identity

| Field | Value |
| --- | --- |
| Dataset name | `fodci-coding` |
| Format/version | `fodci-dataset-manifest` / `1` |
| Dataset SHA-256 | `42c1f12f53dc553c8bfa93c9eb7cd48ff9c86102f568b5b0a4614b84c025c954` |
| Tokenizer version | `1` |
| Vocabulary size | 10,000 |
| Context length | 256 |
| EOS document boundaries | `True` |
| Train/validation leakage | 0 exact content hashes |

## Structure

```text
data/fodci_coding/
├── train/
│   ├── api/routes.py
│   ├── auth/service.py
│   ├── config/settings.py
│   ├── db/repository.py
│   ├── docs/backend_architecture.md
│   ├── tests/test_backend.py
│   └── workers/jobs.py
└── validation/
    ├── api/validation.py
    ├── db/schema.sql
    ├── deployment/service.dockerfile
    └── tests/test_health.py
```

The split directories are explicit and are never merged. Every accepted file is listed in the JSON manifest with its relative path, extension/language, UTF-8 byte count, character count, token count including EOS, and content SHA-256.

## Statistics

| Split | Documents | Bytes | Characters | Tokens incl. EOS | Training examples | Duplicates | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 7 | 9,698 | 9,698 | 9,705 | 34 | 0 | 0 |
| Validation | 4 | 3,044 | 3,044 | 3,048 | 10 | 0 | 0 |

### Language/file-type distribution

| Split | Distribution |
| --- | --- |
| Train | `{"md": 1, "py": 6}` |
| Validation | `{"dockerfile": 1, "py": 2, "sql": 1}` |

## Quality controls

The manifest is generated through the existing `FodciDatasetPipeline`, preserving deterministic path ordering, exact UTF-8 source text, maximum file size validation, empty/whitespace-only rejection, invalid UTF-8 rejection, exact duplicate-content detection, and EOS-aware chunk counts. The builder fails if a split is missing, malformed, contains rejected/duplicate files in strict mode, or shares an exact content hash with the other split.

No model training, checkpoint creation, generation, inference, CLI integration, or external dataset download is part of Phase 2.9.
