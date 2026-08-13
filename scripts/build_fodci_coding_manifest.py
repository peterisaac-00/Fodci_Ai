"""Build the Phase 2.9 local coding-dataset manifest."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.dataset import CodingDatasetManifestBuilder  # noqa: E402

DATA_ROOT = Path("data/fodci_coding")
MANIFEST_JSON = ROOT / "docs" / "datasets" / "fodci-coding-manifest.json"
REPORT_MD = ROOT / "docs" / "datasets" / "fodci-coding.md"


def render_report(manifest: dict) -> str:
    train = manifest["train"]
    validation = manifest["validation"]
    return f"""# Fodci Coding Dataset

> **Phase 2.9 only improves the local training corpus. It does not claim that Fodci has gained intelligence or useful coding ability.**

## Identity

| Field | Value |
| --- | --- |
| Dataset name | `{manifest['dataset_name']}` |
| Format/version | `{manifest['format']}` / `{manifest['version']}` |
| Dataset SHA-256 | `{manifest['dataset_sha256']}` |
| Tokenizer version | `{manifest['tokenizer_version']}` |
| Vocabulary size | {manifest['vocabulary_size']:,} |
| Context length | {manifest['context_length']} |
| EOS document boundaries | `{manifest['use_eos_document_boundaries']}` |
| Train/validation leakage | {manifest['train_validation_leakage_count']} exact content hashes |

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
| Train | {train['document_count']} | {train['total_bytes']:,} | {train['total_characters']:,} | {train['total_tokens']:,} | {train['training_example_count']} | {train['duplicate_count']} | {train['rejected_file_count']} |
| Validation | {validation['document_count']} | {validation['total_bytes']:,} | {validation['total_characters']:,} | {validation['total_tokens']:,} | {validation['training_example_count']} | {validation['duplicate_count']} | {validation['rejected_file_count']} |

### Language/file-type distribution

| Split | Distribution |
| --- | --- |
| Train | `{json.dumps(train['language_distribution'], sort_keys=True)}` |
| Validation | `{json.dumps(validation['language_distribution'], sort_keys=True)}` |

## Quality controls

The manifest is generated through the existing `FodciDatasetPipeline`, preserving deterministic path ordering, exact UTF-8 source text, maximum file size validation, empty/whitespace-only rejection, invalid UTF-8 rejection, exact duplicate-content detection, and EOS-aware chunk counts. The builder fails if a split is missing, malformed, contains rejected/duplicate files in strict mode, or shares an exact content hash with the other split.

No model training, checkpoint creation, generation, inference, CLI integration, or external dataset download is part of Phase 2.9.
"""


def main() -> None:
    manifest = CodingDatasetManifestBuilder(DATA_ROOT, strict=True).build()
    payload = manifest.to_dict()
    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
