"""Build the Phase 2.10 instruction-training dataset manifest."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.dataset import InstructionDatasetManifestBuilder  # noqa: E402

DATA_ROOT = Path("data/fodci_instructions")
MANIFEST_JSON = ROOT / "docs" / "datasets" / "fodci-instruction-manifest.json"
REPORT_MD = ROOT / "docs" / "datasets" / "fodci-instructions.md"


def render_report(manifest: dict) -> str:
    train = manifest["train"]
    validation = manifest["validation"]
    return f"""# Fodci Instruction Dataset

> **Phase 2.10 teaches only the textual structure `Instruction → Input → Response`; it does not claim that Fodci has gained intelligence or production-ready coding ability.**

## Format

Each `.txt` file contains exactly three ordinary textual sections. No new tokenizer special tokens are used:

```text
### Instruction
{{instruction}}

### Input
{{context}}

### Response
{{response}}
```

The serializer, parser, tokenizer, training samples, and evaluation workflow use this same deterministic format. The response boundary is the first token after `### Response` and the training loss mask activates only target positions belonging to the response plus its EOS boundary.

## Identity

| Field | Value |
| --- | --- |
| Dataset name | `{manifest['dataset_name']}` |
| Manifest format/version | `{manifest['format']}` / `{manifest['version']}` |
| Instruction format version | `{manifest['instruction_format_version']}` |
| Dataset SHA-256 | `{manifest['dataset_sha256']}` |
| Tokenizer version | `{manifest['tokenizer_version']}` |
| Vocabulary size | {manifest['vocabulary_size']:,} |
| Context length | {manifest['context_length']} |
| Train/validation leakage | {manifest['train_validation_leakage_count']} exact instruction hashes |

## Structure

```text
data/fodci_instructions/
├── train/       8 backend instruction examples
└── validation/  3 held-out backend instruction examples
```

## Statistics

| Split | Instructions | Serialized tokens | Response tokens | Training examples | Duplicates | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | {train['instruction_count']} | {train['total_tokens']:,} | {train['response_tokens']:,} | {train['training_example_count']} | {train['duplicate_count']} | {train['rejected_file_count']} |
| Validation | {validation['instruction_count']} | {validation['total_tokens']:,} | {validation['response_tokens']:,} | {validation['training_example_count']} | {validation['duplicate_count']} | {validation['rejected_file_count']} |

The exact file names, byte/character counts, serialized-token counts, content hashes, and instruction hashes are in the tracked JSON manifest. Train and validation examples have no identical instruction/response content hash.

## Training objective

The existing `FodciTrainer` remains the optimizer and causal language-model training engine. Instruction samples provide the serialized instruction and input as conditioning context; only response target positions contribute to cross-entropy when a `loss_mask` is present. This is a minimal response-only masking extension, not a new model architecture or fine-tuning framework.

The smoke experiment is CPU-first, deterministic, and intentionally small. It must be interpreted as an engineering validation of data parsing, masking, checkpointing, and objective metrics on a tiny local corpus—not as evidence of general coding ability.

No external dataset, pretrained model, pretrained tokenizer, pretrained weights, generation, inference, CLI integration, Agent behavior, or Phase 3 functionality is part of Phase 2.10.
"""


def main() -> None:
    payload = InstructionDatasetManifestBuilder(DATA_ROOT, strict=True).build().to_dict()
    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
