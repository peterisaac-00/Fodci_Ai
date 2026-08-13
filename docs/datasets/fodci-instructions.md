# Fodci Instruction Dataset

> **Phase 2.10 teaches only the textual structure `Instruction → Input → Response`; it does not claim that Fodci has gained intelligence or production-ready coding ability.**

## Format

Each `.txt` file contains exactly three ordinary textual sections. No new tokenizer special tokens are used:

```text
### Instruction
{instruction}

### Input
{context}

### Response
{response}
```

The serializer, parser, tokenizer, training samples, and evaluation workflow use this same deterministic format. The response boundary is the first token after `### Response` and the training loss mask activates only target positions belonging to the response plus its EOS boundary.

## Identity

| Field | Value |
| --- | --- |
| Dataset name | `fodci-instructions` |
| Manifest format/version | `fodci-instruction-manifest` / `1` |
| Instruction format version | `1` |
| Dataset SHA-256 | `c42a4ad2552bb832ced35603eebe15d2df430fec7f463d054df60806ff46af5c` |
| Tokenizer version | `1` |
| Vocabulary size | 10,000 |
| Context length | 256 |
| Train/validation leakage | 0 exact instruction hashes |

## Structure

```text
data/fodci_instructions/
├── train/       8 backend instruction examples
└── validation/  3 held-out backend instruction examples
```

## Statistics

| Split | Instructions | Serialized tokens | Response tokens | Training examples | Duplicates | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 8 | 7,135 | 2,994 | 15 | 0 | 0 |
| Validation | 3 | 2,237 | 901 | 4 | 0 | 0 |

The exact file names, byte/character counts, serialized-token counts, content hashes, and instruction hashes are in the tracked JSON manifest. Train and validation examples have no identical instruction/response content hash.

## Training objective

The existing `FodciTrainer` remains the optimizer and causal language-model training engine. Instruction samples provide the serialized instruction and input as conditioning context; only response target positions contribute to cross-entropy when a `loss_mask` is present. This is a minimal response-only masking extension, not a new model architecture or fine-tuning framework.

The smoke experiment is CPU-first, deterministic, and intentionally small. It must be interpreted as an engineering validation of data parsing, masking, checkpointing, and objective metrics on a tiny local corpus—not as evidence of general coding ability.

No external dataset, pretrained model, pretrained tokenizer, pretrained weights, generation, inference, CLI integration, Agent behavior, or Phase 3 functionality is part of Phase 2.10.
