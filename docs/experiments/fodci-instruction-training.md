# Fodci Instruction Training — Tiny v1

> **This is a bounded from-scratch engineering experiment. It does not claim intelligence, useful general coding ability, or production readiness.**

## Objective and format

The dataset uses ordinary textual delimiters, not new tokenizer special tokens:

```text
### Instruction
{instruction}

### Input
{context}

### Response
{response}
```

The existing causal language-model training engine is reused. **Response-only loss masking is implemented**: instruction and input tokens provide conditioning context, while response target tokens and the response EOS boundary contribute to cross-entropy. The reported loss and perplexity below are therefore response-only metrics.

## Reproducibility

| Field | Value |
| --- | --- |
| Model version | `fodci-tiny-v1` |
| Model parameters | 11,424,400 |
| Dataset version | `fodci-instructions-v1` |
| Dataset SHA-256 | `c42a4ad2552bb832ced35603eebe15d2df430fec7f463d054df60806ff46af5c` |
| Tokenizer version | 1 |
| Vocabulary size | 10,000 |
| Context length | 256 |
| Seed | 2026 |
| Device | `cpu` |

## Training configuration

| Field | Value |
| --- | ---: |
| Batch size | 2 |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Epochs | 1 |
| Optimization steps | 4 |
| Training time (seconds) | 0.7070 |
| Checkpoint | `artifacts/checkpoints/fodci-instruction-v1.pt` |

## Dataset

| Split | Instructions | Serialized tokens | Response tokens | Training examples |
| --- | ---: | ---: | ---: | ---: |
| Train | 8 | 7,135 | 2,994 | 15 |
| Validation | 3 | 2,237 | 901 | 4 |

## Before/after evaluation

Both states were evaluated on the same validation instruction source and the same response-only mask.

| Metric | Random Fodci Tiny v1 | After instruction training |
| --- | ---: | ---: |
| Validation loss | 9.410560374 | 7.286223404 |
| Response loss | 9.410560374 | 7.286223404 |
| Perplexity | 12216.714988 | 1460.046267 |
| Evaluated examples | 2 | 2 |
| Evaluated response tokens | 901 | 901 |
| Checkpoint identity | `random-initialization` | `instruction-trained` |
| Global step | `None` | `4` |

| Comparison | Value |
| --- | ---: |
| Loss improvement | 2.124336970 |
| Relative loss improvement | 22.5740% |
| Perplexity improvement | 10756.668721 |
| Relative perplexity improvement | 88.0488% |

The result validates the data path, masking path, optimizer update, checkpoint compatibility, and objective measurement on a tiny local dataset. It is not evidence that the model can reliably follow arbitrary instructions or write production backend code.

No pretrained model, tokenizer, or weights were used. No external data was downloaded. No generation, inference, CLI integration, Agent behavior, or Phase 3 functionality is part of Phase 2.10.
