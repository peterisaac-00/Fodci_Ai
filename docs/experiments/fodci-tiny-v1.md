# Fodci Tiny v1 Experiment

> **This report documents an engineering training run from random initialization. It is not evidence of useful language capability.**

## Model

| Field | Value |
| --- | --- |
| Version | `fodci-tiny-v1` |
| Parameters | 11,424,400 |
| Vocabulary size | 10,000 |
| Context length | 256 |
| Hidden size | 320 |
| Attention heads | 5 |
| Transformer blocks | 4 |
| Feed-forward size | 1,280 |
| Initialization | Random, seed `2026` |

## Dataset

The corpus was authored locally for this repository and contains only small backend-engineering examples. No internet source, external dataset, GitHub repository, API, secret, or pretrained artifact was used.

| Split | Directory | Documents | Tokens including EOS | Training examples | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| Train | `data/fodci_tiny_v1/train` | 4 | 6749 | 24 | `40b3335d164c3dad41d6bee790fd8d77ca28c0c9210c1349dbc7e3dc0298cc21` |
| Validation | `data/fodci_tiny_v1/validation` | 2 | 2484 | 9 | `c994d8b4d2a7892224a812bffe59b20e420529cc0f7f9a170794ddd92674cec3` |

The train and validation directories are separate and are consumed through the existing `FodciDatasetPipeline`. Validation examples are never passed to the optimizer.

Train files: `data/fodci_tiny_v1/train/api_routes.py, data/fodci_tiny_v1/train/auth_service.py, data/fodci_tiny_v1/train/database.py, data/fodci_tiny_v1/train/test_backend.py`

Validation files: `data/fodci_tiny_v1/validation/api_validation.py, data/fodci_tiny_v1/validation/config_validation.py`

## Training

| Field | Value |
| --- | --- |
| Device | `cpu` |
| Epoch budget | 2 |
| Max optimization steps | 12 |
| Completed epochs | 1 |
| Optimization steps | 12 |
| Batch size | 2 |
| Learning rate | `0.0003` |
| Weight decay | `0.01` |
| Gradient clipping | `1.0` |
| Seed | `2026` |
| Tokenizer version | `1` |
| Elapsed seconds | `5.356` |

## Results

| Metric | Value |
| --- | ---: |
| Baseline validation loss | `9.340518210` |
| Baseline perplexity | `11390.306640625` |
| Final training loss | `7.251703342` |
| Final validation loss | `6.297471311` |
| Final train perplexity | `1410.505375933` |
| Final validation perplexity | `543.196596924` |
| Training tokens processed | 6,144 |
| Validation tokens evaluated | 2,304 |
| Parameters changed | `True` |
| Checkpoint loaded | `True` |

The baseline is measured on the same validation source before the official optimizer steps. Any loss change is reported as observed; it is not manipulated or fabricated. A small dataset can overfit, so training and validation metrics must be interpreted together.

## Checkpoint

| Field | Value |
| --- | --- |
| Model version | `fodci-tiny-v1` |
| Local path | `artifacts/checkpoints/fodci-tiny-v1.pt` |
| Ignored by Git | `True` |
| Loaded successfully | `True` |

The checkpoint remains a local generated artifact and is intentionally not committed or pushed.
