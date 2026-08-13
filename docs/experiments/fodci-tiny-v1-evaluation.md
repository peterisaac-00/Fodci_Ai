# Fodci Tiny v1 Evaluation

> **This is an early small-scale evaluation of a tiny model trained on a very small backend-focused corpus. It is not a capability or production-readiness claim.**

## Model

| Field | Value |
| --- | --- |
| Model version | `fodci-tiny-v1` |
| Parameters | 11,424,400 |
| Vocabulary size | 10,000 |
| Context length | 256 |
| Hidden size | 320 |
| Attention heads | 5 |
| Transformer blocks | 4 |
| Feed-forward size | 1,280 |
| Tokenizer version | 1 |
| Device | `cpu` |
| Seed | 2026 |

## Dataset

The same existing validation split was used for the random baseline and trained checkpoint. It was loaded through `FodciDatasetPipeline`; no training examples or external data were used during evaluation.

| Field | Value |
| --- | --- |
| Path | `data/fodci_tiny_v1/validation` |
| Split | `validation` |
| Documents | 2 |
| Tokens including EOS | 2484 |
| Evaluation examples | 9 |
| Dataset hash | `c994d8b4d2a7892224a812bffe59b20e420529cc0f7f9a170794ddd92674cec3` |
| Files | `data/fodci_tiny_v1/validation/api_validation.py, data/fodci_tiny_v1/validation/config_validation.py` |

## Evaluation

| Field | Value |
| --- | --- |
| Batch size | 2 |
| Device | `cpu` |
| Baseline evaluation seconds | `0.4412` |
| Trained evaluation seconds | `0.4000` |
| Parameters changed during evaluation | `False` |
| Optimizer changed during evaluation | `False` |

Both evaluations use `model.eval()` and `torch.no_grad()`. The evaluator does not call `backward()` or an optimizer step.

## Results

| Metric | Random baseline | Trained checkpoint | Difference / improvement |
| --- | ---: | ---: | ---: |
| Loss | `9.340518210` | `6.297471311` | delta `-3.043046898` |
| Perplexity | `11390.309279467` | `543.196596924` | delta `-10847.112682543` |
| Examples | 5 | 5 | same split |
| Tokens | 2304 | 2304 | same split |

Measured loss improvement: **3.043046898**, or **32.58%** relative to the random baseline. Measured perplexity improvement: **10847.112682543**, or **95.23%**.

## Checkpoint comparison

| Field | Value |
| --- | --- |
| Checkpoint path | `/home/ubuntu/backend-ai/artifacts/checkpoints/fodci-tiny-v1.pt` |
| Checkpoint identifier | `fodci-tiny-v1` |
| Epoch | 1 |
| Global step | 12 |
| Compatibility validation | `True` |
| Independently inspected metadata | `True` |
| Available valid checkpoints | 1 |
| Best checkpoint identifier | `fodci-tiny-v1` |

## Interpretation

This comparison shows the measured language-model objective on one fixed validation split before and after the small Fodci Tiny v1 training run. It does not establish that Fodci is intelligent, understands programming, generalizes beyond the corpus, or is production ready. The dataset is intentionally small, and any future interpretation must consider possible overfitting.
