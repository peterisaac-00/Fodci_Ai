# Phase 14.3 — Experimental Pretrained Provider Contract

> This phase adds an optional local provider boundary without downloading, training, or activating a pretrained model.

## Design

`PretrainedCodeProvider` implements the existing typed `LLMProvider` boundary. It accepts an explicitly selected model identifier, tokenizer, and causal model, formats typed messages into a bounded prompt, performs local generation, and returns `LLMResponse`. The provider is lazy: importing the default Fodci package does not import Transformers or download a model.

The explicit loading path uses `local_files_only=True` and `trust_remote_code=False`. The default Fodci provider remains unchanged, and no model artifact or external API was used in this phase.

| Field | Result |
|---|---|
| Provider | `PretrainedCodeProvider` |
| Model selection | `local-cache-test-model` contract double |
| Device | CPU |
| Maximum new tokens | 32 |
| Optional runtime | `transformers` |
| Loading policy | `optional-lazy-local-files-only` |
| Stable runtime replaced | `False` |
| Model downloaded | `False` |
| External API used | `False` |
| All phase gates | `True` |

The provider contract was verified with injected tokenizer/model doubles, including typed request formatting, bounded generation settings, invalid-conversation rejection, and response decoding. This avoids confusing interface validation with real model quality.

A concrete Qwen artifact is intentionally deferred to Phase 14.4. That phase will use this same provider boundary and the fixed Phase 14.1 benchmark.
