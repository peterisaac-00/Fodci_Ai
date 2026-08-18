# Phase 14.4 — Qwen 0.5B Backend Benchmark

> This is an experimental local evaluation. It does not activate Qwen as the stable Fodci runtime.

## Experimental model and protocol

The selected model was `Qwen/Qwen2.5-Coder-0.5B-Instruct`, loaded from a local cache directory with Transformers, CPU-only inference, greedy decoding, `max_new_tokens=64`, `trust_remote_code=False`, and `local_files_only=True`. The downloaded artifact was the standard `model.safetensors` representation, not a 4-bit GGUF or GPTQ artifact. The report therefore records `none-fp16-safetensors`; the experiment must not be described as a Q4 benchmark.

| Field | Value |
|---|---:|
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| Parameters | 494,032,768 |
| Representation | FP16 safetensors; no Q4 quantization |
| Device | CPU |
| Cases | 24 |
| Completed cases | 24 |
| Stable runtime replaced | `False` |
| Default Fodci checkpoint untouched | `True` |
| Phase gates | `True` |

## Quantitative result

| Diagnostic | Qwen result | Fodci 11M baseline |
|---|---:|---:|
| Non-empty rate | 1.0000 | 1.0000 |
| Understandable heuristic rate | 0.9167 | 0.0000 |
| Average keyword coverage | 0.7188 | 0.0000 |
| Average repeated-token rate | 0.2366 | 0.3278 |
| Manual review required | `True` | `True` |

The result is a clear language-quality improvement over the stable Fodci baseline: Qwen produced readable English for most cases. However, the heuristic is not a correctness judge. Manual review identified likely technical problems, including suggesting Flask-style `jsonify` for FastAPI, answering an asynchronous database question with an `aiohttp` HTTP example, mixing password hashing and encryption terminology, and recommending `pytest-django` for a FastAPI test. Consequently, the result supports Qwen as a promising experimental language provider, but it does not justify blind activation or claim backend correctness.

## Resource and quantization limitation

The full-precision local run placed substantial memory pressure on the sandbox and took several minutes on CPU. A dynamic INT8 conversion attempt was terminated under the constrained environment and was not counted as a successful quantized result. A real Q4 CPU artifact therefore remains a separate experiment; no unsupported Q4 claim is made here.

The machine-readable report is `artifacts/evaluation/phase144_qwen_benchmark.json`. The runner is:

```text
PYTHONPATH=src python scripts/run_phase144_qwen_benchmark.py --max-new-tokens 64
```

The stable `fodci-testing-qa-v1` checkpoint remains unchanged. Phase 14.5 will add domain policy and output guards around the provider rather than modifying Qwen weights.
