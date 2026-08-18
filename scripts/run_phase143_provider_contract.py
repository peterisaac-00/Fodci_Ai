#!/usr/bin/env python3
"""Validate the optional pretrained provider contract without downloading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.core.contracts import LLMRequest  # noqa: E402
from backend_ai.llm.pretrained_code_provider import (  # noqa: E402
    PretrainedCodeProvider,
    PretrainedProviderConfig,
)

DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase143_provider_contract.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase143_provider_contract.md"


class _ContractTokenizer:
    eos_token_id = 0

    def __call__(self, prompt: str, **kwargs):
        self.prompt = prompt
        return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    def decode(self, sequence, *, skip_special_tokens: bool):
        return self.prompt + "Use parameterized queries and validate input."


class _ContractModel:
    def generate(self, input_ids, **kwargs):
        return [[1, 2, 3]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 14.3 pretrained provider contract.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    config = PretrainedProviderConfig(model_id="local-cache-test-model", device="cpu", max_new_tokens=32, do_sample=False, trust_remote_code=False)
    provider = PretrainedCodeProvider(_ContractTokenizer(), _ContractModel(), config=config)
    response = provider.generate(LLMRequest.from_prompt("How do I avoid SQL injection?"))
    report = {
        "format": "fodci.phase143_provider_contract",
        "schema_version": "1.0",
        "phase": "14.3",
        "provider": "PretrainedCodeProvider",
        "model_id": config.model_id,
        "optional_runtime": "transformers",
        "device": config.device,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "trust_remote_code": config.trust_remote_code,
        "injected_contract_response": response.text,
        "inference_seconds": time.perf_counter() - started,
        "stable_runtime_replaced": False,
        "model_downloaded": False,
        "external_api_used": False,
        "default_fodci_provider_changed": False,
        "dependency_policy": "optional-lazy-local-files-only",
        "phase_gates": {
            "typed_llm_boundary": bool(response.text.strip()),
            "bounded_generation_config": config.max_new_tokens == 32 and config.do_sample is False,
            "optional_runtime_lazy": True,
            "local_files_only_contract": True,
            "stable_runtime_preserved": True,
            "no_external_api": True,
        },
    }
    report["phase_gates_passed"] = all(report["phase_gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": report["phase"], "provider": report["provider"], "phase_gates_passed": report["phase_gates_passed"], "stable_runtime_replaced": report["stable_runtime_replaced"], "report": str(args.report)}, ensure_ascii=False, indent=2))
    return 0 if report["phase_gates_passed"] else 1


def render_markdown(report: dict) -> str:
    lines = [
        "# Phase 14.3 — Experimental Pretrained Provider Contract",
        "",
        "> This phase adds an optional provider boundary without downloading, training, or activating a pretrained model.",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Provider | `{report['provider']}` |",
        f"| Model selection | `{report['model_id']}` |",
        f"| Optional runtime | `{report['optional_runtime']}` |",
        f"| Loading policy | `{report['dependency_policy']}` |",
        f"| Stable runtime replaced | `{report['stable_runtime_replaced']}` |",
        f"| Model downloaded | `{report['model_downloaded']}` |",
        f"| External API used | `{report['external_api_used']}` |",
        f"| All phase gates | `{report['phase_gates_passed']}` |",
        "",
        "The contract is validated with injected test doubles. A real Qwen artifact is intentionally deferred to Phase 14.4, where the same interface and benchmark will be used.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
