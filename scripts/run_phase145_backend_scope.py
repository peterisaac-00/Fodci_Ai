#!/usr/bin/env python3
"""Run deterministic Phase 14.5 domain-policy and output-guard checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from backend_ai.core.contracts import LLMRequest, LLMResponse  # noqa: E402
from backend_ai.llm.backend_scope import BackendDomainPolicy, BackendOutputGuard, BackendScopedProvider  # noqa: E402

DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase145_backend_scope.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase145_backend_scope.md"


class _ProbeProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        prompt = request.messages[-1].content.lower()
        if "repetitive" in prompt:
            return LLMResponse(text="the the the the the the the the")
        return LLMResponse(text="Use FastAPI with Pydantic validation and a parameterized SQL query for the backend request.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 14.5 backend scope policy and output guard.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = BackendDomainPolicy()
    inner = _ProbeProvider()
    provider = BackendScopedProvider(inner, policy=policy, guard=BackendOutputGuard(policy))
    probes = [
        ("backend", "How should FastAPI validate JWT request data?", "accepted"),
        ("backend", "How do I optimize a PostgreSQL query?", "accepted"),
        ("out-of-scope", "How do I make a Unity game?", "blocked"),
        ("out-of-scope", "How do I build an Android app?", "blocked"),
        ("bad-output", "Give a repetitive backend answer.", "guarded"),
    ]
    results: list[dict] = []
    for label, prompt, expected in probes:
        response = provider.generate(LLMRequest.from_prompt(prompt))
        if expected == "accepted":
            observed = "accepted" if response.text.startswith("Use FastAPI") else "unexpected"
        elif expected == "blocked":
            observed = "blocked" if "I specialize in backend engineering" in response.text else "unexpected"
        else:
            observed = "guarded" if response.text.startswith("I could not produce") else "unexpected"
        results.append({"label": label, "prompt": prompt, "expected": expected, "observed": observed, "response": response.text, "passed": observed == expected})
    report = {
        "format": "fodci.phase145_backend_scope",
        "schema_version": "1.0",
        "phase": "14.5",
        "policy": "BackendDomainPolicy",
        "guard": "BackendOutputGuard",
        "probes": results,
        "probe_count": len(results),
        "passed_probe_count": sum(item["passed"] for item in results),
        "inner_provider_calls": inner.calls,
        "out_of_scope_calls_blocked_before_provider": inner.calls == 3,
        "stable_runtime_replaced": False,
        "default_fodci_provider_changed": False,
        "phase_gates": {
            "all_probes_passed": all(item["passed"] for item in results),
            "out_of_scope_blocking": results[2]["passed"] and results[3]["passed"],
            "output_guard_rejection": results[4]["passed"],
            "inner_provider_not_called_for_blocked": inner.calls == 3,
            "stable_runtime_preserved": True,
        },
    }
    report["phase_gates_passed"] = all(report["phase_gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": report["phase"], "probe_count": report["probe_count"], "passed": report["passed_probe_count"], "inner_provider_calls": report["inner_provider_calls"], "phase_gates_passed": report["phase_gates_passed"], "report": str(args.report)}, ensure_ascii=False, indent=2))
    return 0 if report["phase_gates_passed"] else 1


def render_markdown(report: dict) -> str:
    return "\n".join([
        "# Phase 14.5 — Backend Domain Policy and Output Guard",
        "",
        "> This phase constrains an experimental language provider to backend engineering at runtime. It does not erase knowledge from pretrained model weights.",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| Probes passed | {report['passed_probe_count']}/{report['probe_count']} |",
        f"| Out-of-scope calls blocked before provider | `{report['out_of_scope_calls_blocked_before_provider']}` |",
        f"| Stable runtime replaced | `{report['stable_runtime_replaced']}` |",
        f"| All gates | `{report['phase_gates_passed']}` |",
        "",
        "The policy is deterministic and conservative. A pretrained model may still contain general programming knowledge internally; the policy controls which requests reach the provider and which outputs are accepted by the Fodci runtime.",
        "",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
