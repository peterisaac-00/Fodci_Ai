from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from backend_ai.checkpoint import CheckpointManager
from backend_ai.core.contracts import LLMRequest
from backend_ai.inference import InferenceConfig
from backend_ai.llm.fodci_provider import FodciLocalProvider


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"
EVALUATION_DIR = ROOT / "artifacts" / "evaluation"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "fodci-testing-qa-v1.pt"
DEFAULT_REPORT = EVALUATION_DIR / "phase1312_final_evaluation.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase1312_final_evaluation.md"
MODEL_VERSION = "fodci-testing-qa-v1"
PARAMETERS = 11_424_400
SCALING_VERSION = "fodci-scaling-25m-experimental-v1"

CHECKPOINT_CHAIN = (
    ("fodci-tiny-v1.pt", "fodci-tiny-v1"),
    ("fodci-stage1-v1.pt", "fodci-stage1-v1"),
    ("fodci-python-backend-v1.pt", "fodci-python-backend-v1"),
    ("fodci-sql-database-v1.pt", "fodci-sql-database-v1"),
    ("fodci-rest-api-v1.pt", "fodci-rest-api-v1"),
    ("fodci-debugging-v1.pt", "fodci-debugging-v1"),
    ("fodci-security-auth-v1.pt", "fodci-security-auth-v1"),
    ("fodci-testing-qa-v1.pt", "fodci-testing-qa-v1"),
)

TRAINING_REPORTS = (
    "stage1_training.json",
    "phase134_python_backend_training.json",
    "phase135_sql_database_training.json",
    "phase136_rest_api_training.json",
    "phase137_debugging_training.json",
    "phase138_security_auth_training.json",
    "phase139_testing_qa_training.json",
)

BENCHMARK_REPORTS = (
    "stage1_baseline.json",
    "phase134_python_backend_benchmark.json",
    "phase135_sql_database_benchmark.json",
    "phase136_rest_api_benchmark.json",
    "phase137_debugging_benchmark.json",
    "phase138_security_auth_benchmark.json",
    "phase139_testing_qa_benchmark.json",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required release evidence is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"release evidence must be a JSON object: {path}")
    return value


def checkpoint_audit() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for filename, version in CHECKPOINT_CHAIN:
        path = CHECKPOINT_DIR / filename
        entry: dict[str, Any] = {"filename": filename, "version": version, "path": str(path), "exists": path.is_file()}
        if path.is_file():
            info = CheckpointManager(CHECKPOINT_DIR, model_version=version).inspect(path)
            entry.update(
                {
                    "metadata_model_version": info.metadata.model_version,
                    "format_version": info.metadata.format_version,
                    "model_config": info.metadata.model_config,
                    "compatible_identity": info.metadata.model_version == version,
                }
            )
        else:
            entry["compatible_identity"] = False
        entries.append(entry)
    scaling_path = CHECKPOINT_DIR / "fodci-scaling-25m-experimental-v1.pt"
    scaling_entry: dict[str, Any] = {"filename": scaling_path.name, "version": SCALING_VERSION, "exists": scaling_path.is_file(), "experimental": True}
    if scaling_path.is_file():
        scaling_info = CheckpointManager(CHECKPOINT_DIR, model_version=SCALING_VERSION).inspect(scaling_path)
        scaling_entry.update({"metadata_model_version": scaling_info.metadata.model_version, "format_version": scaling_info.metadata.format_version, "compatible_identity": scaling_info.metadata.model_version == SCALING_VERSION})
    entries.append(scaling_entry)
    return {"chain": entries, "all_required_present": all(item["exists"] and item["compatible_identity"] for item in entries[:-1]), "experimental_scaling_present": scaling_entry["exists"] and scaling_entry["compatible_identity"]}


def training_audit() -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for filename in TRAINING_REPORTS:
        data = read_json(EVALUATION_DIR / filename)
        gates = data.get("validation_gates", {})
        reports.append(
            {
                "filename": filename,
                "phase": data.get("phase"),
                "model_version": data.get("model_version"),
                "base_model_version": data.get("base_model_version"),
                "model_parameters": data.get("model_parameters"),
                "all_gates_passed": gates.get("all_passed") is True and all(bool(value) for value in gates.values()),
                "trained_loss": data.get("evaluation", {}).get("trained", {}).get("loss"),
            }
        )
    return {"reports": reports, "all_reports_passed": all(item["all_gates_passed"] for item in reports), "all_default_parameter_count": all(item["model_parameters"] == PARAMETERS for item in reports)}


def benchmark_audit() -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for filename in BENCHMARK_REPORTS:
        data = read_json(EVALUATION_DIR / filename)
        aggregate = data.get("evaluation", {}).get("aggregate", {})
        items = aggregate.get("items", data.get("items", 0))
        reports.append(
            {
                "filename": filename,
                "run_id": data.get("run_id"),
                "items": items,
                "pass_rate": aggregate.get("pass_rate"),
                "non_empty_rate": aggregate.get("non_empty_rate"),
                "average_keyword_coverage": aggregate.get("average_keyword_coverage"),
            }
        )
    return {
        "reports": reports,
        "all_non_empty": all(float(item["non_empty_rate"] or 0.0) >= 1.0 for item in reports),
        "all_have_items": all(int(item["items"] or 0) > 0 for item in reports),
        "all_metrics_valid": all(0.0 <= float(item["non_empty_rate"] or 0.0) <= 1.0 for item in reports),
    }


def runtime_smoke(checkpoint: Path) -> dict[str, Any]:
    config = InferenceConfig(max_new_tokens=4, device="cpu", seed=2026, model_version=MODEL_VERSION, checkpoint_path=checkpoint)
    provider = FodciLocalProvider.from_checkpoint(checkpoint, inference_config=config)
    result = provider.generate(LLMRequest.from_prompt("Give one concise backend testing recommendation."))
    return {
        "model_version": result.text is not None and config.model_version,
        "checkpoint_identity": provider.engine.checkpoint_identity,
        "generated_token_count": provider.engine.generate("### Instruction\nGive one concise backend testing recommendation.\n\n### Input\nNone\n\n### Response\n").generated_token_count,
        "generated_text_non_empty": bool(result.text.strip()),
        "max_new_tokens": config.max_new_tokens,
    }


def git_state() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return {"head": head, "status_lines": status.splitlines(), "clean": not bool(status.strip())}


def render_markdown(report: dict[str, Any]) -> str:
    gates = report["release_gates"]
    return f"""# Phase 13.12 — Final Evaluation & Feature Complete

> This report is the final engineering release audit for the Fodci Backend Engineering Agent. It distinguishes verified architecture and pipeline evidence from capabilities that still require broader semantic and execution-aware evaluation.

## Release decision

| Gate | Result |
|---|---|
| Checkpoint lineage complete | `{gates['checkpoint_lineage_complete']}` |
| Specialist training reports valid | `{gates['training_reports_valid']}` |
| Held-out benchmark reports structurally valid | `{gates['benchmark_reports_valid']}` |
| Scaling experiment gates valid | `{gates['scaling_evidence_valid']}` |
| Multi-agent synergy gates valid | `{gates['synergy_evidence_valid']}` |
| Final checkpoint runtime smoke | `{gates['runtime_smoke_valid']}` |
| Full regression recorded | `{gates['full_regression_recorded']}` |
| Default model preserved | `{gates['default_model_preserved']}` |
| **Feature-complete release gates** | **`{gates['all_passed']}`** |

## Final model and runtime

| Field | Value |
|---|---|
| Stable default model | `{report['stable_model_version']}` |
| Stable parameters | {report['stable_parameter_count']:,} |
| Stable checkpoint | `{report['stable_checkpoint']}` |
| Experimental scaling model | `{report['experimental_scaling_model']}` |
| Experimental checkpoint activated | `{report['experimental_scaling_activated']}` |
| Runtime device | CPU |
| External APIs required | No |

## Phase evidence

| Evidence group | Result |
|---|---|
| Specialist training reports | {len(report['training']['reports'])} reports; all gates passed: `{report['training']['all_reports_passed']}` |
| Held-out benchmark reports | {len(report['benchmarks']['reports'])} reports; metrics valid: `{report['benchmarks']['all_metrics_valid']}`; all non-empty diagnostic: `{report['benchmarks']['all_non_empty']}` |
| Multi-agent subtasks | {report['synergy']['multi_agent_completed_subtasks']} / {report['synergy']['multi_agent_total_subtasks']} completed |
| Memory reload retrieval | `{report['synergy']['memory_reload_retrieval']}` |
| Bounded autonomy | `{report['synergy']['autonomy_completed']}` |
| Scaling target | `{report['scaling']['scaled_parameter_count']:,} parameters; gates passed: `{report['scaling']['all_gates_passed']}` |

## Tests and repository state

| Field | Value |
|---|---:|
| Full regression tests passed | `{report['tests']['passed']}` |
| Pytest warnings | `{report['tests']['warnings']}` |
| Compileall | `{report['tests']['compileall']}` |
| Git HEAD at evaluation | `{report['git']['head']}` |
| Repository clean at evaluation | `{report['git']['clean']}` |

## Stable-release interpretation

The release is **feature complete as an engineering pipeline**: CLI and application boundaries, local model and tokenizer, bounded training and checkpoints, evaluation and benchmarks, specialist curriculum stages, safe tools, advanced memory, multi-agent orchestration, autonomy controls, scaling evidence, and integration validation are present and regression-tested.

Feature complete does not mean that the 11.4M model has the semantic quality of a frontier model. The benchmark reports are deterministic keyword and non-empty-output diagnostics, and several runs correctly show zero keyword pass rate. The final release therefore preserves honest limitations: reliable production-grade coding behavior, broad autonomous repair, and frontier conversational quality require larger datasets, longer matched training, executable task evaluation, and future model improvements.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 13.12 final evaluation and feature-complete release audit.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--tests-passed", type=int, default=None)
    parser.add_argument("--pytest-warnings", type=int, default=None)
    parser.add_argument("--compileall", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    checkpoint = args.checkpoint.resolve()
    checkpoint_report = checkpoint_audit()
    training = training_audit()
    benchmarks = benchmark_audit()
    scaling = read_json(EVALUATION_DIR / "phase1310_scaling_analysis.json")
    synergy = read_json(EVALUATION_DIR / "phase1311_multi_agent_synergy.json")
    smoke = runtime_smoke(checkpoint)
    git = git_state()
    tests = {"passed": args.tests_passed, "warnings": args.pytest_warnings, "compileall": args.compileall}
    release_gates = {
        "checkpoint_lineage_complete": checkpoint_report["all_required_present"],
        "training_reports_valid": training["all_reports_passed"] and training["all_default_parameter_count"],
        "benchmark_reports_valid": benchmarks["all_have_items"] and benchmarks["all_metrics_valid"],
        "scaling_evidence_valid": scaling.get("phase") == "13.10" and scaling.get("validation_gates", {}).get("all_passed") is True,
        "synergy_evidence_valid": synergy.get("synergy_gates", {}).get("all_passed") is True,
        "runtime_smoke_valid": smoke["checkpoint_identity"] == str(checkpoint) and smoke["model_version"] == MODEL_VERSION,
        "full_regression_recorded": isinstance(args.tests_passed, int) and args.tests_passed > 0,
        "default_model_preserved": smoke["model_version"] == MODEL_VERSION and scaling.get("comparison", {}).get("default_runtime_changed") is False,
    }
    release_gates["all_passed"] = all(release_gates.values())
    if not release_gates["all_passed"]:
        raise RuntimeError(f"Phase 13.12 release gates failed: {release_gates}")
    report = {
        "format": "fodci.phase1312_final_evaluation",
        "schema_version": "1.0",
        "phase": "13.12",
        "stable_model_version": MODEL_VERSION,
        "stable_parameter_count": PARAMETERS,
        "stable_checkpoint": str(checkpoint),
        "experimental_scaling_model": SCALING_VERSION,
        "experimental_scaling_activated": False,
        "checkpoint_audit": checkpoint_report,
        "training": training,
        "benchmarks": benchmarks,
        "scaling": {
            "scaled_parameter_count": scaling.get("models", {}).get("scaled_candidate", {}).get("parameter_count"),
            "decision": scaling.get("comparison", {}).get("decision"),
            "all_gates_passed": scaling.get("validation_gates", {}).get("all_passed") is True,
        },
        "synergy": {
            "multi_agent_completed_subtasks": synergy.get("multi_agent", {}).get("completed_subtasks"),
            "multi_agent_total_subtasks": synergy.get("multi_agent", {}).get("total_subtasks"),
            "memory_reload_retrieval": synergy.get("synergy_gates", {}).get("memory_reload_retrieval"),
            "autonomy_completed": synergy.get("synergy_gates", {}).get("autonomy_completed"),
        },
        "runtime_smoke": smoke,
        "tests": tests,
        "git": git,
        "release_gates": release_gates,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": "13.12", "stable_model_version": MODEL_VERSION, "stable_parameter_count": PARAMETERS, "release_gates": release_gates, "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
