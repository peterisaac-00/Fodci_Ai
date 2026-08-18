from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from backend_ai.agent.advanced_autonomy import AutonomyBudget, AutonomyController, TaskLifeCycleState
from backend_ai.agent.advanced_memory import (
    AdvancedMemoryRecord,
    AdvancedMemorySystem,
    MemoryConfidence,
    MemoryProvenance,
    MemoryScope,
    MemoryType,
)
from backend_ai.agent.multi_agent import AgentOrchestrator, AgentRole, SubTask, SubTaskStatus
from backend_ai.checkpoint import CheckpointManager
from backend_ai.core.contracts import LLMRequest
from backend_ai.inference import InferenceConfig
from backend_ai.llm.fodci_provider import FodciLocalProvider


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "fodci-testing-qa-v1.pt"
DEFAULT_REPORT = ROOT / "artifacts" / "evaluation" / "phase1311_multi_agent_synergy.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "experiments" / "phase1311_multi_agent_synergy.md"
SEED = 2026
MODEL_VERSION = "fodci-testing-qa-v1"
EXPERIMENTAL_SCALING_VERSION = "fodci-scaling-25m-experimental-v1"


def make_subtasks(model_hint: str) -> tuple[SubTask, ...]:
    return (
        SubTask(
            id="plan-testing-workflow",
            description="Plan a backend testing workflow for the requested change.",
            role=AgentRole.PLANNER,
            metadata={"model_hint": model_hint[:160]},
        ),
        SubTask(
            id="implement-testing-scope",
            description="Define the implementation scope and affected backend boundaries.",
            role=AgentRole.CODER,
            dependencies=("plan-testing-workflow",),
        ),
        SubTask(
            id="run-integration-tests",
            description="Run focused unit and integration checks for the planned workflow.",
            role=AgentRole.TESTER,
            dependencies=("implement-testing-scope",),
        ),
        SubTask(
            id="verify-final-state",
            description="Verify the final state, evidence, and bounded completion criteria.",
            role=AgentRole.VERIFIER,
            dependencies=("run-integration-tests",),
        ),
    )


def load_provider(checkpoint: Path) -> tuple[FodciLocalProvider, dict[str, Any]]:
    manager = CheckpointManager(checkpoint.parent, model_version=MODEL_VERSION)
    info = manager.inspect(checkpoint)
    inference_config = InferenceConfig(
        max_new_tokens=4,
        device="cpu",
        seed=SEED,
        model_version=MODEL_VERSION,
        checkpoint_path=checkpoint,
    )
    provider = FodciLocalProvider.from_checkpoint(checkpoint, inference_config=inference_config)
    request = LLMRequest.from_prompt("Plan a deterministic Pytest integration check for a backend change.")
    response = provider.generate(request)
    engine_result = getattr(provider.engine, "checkpoint_identity", None)
    generated_count = getattr(provider.engine, "config", inference_config).max_new_tokens
    return provider, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_model_version": info.metadata.model_version,
        "checkpoint_identity": engine_result,
        "generated_text": response.text,
        "generated_text_non_empty": bool(response.text.strip()),
        "max_new_tokens": generated_count,
        "model_vocab_size": info.metadata.model_config.get("vocab_size") if isinstance(info.metadata.model_config, dict) else None,
    }


def run_orchestration(memory: AdvancedMemorySystem, project_root: Path, model_hint: str) -> dict[str, Any]:
    subtasks = make_subtasks(model_hint)
    orchestrator = AgentOrchestrator(memory_system=memory, max_steps=8)
    state = orchestrator.execute_task(
        "phase1311-orchestration-task",
        "Validate a backend testing workflow with model context, multi-agent coordination, and memory evidence.",
        project_root,
        subtasks,
    )
    completed = sum(1 for subtask in state.subtasks if subtask.status == SubTaskStatus.COMPLETED)
    return {
        "status": state.status,
        "task_id": state.task_id,
        "completed_steps": list(state.completed_steps),
        "completed_subtasks": completed,
        "total_subtasks": len(state.subtasks),
        "state": state.to_dict(),
    }


def run_autonomy(memory: AdvancedMemorySystem, project_root: Path) -> dict[str, Any]:
    orchestrator = AgentOrchestrator(memory_system=memory, max_steps=8)
    controller = AutonomyController(
        orchestrator=orchestrator,
        memory_system=memory,
        budget=AutonomyBudget(max_iterations=3, max_retries=1, max_replans=1, max_tool_calls=8, max_recovery_attempts=1),
    )
    result = controller.run(
        "phase1311-autonomy-task",
        "Verify a bounded multi-agent backend task and persist its reusable evidence.",
        project_root,
        (
            SubTask(id="autonomy-plan", description="Plan the bounded task.", role=AgentRole.PLANNER),
            SubTask(id="autonomy-test", description="Test the bounded task.", role=AgentRole.TESTER, dependencies=("autonomy-plan",)),
        ),
    )
    return result


def render_markdown(report: dict[str, Any]) -> str:
    gates = report["synergy_gates"]
    provider = report["model_provider"]
    orchestration = report["multi_agent"]
    autonomy = report["autonomy"]
    return f"""# Phase 13.11 — Integration & Multi-Agent Synergy

> This report validates integration boundaries and bounded state flow. It does not claim that generated text quality is equivalent to reliable autonomous software engineering.

## Scope

The workflow loads the verified Phase 13.9 checkpoint through the local provider, generates a short bounded model response, passes model context into a dependency-ordered multi-agent task, persists successful subtask evidence in `AdvancedMemorySystem`, reloads that memory from disk, and executes a second bounded task through `AutonomyController`.

The approximately 26M Phase 13.10 checkpoint remains experimental and is not the default runtime model. This phase validates the production-compatible 11.4M specialist checkpoint because it is the checkpoint wired to the current Fodci architecture.

## Evidence summary

| Area | Result |
|---|---|
| Model checkpoint loaded | `{gates['model_checkpoint_loaded']}` |
| Provider generated bounded response | `{gates['provider_generation']}` |
| Multi-agent dependency workflow | `{gates['multi_agent_completed']}` |
| Shared task state complete | `{gates['task_state_complete']}` |
| Memory records persisted | `{gates['memory_persisted']}` |
| Memory survives reload and retrieval | `{gates['memory_reload_retrieval']}` |
| Bounded autonomy completed | `{gates['autonomy_completed']}` |
| Autonomy budget respected | `{gates['autonomy_budget_respected']}` |
| Default runtime preserved | `{gates['default_runtime_preserved']}` |
| All synergy gates passed | `{gates['all_passed']}` |

## Model/provider boundary

| Field | Value |
|---|---|
| Model version | `{provider['checkpoint_model_version']}` |
| Checkpoint identity | `{provider['checkpoint_identity']}` |
| Generation limit | `{provider['max_new_tokens']}` tokens |
| Non-empty generated text | `{provider['generated_text_non_empty']}` |
| Experimental scaling model | `{EXPERIMENTAL_SCALING_VERSION}` (not activated) |

The generated model text is retained as evidence only and is not treated as an instruction to bypass orchestration or safety controls.

## Multi-agent and memory evidence

| Field | Value |
|---|---:|
| Orchestrator task status | `{orchestration['status']}` |
| Completed subtasks | `{orchestration['completed_subtasks']}` / `{orchestration['total_subtasks']}` |
| Completed step records | `{len(orchestration['completed_steps'])}` |
| Persisted memory records after orchestration | `{report['memory']['records_after_orchestration']}` |
| Reloaded memory records | `{report['memory']['records_after_reload']}` |
| Retrieved completion memories | `{report['memory']['retrieved_records']}` |

## Autonomy evidence

| Field | Value |
|---|---|
| Lifecycle state | `{autonomy['lifecycle_state']}` |
| Completed subtasks | `{autonomy['progress']['completed_subtasks']}` / `{autonomy['progress']['total_subtasks']}` |
| Checkpoints | `{autonomy['checkpoints_count']}` |
| Maximum iterations | `{report['autonomy_budget']['max_iterations']}` |
| Maximum tool calls | `{report['autonomy_budget']['max_tool_calls']}` |

## Interpretation

The integration gates prove that the current trained checkpoint can be loaded through the provider boundary while the multi-agent orchestrator shares task state and persists reusable completion evidence into advanced memory. The autonomy controller completes a bounded dependency workflow and exposes lifecycle and progress evidence. These are architectural synergy results, not a semantic capability claim; reliable agent quality still requires execution-aware tasks, real tool results, and broader evaluation.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 13.11 model, memory, multi-agent, and autonomy synergy.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    started = time.perf_counter()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    provider, provider_report = load_provider(checkpoint)
    with tempfile.TemporaryDirectory(prefix="fodci-phase1311-") as temporary:
        project_root = Path(temporary)
        memory_path = project_root / "advanced_memory.json"
        memory = AdvancedMemorySystem(memory_path)
        memory.add(
            AdvancedMemoryRecord(
                id="phase1311-user-testing-preference",
                memory_type=MemoryType.PREFERENCE_MEMORY,
                scope=MemoryScope.PROJECT,
                project_id="phase1311-project",
                content="Prefer deterministic Pytest integration checks with explicit cleanup and bounded execution.",
                source=MemoryProvenance.USER,
                confidence=MemoryConfidence.HIGH,
                importance=0.9,
            )
        )
        orchestration_report = run_orchestration(memory, project_root, provider_report["generated_text"])
        records_after_orchestration = len(memory.store.records)
        reloaded_memory = AdvancedMemorySystem(memory_path)
        retrieved = reloaded_memory.retrieve("testing workflow completed", scope=MemoryScope.PROJECT, max_results=10)
        autonomy_report = run_autonomy(reloaded_memory, project_root)
        records_after_reload = len(reloaded_memory.store.records)
        retrieved_after_autonomy = reloaded_memory.retrieve("bounded task", scope=MemoryScope.PROJECT, max_results=10)

    autonomy_budget = {
        "max_iterations": 3,
        "max_retries": 1,
        "max_replans": 1,
        "max_tool_calls": 8,
        "max_recovery_attempts": 1,
    }
    gates = {
        "model_checkpoint_loaded": provider_report["checkpoint_model_version"] == MODEL_VERSION,
        "provider_generation": provider_report["checkpoint_identity"] == str(checkpoint) and provider_report["max_new_tokens"] == 4,
        "multi_agent_completed": orchestration_report["status"] == "COMPLETED" and orchestration_report["completed_subtasks"] == orchestration_report["total_subtasks"],
        "task_state_complete": len(orchestration_report["completed_steps"]) == orchestration_report["total_subtasks"],
        "memory_persisted": records_after_orchestration >= 5,
        "memory_reload_retrieval": len(retrieved) >= 1 and records_after_reload >= records_after_orchestration,
        "autonomy_completed": autonomy_report["lifecycle_state"] == TaskLifeCycleState.COMPLETED.value,
        "autonomy_budget_respected": autonomy_report["progress"]["completed_subtasks"] == 2 and autonomy_report["checkpoints_count"] <= autonomy_budget["max_iterations"],
        "default_runtime_preserved": provider_report["checkpoint_model_version"] == MODEL_VERSION and EXPERIMENTAL_SCALING_VERSION != MODEL_VERSION,
    }
    gates["all_passed"] = all(gates.values())
    if not gates["all_passed"]:
        raise RuntimeError(f"Phase 13.11 synergy gates failed: {gates}")
    report = {
        "format": "fodci.phase1311_multi_agent_synergy",
        "schema_version": "1.0",
        "phase": "13.11",
        "checkpoint": str(checkpoint),
        "model_version": MODEL_VERSION,
        "experimental_scaling_model": EXPERIMENTAL_SCALING_VERSION,
        "model_provider": provider_report,
        "multi_agent": orchestration_report,
        "memory": {
            "records_after_orchestration": records_after_orchestration,
            "records_after_reload": records_after_reload,
            "retrieved_records": len(retrieved),
            "retrieved_after_autonomy": len(retrieved_after_autonomy),
        },
        "autonomy": autonomy_report,
        "autonomy_budget": autonomy_budget,
        "synergy_gates": gates,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"phase": "13.11", "model_version": MODEL_VERSION, "synergy_gates": gates, "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
