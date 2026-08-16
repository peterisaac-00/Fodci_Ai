from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from backend_ai.evaluation import (
    AllowedScope,
    BenchmarkConfig,
    BenchmarkExecutionResult,
    BenchmarkRequest,
    BenchmarkRunner,
    BenchmarkStatus,
    BenchmarkTaskStatus,
    EvaluationConstraint,
    EvaluationDifficulty,
    EvaluationTask,
    EvaluationTaskCategory,
    EvaluationTestType,
    ExpectedArea,
    ExpectedAreaType,
    ExpectedBehavior,
    GroundTruth,
    ProjectDefinition,
    Requirement,
    SuccessCriterion,
    SuccessCriterionType,
    TestDefinition,
)


def task(task_id: str = "EVAL-001") -> EvaluationTask:
    return EvaluationTask(
        task_id=task_id,
        title="Fix addition",
        description="Fix the broken addition function.",
        version="1.0",
        category=EvaluationTaskCategory.BUG_FIX,
        difficulty=EvaluationDifficulty.EASY,
        project_definition=ProjectDefinition(project_type="backend", language="Python", runtime="Python 3.12", test_framework="pytest"),
        user_intent="Make addition correct.",
        requirements=(Requirement("REQ-001", "addition returns the arithmetic sum"),),
        expected_behaviors=(ExpectedBehavior("BEH-001", "add(2, 3)", "call add", "5", "5"),),
        allowed_scope=AllowedScope(allowed_files=("src/value.py",), allowed_change_types=("EDIT",), forbidden_paths=(".env",)),
        expected_areas=(ExpectedArea("addition implementation", ("src/value.py",), ExpectedAreaType.REQUIRED_CHANGE),),
        tests=(TestDefinition("TEST-001", "addition test", EvaluationTestType.UNIT, "tests/test_value.py", True, "PASS", ("REQ-001",), ("BEH-001",)),),
        success_criteria=(SuccessCriterion("CRIT-001", "test passes", SuccessCriterionType.TEST_PASS, True, "PASS", test_ids=("TEST-001",)),),
        ground_truth=GroundTruth(expected_behavior_ids=("BEH-001",), required_outcomes=("sum is correct",)),
        constraints=EvaluationConstraint(max_files_expected=1),
    )


class Runtime:
    def __init__(self, status: BenchmarkTaskStatus = BenchmarkTaskStatus.PASSED, *, mutate: bool = True, error: Exception | None = None) -> None:
        self.status = status
        self.mutate = mutate
        self.error = error
        self.calls: list[Path] = []

    def execute(self, evaluation_task, workspace_root: Path, *, max_wall_time: float) -> BenchmarkExecutionResult:
        self.calls.append(workspace_root)
        if self.error is not None:
            raise self.error
        source = workspace_root / "src" / "value.py"
        if self.mutate:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        return BenchmarkExecutionResult(
            self.status,
            execution_status=self.status.value,
            termination_reason=self.status.value,
            test_evidence={"status": "PASS" if self.status is BenchmarkTaskStatus.PASSED else "FAIL", "passed": 1},
            completion_evidence={"status": "COMPLETE"},
            final_verification_evidence={"status": "VERIFIED" if self.status is BenchmarkTaskStatus.PASSED else "NOT_VERIFIED"},
            stop_condition_evidence={"status": "DONE" if self.status is BenchmarkTaskStatus.PASSED else "FAILED"},
            failure_information=("password=secret-token",) if self.status is not BenchmarkTaskStatus.PASSED else (),
            logs=("authorization=Bearer-secret-token", "bounded log"),
            tests_requested=True,
            tests_executed=True,
        )


def fixture(_task: EvaluationTask, root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "value.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")


def request(runtime: Runtime, tasks: tuple[EvaluationTask, ...] = (task(),), **config_kwargs) -> BenchmarkRequest:
    return BenchmarkRequest(tasks, config=BenchmarkConfig(**config_kwargs), runtime=runtime, fixture_provider=fixture)


def test_valid_config_and_successful_lifecycle_collect_evidence() -> None:
    runtime = Runtime()
    result = BenchmarkRunner().run(request(runtime))
    assert result.status is BenchmarkStatus.COMPLETED
    assert result.summary.total_tasks == 1
    assert result.summary.completed_tasks == 1
    run = result.task_runs[0]
    assert run.status is BenchmarkTaskStatus.PASSED
    assert run.evidence.execution_started and run.evidence.execution_completed
    assert run.evidence.changed_paths == ("src/value.py",)
    assert run.evidence.expected_paths_touched == ("src/value.py",)
    assert run.evidence.tests_executed is True
    assert run.evidence.completion_evidence == {"status": "COMPLETE"}
    assert run.evidence.final_verification_evidence == {"status": "VERIFIED"}
    assert run.evidence.failure_information == ()
    assert runtime.calls[0] != Path.cwd()


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(max_tasks=0)
    with pytest.raises(ValueError):
        BenchmarkConfig(max_total_wall_time=-1)
    with pytest.raises(ValueError):
        BenchmarkConfig(max_artifact_bytes=0)


def test_empty_collection_and_duplicate_ids_are_structured_failures() -> None:
    runtime = Runtime()
    empty = BenchmarkRunner().run(BenchmarkRequest((), runtime=runtime))
    assert empty.status is BenchmarkStatus.FAILED
    duplicate = BenchmarkRunner().run(BenchmarkRequest((task(), task()), runtime=runtime))
    assert duplicate.status is BenchmarkStatus.FAILED
    assert "task IDs" in duplicate.warnings[0]


def test_missing_runtime_is_not_silently_executed() -> None:
    result = BenchmarkRunner().run(BenchmarkRequest((task(),)))
    assert result.status is BenchmarkStatus.FAILED
    assert "runtime adapter" in result.warnings[0]


def test_task_validation_failure_is_recorded_without_workspace_execution() -> None:
    invalid = EvaluationTask(task_id="bad")
    runtime = Runtime()
    result = BenchmarkRunner().run(BenchmarkRequest((invalid,), runtime=runtime))
    assert result.task_runs[0].status is BenchmarkTaskStatus.FAILED
    assert result.task_runs[0].evidence.execution_started is False
    assert runtime.calls == []


def test_isolation_creates_distinct_workspaces_and_cleans_them() -> None:
    runtime = Runtime()
    tasks = (task("EVAL-001"), task("EVAL-002"))
    result = BenchmarkRunner().run(request(runtime, tasks=tasks))
    assert result.status is BenchmarkStatus.COMPLETED
    assert len(runtime.calls) == 2
    assert runtime.calls[0] != runtime.calls[1]
    assert not runtime.calls[0].exists()
    assert not runtime.calls[1].exists()


def test_cleanup_can_be_disabled_for_artifact_inspection() -> None:
    runtime = Runtime()
    result = BenchmarkRunner().run(request(runtime, cleanup_workspaces=False))
    workspace = runtime.calls[0]
    assert workspace.exists()
    shutil.rmtree(workspace.parent, ignore_errors=True)
    assert result.task_runs[0].evidence.cleanup_status == "preserved"


def test_fail_fast_stops_after_terminal_failure() -> None:
    runtime = Runtime(BenchmarkTaskStatus.FAILED, mutate=False)
    result = BenchmarkRunner().run(request(runtime, tasks=(task("EVAL-001"), task("EVAL-002")), fail_fast=True))
    assert len(result.task_runs) == 2
    assert result.task_runs[1].status is BenchmarkTaskStatus.SKIPPED
    assert result.summary.skipped_tasks == 1
    assert result.fail_fast_triggered is True
    assert result.fail_fast_task_id == "EVAL-001"
    assert result.status is BenchmarkStatus.PARTIAL


def test_continue_on_failure_runs_remaining_tasks() -> None:
    class SequenceRuntime(Runtime):
        def __init__(self) -> None:
            super().__init__()
            self.index = 0

        def execute(self, evaluation_task, workspace_root: Path, *, max_wall_time: float) -> BenchmarkExecutionResult:
            self.calls.append(workspace_root)
            self.index += 1
            status = BenchmarkTaskStatus.FAILED if self.index == 1 else BenchmarkTaskStatus.PASSED
            return BenchmarkExecutionResult(status, execution_status=status.value, termination_reason=status.value)

    runtime = SequenceRuntime()
    result = BenchmarkRunner().run(request(runtime, tasks=(task("EVAL-001"), task("EVAL-002")), fail_fast=False))
    assert len(result.task_runs) == 2
    assert result.task_runs[0].status is BenchmarkTaskStatus.FAILED
    assert result.task_runs[1].status is BenchmarkTaskStatus.PASSED
    assert result.status is BenchmarkStatus.PARTIAL


def test_continue_on_task_failure_false_stops_without_fail_fast_flag() -> None:
    runtime = Runtime(BenchmarkTaskStatus.BLOCKED, mutate=False)
    result = BenchmarkRunner().run(request(runtime, tasks=(task("EVAL-001"), task("EVAL-002")), continue_on_task_failure=False))
    assert len(result.task_runs) == 2
    assert result.task_runs[1].status is BenchmarkTaskStatus.SKIPPED
    assert result.fail_fast_triggered is True


@pytest.mark.parametrize("status", [BenchmarkTaskStatus.BLOCKED, BenchmarkTaskStatus.UNAVAILABLE, BenchmarkTaskStatus.TIMED_OUT, BenchmarkTaskStatus.INCOMPLETE_EVIDENCE])
def test_task_statuses_are_preserved(status: BenchmarkTaskStatus) -> None:
    result = BenchmarkRunner().run(request(Runtime(status, mutate=False)))
    assert result.task_runs[0].status is status
    assert result.summary.total_tasks == 1


def test_timeout_and_infrastructure_failures_are_distinguished() -> None:
    timeout = BenchmarkRunner().run(request(Runtime(error=TimeoutError("task timeout"))))
    assert timeout.task_runs[0].status is BenchmarkTaskStatus.TIMED_OUT
    infra = BenchmarkRunner().run(request(Runtime(error=RuntimeError("runner broken"))))
    assert infra.task_runs[0].status is BenchmarkTaskStatus.INFRASTRUCTURE_ERROR
    assert infra.summary.infrastructure_failures == 1


def test_evidence_redacts_secrets_and_bounds_artifacts() -> None:
    class ArtifactRuntime(Runtime):
        def execute(self, evaluation_task, workspace_root: Path, *, max_wall_time: float) -> BenchmarkExecutionResult:
            self.calls.append(workspace_root)
            (workspace_root / "artifact.bin").write_bytes(b"x" * 100)
            return super().execute(evaluation_task, workspace_root, max_wall_time=max_wall_time)

    runtime = ArtifactRuntime()
    result = BenchmarkRunner().run(request(runtime, collect_artifacts=True, max_artifact_bytes=20, max_log_chars=30))
    evidence = result.task_runs[0].evidence
    assert all("secret-token" not in item for item in evidence.bounded_logs)
    assert all("secret-token" not in item for item in evidence.failure_information)
    assert evidence.artifacts == () or sum((runtime.calls[0] / item).stat().st_size for item in evidence.artifacts if (runtime.calls[0] / item).exists()) <= 20


def test_deterministic_order_and_serialization() -> None:
    runtime = Runtime()
    result = BenchmarkRunner().run(request(runtime, tasks=(task("EVAL-002"), task("EVAL-001"))))
    assert result.deterministic_metadata["task_order"] == ("EVAL-002", "EVAL-001")
    assert result.to_json() == result.to_json()
    assert "scoring" in result.to_json()


def test_benchmark_does_not_expose_quality_score_or_percentage() -> None:
    result = BenchmarkRunner().run(request(Runtime()))
    payload = result.to_dict()
    assert "score" not in payload
    assert "percentage" not in payload
    assert "weighted_score" not in payload


def test_project_root_is_copied_into_isolated_workspace(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("fixture", encoding="utf-8")
    runtime = Runtime(mutate=False)
    result = BenchmarkRunner().run(BenchmarkRequest((task(),), project_root=tmp_path, config=BenchmarkConfig(cleanup_workspaces=False), runtime=runtime))
    assert result.status is BenchmarkStatus.COMPLETED
    assert (runtime.calls[0] / "existing.txt").exists()
    shutil.rmtree(runtime.calls[0].parent, ignore_errors=True)
