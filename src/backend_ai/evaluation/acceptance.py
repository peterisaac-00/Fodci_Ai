"""Phase 11.6 regression analysis and fail-closed model acceptance.

The evaluator consumes persisted Phase 11.5 evidence.  It never reruns a
benchmark, trains a model, changes weights, promotes a registry pointer, or
changes the Agent runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

from backend_ai.evaluation.benchmark import (
    BENCHMARK_PROTOCOL_VERSION,
    BenchmarkComparison,
    BenchmarkDataset,
    BenchmarkRun,
    BenchmarkStatus,
    MetricDirection,
)
from backend_ai.model_artifact import ModelArtifact
from backend_ai.evaluation.regression import RegressionSeverity


ACCEPTANCE_FORMAT = "fodci.model_acceptance"
ACCEPTANCE_SCHEMA_VERSION = "1.0"
ACCEPTANCE_POLICY_VERSION = "11.6-v1"
DEFAULT_ACCEPTANCE_STORE_PATH = Path("artifacts") / "evaluation" / "acceptance_reports.json"
_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SEVERITY_ORDER = {RegressionSeverity.NONE: 0, RegressionSeverity.LOW: 1, RegressionSeverity.MEDIUM: 2, RegressionSeverity.HIGH: 3, RegressionSeverity.CRITICAL: 4}


class AcceptanceError(ValueError):
    """Invalid acceptance input or policy."""


class AcceptanceConflictError(AcceptanceError):
    """Immutable acceptance report ID conflict."""


class AcceptanceDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INVALID_EVALUATION = "INVALID_EVALUATION"


class RegressionCategory(str, Enum):
    CAPABILITY = "CAPABILITY"
    METRIC = "METRIC"
    TOOL = "TOOL"
    DEBUGGING = "DEBUGGING"
    DOMAIN = "DOMAIN"
    OVERFITTING = "OVERFITTING"
    CONTAMINATION = "CONTAMINATION"
    REPRODUCIBILITY = "REPRODUCIBILITY"


@dataclass(frozen=True, slots=True)
class AcceptancePolicy:
    """Conservative, versioned thresholds; no threshold is hidden in code."""

    policy_version: str = ACCEPTANCE_POLICY_VERSION
    minimum_task_success_rate: float = 0.50
    minimum_test_pass_rate: float = 0.80
    minimum_tool_success_rate: float = 0.80
    minimum_error_recovery_rate: float = 0.50
    maximum_failure_rate: float = 0.20
    maximum_average_attempts: float = 8.0
    maximum_allowed_regression: int = 0
    minimum_required_improvement: float = 0.05
    minimum_improved_metrics: int = 2
    regression_tolerance: float = 0.01
    critical_regression_delta: float = 0.10
    maximum_overfitting_gap: float = 0.20
    require_held_out_test: bool = True
    reject_on_overfitting_gap: bool = True
    require_complete_evidence: bool = True
    require_reproducibility: bool = True
    critical_categories: tuple[str, ...] = ("API_ENDPOINT", "AUTHENTICATION", "DATABASE", "BUG_FIX", "TESTING", "SECURITY", "ARCHITECTURE")

    def __post_init__(self) -> None:
        if self.policy_version != ACCEPTANCE_POLICY_VERSION:
            raise AcceptanceError("unsupported acceptance policy version")
        for name in ("minimum_task_success_rate", "minimum_test_pass_rate", "minimum_tool_success_rate", "minimum_error_recovery_rate", "maximum_failure_rate"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise AcceptanceError(f"{name} must be a finite rate in [0, 1]")
        for name in ("maximum_average_attempts", "minimum_required_improvement", "regression_tolerance", "critical_regression_delta", "maximum_overfitting_gap"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or float(value) < 0:
                raise AcceptanceError(f"{name} must be a finite non-negative number")
        for name in ("maximum_allowed_regression", "minimum_improved_metrics"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AcceptanceError(f"{name} must be a non-negative integer")
        if self.minimum_improved_metrics < 2:
            raise AcceptanceError("minimum_improved_metrics must prevent single-metric acceptance")
        if not all(isinstance(item, str) and item.strip() for item in self.critical_categories):
            raise AcceptanceError("critical_categories must contain text")
        for name in ("require_held_out_test", "reject_on_overfitting_gap", "require_complete_evidence", "require_reproducibility"):
            if not isinstance(getattr(self, name), bool):
                raise AcceptanceError(f"{name} must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ModelRegressionFinding:
    finding_id: str
    category: RegressionCategory
    dimension: str
    scope: str
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    severity: RegressionSeverity
    critical: bool
    evidence_ids: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not self.finding_id.strip() or not self.dimension.strip() or not self.scope.strip():
            raise AcceptanceError("regression finding identity must contain text")
        if not isinstance(self.category, RegressionCategory) or not isinstance(self.severity, RegressionSeverity):
            raise AcceptanceError("regression finding category/severity is invalid")
        for value in (self.baseline_value, self.candidate_value, self.delta):
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
                raise AcceptanceError("regression finding values must be finite")
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))

    def to_dict(self) -> dict[str, Any]:
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ReproducibilityReport:
    valid: bool
    checks: Mapping[str, bool]
    missing: tuple[str, ...]
    mismatches: tuple[str, ...]
    fingerprints: Mapping[str, str | None]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", _freeze(self.checks))
        object.__setattr__(self, "fingerprints", _freeze(self.fingerprints))
        object.__setattr__(self, "missing", tuple(sorted(set(self.missing))))
        object.__setattr__(self, "mismatches", tuple(sorted(set(self.mismatches))))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))

    def to_dict(self) -> dict[str, Any]:
        return {name: _serialize(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class RegressionAnalysis:
    status: str
    findings: tuple[ModelRegressionFinding, ...]
    improved_metrics: tuple[str, ...]
    checked_metrics: tuple[str, ...]
    warnings: tuple[str, ...]
    highest_severity: RegressionSeverity

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL", "INCONCLUSIVE"}:
            raise AcceptanceError("regression analysis status is invalid")
        object.__setattr__(self, "findings", tuple(sorted(self.findings, key=lambda item: item.finding_id)))
        object.__setattr__(self, "improved_metrics", tuple(sorted(set(self.improved_metrics))))
        object.__setattr__(self, "checked_metrics", tuple(sorted(set(self.checked_metrics))))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "findings": [item.to_dict() for item in self.findings], "improved_metrics": list(self.improved_metrics), "checked_metrics": list(self.checked_metrics), "warnings": list(self.warnings), "highest_severity": self.highest_severity.value}


@dataclass(frozen=True, slots=True)
class AcceptanceRequest:
    evaluation_id: str
    comparison: BenchmarkComparison
    base_run: BenchmarkRun
    candidate_run: BenchmarkRun
    dataset: BenchmarkDataset
    policy: AcceptancePolicy = AcceptancePolicy()
    candidate_artifact: ModelArtifact | None = None
    candidate_training_config: Mapping[str, Any] | None = None
    training_dataset_fingerprint: str | None = None
    training_source_record_ids: tuple[str, ...] = ()
    validation_success_rate: float | None = None
    held_out_test: bool = True

    def __post_init__(self) -> None:
        if not self.evaluation_id.strip():
            raise AcceptanceError("evaluation_id must contain text")
        for name in ("comparison", "base_run", "candidate_run", "dataset", "policy"):
            if getattr(self, name) is None:
                raise AcceptanceError(f"{name} is required")
        if self.candidate_training_config is not None:
            object.__setattr__(self, "candidate_training_config", _freeze(self.candidate_training_config))
        object.__setattr__(self, "training_source_record_ids", tuple(sorted(set(self.training_source_record_ids))))
        if self.training_dataset_fingerprint is not None and not _fingerprint(self.training_dataset_fingerprint):
            raise AcceptanceError("training_dataset_fingerprint is invalid")
        if self.validation_success_rate is not None and (not isinstance(self.validation_success_rate, (int, float)) or not 0.0 <= float(self.validation_success_rate) <= 1.0):
            raise AcceptanceError("validation_success_rate must be in [0, 1]")
        if not isinstance(self.held_out_test, bool):
            raise AcceptanceError("held_out_test must be boolean")


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    format: str
    schema_version: str
    evaluation_id: str
    decision: AcceptanceDecision
    reason: str
    model_version: str
    base_model_version: str
    dataset_version: str
    benchmark_version: str
    benchmark_fingerprint: str
    metrics: Mapping[str, Any]
    regressions: tuple[ModelRegressionFinding, ...]
    warnings: tuple[str, ...]
    policy: AcceptancePolicy
    reproducibility: ReproducibilityReport
    fingerprints: Mapping[str, str | None]
    lineage: Mapping[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        if self.format != ACCEPTANCE_FORMAT or self.schema_version != ACCEPTANCE_SCHEMA_VERSION or not self.evaluation_id.strip():
            raise AcceptanceError("acceptance report identity is invalid")
        if not isinstance(self.decision, AcceptanceDecision) or not _fingerprint(self.benchmark_fingerprint):
            raise AcceptanceError("acceptance report decision/fingerprint is invalid")
        if not isinstance(self.policy, AcceptancePolicy) or not isinstance(self.reproducibility, ReproducibilityReport):
            raise AcceptanceError("acceptance report policy/reproducibility is invalid")
        object.__setattr__(self, "metrics", _freeze(self.metrics))
        object.__setattr__(self, "fingerprints", _freeze(self.fingerprints))
        object.__setattr__(self, "lineage", _freeze(self.lineage))
        object.__setattr__(self, "regressions", tuple(sorted(self.regressions, key=lambda item: item.finding_id)))
        object.__setattr__(self, "warnings", tuple(sorted(set(self.warnings))))

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "schema_version": self.schema_version, "evaluation_id": self.evaluation_id, "decision": self.decision.value, "reason": self.reason, "model_version": self.model_version, "base_model_version": self.base_model_version, "dataset_version": self.dataset_version, "benchmark_version": self.benchmark_version, "benchmark_fingerprint": self.benchmark_fingerprint, "metrics": _thaw(self.metrics), "regressions": [item.to_dict() for item in self.regressions], "warnings": list(self.warnings), "policy": self.policy.to_dict(), "reproducibility": self.reproducibility.to_dict(), "fingerprints": _thaw(self.fingerprints), "lineage": _thaw(self.lineage), "created_at": self.created_at}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


class AcceptanceStore:
    """Atomic immutable local acceptance report store."""

    def __init__(self, path: Path | str | None = DEFAULT_ACCEPTANCE_STORE_PATH) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self._reports: dict[str, AcceptanceReport] = {}
        self._loaded_digest: str | None = None
        self.reload()

    def reload(self) -> None:
        if self.path is None:
            self._reports = {}
            self._loaded_digest = None
            return
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise AcceptanceError("acceptance store must not use symlinks")
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            self._reports = {}
            self._loaded_digest = None
            return
        except OSError as exc:
            raise AcceptanceError("acceptance store is unavailable") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
            if set(payload) != {"format", "schema_version", "reports"} or payload["format"] != ACCEPTANCE_FORMAT or payload["schema_version"] != ACCEPTANCE_SCHEMA_VERSION:
                raise AcceptanceError("acceptance store header is invalid")
            self._reports = {str(key): _report_from_dict(value) for key, value in sorted(payload["reports"].items())}
        except AcceptanceError:
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AcceptanceError("acceptance store is malformed") from exc
        self._loaded_digest = _digest(raw)

    def get(self, evaluation_id: str) -> AcceptanceReport | None:
        return self._reports.get(evaluation_id)

    def list_reports(self) -> tuple[AcceptanceReport, ...]:
        return tuple(self._reports[key] for key in sorted(self._reports))

    def save(self, report: AcceptanceReport) -> AcceptanceReport:
        existing = self._reports.get(report.evaluation_id)
        if existing is not None:
            if existing.to_json() == report.to_json():
                return existing
            raise AcceptanceConflictError("evaluation_id already has a different immutable acceptance report")
        self._reports[report.evaluation_id] = report
        try:
            self._persist()
        except Exception:
            self._reports.pop(report.evaluation_id, None)
            raise
        return report

    def _persist(self) -> None:
        if self.path is None:
            return
        if self.path.exists() and (self._loaded_digest is None or _digest(self.path.read_bytes()) != self._loaded_digest):
            raise AcceptanceConflictError("acceptance store changed since it was loaded")
        payload = {"format": ACCEPTANCE_FORMAT, "schema_version": ACCEPTANCE_SCHEMA_VERSION, "reports": {key: value.to_dict() for key, value in sorted(self._reports.items())}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (_canonical_json(payload) + "\n").encode("utf-8")
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            self._loaded_digest = _digest(encoded)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass


class ModelAcceptanceEvaluator:
    """Evaluate persisted benchmark evidence without rerunning it."""

    def evaluate(self, request: AcceptanceRequest) -> AcceptanceReport:
        reproducibility = self._reproducibility(request)
        analysis = self._regressions(request)
        metrics = _metrics_snapshot(request)
        warnings = list(reproducibility.warnings) + list(analysis.warnings)
        invalid_reasons = list(reproducibility.missing) + list(reproducibility.mismatches)
        if request.policy.require_complete_evidence:
            invalid_reasons.extend(self._completeness_failures(request))
        if invalid_reasons and request.policy.require_reproducibility:
            decision = AcceptanceDecision.INVALID_EVALUATION
            reason = "Evaluation is invalid or incomplete: " + "; ".join(sorted(set(invalid_reasons)))
        else:
            policy_failures = self._policy_failures(request, analysis)
            if policy_failures:
                decision = AcceptanceDecision.REJECT
                reason = "Acceptance policy failed: " + "; ".join(policy_failures)
            elif analysis.status != "PASS":
                decision = AcceptanceDecision.REJECT
                reason = "Regression analysis did not pass"
            else:
                decision = AcceptanceDecision.ACCEPT
                reason = "Candidate met the configured improvement, regression, threshold, held-out, and reproducibility requirements"
        lineage, fingerprints = _lineage(request)
        return AcceptanceReport(ACCEPTANCE_FORMAT, ACCEPTANCE_SCHEMA_VERSION, request.evaluation_id, decision, reason, request.candidate_run.model.model_version, request.base_run.model.model_version, request.dataset.dataset_version, request.dataset.benchmark_version, compute_benchmark_fingerprint(request.comparison), metrics, analysis.findings, tuple(sorted(set(warnings))), request.policy, reproducibility, fingerprints, lineage, _utc_now())

    def _completeness_failures(self, request: AcceptanceRequest) -> list[str]:
        failures: list[str] = []
        expected_ids = {task.task_id for task in request.dataset.tasks}
        for label, run in (("base", request.base_run), ("candidate", request.candidate_run)):
            if run.status is BenchmarkStatus.INVALID:
                failures.append(f"{label}_benchmark_status_invalid")
            if {item.task_id for item in run.task_results} != expected_ids:
                failures.append(f"{label}_task_set_incomplete")
        if request.comparison.base_run_id != request.base_run.run_id or request.comparison.candidate_run_id != request.candidate_run.run_id:
            failures.append("comparison_run_ids_do_not_match")
        if request.comparison.dataset_fingerprint != request.dataset.dataset_fingerprint:
            failures.append("comparison_dataset_fingerprint_mismatch")
        if request.base_run.protocol != request.candidate_run.protocol:
            failures.append("evaluation_protocol_mismatch")
        if not request.held_out_test and request.policy.require_held_out_test:
            failures.append("held_out_test_not_confirmed")
        return failures

    def _reproducibility(self, request: AcceptanceRequest) -> ReproducibilityReport:
        missing: list[str] = []
        mismatches: list[str] = []
        warnings: list[str] = []
        base_identity = request.base_run.model.model_identity
        candidate_identity = request.candidate_run.model.model_identity
        candidate_config = request.candidate_training_config
        artifact = request.candidate_artifact
        if artifact is not None:
            try:
                artifact.assert_valid()
            except Exception as exc:
                mismatches.append(f"candidate_artifact_invalid:{type(exc).__name__}")
            if candidate_config is None:
                candidate_config = artifact.metadata.training_config
            if request.training_dataset_fingerprint is None:
                object.__setattr__(request, "training_dataset_fingerprint", artifact.metadata.dataset_fingerprint)
            if artifact.model_version != candidate_identity.model_version:
                mismatches.append("candidate_artifact_model_version_mismatch")
            if artifact.metadata.base_model.model_fingerprint != base_identity.model_fingerprint:
                mismatches.append("candidate_artifact_base_model_mismatch")
            if request.candidate_run.model.artifact_fingerprint and artifact.fingerprint != request.candidate_run.model.artifact_fingerprint:
                mismatches.append("candidate_artifact_fingerprint_mismatch")
        if not base_identity.model_fingerprint or not _fingerprint(base_identity.model_fingerprint):
            missing.append("base_model_fingerprint")
        if not candidate_identity.model_fingerprint or not _fingerprint(candidate_identity.model_fingerprint):
            missing.append("candidate_model_fingerprint")
        if not request.dataset.dataset_fingerprint or not _fingerprint(request.dataset.dataset_fingerprint):
            missing.append("dataset_fingerprint")
        if not request.comparison.dataset_fingerprint or not _fingerprint(request.comparison.dataset_fingerprint):
            missing.append("benchmark_dataset_fingerprint")
        if not request.base_run.protocol or not request.candidate_run.protocol:
            missing.append("evaluation_protocol")
        for key in ("seed", "temperature", "max_tokens", "max_iterations", "timeout_seconds", "system_prompt_version", "agent_version", "tool_version"):
            if key not in request.base_run.protocol:
                missing.append(f"evaluation_config.{key}")
        if not request.training_dataset_fingerprint:
            missing.append("training_dataset_fingerprint")
        elif not _fingerprint(request.training_dataset_fingerprint):
            mismatches.append("training_dataset_fingerprint_invalid")
        if not candidate_config:
            missing.append("training_configuration")
        if request.policy.require_held_out_test and not request.held_out_test:
            missing.append("held_out_test_identity")
        if request.base_run.protocol.get("runs_per_task") == 1:
            warnings.append("single-run benchmark has no variance estimate; statistical significance is not claimed")
        try:
            request.dataset.validate_contamination(training_dataset_fingerprint=request.training_dataset_fingerprint, training_source_record_ids=request.training_source_record_ids)
        except Exception as exc:
            mismatches.append(f"dataset_contamination:{type(exc).__name__}")
        checks = {
            "base_model_identity": bool(base_identity.model_version and base_identity.model_fingerprint),
            "candidate_model_identity": bool(candidate_identity.model_version and candidate_identity.model_fingerprint),
            "dataset_identity": bool(request.dataset.dataset_version and request.dataset.dataset_fingerprint),
            "benchmark_identity": request.comparison.benchmark_version == request.dataset.benchmark_version,
            "evaluation_protocol": request.base_run.protocol == request.candidate_run.protocol and request.comparison.protocol_version == BENCHMARK_PROTOCOL_VERSION,
            "training_configuration": bool(candidate_config),
            "random_seed": "seed" in request.base_run.protocol,
            "evaluation_configuration": all(key in request.base_run.protocol for key in ("temperature", "max_tokens", "max_iterations", "timeout_seconds")),
            "model_fingerprint": bool(_fingerprint(base_identity.model_fingerprint) and _fingerprint(candidate_identity.model_fingerprint)),
            "dataset_fingerprint": _fingerprint(request.dataset.dataset_fingerprint),
            "benchmark_fingerprint": _fingerprint(compute_benchmark_fingerprint(request.comparison)),
            "policy_version": request.policy.policy_version == ACCEPTANCE_POLICY_VERSION,
            "held_out_test": request.held_out_test,
        }
        fingerprints = {"base_model": base_identity.model_fingerprint, "candidate_model": candidate_identity.model_fingerprint, "candidate_artifact": artifact.fingerprint if artifact is not None else request.candidate_run.model.artifact_fingerprint, "training_dataset": request.training_dataset_fingerprint, "benchmark_dataset": request.dataset.dataset_fingerprint, "benchmark": compute_benchmark_fingerprint(request.comparison), "training_config": _training_config_fingerprint(candidate_config)}
        valid = not missing and not mismatches and all(checks.values())
        return ReproducibilityReport(valid, checks, tuple(missing), tuple(mismatches), fingerprints, tuple(warnings))

    def _regressions(self, request: AcceptanceRequest) -> RegressionAnalysis:
        policy = request.policy
        findings: list[ModelRegressionFinding] = []
        warnings: list[str] = []
        checked: list[str] = []
        improved: list[str] = []
        base = request.base_run.aggregate
        candidate = request.candidate_run.aggregate
        metric_specs = (("task_success_rate", MetricDirection.HIGHER_IS_BETTER, RegressionCategory.METRIC, True), ("test_pass_rate", MetricDirection.HIGHER_IS_BETTER, RegressionCategory.METRIC, True), ("tool_success_rate", MetricDirection.HIGHER_IS_BETTER, RegressionCategory.TOOL, True), ("error_recovery_rate", MetricDirection.HIGHER_IS_BETTER, RegressionCategory.DEBUGGING, True), ("average_attempts", MetricDirection.LOWER_IS_BETTER, RegressionCategory.METRIC, False), ("failure_rate", MetricDirection.LOWER_IS_BETTER, RegressionCategory.METRIC, True))
        for name, direction, category, critical_metric in metric_specs:
            b_value = getattr(base, name)
            c_value = getattr(candidate, name)
            if b_value is None or c_value is None:
                warnings.append(f"metric evidence unavailable: {name}")
                continue
            checked.append(name)
            delta = float(c_value) - float(b_value)
            improvement = delta >= policy.minimum_required_improvement if direction is MetricDirection.HIGHER_IS_BETTER else delta <= -policy.minimum_required_improvement
            if improvement:
                improved.append(name)
            bad = delta < -policy.regression_tolerance if direction is MetricDirection.HIGHER_IS_BETTER else delta > policy.regression_tolerance
            if bad:
                magnitude = abs(delta)
                severity = _metric_severity(magnitude, policy, critical_metric)
                findings.append(ModelRegressionFinding(f"METRIC-{name}", category, name, "overall", float(b_value), float(c_value), delta, severity, severity is RegressionSeverity.CRITICAL, ("benchmark_metrics",), f"{name} changed unfavorably beyond tolerance {policy.regression_tolerance:.4f}"))
        self._task_findings(request, findings, policy)
        self._group_findings(request, findings, policy)
        if request.validation_success_rate is not None and candidate.task_success_rate is not None:
            gap = float(request.validation_success_rate) - float(candidate.task_success_rate)
            if gap > policy.maximum_overfitting_gap:
                severity = RegressionSeverity.CRITICAL if policy.reject_on_overfitting_gap else RegressionSeverity.HIGH
                findings.append(ModelRegressionFinding("OVERFITTING-GAP", RegressionCategory.OVERFITTING, "validation_test_gap", "held-out", float(request.validation_success_rate), float(candidate.task_success_rate), -gap, severity, policy.reject_on_overfitting_gap, ("validation_test_gap",), f"validation-to-held-out gap {gap:.4f} exceeds {policy.maximum_overfitting_gap:.4f}"))
            elif gap > policy.regression_tolerance:
                warnings.append(f"validation-to-held-out gap is {gap:.4f}; possible overfitting warning")
        else:
            warnings.append("validation performance was not supplied; overfitting gap is not assessed")
        highest = max((item.severity for item in findings), key=lambda item: _SEVERITY_ORDER[item], default=RegressionSeverity.NONE)
        status = "FAIL" if findings else "PASS" if checked else "INCONCLUSIVE"
        return RegressionAnalysis(status, tuple(findings), tuple(improved), tuple(checked), tuple(warnings), highest)

    @staticmethod
    def _task_findings(request: AcceptanceRequest, findings: list[ModelRegressionFinding], policy: AcceptancePolicy) -> None:
        base_items = {item.task_id: item for item in request.base_run.task_results}
        candidate_items = {item.task_id: item for item in request.candidate_run.task_results}
        for task_id in sorted(set(base_items) & set(candidate_items)):
            left, right = base_items[task_id], candidate_items[task_id]
            if left.success and not right.success:
                category = right.category or left.category
                critical = category in policy.critical_categories
                findings.append(ModelRegressionFinding(f"CAPABILITY-{task_id}", RegressionCategory.CAPABILITY, "task_success", category, 1.0, 0.0, -1.0, RegressionSeverity.CRITICAL if critical else RegressionSeverity.HIGH, critical, (task_id,), f"task passed with Base but failed with Candidate in {category}"))
            if left.tool_calls and right.tool_calls and right.successful_tool_calls / right.tool_calls + policy.regression_tolerance < left.successful_tool_calls / left.tool_calls:
                findings.append(ModelRegressionFinding(f"TOOL-{task_id}", RegressionCategory.TOOL, "tool_success_rate", right.category, left.successful_tool_calls / left.tool_calls, right.successful_tool_calls / right.tool_calls, right.successful_tool_calls / right.tool_calls - left.successful_tool_calls / left.tool_calls, RegressionSeverity.HIGH, right.category in policy.critical_categories, (task_id,), "tool success decreased for a task"))
            if left.recovery_encountered and left.recovery_success and (not right.recovery_success):
                findings.append(ModelRegressionFinding(f"DEBUG-{task_id}", RegressionCategory.DEBUGGING, "recovery_success", right.category, 1.0, 0.0, -1.0, RegressionSeverity.CRITICAL if right.category == "BUG_FIX" else RegressionSeverity.HIGH, right.category == "BUG_FIX", (task_id,), "recovery succeeded with Base but not with Candidate"))

    @staticmethod
    def _group_findings(request: AcceptanceRequest, findings: list[ModelRegressionFinding], policy: AcceptancePolicy) -> None:
        base_groups = request.base_run.aggregate.by_category
        candidate_groups = request.candidate_run.aggregate.by_category
        for group_name in sorted(set(base_groups) & set(candidate_groups)):
            left, right = base_groups[group_name], candidate_groups[group_name]
            for metric in ("task_success_rate", "test_pass_rate", "tool_success_rate", "error_recovery_rate"):
                b_value, c_value = left.get(metric), right.get(metric)
                if not isinstance(b_value, (int, float)) or not isinstance(c_value, (int, float)):
                    continue
                delta = float(c_value) - float(b_value)
                if delta < -policy.regression_tolerance:
                    critical = group_name in policy.critical_categories and metric in {"task_success_rate", "test_pass_rate"}
                    severity = _metric_severity(abs(delta), policy, critical)
                    findings.append(ModelRegressionFinding(f"DOMAIN-{group_name}-{metric}", RegressionCategory.DOMAIN, metric, group_name, float(b_value), float(c_value), delta, severity, critical or severity is RegressionSeverity.CRITICAL, (f"category:{group_name}",), f"{group_name} {metric} decreased beyond tolerance"))

    def _policy_failures(self, request: AcceptanceRequest, analysis: RegressionAnalysis) -> list[str]:
        policy = request.policy
        aggregate = request.candidate_run.aggregate
        failures: list[str] = []
        thresholds = (("minimum_task_success_rate", aggregate.task_success_rate), ("minimum_test_pass_rate", aggregate.test_pass_rate), ("minimum_tool_success_rate", aggregate.tool_success_rate), ("minimum_error_recovery_rate", aggregate.error_recovery_rate))
        for name, value in thresholds:
            threshold = getattr(policy, name)
            if value is None:
                failures.append(f"{name}:missing")
            elif value < threshold:
                failures.append(f"{name}:{value:.4f}<{threshold:.4f}")
        if aggregate.failure_rate is None:
            failures.append("maximum_failure_rate:missing")
        elif aggregate.failure_rate > policy.maximum_failure_rate:
            failures.append(f"maximum_failure_rate:{aggregate.failure_rate:.4f}>{policy.maximum_failure_rate:.4f}")
        if aggregate.average_attempts is None:
            failures.append("maximum_average_attempts:missing")
        elif aggregate.average_attempts > policy.maximum_average_attempts:
            failures.append(f"maximum_average_attempts:{aggregate.average_attempts:.4f}>{policy.maximum_average_attempts:.4f}")
        if len(analysis.findings) > policy.maximum_allowed_regression:
            failures.append(f"maximum_allowed_regression:{len(analysis.findings)}>{policy.maximum_allowed_regression}")
        critical = [item.finding_id for item in analysis.findings if item.critical or item.severity in {RegressionSeverity.CRITICAL, RegressionSeverity.HIGH}]
        if critical:
            failures.append("critical_regression:" + ",".join(sorted(critical)))
        if len(analysis.improved_metrics) < policy.minimum_improved_metrics:
            failures.append(f"minimum_improved_metrics:{len(analysis.improved_metrics)}<{policy.minimum_improved_metrics}")
        if policy.require_held_out_test and not request.held_out_test:
            failures.append("held_out_test_required")
        return failures


def compute_benchmark_fingerprint(comparison: BenchmarkComparison) -> str:
    payload = {"format": "fodci.benchmark_protocol", "protocol_version": comparison.protocol_version, "benchmark_version": comparison.benchmark_version, "dataset_version": comparison.dataset_version, "dataset_fingerprint": comparison.dataset_fingerprint}
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def render_acceptance_report(report: AcceptanceReport) -> str:
    lines = ["=" * 56, "FODCI MODEL ACCEPTANCE REPORT", "=" * 56, "", f"Base Model: {report.base_model_version}", f"Candidate:  {report.model_version}", f"Benchmark:  {report.benchmark_version}", f"Dataset:    {report.dataset_version}", "", "RESULT", "-" * 56, f"Decision: {report.decision.value}", f"Reason: {report.reason}", "", "METRICS", "-" * 56]
    for name, values in sorted(_thaw(report.metrics).items()):
        if isinstance(values, Mapping):
            lines.append(f"{name}:")
            lines.append(f"  Base: {_format(values.get('base'))}  Candidate: {_format(values.get('candidate'))}  Delta: {_format(values.get('delta'))}")
    lines.extend(["", "REGRESSIONS", "-" * 56])
    if report.regressions:
        for finding in report.regressions:
            lines.append(f"{finding.finding_id}: {finding.category.value} / {finding.severity.value} / {finding.scope} — {finding.explanation}")
    else:
        lines.append("None")
    lines.extend(["", "WARNINGS", "-" * 56])
    lines.extend(report.warnings or ("None",))
    lines.extend(["", "REPRODUCIBILITY", "-" * 56, "PASS" if report.reproducibility.valid else "FAIL"])
    if report.reproducibility.missing:
        lines.append("Missing: " + ", ".join(report.reproducibility.missing))
    if report.reproducibility.mismatches:
        lines.append("Mismatches: " + ", ".join(report.reproducibility.mismatches))
    lines.extend(["", "FINAL DECISION", "-" * 56, report.decision.value, "=" * 56])
    return "\n".join(lines)


def _metrics_snapshot(request: AcceptanceRequest) -> Mapping[str, Any]:
    names = ("task_success_rate", "test_pass_rate", "tool_success_rate", "error_recovery_rate", "average_attempts", "failure_rate")
    return {name: {"base": getattr(request.base_run.aggregate, name), "candidate": getattr(request.candidate_run.aggregate, name), "delta": _delta(getattr(request.base_run.aggregate, name), getattr(request.candidate_run.aggregate, name)), "direction": "LOWER_IS_BETTER" if name in {"average_attempts", "failure_rate"} else "HIGHER_IS_BETTER"} for name in names}


def _lineage(request: AcceptanceRequest) -> tuple[Mapping[str, Any], Mapping[str, str | None]]:
    artifact = request.candidate_artifact
    config = request.candidate_training_config or (artifact.metadata.training_config if artifact is not None else {})
    return ({"model_version": request.candidate_run.model.model_version, "base_model_version": request.base_run.model.model_version, "dataset_version": request.dataset.dataset_version, "benchmark_version": request.dataset.benchmark_version, "training_config": _thaw(config), "checkpoint": str(request.candidate_run.model.checkpoint_path), "evaluation_id": request.evaluation_id, "acceptance_policy_version": request.policy.policy_version, "candidate_artifact_id": artifact.model_id if artifact is not None else request.candidate_run.model.model_artifact_id}, {"base_model": request.base_run.model.model_identity.model_fingerprint, "candidate_model": request.candidate_run.model.model_identity.model_fingerprint, "candidate_artifact": artifact.fingerprint if artifact is not None else request.candidate_run.model.artifact_fingerprint, "training_dataset": request.training_dataset_fingerprint, "benchmark_dataset": request.dataset.dataset_fingerprint, "benchmark": compute_benchmark_fingerprint(request.comparison), "training_config": _training_config_fingerprint(config)})


def _training_config_fingerprint(config: Mapping[str, Any] | None) -> str | None:
    if not config:
        return None
    return "sha256:" + hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _metric_severity(magnitude: float, policy: AcceptancePolicy, critical_metric: bool) -> RegressionSeverity:
    if critical_metric and magnitude >= policy.critical_regression_delta:
        return RegressionSeverity.CRITICAL
    if magnitude >= policy.critical_regression_delta:
        return RegressionSeverity.HIGH
    return RegressionSeverity.MEDIUM


def _delta(base: Any, candidate: Any) -> float | None:
    return float(candidate) - float(base) if isinstance(base, (int, float)) and isinstance(candidate, (int, float)) else None


def _format(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)


def _report_from_dict(payload: Mapping[str, Any]) -> AcceptanceReport:
    policy = AcceptancePolicy(**payload["policy"])
    repro_payload = payload["reproducibility"]
    reproducibility = ReproducibilityReport(**repro_payload)
    findings = tuple(ModelRegressionFinding(item["finding_id"], RegressionCategory(item["category"]), item["dimension"], item["scope"], item["baseline_value"], item["candidate_value"], item["delta"], RegressionSeverity(item["severity"]), item["critical"], tuple(item.get("evidence_ids", ())), item["explanation"]) for item in payload["regressions"])
    return AcceptanceReport(payload["format"], payload["schema_version"], payload["evaluation_id"], AcceptanceDecision(payload["decision"]), payload["reason"], payload["model_version"], payload["base_model_version"], payload["dataset_version"], payload["benchmark_version"], payload["benchmark_fingerprint"], payload["metrics"], findings, tuple(payload.get("warnings", ())), policy, reproducibility, payload["fingerprints"], payload["lineage"], payload["created_at"])


def _fingerprint(value: str | None) -> bool:
    return isinstance(value, str) and bool(_FINGERPRINT_PATTERN.fullmatch(value))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if hasattr(value, "to_dict"):
        return _serialize(value.to_dict())
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_serialize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ACCEPTANCE_FORMAT",
    "ACCEPTANCE_POLICY_VERSION",
    "ACCEPTANCE_SCHEMA_VERSION",
    "AcceptanceConflictError",
    "AcceptanceDecision",
    "AcceptanceError",
    "AcceptancePolicy",
    "AcceptanceReport",
    "AcceptanceRequest",
    "AcceptanceStore",
    "ModelAcceptanceEvaluator",
    "ModelRegressionFinding",
    "RegressionAnalysis",
    "RegressionCategory",
    "ReproducibilityReport",
    "compute_benchmark_fingerprint",
    "render_acceptance_report",
]
