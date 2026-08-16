from backend_ai.core.contracts import Evaluator
from backend_ai.evaluation.config import EvaluationConfig
try:
    from backend_ai.evaluation.evaluator import EvaluationComparison, EvaluationResult as ModelEvaluationResult, FodciEvaluator
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    EvaluationComparison = None  # type: ignore[assignment]
    ModelEvaluationResult = None  # type: ignore[assignment]
    FodciEvaluator = None  # type: ignore[assignment]
from backend_ai.evaluation.benchmark_runner import (
    BenchmarkConfig, BenchmarkEvidence, BenchmarkExecutionResult, BenchmarkRequest,
    BenchmarkResult, BenchmarkRunSummary, BenchmarkRunner, BenchmarkStatus,
    BenchmarkTaskRun, BenchmarkTaskStatus, BenchmarkTerminationReason,
)
from backend_ai.evaluation.scoring import (
    BenchmarkScorer, BenchmarkScore, CriterionEvaluation, EvaluationResult,
    EvaluationScorer, EvaluationStatus, EvaluationWeights, EvidenceStatus, ScoreDimension,
    ScoringPolicy, TaskEvaluation, TaskScore, evaluate_benchmark,
)
from backend_ai.evaluation.regression import (
    AggregateComparison, ComparisonClassification, ComparisonConfig, ComparisonResult,
    ComparisonStatus, DimensionComparison, EvaluationComparisonRequest,
    EvaluationComparisonResult, EvaluationRegressionComparator, EvaluationSnapshot,
    EvaluationVersion, RegressionFinding, RegressionSeverity, RegressionSummary,
    TaskComparison, compare_evaluations,
)
from backend_ai.evaluation.metrics import (
    BenchmarkMetrics, CategoryMetrics, DifficultyMetrics, MetricName, MetricStatus,
    SingleMetric, TaskMetrics, TaskMetricsCollection, collect_benchmark_metrics,
    collect_metrics,
)
from backend_ai.evaluation.report import (
    EvaluationReport, ReportConfig, ReportInputs, ReportLimit, TruncationInfo,
    generate_evaluation_report,
)
from backend_ai.evaluation.version_comparison import (
    ComparisonClassification as VersionComparisonClassification, ComparisonDimension,
    GroupMetricComparison, VersionMetricsComparison, compare_evaluation_metrics,
    compare_evaluations_detailed,
)
from backend_ai.evaluation.regression_evaluation import (
    RegressionEvaluator, RegressionGate, RegressionGateResult, RegressionType,
    RegressionVerdict, RegressionEvaluationResult, evaluate_regression,
)
from backend_ai.evaluation.benchmark_validation import (
    BenchmarkHealth, BenchmarkValidationResult, BenchmarkValidator, IssueLevel,
    ValidationIssue as BenchmarkValidationIssue, ValidationStatus, validate_benchmark,
)
from backend_ai.evaluation.task_model import (
    AllowedScope, ChangeType, EvaluationConstraint, EvaluationDifficulty,
    EvaluationTask, EvaluationTaskCategory, EvaluationTaskValidationResult,
    EvaluationTaskValidator, EvaluationTestType, ExpectedArea, ExpectedAreaType,
    ExpectedBehavior, ForbiddenChange, ForbiddenChangeType, GroundTruth,
    ProjectDefinition, Requirement, SuccessCriterion, SuccessCriterionType,
    TestDefinition, ValidationIssue, ValidationSeverity, create_evaluation_task,
    serialize_evaluation_task, validate_evaluation_task,
)

__all__ = [
    "BenchmarkConfig", "BenchmarkEvidence", "BenchmarkExecutionResult", "BenchmarkRequest",
    "BenchmarkResult", "BenchmarkRunSummary", "BenchmarkRunner", "BenchmarkStatus",
    "BenchmarkTaskRun", "BenchmarkTaskStatus", "BenchmarkTerminationReason",
    "BenchmarkScorer", "BenchmarkScore", "CriterionEvaluation", "EvaluationResult",
    "EvaluationScorer", "EvaluationStatus", "EvaluationWeights", "EvidenceStatus", "ScoreDimension",
    "ScoringPolicy", "TaskEvaluation", "TaskScore", "evaluate_benchmark",
    "AggregateComparison", "ComparisonClassification", "ComparisonConfig", "ComparisonResult",
    "ComparisonStatus", "DimensionComparison", "EvaluationComparisonRequest", "EvaluationComparisonResult",
    "EvaluationRegressionComparator", "EvaluationSnapshot", "EvaluationVersion", "RegressionFinding",
    "RegressionSeverity", "RegressionSummary", "TaskComparison", "compare_evaluations",
    "EvaluationComparison", "ModelEvaluationResult", "EvaluationConfig", "Evaluator", "FodciEvaluator",
    "AllowedScope", "ChangeType", "EvaluationConstraint", "EvaluationDifficulty", "EvaluationTask",
    "EvaluationTaskCategory", "EvaluationTaskValidationResult", "EvaluationTaskValidator",
    "EvaluationTestType", "ExpectedArea", "ExpectedAreaType", "ExpectedBehavior", "ForbiddenChange",
    "ForbiddenChangeType", "GroundTruth", "ProjectDefinition", "Requirement", "SuccessCriterion",
    "SuccessCriterionType", "TestDefinition", "ValidationIssue", "ValidationSeverity",
    "create_evaluation_task", "serialize_evaluation_task", "validate_evaluation_task",
    "BenchmarkMetrics", "CategoryMetrics", "DifficultyMetrics", "MetricName", "MetricStatus",
    "SingleMetric", "TaskMetrics", "TaskMetricsCollection", "collect_benchmark_metrics",
    "collect_metrics",
    "EvaluationReport", "ReportConfig", "ReportInputs", "ReportLimit", "TruncationInfo",
    "generate_evaluation_report",
    "VersionComparisonClassification", "ComparisonDimension", "GroupMetricComparison",
    "VersionMetricsComparison", "compare_evaluation_metrics", "compare_evaluations_detailed",
    "RegressionEvaluator", "RegressionGate", "RegressionGateResult", "RegressionType",
    "RegressionVerdict", "RegressionEvaluationResult", "evaluate_regression",
    "BenchmarkHealth", "BenchmarkValidationResult", "BenchmarkValidator",
    "BenchmarkValidationIssue", "IssueLevel", "ValidationStatus", "validate_benchmark",
]
