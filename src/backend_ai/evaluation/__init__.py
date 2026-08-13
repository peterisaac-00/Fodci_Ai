"""Phase 2.8 evaluation layer for Fodci."""

from backend_ai.core.contracts import Evaluator
from backend_ai.evaluation.config import EvaluationConfig
from backend_ai.evaluation.evaluator import EvaluationComparison, EvaluationResult, FodciEvaluator

__all__ = [
    "EvaluationComparison",
    "EvaluationConfig",
    "EvaluationResult",
    "Evaluator",
    "FodciEvaluator",
]
