"""CPU-first evaluation pipeline for Fodci."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch import nn

from backend_ai.checkpoint import CheckpointInfo, CheckpointManager
from backend_ai.dataset.samples import TrainingExample
from backend_ai.training.config import TrainingConfig
from backend_ai.training.metrics import perplexity
from backend_ai.training.trainer import FodciTrainer, ExampleSource, seed_everything
from backend_ai.evaluation.config import EvaluationConfig


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics for one model state on one explicit dataset split."""

    checkpoint_id: str
    checkpoint_path: str | None
    model_version: str
    tokenizer_version: int
    device: str
    dataset_path: str
    dataset_split: str
    dataset_hash: str | None
    document_count: int | None
    loss: float
    perplexity: float
    evaluation_examples: int
    evaluated_tokens: int
    evaluation_seconds: float
    epoch: int | None = None
    global_step: int | None = None
    loss_type: str = "token_loss"
    response_loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationComparison:
    """Measured comparison between random baseline and trained checkpoint."""

    baseline: EvaluationResult
    trained: EvaluationResult
    loss_delta: float
    loss_improvement: float
    loss_relative_improvement_percent: float
    perplexity_delta: float
    perplexity_improvement: float
    perplexity_relative_improvement_percent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "trained": self.trained.to_dict(),
            "loss_delta": self.loss_delta,
            "loss_improvement": self.loss_improvement,
            "loss_relative_improvement_percent": self.loss_relative_improvement_percent,
            "perplexity_delta": self.perplexity_delta,
            "perplexity_improvement": self.perplexity_improvement,
            "perplexity_relative_improvement_percent": self.perplexity_relative_improvement_percent,
        }


class FodciEvaluator:
    """Evaluate Fodci states without optimizer updates or gradient accumulation."""

    def __init__(
        self,
        model: nn.Module,
        config: EvaluationConfig | None = None,
        *,
        dataset_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or EvaluationConfig()
        self.dataset_metadata = dict(dataset_metadata or {})
        self.device = _resolve_device(self.config.device)
        seed_everything(self.config.seed)
        self.model = model.to(self.device)
        self._validate_model_contract()
        self.checkpoint_manager = CheckpointManager(
            self.config.checkpoint_dir,
            model_version=self.config.model_version,
            tokenizer_version=self.config.tokenizer_version,
        )

    def evaluate(
        self,
        dataset: ExampleSource,
        *,
        checkpoint_id: str = "random-initialization",
        checkpoint_path: Path | str | None = None,
    ) -> EvaluationResult:
        """Evaluate the current model, optionally after validated checkpoint loading."""

        epoch: int | None = None
        global_step: int | None = None
        actual_path: str | None = None
        if checkpoint_path is not None:
            loaded = self.checkpoint_manager.load(
                checkpoint_path,
                self.model,
                self._optimizer,
                device=self.device,
            )
            epoch = loaded.epoch
            global_step = loaded.global_step
            actual_path = str(Path(checkpoint_path))

        self.model.eval()
        start = time.perf_counter()
        loss, examples, tokens = self._runtime.evaluate(dataset)
        elapsed = time.perf_counter() - start
        model_version = self.config.model_version
        if checkpoint_path is not None:
            model_version = self.checkpoint_manager.inspect(checkpoint_path).metadata.model_version
        return EvaluationResult(
            checkpoint_id=checkpoint_id,
            checkpoint_path=actual_path,
            model_version=model_version,
            tokenizer_version=self.config.tokenizer_version,
            device=str(self.device),
            dataset_path=str(self.config.dataset_path),
            dataset_split=self.config.dataset_split,
            dataset_hash=_optional_string(self.dataset_metadata.get("dataset_hash")),
            document_count=_optional_int(self.dataset_metadata.get("document_count")),
            loss=loss,
            perplexity=perplexity(loss) or float("inf"),
            evaluation_examples=examples,
            evaluated_tokens=tokens,
            evaluation_seconds=elapsed,
            epoch=epoch,
            global_step=global_step,
            loss_type=str(self.dataset_metadata.get("loss_type", "token_loss")),
            response_loss=(
                loss if self.dataset_metadata.get("loss_type") == "response_only" else None
            ),
        )

    def evaluate_checkpoint(
        self,
        checkpoint_path: Path | str,
        dataset: ExampleSource,
        *,
        checkpoint_id: str | None = None,
    ) -> EvaluationResult:
        """Evaluate one compatible checkpoint on the supplied dataset source."""

        path = Path(checkpoint_path)
        return self.evaluate(
            dataset,
            checkpoint_id=checkpoint_id or path.stem,
            checkpoint_path=path,
        )

    def evaluate_checkpoints(
        self,
        checkpoint_paths: Iterable[Path | str],
        dataset: ExampleSource,
    ) -> tuple[EvaluationResult, ...]:
        """Evaluate multiple checkpoints against the same source factory."""

        return tuple(self.evaluate_checkpoint(path, dataset) for path in checkpoint_paths)

    def evaluate_best(
        self,
        dataset: ExampleSource,
        *,
        manager: CheckpointManager | None = None,
    ) -> EvaluationResult:
        """Select and evaluate the manager's best validation-loss checkpoint."""

        selected_manager = manager or self.checkpoint_manager
        info = selected_manager.best()
        if info is None:
            raise FileNotFoundError("No checkpoint with validation_loss metadata is available.")
        return self.evaluate_checkpoint(info.path, dataset, checkpoint_id="best")

    @staticmethod
    def compare(
        baseline: EvaluationResult,
        trained: EvaluationResult,
    ) -> EvaluationComparison:
        """Compare two measured results and reject different evaluation contexts."""

        if baseline.dataset_path != trained.dataset_path or baseline.dataset_split != trained.dataset_split:
            raise ValueError("Baseline and trained evaluations must use the same dataset split.")
        loss_delta = trained.loss - baseline.loss
        loss_improvement = baseline.loss - trained.loss
        loss_relative = _relative_percent(loss_improvement, baseline.loss)
        perplexity_delta = trained.perplexity - baseline.perplexity
        perplexity_improvement = baseline.perplexity - trained.perplexity
        perplexity_relative = _relative_percent(perplexity_improvement, baseline.perplexity)
        return EvaluationComparison(
            baseline=baseline,
            trained=trained,
            loss_delta=loss_delta,
            loss_improvement=loss_improvement,
            loss_relative_improvement_percent=loss_relative,
            perplexity_delta=perplexity_delta,
            perplexity_improvement=perplexity_improvement,
            perplexity_relative_improvement_percent=perplexity_relative,
        )

    def _validate_model_contract(self) -> None:
        config = getattr(self.model, "config", None)
        if config is None or not hasattr(config, "context_length") or not hasattr(config, "vocab_size"):
            raise TypeError("FodciEvaluator requires a model with config.context_length and config.vocab_size.")
        self._runtime = FodciTrainer(
            self.model,
            train_dataset=(),
            validation_dataset=None,
            config=TrainingConfig(
                batch_size=self.config.batch_size,
                device=str(self.device),
                seed=self.config.seed,
                checkpoint_interval=0,
                output_dir=self.config.checkpoint_dir,
            ),
            model_version=self.config.model_version,
        )
        self._optimizer = self._runtime.optimizer
        self._context_length = int(config.context_length)
        self._vocab_size = int(config.vocab_size)


def _resolve_device(value: str):
    import torch

    if value == "cpu":
        return torch.device("cpu")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{value}' is unavailable; CUDA is not available.")
    return torch.device(value)


def _relative_percent(improvement: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0
    return (improvement / baseline) * 100.0


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
