"""Minimal CPU-friendly training loop for the Fodci language model."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

from backend_ai.dataset.samples import TrainingExample
from backend_ai.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from backend_ai.training.config import TrainingConfig
from backend_ai.training.metrics import EpochMetrics, TrainingResult, perplexity

logger = logging.getLogger("backend_ai.training")
ExampleSource: TypeAlias = Iterable[TrainingExample] | Callable[[], Iterable[TrainingExample]]


class FodciTrainer:
    """Train an existing FodciModel without changing its architecture."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: ExampleSource,
        validation_dataset: ExampleSource | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = self.config.resolve_device()
        seed_everything(self.config.seed)
        self.model = model.to(self.device)
        self._validate_model_contract()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.global_step = 0
        self._next_epoch = 1
        self._history: list[EpochMetrics] = []
        self._last_checkpoint: Path | None = None

    @property
    def history(self) -> tuple[EpochMetrics, ...]:
        """Return metrics recorded by this trainer instance."""

        return tuple(self._history)

    def train(self) -> TrainingResult:
        """Run configured epochs and return lightweight training metrics."""

        experiment_start = time.perf_counter()
        for epoch in range(self._next_epoch, self.config.epochs + 1):
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break
            epoch_start = time.perf_counter()
            self.model.train()
            train_loss, training_steps, training_tokens = self._train_epoch(epoch)
            validation_loss, validation_steps, validation_tokens = self._validate_epoch(epoch)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                training_steps=training_steps,
                validation_steps=validation_steps,
                learning_rate=learning_rate,
                train_perplexity=perplexity(train_loss) or float("inf"),
                validation_perplexity=perplexity(validation_loss),
                training_tokens=training_tokens,
                validation_tokens=validation_tokens,
                elapsed_seconds=time.perf_counter() - epoch_start,
            )
            self._history.append(metrics)
            self._next_epoch = epoch + 1
            if self.config.checkpoint_interval and epoch % self.config.checkpoint_interval == 0:
                self._last_checkpoint = self.save_checkpoint(
                    self.config.output_dir / f"epoch-{epoch:04d}.pt",
                )
            logger.info(
                "Epoch %d/%d train_loss=%.6f val_loss=%s lr=%.6g steps=%d tokens=%d",
                epoch,
                self.config.epochs,
                train_loss,
                f"{validation_loss:.6f}" if validation_loss is not None else "n/a",
                learning_rate,
                training_steps,
                training_tokens,
            )
        return TrainingResult(
            history=tuple(self._history),
            global_step=self.global_step,
            last_checkpoint=str(self._last_checkpoint) if self._last_checkpoint else None,
            elapsed_seconds=time.perf_counter() - experiment_start,
        )

    def resume(self, checkpoint_path: Path | str) -> CheckpointState:
        """Restore model/optimizer state and continue at the following epoch."""

        state = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.device)
        self.global_step = state.global_step
        self._next_epoch = state.epoch + 1
        self._last_checkpoint = Path(checkpoint_path)
        return state

    def save_checkpoint(self, path: Path | str | None = None) -> Path:
        """Save the current state, defaulting to a manual checkpoint path."""

        checkpoint_path = Path(path) if path is not None else self.config.output_dir / "latest.pt"
        latest_metrics = self._history[-1].to_dict() if self._history else {}
        epoch = self._next_epoch - 1
        return save_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            epoch=epoch,
            global_step=self.global_step,
            config=self.config,
            metrics=latest_metrics,
        )

    def _train_epoch(self, epoch: int) -> tuple[float, int, int]:
        total_loss = 0.0
        total_tokens = 0
        steps = 0
        for batch in _iter_batches(self.train_dataset, self.config.batch_size):
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break
            input_ids, target_ids = self._batch_to_tensors(batch)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(input_ids)
            loss = _cross_entropy(logits, target_ids)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
            self.optimizer.step()
            steps += 1
            self.global_step += 1
            token_count = target_ids.numel()
            total_loss += float(loss.detach().item()) * token_count
            total_tokens += token_count
            if self.config.log_interval and self.global_step % self.config.log_interval == 0:
                logger.info("step=%d train_loss=%.6f", self.global_step, loss.item())
        if steps == 0 or total_tokens == 0:
            raise ValueError("Training dataset produced no examples.")
        return total_loss / total_tokens, steps, total_tokens

    def _validate_epoch(self, epoch: int) -> tuple[float | None, int, int]:
        if self.validation_dataset is None:
            return None, 0, 0
        if self.config.validation_interval == 0 or epoch % self.config.validation_interval != 0:
            return None, 0, 0
        return self.evaluate(self.validation_dataset)

    def evaluate(
        self,
        dataset: ExampleSource,
    ) -> tuple[float, int, int]:
        """Evaluate a source without updating parameters or optimizer state."""

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        steps = 0
        with torch.no_grad():
            for batch in _iter_batches(dataset, self.config.batch_size):
                input_ids, target_ids = self._batch_to_tensors(batch)
                logits = self.model(input_ids)
                loss = _cross_entropy(logits, target_ids)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite evaluation loss.")
                steps += 1
                token_count = target_ids.numel()
                total_loss += float(loss.item()) * token_count
                total_tokens += token_count
        if steps == 0 or total_tokens == 0:
            raise ValueError("Evaluation dataset produced no examples.")
        return total_loss / total_tokens, steps, total_tokens

    def _batch_to_tensors(
        self,
        batch: list[TrainingExample],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not batch:
            raise ValueError("Training batch cannot be empty.")
        sequence_lengths = {len(example.input_ids) for example in batch}
        if len(sequence_lengths) != 1:
            raise ValueError("All examples in a batch must have the same sequence length.")
        input_rows: list[tuple[int, ...]] = []
        target_rows: list[tuple[int, ...]] = []
        for example in batch:
            if len(example.input_ids) != len(example.target_ids):
                raise ValueError("Each example must have equal input and target lengths.")
            if not 0 < len(example.input_ids) <= self._context_length:
                raise ValueError("Example sequence length must be within model context length.")
            input_rows.append(example.input_ids)
            target_rows.append(example.target_ids)
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
        target_ids = torch.tensor(target_rows, dtype=torch.long, device=self.device)
        self._validate_token_range(input_ids, "input_ids")
        self._validate_token_range(target_ids, "target_ids")
        return input_ids, target_ids

    def _validate_model_contract(self) -> None:
        config = getattr(self.model, "config", None)
        if config is None or not hasattr(config, "context_length") or not hasattr(config, "vocab_size"):
            raise TypeError("FodciTrainer requires a model with config.context_length and config.vocab_size.")
        if int(config.context_length) < 1 or int(config.vocab_size) < 1:
            raise ValueError("Model context_length and vocab_size must be positive.")
        self._context_length = int(config.context_length)
        self._vocab_size = int(config.vocab_size)

    def _validate_token_range(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.numel() == 0:
            raise ValueError(f"{name} cannot be empty.")
        if int(tensor.min().item()) < 0 or int(tensor.max().item()) >= self._vocab_size:
            raise ValueError(f"{name} contains a token outside the model vocabulary.")


def _iter_batches(source: ExampleSource, batch_size: int) -> Iterator[list[TrainingExample]]:
    examples = source() if callable(source) else source
    batch: list[TrainingExample] = []
    for example in examples:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("Model logits must have shape (batch_size, sequence_length, vocab_size).")
    if target_ids.ndim != 2 or logits.shape[:2] != target_ids.shape:
        raise ValueError("Model logits and target_ids have incompatible shapes.")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
    )


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible CPU training."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
