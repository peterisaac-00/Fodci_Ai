"""Phase 11.3 offline fine-tuning orchestration.

This module is deliberately outside the Agent runtime.  It consumes only a
validated Phase 11.2 ``TrainingDatasetArtifact`` and an explicit model adapter,
then delegates optimization and objective metrics to the existing
``FodciTrainer``.  It never adds an Agent command, online learning path, network
access, cloud orchestration, benchmark acceptance, or model promotion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Any, Protocol

import torch
from torch import nn

from backend_ai.evaluation.baseline import ModelIdentity, model_identity_from_checkpoint
from backend_ai.checkpoint import CheckpointCompatibilityError, CheckpointManager, CheckpointMetadata
from backend_ai.dataset.samples import TrainingExample as TokenTrainingExample
from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import (
    FodciTokenizer,
    PAD_ID,
    TOKENIZER_FORMAT,
    TOKENIZER_VERSION,
)
from backend_ai.training.config import TrainingConfig
from backend_ai.training.metrics import EpochMetrics, TrainingResult
from backend_ai.training.trainer import FodciTrainer, seed_everything
from backend_ai.agent.training_dataset import (
    TRAINING_DATASET_SCHEMA_VERSION,
    TrainingDatasetArtifact,
    TrainingDatasetArtifactError,
    TrainingDatasetLoader,
    TrainingExample as TextTrainingExample,
)


FINE_TUNING_FORMAT = "fodci.fine_tuning_run"
FINE_TUNING_PROTOCOL_VERSION = "11.3"
FINE_TUNING_SCHEMA_VERSION = "1.0"
CANDIDATE_MODEL_VERSION_PATTERN = re.compile(r"^candidate-v[0-9]+(?:\.[0-9]+)?$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_MODEL_PARAMETERS = 20_000_000


class FineTuningError(RuntimeError):
    """Base error for explicit Phase 11.3 failures."""


class FineTuningConfigurationError(FineTuningError, ValueError):
    """Invalid or unsafe fine-tuning configuration."""


class FineTuningDatasetError(FineTuningError, ValueError):
    """Missing, invalid, mutable, or leaked training data."""


class FineTuningModelError(FineTuningError, ValueError):
    """Missing or incompatible base model/tokenizer identity."""


class FineTuningCheckpointError(FineTuningError, ValueError):
    """Missing, corrupted, incompatible, or unrelated resume checkpoint."""


class FineTuningRunConflictError(FineTuningError):
    """A run ID already belongs to a different immutable run artifact."""


class FineTuningStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class TokenizerIdentity:
    """Canonical tokenizer identity recorded with every fine-tuning run."""

    format: str
    version: int
    vocab_size: int
    fingerprint: str

    def __post_init__(self) -> None:
        if self.format != TOKENIZER_FORMAT or not isinstance(self.version, int) or self.version < 0:
            raise FineTuningModelError("tokenizer format/version is invalid")
        if not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise FineTuningModelError("tokenizer vocab_size is invalid")
        if not isinstance(self.fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(self.fingerprint):
            raise FineTuningModelError("tokenizer fingerprint must use sha256 format")

    @classmethod
    def from_tokenizer(cls, tokenizer: FodciTokenizer) -> "TokenizerIdentity":
        if not isinstance(tokenizer, FodciTokenizer):
            raise FineTuningModelError("Phase 11.3 currently requires a FodciTokenizer adapter")
        payload = {
            "format": TOKENIZER_FORMAT,
            "version": TOKENIZER_VERSION,
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": tokenizer.special_tokens,
            "merges": [list(item) for item in tokenizer.merges],
        }
        fingerprint = "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return cls(TOKENIZER_FORMAT, TOKENIZER_VERSION, tokenizer.vocab_size, fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "version": self.version, "vocab_size": self.vocab_size, "fingerprint": self.fingerprint}


class FineTuningModelAdapter(Protocol):
    """Minimal model adapter needed by the model-agnostic runner."""

    model: nn.Module
    tokenizer: FodciTokenizer
    model_identity: ModelIdentity
    tokenizer_identity: TokenizerIdentity


@dataclass(slots=True)
class FodciModelAdapter:
    """Adapter for the current FodciModel while keeping the runner generic."""

    model: nn.Module
    tokenizer: FodciTokenizer
    model_identity: ModelIdentity
    tokenizer_identity: TokenizerIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise FineTuningModelError("model adapter requires a torch.nn.Module")
        if not isinstance(self.tokenizer, FodciTokenizer):
            raise FineTuningModelError("model adapter requires a FodciTokenizer")
        if not isinstance(self.model_identity, ModelIdentity):
            raise FineTuningModelError("model_identity must use the existing ModelIdentity contract")
        if not isinstance(self.tokenizer_identity, TokenizerIdentity):
            raise FineTuningModelError("tokenizer_identity is invalid")
        if self.model_identity.model_fingerprint is None:
            raise FineTuningModelError("base model must have a unique checkpoint fingerprint")
        config = getattr(self.model, "config", None)
        if config is None or not hasattr(config, "vocab_size") or not hasattr(config, "context_length"):
            raise FineTuningModelError("model must expose config.vocab_size and config.context_length")
        if int(config.vocab_size) != self.tokenizer.vocab_size:
            raise FineTuningModelError("model and tokenizer vocabulary sizes are incompatible")
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        if parameter_count > _MAX_MODEL_PARAMETERS:
            raise FineTuningModelError(f"model has {parameter_count} parameters; Phase 11.3 hard limit is {_MAX_MODEL_PARAMETERS}")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        tokenizer_path: Path | str | None = None,
        device: str = "cpu",
    ) -> "FodciModelAdapter":
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FineTuningModelError(f"base checkpoint is unavailable: {path}")
        try:
            info = CheckpointManager(path.parent).inspect(path)
            model_config = ModelConfig(**{key: value for key, value in info.metadata.model_config.items() if key in ModelConfig.__dataclass_fields__})
            model = FodciModel(model_config)
            tokenizer = FodciTokenizer(vocab_size=info.metadata.vocabulary_size)
            if tokenizer_path is not None:
                tokenizer = FodciTokenizer.load(tokenizer_path)
            if tokenizer.vocab_size != info.metadata.vocabulary_size:
                raise FineTuningModelError("tokenizer vocabulary does not match the base checkpoint")
            resolved_device = _resolve_device(device)
            manager = CheckpointManager(path.parent, model_version=info.metadata.model_version, tokenizer_version=info.metadata.tokenizer_version)
            manager.load_model(path, model, device=resolved_device)
            model_identity = model_identity_from_checkpoint(path, model_version=info.metadata.model_version, tokenizer_version=info.metadata.tokenizer_version)
            return cls(model, tokenizer, model_identity, TokenizerIdentity.from_tokenizer(tokenizer))
        except FineTuningError:
            raise
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, CheckpointCompatibilityError) as exc:
            raise FineTuningModelError(f"unable to load compatible base checkpoint: {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class FineTuningConfig:
    """Explicit, versionable offline fine-tuning configuration."""

    run_id: str
    candidate_model_version: str = "candidate-v1"
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float | None = 1.0
    seed: int = 2026
    device: str = "cpu"
    checkpoint_interval: int = 1
    validation_interval: int = 1
    log_interval: int = 0
    output_directory: Path | str = Path("artifacts/training_runs")

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise FineTuningConfigurationError("run_id must be a bounded stable identifier")
        if not CANDIDATE_MODEL_VERSION_PATTERN.fullmatch(self.candidate_model_version):
            raise FineTuningConfigurationError("candidate_model_version must use candidate-vN format")
        if self.epochs <= 0 or (self.max_steps is not None and self.max_steps <= 0):
            raise FineTuningConfigurationError("epochs must be positive and max_steps must be positive or None")
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise FineTuningConfigurationError("batch_size and gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise FineTuningConfigurationError("learning_rate must be positive and weight_decay cannot be negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise FineTuningConfigurationError("max_grad_norm must be positive or None")
        if self.seed < 0:
            raise FineTuningConfigurationError("seed cannot be negative")
        for name in ("checkpoint_interval", "validation_interval", "log_interval"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise FineTuningConfigurationError(f"{name} must be a non-negative integer")
        try:
            TrainingConfig(device=self.device)
        except (ValueError, RuntimeError) as exc:
            raise FineTuningConfigurationError(str(exc)) from exc
        object.__setattr__(self, "device", self.device.strip().lower())
        object.__setattr__(self, "output_directory", Path(self.output_directory).expanduser())

    @property
    def run_directory(self) -> Path:
        return self.output_directory / self.run_id

    @property
    def checkpoint_directory(self) -> Path:
        return self.run_directory / "checkpoints"

    def to_training_config(self) -> TrainingConfig:
        return TrainingConfig(
            epochs=self.epochs,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            device=self.device,
            seed=self.seed,
            log_interval=self.log_interval,
            validation_interval=self.validation_interval,
            checkpoint_interval=self.checkpoint_interval,
            output_dir=self.checkpoint_directory,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "candidate_model_version": self.candidate_model_version,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "device": self.device,
            "checkpoint_interval": self.checkpoint_interval,
            "validation_interval": self.validation_interval,
            "log_interval": self.log_interval,
            "output_directory": str(self.output_directory),
            "run_directory": str(self.run_directory),
            "checkpoint_directory": str(self.checkpoint_directory),
        }


@dataclass(frozen=True, slots=True)
class FineTuningDatasetIdentity:
    """Immutable identity of the validated Phase 11.2 input."""

    dataset_version: str
    dataset_fingerprint: str
    schema_version: str
    training_examples: int
    validation_examples: int

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_version, str) or not self.dataset_version.strip() or not _FINGERPRINT_PATTERN.fullmatch(self.dataset_fingerprint):
            raise FineTuningDatasetError("dataset identity is invalid")
        if self.schema_version != TRAINING_DATASET_SCHEMA_VERSION:
            raise FineTuningDatasetError("unsupported training dataset schema version")
        if self.training_examples <= 0 or self.validation_examples <= 0:
            raise FineTuningDatasetError("training and validation splits must both be non-empty")

    @classmethod
    def from_artifact(cls, artifact: TrainingDatasetArtifact) -> "FineTuningDatasetIdentity":
        if not isinstance(artifact, TrainingDatasetArtifact):
            raise FineTuningDatasetError("fine-tuning requires a validated TrainingDatasetArtifact")
        if not artifact.train or not artifact.validation:
            raise FineTuningDatasetError("fine-tuning requires non-empty train and validation partitions")
        train_ids = {item.example_id for item in artifact.train}
        validation_ids = {item.example_id for item in artifact.validation}
        test_ids = {item.example_id for item in artifact.test}
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise FineTuningDatasetError("dataset partitions overlap; test leakage is rejected")
        return cls(artifact.manifest.dataset_version, artifact.manifest.dataset_fingerprint, artifact.manifest.training_schema_version, len(artifact.train), len(artifact.validation))

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_version": self.dataset_version, "dataset_fingerprint": self.dataset_fingerprint, "schema_version": self.schema_version, "training_examples": self.training_examples, "validation_examples": self.validation_examples}


@dataclass(frozen=True, slots=True)
class FineTuningRunResult:
    """Traceable outcome of one offline candidate training run."""

    format: str
    protocol_version: str
    schema_version: str
    run_id: str
    status: str
    base_model: ModelIdentity
    candidate_model: ModelIdentity | None
    tokenizer: TokenizerIdentity
    dataset: FineTuningDatasetIdentity
    configuration: Mapping[str, Any]
    metrics: tuple[Mapping[str, Any], ...]
    checkpoints: tuple[Mapping[str, Any], ...]
    run_directory: str
    resumed_from: str | None
    software: Mapping[str, Any]
    hardware: Mapping[str, Any]
    started_at_utc: str
    ended_at_utc: str | None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.format != FINE_TUNING_FORMAT or self.protocol_version != FINE_TUNING_PROTOCOL_VERSION or self.schema_version != FINE_TUNING_SCHEMA_VERSION:
            raise FineTuningError("unsupported fine-tuning run identity")
        if self.status not in {FineTuningStatus.COMPLETED, FineTuningStatus.FAILED}:
            raise FineTuningError("unsupported fine-tuning status")
        if self.status == FineTuningStatus.COMPLETED and self.candidate_model is None:
            raise FineTuningError("completed run must have a candidate model identity")
        if self.status == FineTuningStatus.FAILED and not self.error:
            raise FineTuningError("failed run must have an error")
        if not isinstance(self.base_model, ModelIdentity) or not isinstance(self.tokenizer, TokenizerIdentity) or not isinstance(self.dataset, FineTuningDatasetIdentity):
            raise FineTuningError("run identity objects are invalid")
        for name, value in (("configuration", self.configuration), ("software", self.software), ("hardware", self.hardware)):
            if not isinstance(value, Mapping):
                raise FineTuningError(f"{name} must be a mapping")
        if not isinstance(self.metrics, tuple) or not isinstance(self.checkpoints, tuple):
            raise FineTuningError("metrics and checkpoints must be tuples")
        _validate_json_value(self.configuration, "configuration", 0, 8)
        _validate_json_value(self.metrics, "metrics", 0, 8)
        _validate_json_value(self.checkpoints, "checkpoints", 0, 8)
        _validate_json_value(self.software, "software", 0, 4)
        _validate_json_value(self.hardware, "hardware", 0, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "protocol_version": self.protocol_version,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status,
            "base_model": self.base_model.to_dict(),
            "base_model_version": self.base_model.model_version,
            "model_identifier": {"model_path": self.base_model.model_path, "model_fingerprint": self.base_model.model_fingerprint},
            "candidate_model": self.candidate_model.to_dict() if self.candidate_model else None,
            "tokenizer": self.tokenizer.to_dict(),
            "dataset": self.dataset.to_dict(),
            "dataset_version": self.dataset.dataset_version,
            "dataset_fingerprint": self.dataset.dataset_fingerprint,
            "dataset_schema_version": self.dataset.schema_version,
            "number_of_training_examples": self.dataset.training_examples,
            "number_of_validation_examples": self.dataset.validation_examples,
            "configuration": _thaw(self.configuration),
            "training_configuration": _thaw(self.configuration),
            "metrics": [_thaw(item) for item in self.metrics],
            "checkpoints": [_thaw(item) for item in self.checkpoints],
            "run_directory": self.run_directory,
            "resumed_from": self.resumed_from,
            "software": _thaw(self.software),
            "hardware": _thaw(self.hardware),
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "error": self.error,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def create_model_artifact(self, directory: Path | str, *, model_id: str | None = None, evaluation_reference: Any | None = None, created_at: str | None = None) -> Any:
        """Explicitly wrap this completed run as a Phase 11.4 ModelArtifact."""

        from backend_ai.model_artifact import ModelArtifact

        return ModelArtifact.create_from_fine_tuning_run(self, directory, model_id=model_id, evaluation_reference=evaluation_reference, created_at=created_at)


class FineTuningRunner:
    """Run one validated, offline fine-tuning experiment."""

    def __init__(self, adapter: FineTuningModelAdapter, artifact: TrainingDatasetArtifact, config: FineTuningConfig) -> None:
        self.adapter = adapter
        self.artifact = artifact
        self.config = config
        self.dataset_identity = FineTuningDatasetIdentity.from_artifact(artifact)
        self._validate_adapter()
        self._run_metadata_base = self._build_checkpoint_run_metadata()

    @classmethod
    def from_paths(
        cls,
        *,
        base_checkpoint: Path | str,
        dataset_directory: Path | str,
        config: FineTuningConfig,
        tokenizer_path: Path | str | None = None,
    ) -> "FineTuningRunner":
        try:
            artifact = TrainingDatasetLoader.load_artifact(dataset_directory)
        except (TrainingDatasetArtifactError, OSError, ValueError) as exc:
            raise FineTuningDatasetError(f"unable to load validated training artifact: {dataset_directory}: {exc}") from exc
        adapter = FodciModelAdapter.from_checkpoint(base_checkpoint, tokenizer_path=tokenizer_path, device=config.device)
        return cls(adapter, artifact, config)

    def run(self, *, resume_checkpoint: Path | str | None = None) -> FineTuningRunResult:
        started = _utc_now()
        self._prepare_run_directory()
        resumed_from: str | None = None
        checkpoints: list[Mapping[str, Any]] = []
        try:
            train_examples = tuple(_tokenize_examples(self.artifact.train, self.adapter.tokenizer, self._context_length))
            validation_examples = tuple(_tokenize_examples(self.artifact.validation, self.adapter.tokenizer, self._context_length))
            if not train_examples or not validation_examples:
                raise FineTuningDatasetError("tokenization produced an empty train or validation split")
            training_config = self.config.to_training_config()
            trainer = FodciTrainer(
                self.adapter.model,
                lambda: iter(train_examples),
                lambda: iter(validation_examples),
                training_config,
                model_version=self.config.candidate_model_version,
                checkpoint_run_metadata=self._run_metadata_base,
            )
            if resume_checkpoint is not None:
                resume_info = self._validate_resume_checkpoint(resume_checkpoint)
                trainer.resume(resume_checkpoint)
                resumed_from = str(Path(resume_checkpoint).expanduser())
                if self.config.epochs <= resume_info.epoch:
                    raise FineTuningCheckpointError("resume configuration epochs must exceed checkpoint epoch")
            initial_path = self.config.checkpoint_directory / "initial.pt"
            if resume_checkpoint is None:
                trainer.save_checkpoint(initial_path, run_metadata=self._run_metadata_base)
                checkpoints.append(_checkpoint_entry(initial_path, "initial", trainer.global_step, 0, {}))
            training_result = trainer.train()
            final_path = self.config.checkpoint_directory / "final.pt"
            trainer.save_checkpoint(final_path, run_metadata={**self._run_metadata_base, "resumed_from": resumed_from})
            checkpoints.extend(_checkpoint_entries(self.config.checkpoint_directory, exclude={initial_path, final_path}))
            checkpoints.append(_checkpoint_entry(final_path, "final", training_result.global_step, training_result.final_metrics.epoch if training_result.final_metrics else 0, training_result.final_metrics.to_dict() if training_result.final_metrics else {}))
            candidate_identity = ModelIdentity(
                self.adapter.model_identity.model_name,
                self.config.candidate_model_version,
                str(final_path),
                "sha256:" + _file_sha256(final_path),
                self.adapter.tokenizer_identity.version,
            )
            result = FineTuningRunResult(
                FINE_TUNING_FORMAT,
                FINE_TUNING_PROTOCOL_VERSION,
                FINE_TUNING_SCHEMA_VERSION,
                self.config.run_id,
                FineTuningStatus.COMPLETED,
                self.adapter.model_identity,
                candidate_identity,
                self.adapter.tokenizer_identity,
                self.dataset_identity,
                self.config.to_dict(),
                tuple(metric.to_dict() for metric in training_result.history),
                tuple(checkpoints),
                str(self.config.run_directory),
                resumed_from,
                _software_identity(),
                _hardware_identity(training_config.device),
                started,
                _utc_now(),
            )
        except (FineTuningError, OSError, RuntimeError, ValueError, FloatingPointError) as exc:
            result = FineTuningRunResult(
                FINE_TUNING_FORMAT,
                FINE_TUNING_PROTOCOL_VERSION,
                FINE_TUNING_SCHEMA_VERSION,
                self.config.run_id,
                FineTuningStatus.FAILED,
                self.adapter.model_identity,
                None,
                self.adapter.tokenizer_identity,
                self.dataset_identity,
                self.config.to_dict(),
                (),
                tuple(checkpoints),
                str(self.config.run_directory),
                resumed_from,
                _software_identity(),
                _hardware_identity(self.config.device),
                started,
                _utc_now(),
                _safe_message(exc),
            )
        _write_run_result(self.config.run_directory, result)
        return result

    def _validate_adapter(self) -> None:
        for name in ("model", "tokenizer", "model_identity", "tokenizer_identity"):
            if not hasattr(self.adapter, name):
                raise FineTuningModelError(f"model adapter is missing {name}")
        if self.adapter.model_identity.model_fingerprint is None:
            raise FineTuningModelError("base model fingerprint is required")
        config = getattr(self.adapter.model, "config", None)
        if config is None:
            raise FineTuningModelError("base model must expose a config")
        parameter_count = sum(parameter.numel() for parameter in self.adapter.model.parameters())
        if parameter_count > _MAX_MODEL_PARAMETERS:
            raise FineTuningModelError("model parameter hard limit exceeded")
        if int(config.vocab_size) != self.adapter.tokenizer_identity.vocab_size:
            raise FineTuningModelError("model/tokenizer vocabulary mismatch")
        if self.adapter.tokenizer_identity.version != self.adapter.model_identity.tokenizer_version:
            raise FineTuningModelError("model/tokenizer version mismatch")
        self._context_length = int(config.context_length)
        if self._context_length <= 1:
            raise FineTuningModelError("model context_length must exceed one token")

    def _prepare_run_directory(self) -> None:
        run_dir = self.config.run_directory
        existing = run_dir / "run.json"
        if existing.is_file():
            raise FineTuningRunConflictError(f"run_id already has an immutable run result: {self.config.run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_directory.mkdir(parents=True, exist_ok=True)

    def _build_checkpoint_run_metadata(self) -> dict[str, Any]:
        return {
            "phase": FINE_TUNING_PROTOCOL_VERSION,
            "run_id": self.config.run_id,
            "base_model_fingerprint": self.adapter.model_identity.model_fingerprint,
            "base_model_version": self.adapter.model_identity.model_version,
            "dataset_version": self.dataset_identity.dataset_version,
            "dataset_fingerprint": self.dataset_identity.dataset_fingerprint,
            "dataset_schema_version": self.dataset_identity.schema_version,
            "tokenizer_fingerprint": self.adapter.tokenizer_identity.fingerprint,
            "tokenizer_version": self.adapter.tokenizer_identity.version,
            "candidate_model_version": self.config.candidate_model_version,
        }

    def _validate_resume_checkpoint(self, checkpoint_path: Path | str) -> CheckpointMetadata:
        path = Path(checkpoint_path).expanduser()
        try:
            info = CheckpointManager(self.config.checkpoint_directory, model_version=self.config.candidate_model_version, tokenizer_version=self.adapter.tokenizer_identity.version).inspect(path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise FineTuningCheckpointError(f"resume checkpoint is invalid: {path}: {exc}") from exc
        metadata = info.metadata
        expected = self._run_metadata_base
        actual = metadata.run_metadata
        if not isinstance(actual, Mapping) or actual.get("phase") != FINE_TUNING_PROTOCOL_VERSION:
            raise FineTuningCheckpointError("resume checkpoint is not a Phase 11.3 checkpoint")
        for key in ("base_model_fingerprint", "dataset_version", "dataset_fingerprint", "dataset_schema_version", "tokenizer_fingerprint", "candidate_model_version"):
            if actual.get(key) != expected[key]:
                raise FineTuningCheckpointError(f"resume checkpoint {key} is incompatible")
        if metadata.epoch < 0 or metadata.global_step < 0:
            raise FineTuningCheckpointError("resume checkpoint progress is invalid")
        return metadata


def fine_tune(
    *,
    base_checkpoint: Path | str,
    dataset_directory: Path | str,
    config: FineTuningConfig,
    tokenizer_path: Path | str | None = None,
    resume_checkpoint: Path | str | None = None,
) -> FineTuningRunResult:
    """Convenience entry point for one explicit offline run."""

    return FineTuningRunner.from_paths(base_checkpoint=base_checkpoint, dataset_directory=dataset_directory, config=config, tokenizer_path=tokenizer_path).run(resume_checkpoint=resume_checkpoint)


def load_run_result(path: Path | str) -> FineTuningRunResult:
    """Load and validate an immutable Phase 11.3 run sidecar."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FineTuningError("fine-tuning run metadata must be an object")
    base = payload["base_model"]
    candidate = payload.get("candidate_model")
    tokenizer_payload = payload["tokenizer"]
    dataset_payload = payload["dataset"]
    base_identity = ModelIdentity(base["model_name"], base["model_version"], base.get("model_path"), base.get("model_fingerprint"), base.get("tokenizer_version"))
    candidate_identity = None if candidate is None else ModelIdentity(candidate["model_name"], candidate["model_version"], candidate.get("model_path"), candidate.get("model_fingerprint"), candidate.get("tokenizer_version"))
    tokenizer_identity = TokenizerIdentity(tokenizer_payload["format"], tokenizer_payload["version"], tokenizer_payload["vocab_size"], tokenizer_payload["fingerprint"])
    dataset_identity = FineTuningDatasetIdentity(dataset_payload["dataset_version"], dataset_payload["dataset_fingerprint"], dataset_payload["schema_version"], dataset_payload["training_examples"], dataset_payload["validation_examples"])
    return FineTuningRunResult(payload["format"], payload["protocol_version"], payload["schema_version"], payload["run_id"], FineTuningStatus(payload["status"]), base_identity, candidate_identity, tokenizer_identity, dataset_identity, payload["configuration"], tuple(payload["metrics"]), tuple(payload["checkpoints"]), payload["run_directory"], payload.get("resumed_from"), payload["software"], payload["hardware"], payload["started_at_utc"], payload.get("ended_at_utc"), payload.get("error"))


def _tokenize_examples(examples: Sequence[TextTrainingExample], tokenizer: FodciTokenizer, context_length: int) -> tuple[TokenTrainingExample, ...]:
    result: list[TokenTrainingExample] = []
    for example in examples:
        if not isinstance(example, TextTrainingExample):
            raise FineTuningDatasetError("training artifact contains an invalid text example")
        serialized = f"{example.input}\n\nTarget:\n{example.target}"
        token_ids = tokenizer.encode(serialized, add_bos=True, add_eos=True)
        if len(token_ids) < 2:
            raise FineTuningDatasetError(f"training example {example.example_id} has fewer than two tokens")
        if len(token_ids) > context_length:
            raise FineTuningDatasetError(f"training example {example.example_id} exceeds model context_length {context_length}")
        input_ids = tuple(token_ids[:-1])
        target_ids = tuple(token_ids[1:])
        padding = context_length - len(input_ids)
        result.append(TokenTrainingExample(input_ids + (PAD_ID,) * padding, target_ids + (PAD_ID,) * padding, example.example_id, (True,) * len(input_ids) + (False,) * padding))
    return tuple(result)


def _checkpoint_entries(directory: Path, *, exclude: set[Path]) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    for path in sorted(directory.glob("*.pt")):
        if path in exclude:
            continue
        try:
            info = CheckpointManager(directory).inspect(path)
        except Exception:
            continue
        entries.append(_checkpoint_entry(path, "intermediate", info.metadata.global_step, info.metadata.epoch, info.metadata.metrics))
    return entries


def _checkpoint_entry(path: Path, kind: str, global_step: int, epoch: int, metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"path": str(path), "kind": kind, "epoch": epoch, "global_step": global_step, "fingerprint": "sha256:" + _file_sha256(path), "metrics": dict(metrics)}


def _write_run_result(directory: Path, result: FineTuningRunResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(directory / "run.json", result.to_dict())
    _atomic_write_json(directory / "metrics.json", {"run_id": result.run_id, "status": result.status, "metrics": [_thaw(item) for item in result.metrics]})


def _resolve_device(value: str) -> torch.device:
    return TrainingConfig(device=value).resolve_device()


def _software_identity() -> Mapping[str, Any]:
    return {"python": sys.version.split()[0], "torch": torch.__version__, "fine_tuning_protocol": FINE_TUNING_PROTOCOL_VERSION}


def _hardware_identity(device: str) -> Mapping[str, Any]:
    resolved = _resolve_device(device)
    return {"platform": platform.platform(), "machine": platform.machine(), "device": str(resolved), "cuda_available": bool(torch.cuda.is_available()), "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_thaw(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _validate_json_value(value: Any, name: str, depth: int, maximum_depth: int) -> None:
    if depth > maximum_depth:
        raise FineTuningError(f"{name} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise FineTuningError(f"{name} contains an invalid key")
            _validate_json_value(item, f"{name}.{key}", depth + 1, maximum_depth)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{name}[{index}]", depth + 1, maximum_depth)
        return
    raise FineTuningError(f"{name} contains an unsupported value type")


def _safe_message(exc: Exception) -> str:
    return str(exc).strip()[:1_024] or exc.__class__.__name__


def _atomic_write_json(path: Path, payload: Any) -> None:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


__all__ = [
    "CANDIDATE_MODEL_VERSION_PATTERN",
    "FINE_TUNING_FORMAT",
    "FINE_TUNING_PROTOCOL_VERSION",
    "FINE_TUNING_SCHEMA_VERSION",
    "FineTuningCheckpointError",
    "FineTuningConfigurationError",
    "FineTuningConfig",
    "FineTuningDatasetError",
    "FineTuningDatasetIdentity",
    "FineTuningError",
    "FineTuningModelAdapter",
    "FineTuningModelError",
    "FineTuningRunConflictError",
    "FineTuningRunResult",
    "FineTuningRunner",
    "FineTuningStatus",
    "FodciModelAdapter",
    "TokenizerIdentity",
    "fine_tune",
    "load_run_result",
]
