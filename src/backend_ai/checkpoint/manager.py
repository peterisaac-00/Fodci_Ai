"""Safe metadata-aware checkpoint management for Fodci."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from backend_ai.tokenizer import TOKENIZER_VERSION

if TYPE_CHECKING:
    from backend_ai.training.config import TrainingConfig

CHECKPOINT_FORMAT = "fodci.checkpoint"
CHECKPOINT_FORMAT_VERSION = 2
_STRUCTURAL_MODEL_FIELDS = (
    "vocab_size",
    "context_length",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "feed_forward_size",
    "activation",
)


class CheckpointError(RuntimeError):
    """Base error for checkpoint management failures."""


class CheckpointFormatError(CheckpointError, ValueError):
    """Raised when a checkpoint is missing or has unsupported metadata."""


class CheckpointCompatibilityError(CheckpointError, ValueError):
    """Raised when a checkpoint cannot belong to the requested Fodci model."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Small, independently inspectable identity and resume metadata."""

    model_version: str
    model_config: dict[str, Any]
    tokenizer_version: int
    vocabulary_size: int
    context_length: int
    epoch: int
    global_step: int
    training_config: dict[str, Any]
    metrics: dict[str, Any]
    seed: int
    format: str = CHECKPOINT_FORMAT
    format_version: int = CHECKPOINT_FORMAT_VERSION
    created_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CheckpointMetadata":
        required = {
            "model_version",
            "model_config",
            "tokenizer_version",
            "vocabulary_size",
            "context_length",
            "epoch",
            "global_step",
            "training_config",
            "metrics",
            "seed",
            "format",
            "format_version",
            "created_at_utc",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise CheckpointFormatError(
                f"Checkpoint metadata is missing required fields: {', '.join(missing)}"
            )
        if raw["format"] != CHECKPOINT_FORMAT:
            raise CheckpointFormatError("Checkpoint has an unsupported format identifier.")
        if raw["format_version"] != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointFormatError(
                f"Unsupported checkpoint format version: {raw['format_version']}"
            )
        for name in ("epoch", "global_step", "seed", "tokenizer_version", "vocabulary_size", "context_length"):
            value = raw[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CheckpointFormatError(f"Checkpoint metadata field '{name}' is invalid.")
        if not isinstance(raw["model_version"], str) or not raw["model_version"]:
            raise CheckpointFormatError("Checkpoint model_version must be a non-empty string.")
        if not isinstance(raw["model_config"], Mapping):
            raise CheckpointFormatError("Checkpoint model_config must be an object.")
        if not isinstance(raw["training_config"], Mapping) or not isinstance(raw["metrics"], Mapping):
            raise CheckpointFormatError("Checkpoint training_config and metrics must be objects.")
        return cls(
            model_version=raw["model_version"],
            model_config=dict(raw["model_config"]),
            tokenizer_version=raw["tokenizer_version"],
            vocabulary_size=raw["vocabulary_size"],
            context_length=raw["context_length"],
            epoch=raw["epoch"],
            global_step=raw["global_step"],
            training_config=dict(raw["training_config"]),
            metrics=dict(raw["metrics"]),
            seed=raw["seed"],
            format=raw["format"],
            format_version=raw["format_version"],
            created_at_utc=raw["created_at_utc"],
        )


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """Path plus metadata returned by inspect/list operations."""

    path: Path
    metadata: CheckpointMetadata


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Result of loading weights and optimizer state into existing objects."""

    metadata: CheckpointMetadata

    @property
    def epoch(self) -> int:
        return self.metadata.epoch

    @property
    def global_step(self) -> int:
        return self.metadata.global_step

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self.metadata.metrics)

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.metadata.training_config)


class CheckpointManager:
    """Manage Fodci checkpoints without instantiating duplicate model objects."""

    def __init__(
        self,
        directory: Path | str,
        *,
        model_version: str = "fodci-tiny-v1",
        tokenizer_version: int = TOKENIZER_VERSION,
    ) -> None:
        self.directory = Path(directory)
        self.model_version = model_version
        self.tokenizer_version = tokenizer_version

    def exists(self, path: Path | str) -> bool:
        """Return whether a regular checkpoint file exists."""

        return Path(path).is_file()

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        *,
        epoch: int,
        global_step: int,
        metrics: Mapping[str, Any] | None = None,
        path: Path | str | None = None,
    ) -> Path:
        """Atomically save weights, optimizer state, and inspectable metadata."""

        checkpoint_path = self._resolve_path(path, epoch=epoch, global_step=global_step)
        model_config = _model_config_dict(model)
        metadata = CheckpointMetadata(
            model_version=self.model_version,
            model_config=model_config,
            tokenizer_version=self.tokenizer_version,
            vocabulary_size=int(model_config["vocab_size"]),
            context_length=int(model_config["context_length"]),
            epoch=_nonnegative_int(epoch, "epoch"),
            global_step=_nonnegative_int(global_step, "global_step"),
            training_config=config.to_dict(),
            metrics=dict(metrics or {}),
            seed=config.seed,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        payload = {
            "metadata": metadata.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            torch.save(payload, temporary_path)
            with temporary_path.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_path, checkpoint_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return checkpoint_path

    def inspect(self, path: Path | str) -> CheckpointInfo:
        """Read metadata without constructing a model or optimizer."""

        checkpoint_path = Path(path)
        payload = self._read_payload(checkpoint_path, map_location="cpu")
        metadata = _metadata_from_payload(payload, checkpoint_path)
        return CheckpointInfo(path=checkpoint_path, metadata=metadata)

    def load_model(
        self,
        path: Path | str,
        model: nn.Module,
        *,
        device: torch.device,
    ) -> LoadedCheckpoint:
        """Validate compatibility, then restore model weights without an optimizer."""

        checkpoint_path = Path(path)
        payload, metadata = self._validated_payload(checkpoint_path, model, device=device)
        try:
            model.load_state_dict(payload["model_state_dict"])
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            raise CheckpointCompatibilityError(
                "Checkpoint state is incompatible with the instantiated model."
            ) from exc
        return LoadedCheckpoint(metadata=metadata)

    def load(
        self,
        path: Path | str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
    ) -> LoadedCheckpoint:
        """Validate compatibility, then restore model and optimizer state."""

        checkpoint_path = Path(path)
        payload, metadata = self._validated_payload(checkpoint_path, model, device=device)
        try:
            model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            raise CheckpointCompatibilityError(
                "Checkpoint optimizer state is incompatible with the instantiated optimizer."
            ) from exc
        return LoadedCheckpoint(metadata=metadata)

    def _validated_payload(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        *,
        device: torch.device,
    ) -> tuple[dict[str, Any], CheckpointMetadata]:
        payload = self._read_payload(checkpoint_path, map_location=device)
        metadata = _metadata_from_payload(payload, checkpoint_path)
        self.validate_compatibility(model, metadata)
        return payload, metadata

    def validate_compatibility(
        self,
        model: nn.Module,
        metadata: CheckpointMetadata,
    ) -> None:
        """Fail before mutation when model identity or structure differs."""

        if metadata.model_version != self.model_version:
            raise CheckpointCompatibilityError(
                f"Model version mismatch: checkpoint={metadata.model_version!r}, "
                f"expected={self.model_version!r}."
            )
        if metadata.tokenizer_version != self.tokenizer_version:
            raise CheckpointCompatibilityError(
                f"Tokenizer version mismatch: checkpoint={metadata.tokenizer_version}, "
                f"expected={self.tokenizer_version}."
            )
        current_config = _model_config_dict(model)
        for field in _STRUCTURAL_MODEL_FIELDS:
            expected = current_config.get(field)
            actual = metadata.model_config.get(field)
            if actual != expected:
                label = {
                    "vocab_size": "vocabulary size",
                    "context_length": "context length",
                }.get(field, field)
                raise CheckpointCompatibilityError(
                    f"Model configuration mismatch for {label}: checkpoint={actual!r}, "
                    f"expected={expected!r}."
                )
        if metadata.vocabulary_size != int(current_config["vocab_size"]):
            raise CheckpointCompatibilityError("Checkpoint vocabulary size is incompatible.")
        if metadata.context_length != int(current_config["context_length"]):
            raise CheckpointCompatibilityError("Checkpoint context length is incompatible.")

    def list(self) -> tuple[CheckpointInfo, ...]:
        """List valid ``.pt`` checkpoints sorted by metadata progress."""

        if not self.directory.is_dir():
            return ()
        infos: list[CheckpointInfo] = []
        for path in sorted(self.directory.glob("*.pt")):
            try:
                infos.append(self.inspect(path))
            except CheckpointError:
                continue
        return tuple(sorted(infos, key=_progress_key))

    def latest(self) -> CheckpointInfo | None:
        """Return the checkpoint with greatest metadata global step/epoch."""

        checkpoints = self.list()
        return checkpoints[-1] if checkpoints else None

    def best(self) -> CheckpointInfo | None:
        """Return the lowest validation-loss checkpoint, with latest tie-break."""

        candidates = [
            info
            for info in self.list()
            if _validation_loss(info.metadata.metrics) is not None
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda info: (
                float(_validation_loss(info.metadata.metrics)),
                -info.metadata.global_step,
            ),
        )

    def _resolve_path(self, path: Path | str | None, *, epoch: int, global_step: int) -> Path:
        if path is not None:
            return Path(path)
        return self.directory / f"checkpoint-epoch-{epoch:04d}-step-{global_step:08d}.pt"

    def _read_payload(self, path: Path, *, map_location: str | torch.device) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        try:
            payload = torch.load(path, map_location=map_location)
        except Exception as exc:
            raise CheckpointFormatError(f"Unable to read checkpoint: {path}") from exc
        if not isinstance(payload, dict):
            raise CheckpointFormatError(f"Checkpoint payload must be an object: {path}")
        return payload


def _metadata_from_payload(payload: Mapping[str, Any], path: Path) -> CheckpointMetadata:
    if "metadata" not in payload:
        raise CheckpointFormatError(f"Checkpoint has no metadata: {path}")
    try:
        metadata = CheckpointMetadata.from_dict(payload["metadata"])
    except (TypeError, AttributeError) as exc:
        raise CheckpointFormatError(f"Checkpoint metadata is invalid: {path}") from exc
    for field in ("model_state_dict", "optimizer_state_dict"):
        if field not in payload:
            raise CheckpointFormatError(f"Checkpoint is missing {field}: {path}")
    return metadata


def _model_config_dict(model: nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("Checkpoint management requires a model with a config attribute.")
    try:
        values = asdict(config)
    except TypeError as exc:
        raise TypeError("Model config must be a dataclass-compatible object.") from exc
    for field in _STRUCTURAL_MODEL_FIELDS:
        if field not in values:
            raise TypeError(f"Model config is missing required field: {field}")
    return values


def _nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _validation_loss(metrics: Mapping[str, Any]) -> float | None:
    value = metrics.get("validation_loss")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _progress_key(info: CheckpointInfo) -> tuple[int, int, str]:
    return (
        info.metadata.global_step,
        info.metadata.epoch,
        info.metadata.created_at_utc,
    )


# Imported late so corrupted pickle errors are handled without exposing it as a public API.
import pickle  # noqa: E402  (standard-library exception type)
