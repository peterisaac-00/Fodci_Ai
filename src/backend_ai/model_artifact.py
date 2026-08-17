"""Immutable local Model Artifact and Model Identity system for Fodci.

Phase 11.4 wraps a completed Phase 11.3 fine-tuning checkpoint with an
integrity-verifiable provenance manifest.  This module is local-only and does
not evaluate, promote, deploy, or replace any model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

from backend_ai.evaluation.baseline import ModelIdentity

if TYPE_CHECKING:
    from backend_ai.training.fine_tuning import FineTuningRunResult


MODEL_ARTIFACT_FORMAT = "fodci.model_artifact"
MODEL_ARTIFACT_SCHEMA_VERSION = "1.0"
MODEL_ARTIFACT_REGISTRY_FORMAT = "fodci.model_artifact_registry"
MODEL_ARTIFACT_REGISTRY_SCHEMA_VERSION = "1.0"
MODEL_ARTIFACT_CHECKPOINT = "checkpoint/final.pt"
_MODEL_VERSION_PATTERN = re.compile(r"^(?:candidate-v|model-v|v)[0-9]+(?:\.[0-9]+)?$")
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$")
_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_REGISTRY_BYTES = 32 * 1024 * 1024


class ModelArtifactError(ValueError):
    """Base error for invalid, unavailable, or unsafe model artifacts."""


class ModelArtifactConflictError(ModelArtifactError):
    """Raised when an immutable artifact or identity would be overwritten."""


class ModelArtifactIntegrityError(ModelArtifactError):
    """Raised when artifact files or metadata do not match their fingerprints."""


class ModelArtifactStorageError(ModelArtifactError):
    """Raised when local artifact or registry storage is malformed/unavailable."""


@dataclass(frozen=True, slots=True)
class EvaluationReference:
    """Minimal future-compatible reference; no evaluation is fabricated."""

    status: str = "NOT_EVALUATED"
    evaluation_id: str | None = None
    protocol_version: str | None = None
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"NOT_EVALUATED", "RECORDED"}:
            raise ModelArtifactError("evaluation reference status is unsupported")
        if self.status == "RECORDED" and (not isinstance(self.evaluation_id, str) or not self.evaluation_id.strip()):
            raise ModelArtifactError("RECORDED evaluation reference requires evaluation_id")
        if self.evaluation_id is not None and (not isinstance(self.evaluation_id, str) or not self.evaluation_id.strip()):
            raise ModelArtifactError("evaluation_id must be text or None")
        if self.protocol_version is not None and (not isinstance(self.protocol_version, str) or not self.protocol_version.strip()):
            raise ModelArtifactError("protocol_version must be text or None")
        if self.fingerprint is not None and not _FINGERPRINT_PATTERN.fullmatch(self.fingerprint):
            raise ModelArtifactError("evaluation fingerprint must use sha256 format")

    @classmethod
    def not_evaluated(cls) -> "EvaluationReference":
        return cls()

    @classmethod
    def recorded(cls, evaluation_id: str, *, protocol_version: str | None = None, fingerprint: str | None = None) -> "EvaluationReference":
        return cls("RECORDED", evaluation_id, protocol_version, fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "evaluation_id": self.evaluation_id, "protocol_version": self.protocol_version, "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Any) -> "EvaluationReference":
        if not isinstance(payload, Mapping) or set(payload) != {"status", "evaluation_id", "protocol_version", "fingerprint"}:
            raise ModelArtifactError("evaluation reference fields are invalid")
        return cls(payload["status"], payload["evaluation_id"], payload["protocol_version"], payload["fingerprint"])


@dataclass(frozen=True, slots=True)
class ModelArtifactMetadata:
    """Canonical immutable provenance metadata for one exact model artifact."""

    format: str
    schema_version: str
    model_version: str
    model_id: str
    base_model: ModelIdentity
    dataset_version: str
    dataset_fingerprint: str
    training_config: Mapping[str, Any]
    training_config_fingerprint: str
    checkpoint: str
    checkpoint_fingerprint: str
    evaluation_reference: EvaluationReference
    created_at: str
    artifact_fingerprint: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.format != MODEL_ARTIFACT_FORMAT or self.schema_version != MODEL_ARTIFACT_SCHEMA_VERSION:
            raise ModelArtifactError("unsupported model artifact format/schema")
        if not isinstance(self.model_version, str) or not _MODEL_VERSION_PATTERN.fullmatch(self.model_version):
            raise ModelArtifactError("model_version must use vN, model-vN, or candidate-vN format")
        if not isinstance(self.model_id, str) or not _MODEL_ID_PATTERN.fullmatch(self.model_id):
            raise ModelArtifactError("model_id is invalid")
        if not isinstance(self.base_model, ModelIdentity) or not self.base_model.model_fingerprint or not _FINGERPRINT_PATTERN.fullmatch(self.base_model.model_fingerprint):
            raise ModelArtifactError("base_model must have a valid checkpoint fingerprint")
        if not isinstance(self.dataset_version, str) or not self.dataset_version.strip() or not _FINGERPRINT_PATTERN.fullmatch(self.dataset_fingerprint):
            raise ModelArtifactError("dataset identity is invalid")
        if not isinstance(self.training_config, Mapping) or not self.training_config:
            raise ModelArtifactError("training_config must be a non-empty mapping")
        if not _FINGERPRINT_PATTERN.fullmatch(self.training_config_fingerprint):
            raise ModelArtifactError("training_config_fingerprint is invalid")
        if compute_training_config_fingerprint(self.training_config) != self.training_config_fingerprint:
            raise ModelArtifactIntegrityError("training_config fingerprint does not match training_config")
        if not isinstance(self.checkpoint, str) or not self.checkpoint or Path(self.checkpoint).is_absolute() or ".." in Path(self.checkpoint).parts:
            raise ModelArtifactError("checkpoint must be a safe relative path")
        if not self.checkpoint.startswith("checkpoint/"):
            raise ModelArtifactError("checkpoint must be stored under checkpoint/")
        if not _FINGERPRINT_PATTERN.fullmatch(self.checkpoint_fingerprint) or not isinstance(self.evaluation_reference, EvaluationReference):
            raise ModelArtifactError("checkpoint/evaluation identity is invalid")
        if not isinstance(self.created_at, str) or not self.created_at.strip() or len(self.created_at) > 128:
            raise ModelArtifactError("created_at must be bounded metadata")
        if not _FINGERPRINT_PATTERN.fullmatch(self.artifact_fingerprint):
            raise ModelArtifactError("artifact_fingerprint is invalid")
        if not isinstance(self.provenance, Mapping):
            raise ModelArtifactError("provenance must be a mapping")
        _validate_json_value(self.training_config, "training_config", 0, 8)
        _validate_json_value(self.provenance, "provenance", 0, 8)
        object.__setattr__(self, "training_config", _freeze(self.training_config))
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "model_id": self.model_id,
            "base_model": self.base_model.to_dict(),
            "dataset_version": self.dataset_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "training_config": _thaw(self.training_config),
            "training_config_fingerprint": self.training_config_fingerprint,
            "checkpoint": self.checkpoint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "evaluation_reference": self.evaluation_reference.to_dict(),
            "created_at": self.created_at,
            "artifact_fingerprint": self.artifact_fingerprint,
            "provenance": _thaw(self.provenance),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Any) -> "ModelArtifactMetadata":
        if not isinstance(payload, Mapping):
            raise ModelArtifactStorageError("model artifact metadata must be an object")
        allowed = {"format", "schema_version", "model_version", "model_id", "base_model", "dataset_version", "dataset_fingerprint", "training_config", "training_config_fingerprint", "checkpoint", "checkpoint_fingerprint", "evaluation_reference", "created_at", "artifact_fingerprint", "provenance"}
        if set(payload) != allowed:
            raise ModelArtifactStorageError("model artifact metadata fields are missing or unknown")
        base = payload["base_model"]
        if not isinstance(base, Mapping):
            raise ModelArtifactStorageError("base_model metadata is invalid")
        base_identity = ModelIdentity(base["model_name"], base["model_version"], base.get("model_path"), base.get("model_fingerprint"), base.get("tokenizer_version"))
        return cls(payload["format"], payload["schema_version"], payload["model_version"], payload["model_id"], base_identity, payload["dataset_version"], payload["dataset_fingerprint"], payload["training_config"], payload["training_config_fingerprint"], payload["checkpoint"], payload["checkpoint_fingerprint"], EvaluationReference.from_dict(payload["evaluation_reference"]), payload["created_at"], payload["artifact_fingerprint"], payload["provenance"])


@dataclass(frozen=True, slots=True)
class ModelArtifactVerification:
    valid: bool
    checks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool) or not isinstance(self.checks, tuple) or any(not isinstance(item, str) for item in self.checks):
            raise ModelArtifactError("artifact verification result is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "checks": list(self.checks)}


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A verified artifact directory containing a checkpoint and immutable metadata."""

    root: Path
    metadata: ModelArtifactMetadata

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    @property
    def model_id(self) -> str:
        return self.metadata.model_id

    @property
    def fingerprint(self) -> str:
        return self.metadata.artifact_fingerprint

    @property
    def checkpoint_path(self) -> Path:
        return self.root / self.metadata.checkpoint

    def to_dict(self) -> dict[str, Any]:
        return self.metadata.to_dict()

    def verify(self) -> ModelArtifactVerification:
        checks: list[str] = []
        try:
            if not self.root.is_dir() or self.root.is_symlink():
                raise ModelArtifactIntegrityError("artifact root is unavailable or unsafe")
            metadata_path = self.root / "metadata.json"
            evaluation_path = self.root / "evaluation.json"
            if metadata_path.is_symlink() or evaluation_path.is_symlink() or self.checkpoint_path.is_symlink():
                raise ModelArtifactIntegrityError("artifact files must not be symlinks")
            if not self.checkpoint_path.is_file():
                raise ModelArtifactIntegrityError("artifact checkpoint is missing")
            actual_checkpoint = "sha256:" + _file_sha256(self.checkpoint_path)
            if actual_checkpoint != self.metadata.checkpoint_fingerprint:
                raise ModelArtifactIntegrityError("checkpoint fingerprint mismatch")
            evaluation = _read_json(evaluation_path)
            if EvaluationReference.from_dict(evaluation) != self.metadata.evaluation_reference:
                raise ModelArtifactIntegrityError("evaluation reference mismatch")
            expected = compute_artifact_fingerprint(self.metadata)
            if expected != self.metadata.artifact_fingerprint:
                raise ModelArtifactIntegrityError("artifact fingerprint mismatch")
            checks.extend(("metadata_valid", "checkpoint_fingerprint_valid", "evaluation_reference_valid", "artifact_fingerprint_valid"))
            return ModelArtifactVerification(True, tuple(checks))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ModelArtifactError) as exc:
            return ModelArtifactVerification(False, tuple(checks + [str(exc)]))

    def assert_valid(self) -> None:
        result = self.verify()
        if not result.valid:
            raise ModelArtifactIntegrityError("; ".join(result.checks))

    @classmethod
    def load(cls, directory: Path | str) -> "ModelArtifact":
        root = Path(directory).expanduser()
        if not root.is_dir() or root.is_symlink():
            raise ModelArtifactStorageError(f"model artifact directory is unavailable: {root}")
        metadata_path = root / "metadata.json"
        if metadata_path.is_symlink():
            raise ModelArtifactStorageError("model artifact metadata must not be a symlink")
        try:
            raw = metadata_path.read_bytes()
            if len(raw) > _MAX_METADATA_BYTES:
                raise ModelArtifactStorageError("model artifact metadata exceeds configured limit")
            metadata = ModelArtifactMetadata.from_dict(json.loads(raw.decode("utf-8")))
        except ModelArtifactError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            raise ModelArtifactStorageError(f"model artifact metadata is malformed: {root}") from exc
        artifact = cls(root, metadata)
        artifact.assert_valid()
        return artifact

    @classmethod
    def create_from_fine_tuning_run(
        cls,
        run: "FineTuningRunResult",
        directory: Path | str,
        *,
        model_id: str | None = None,
        evaluation_reference: EvaluationReference | None = None,
        created_at: str | None = None,
    ) -> "ModelArtifact":
        from backend_ai.training.fine_tuning import FineTuningStatus

        if not isinstance(run, object) or getattr(run, "status", None) is not FineTuningStatus.COMPLETED:
            raise ModelArtifactError("model artifact requires a completed Phase 11.3 run")
        candidate = getattr(run, "candidate_model", None)
        if candidate is None or not candidate.model_path:
            raise ModelArtifactError("completed run has no candidate checkpoint")
        source_checkpoint = Path(candidate.model_path).expanduser()
        if not source_checkpoint.is_file():
            raise ModelArtifactStorageError(f"fine-tuning checkpoint is unavailable: {source_checkpoint}")
        root = Path(directory).expanduser()
        if root.exists():
            raise ModelArtifactConflictError(f"model artifact directory already exists: {root}")
        if root.parent.is_symlink():
            raise ModelArtifactStorageError("model artifact parent must not be a symlink")
        artifact_model_id = model_id or f"fodci-{run.configured_candidate_model_version if hasattr(run, 'configured_candidate_model_version') else candidate.model_version}"
        if model_id is None:
            artifact_model_id = f"fodci-{candidate.model_version}"
        eval_ref = evaluation_reference or EvaluationReference.not_evaluated()
        training_config = dict(getattr(run, "configuration"))
        checkpoint_relative = MODEL_ARTIFACT_CHECKPOINT
        destination = root / checkpoint_relative
        checkpoint_fingerprint = "sha256:" + _file_sha256(source_checkpoint)
        provenance = _run_provenance(run, source_checkpoint)
        metadata_without_fingerprint = {
            "format": MODEL_ARTIFACT_FORMAT,
            "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "model_version": candidate.model_version,
            "model_id": artifact_model_id,
            "base_model": run.base_model,
            "dataset_version": run.dataset.dataset_version,
            "dataset_fingerprint": run.dataset.dataset_fingerprint,
            "training_config": training_config,
            "training_config_fingerprint": compute_training_config_fingerprint(training_config),
            "checkpoint": checkpoint_relative,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "evaluation_reference": eval_ref,
            "created_at": created_at or _utc_now(),
            "artifact_fingerprint": "sha256:" + "0" * 64,
            "provenance": provenance,
        }
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
        try:
            temporary_destination = temporary_root / checkpoint_relative
            temporary_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_checkpoint, temporary_destination)
            with temporary_destination.open("rb") as stream:
                os.fsync(stream.fileno())
            actual_checkpoint = "sha256:" + _file_sha256(temporary_destination)
            metadata_without_fingerprint["checkpoint_fingerprint"] = actual_checkpoint
            metadata_without_fingerprint["artifact_fingerprint"] = compute_artifact_fingerprint(ModelArtifactMetadata(**metadata_without_fingerprint))
            metadata = ModelArtifactMetadata(**metadata_without_fingerprint)
            _atomic_write_json(temporary_root / "metadata.json", metadata.to_dict())
            _atomic_write_json(temporary_root / "evaluation.json", eval_ref.to_dict())
            temporary_artifact = cls(temporary_root, metadata)
            temporary_artifact.assert_valid()
            os.replace(temporary_root, root)
            temporary_root = Path()
            return cls(root, metadata)
        except Exception:
            if temporary_root and str(temporary_root) != ".":
                shutil.rmtree(temporary_root, ignore_errors=True)
            raise


@dataclass(frozen=True, slots=True)
class ModelArtifactRegistryEntry:
    model_id: str
    model_version: str
    artifact_directory: str
    artifact_fingerprint: str
    status: str = "UNASSIGNED"

    def __post_init__(self) -> None:
        if not _MODEL_ID_PATTERN.fullmatch(self.model_id) or not _MODEL_VERSION_PATTERN.fullmatch(self.model_version) or not _FINGERPRINT_PATTERN.fullmatch(self.artifact_fingerprint):
            raise ModelArtifactError("registry entry identity is invalid")
        if self.status not in {"UNASSIGNED", "CANDIDATE", "OFFICIAL"}:
            raise ModelArtifactError("registry entry status is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "model_version": self.model_version, "artifact_directory": self.artifact_directory, "artifact_fingerprint": self.artifact_fingerprint, "status": self.status}

    @classmethod
    def from_dict(cls, payload: Any) -> "ModelArtifactRegistryEntry":
        if not isinstance(payload, Mapping) or set(payload) != {"model_id", "model_version", "artifact_directory", "artifact_fingerprint", "status"}:
            raise ModelArtifactStorageError("registry entry fields are invalid")
        return cls(payload["model_id"], payload["model_version"], payload["artifact_directory"], payload["artifact_fingerprint"], payload["status"])


class ModelArtifactRegistry:
    """Small local immutable index; it never auto-promotes a model."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else None
        self._entries: dict[str, ModelArtifactRegistryEntry] = {}
        self._current_candidate: str | None = None
        self._current_official: str | None = None
        self._loaded_digest: str | None = None
        self.reload()

    def reload(self) -> None:
        if self.path is None:
            self._entries = {}
            self._current_candidate = None
            self._current_official = None
            self._loaded_digest = None
            return
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ModelArtifactStorageError("model registry must not use symlinks")
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            self._entries = {}
            self._current_candidate = None
            self._current_official = None
            self._loaded_digest = None
            return
        except OSError as exc:
            raise ModelArtifactStorageError("model registry is unavailable") from exc
        if len(raw) > _MAX_REGISTRY_BYTES:
            raise ModelArtifactStorageError("model registry exceeds configured limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, Mapping) or set(payload) != {"format", "schema_version", "artifacts", "current_candidate_model_id", "current_official_model_id"} or payload["format"] != MODEL_ARTIFACT_REGISTRY_FORMAT or payload["schema_version"] != MODEL_ARTIFACT_REGISTRY_SCHEMA_VERSION:
                raise ModelArtifactStorageError("model registry header is invalid")
            raw_entries = payload["artifacts"]
            if not isinstance(raw_entries, Mapping):
                raise ModelArtifactStorageError("model registry artifacts must be an object")
            self._entries = {key: ModelArtifactRegistryEntry.from_dict(value) for key, value in raw_entries.items()}
            if set(self._entries) != {entry.model_id for entry in self._entries.values()}:
                raise ModelArtifactStorageError("model registry keys do not match entry IDs")
            self._current_candidate = _optional_entry_id(payload["current_candidate_model_id"], self._entries)
            self._current_official = _optional_entry_id(payload["current_official_model_id"], self._entries)
            self._validate_unique_versions()
        except ModelArtifactError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ModelArtifactStorageError("model registry is malformed") from exc
        self._loaded_digest = "sha256:" + hashlib.sha256(raw).hexdigest()

    def list_artifacts(self) -> tuple[ModelArtifactRegistryEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def get(self, model_id: str) -> ModelArtifactRegistryEntry | None:
        return self._entries.get(model_id)

    def require(self, model_id: str) -> ModelArtifactRegistryEntry:
        found = self.get(model_id)
        if found is None:
            raise ModelArtifactError(f"model artifact does not exist: {model_id}")
        return found

    def current_candidate(self) -> ModelArtifactRegistryEntry | None:
        return self._entries.get(self._current_candidate) if self._current_candidate else None

    def current_official(self) -> ModelArtifactRegistryEntry | None:
        return self._entries.get(self._current_official) if self._current_official else None

    def load_artifact(self, model_id: str) -> ModelArtifact:
        """Load and verify the artifact referenced by a registry entry."""

        entry = self.require(model_id)
        artifact = ModelArtifact.load(entry.artifact_directory)
        if artifact.model_id != entry.model_id or artifact.model_version != entry.model_version or artifact.fingerprint != entry.artifact_fingerprint:
            raise ModelArtifactIntegrityError("registry entry does not match the referenced artifact")
        return artifact

    def register(self, artifact: ModelArtifact) -> ModelArtifactRegistryEntry:
        if not isinstance(artifact, ModelArtifact):
            raise ModelArtifactError("register requires ModelArtifact")
        artifact.assert_valid()
        if artifact.model_id in self._entries or any(item.model_version == artifact.model_version for item in self._entries.values()):
            raise ModelArtifactConflictError("model_id or model_version already exists in immutable registry")
        entry = ModelArtifactRegistryEntry(artifact.model_id, artifact.model_version, str(artifact.root), artifact.fingerprint)
        self._entries[entry.model_id] = entry
        try:
            self._persist()
        except Exception:
            self._entries.pop(entry.model_id, None)
            raise
        return entry

    def set_current_candidate(self, model_id: str) -> ModelArtifactRegistryEntry:
        entry = self.require(model_id)
        previous_entries = dict(self._entries)
        previous_pointer = self._current_candidate
        self._current_candidate = model_id
        self._entries[model_id] = ModelArtifactRegistryEntry(entry.model_id, entry.model_version, entry.artifact_directory, entry.artifact_fingerprint, "CANDIDATE")
        try:
            self._persist()
        except Exception:
            self._entries = previous_entries
            self._current_candidate = previous_pointer
            raise
        return self._entries[model_id]

    def set_current_official(self, model_id: str) -> ModelArtifactRegistryEntry:
        """Record an explicit external promotion; this method is never automatic."""

        entry = self.require(model_id)
        previous_entries = dict(self._entries)
        previous_pointer = self._current_official
        self._current_official = model_id
        self._entries[model_id] = ModelArtifactRegistryEntry(entry.model_id, entry.model_version, entry.artifact_directory, entry.artifact_fingerprint, "OFFICIAL")
        try:
            self._persist()
        except Exception:
            self._entries = previous_entries
            self._current_official = previous_pointer
            raise
        return self._entries[model_id]

    def _validate_unique_versions(self) -> None:
        versions = [entry.model_version for entry in self._entries.values()]
        if len(versions) != len(set(versions)):
            raise ModelArtifactStorageError("model registry contains duplicate model versions")

    def _persist(self) -> None:
        if self.path is None:
            self._loaded_digest = None
            return
        if self.path.exists():
            current = "sha256:" + hashlib.sha256(self.path.read_bytes()).hexdigest()
            if self._loaded_digest is None or current != self._loaded_digest:
                raise ModelArtifactConflictError("model registry changed since it was loaded")
        payload = {"format": MODEL_ARTIFACT_REGISTRY_FORMAT, "schema_version": MODEL_ARTIFACT_REGISTRY_SCHEMA_VERSION, "artifacts": {key: self._entries[key].to_dict() for key in sorted(self._entries)}, "current_candidate_model_id": self._current_candidate, "current_official_model_id": self._current_official}
        encoded = (_canonical_json(payload) + "\n").encode("utf-8")
        if len(encoded) > _MAX_REGISTRY_BYTES:
            raise ModelArtifactStorageError("model registry exceeds configured limit")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=self.path.parent, prefix=".model_registry.", suffix=".tmp", delete=False) as stream:
                temporary_path = stream.name
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            self._loaded_digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass


def create_model_artifact_from_fine_tuning_run(run: "FineTuningRunResult", directory: Path | str, *, model_id: str | None = None, evaluation_reference: EvaluationReference | None = None, created_at: str | None = None) -> ModelArtifact:
    return ModelArtifact.create_from_fine_tuning_run(run, directory, model_id=model_id, evaluation_reference=evaluation_reference, created_at=created_at)


def compute_training_config_fingerprint(config: Mapping[str, Any]) -> str:
    if not isinstance(config, Mapping) or not config:
        raise ModelArtifactError("training configuration must be a non-empty mapping")
    return "sha256:" + hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def compute_artifact_fingerprint(metadata: ModelArtifactMetadata) -> str:
    if not isinstance(metadata, ModelArtifactMetadata):
        raise ModelArtifactError("compute_artifact_fingerprint requires ModelArtifactMetadata")
    payload = {
        "format": metadata.format,
        "schema_version": metadata.schema_version,
        "model_version": metadata.model_version,
        "model_id": metadata.model_id,
        "base_model": {"model_name": metadata.base_model.model_name, "model_version": metadata.base_model.model_version, "model_fingerprint": metadata.base_model.model_fingerprint, "tokenizer_version": metadata.base_model.tokenizer_version},
        "dataset_version": metadata.dataset_version,
        "dataset_fingerprint": metadata.dataset_fingerprint,
        "training_config": _thaw(metadata.training_config),
        "training_config_fingerprint": metadata.training_config_fingerprint,
        "checkpoint": metadata.checkpoint,
        "checkpoint_fingerprint": metadata.checkpoint_fingerprint,
        "evaluation_reference": metadata.evaluation_reference.to_dict(),
        "provenance": _thaw(metadata.provenance),
    }
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _run_provenance(run: "FineTuningRunResult", source_checkpoint: Path) -> dict[str, Any]:
    return {
        "fine_tuning_protocol_version": run.protocol_version,
        "training_run_id": run.run_id,
        "source_checkpoint": str(source_checkpoint),
        "resumed_from": run.resumed_from,
        "metrics": [_thaw(item) for item in run.metrics],
        "checkpoints": [_thaw(item) for item in run.checkpoints],
        "software": _thaw(run.software),
        "hardware": _thaw(run.hardware),
    }


def _optional_entry_id(value: Any, entries: Mapping[str, ModelArtifactRegistryEntry]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in entries:
        raise ModelArtifactStorageError("registry pointer references an unknown model")
    return value


def _read_json(path: Path) -> Any:
    if path.is_symlink():
        raise ModelArtifactStorageError("artifact JSON files must not be symlinks")
    return json.loads(path.read_text(encoding="utf-8"))


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_thaw(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _validate_json_value(value: Any, name: str, depth: int, maximum_depth: int) -> None:
    if depth > maximum_depth:
        raise ModelArtifactError(f"{name} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ModelArtifactError(f"{name} contains an invalid key")
            _validate_json_value(item, f"{name}.{key}", depth + 1, maximum_depth)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{name}[{index}]", depth + 1, maximum_depth)
        return
    raise ModelArtifactError(f"{name} contains an unsupported value type")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "EvaluationReference",
    "MODEL_ARTIFACT_CHECKPOINT",
    "MODEL_ARTIFACT_FORMAT",
    "MODEL_ARTIFACT_REGISTRY_FORMAT",
    "MODEL_ARTIFACT_REGISTRY_SCHEMA_VERSION",
    "MODEL_ARTIFACT_SCHEMA_VERSION",
    "ModelArtifact",
    "ModelArtifactConflictError",
    "ModelArtifactError",
    "ModelArtifactIntegrityError",
    "ModelArtifactMetadata",
    "ModelArtifactRegistry",
    "ModelArtifactRegistryEntry",
    "ModelArtifactStorageError",
    "ModelArtifactVerification",
    "compute_artifact_fingerprint",
    "compute_training_config_fingerprint",
    "create_model_artifact_from_fine_tuning_run",
]
