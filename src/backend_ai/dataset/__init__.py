"""Local streaming dataset pipeline for Phase 2.4."""

from backend_ai.dataset.config import DEFAULT_EXTENSIONS, DatasetConfig
from backend_ai.dataset.documents import Document, DocumentLoadResult, LoadIssue
from backend_ai.dataset.loader import LocalDocumentLoader
from backend_ai.dataset.pipeline import FodciDatasetPipeline
from backend_ai.dataset.samples import TokenSequenceBuilder, TrainingExample

__all__ = [
    "DEFAULT_EXTENSIONS",
    "DatasetConfig",
    "Document",
    "DocumentLoadResult",
    "FodciDatasetPipeline",
    "LoadIssue",
    "LocalDocumentLoader",
    "TokenSequenceBuilder",
    "TrainingExample",
]
