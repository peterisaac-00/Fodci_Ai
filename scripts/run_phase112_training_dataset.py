#!/usr/bin/env python3
"""Build the Phase 11.2 training artifact from an explicit local Experience store."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from backend_ai.agent.dataset_versioning import DatasetVersionRegistry, DatasetVersioner
from backend_ai.agent.experience_records import ExperienceRecordStore
from backend_ai.agent.training_dataset import TrainingDatasetBuilder, TrainingDatasetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic Phase 11.2 training dataset.")
    parser.add_argument("--experience-store", required=True, type=Path, help="Path to an existing ExperienceRecord JSON store.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for manifest, metadata, and split artifacts.")
    parser.add_argument("--version", default="dataset-v1", help="Immutable dataset version name, for example dataset-v1.")
    parser.add_argument("--seed", default=2026, type=int, help="Deterministic train/validation/test split seed.")
    parser.add_argument("--created-at", default="deterministic-build", help="Bounded audit metadata; it is not part of the content fingerprint.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = DatasetVersionRegistry()
    versioner = DatasetVersioner(registry=registry)
    config = TrainingDatasetConfig(dataset_version=args.version, created_at=args.created_at)
    if config.split_policy.seed != args.seed:
        from dataclasses import replace

        config = replace(config, split_policy=replace(config.split_policy, seed=args.seed))
    store = ExperienceRecordStore(args.experience_store)
    builder = TrainingDatasetBuilder(config=config, versioner=versioner)
    try:
        result = builder.build_from_store(store)
        artifact_path = result.artifact.write(args.output)
    except Exception as exc:
        print(f"Phase 11.2 dataset build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Training artifact: {artifact_path}")
    print(f"Dataset version: {result.artifact.manifest.dataset_version}")
    print(f"Dataset fingerprint: {result.artifact.manifest.dataset_fingerprint}")
    print(f"Source records: {result.report.source_record_count}")
    print(f"Accepted records: {result.report.accepted_record_count}")
    print(f"Rejected records: {result.report.rejected_record_count}")
    print(f"Duplicates: {result.report.duplicate_count}")
    print(f"Train/validation/test: {len(result.artifact.train)}/{len(result.artifact.validation)}/{len(result.artifact.test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
