# Scripts

This directory is reserved for small, reviewed project-maintenance scripts. Phase 0 intentionally adds no operational automation because the project does not yet execute tools, manage environments, or run an agent loop.


## Phase 11.2 training dataset

`run_phase112_training_dataset.py` consumes only an existing local Experience Record store and writes the deterministic training artifact directory. It never creates source experiences, generates synthetic examples, loads a model, tokenizes data, trains, or changes weights.

```text
python scripts/run_phase112_training_dataset.py \
  --experience-store path/to/experience_records.json \
  --output artifacts/training/dataset-v1 \
  --version dataset-v1 \
  --seed 2026
```

The command reports the source/accepted/rejected/duplicate counts, train/validation/test counts, version, and final fingerprint. The output contains `manifest.json`, `metadata.json`, `train.json`, `validation.json`, and `test.json`.


## Phase 11.3 offline fine-tuning

`run_phase113_fine_tuning.py` is a developer-only workflow. It is intentionally separate from the normal `fodci` Agent runtime and consumes only an existing Phase 11.2 artifact plus a compatible local base checkpoint.

```text
python scripts/run_phase113_fine_tuning.py \
  --base-checkpoint artifacts/checkpoints/fodci-tiny-v1.pt \
  --dataset-directory artifacts/training/dataset-v1 \
  --run-id candidate-v1-smoke \
  --candidate-model-version candidate-v1 \
  --epochs 1 \
  --max-steps 1 \
  --batch-size 1 \
  --device cpu \
  --output-directory artifacts/training_runs
```

The run writes `run.json`, `metrics.json`, and run-linked `initial.pt`, intermediate, and `final.pt` checkpoints under `artifacts/training_runs/<run_id>/`. A compatible Phase 11.3 checkpoint can be resumed with `--resume-checkpoint`; resume requires a new run ID and preserves `resumed_from` lineage. The output is a candidate trained artifact only and is not production acceptance.
