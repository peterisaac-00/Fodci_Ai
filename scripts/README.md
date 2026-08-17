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
