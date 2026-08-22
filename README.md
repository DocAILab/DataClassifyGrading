# DataClassifyGrading

A two-stage leaf-classification framework with strict JSON model outputs:

1. Stage 1 selects five prompt choice IDs from a runtime-loaded leaf registry.
2. Stage 2 selects one local bundle ID and immediately decodes it to the canonical `category_id`.

## Repository data policy

The repository contains code, engineering configuration, synthetic examples/tests, and the approved `data/knowledge/**/*.json` knowledge base. Production datasets, registries, corpora, model inputs, reports, and training artifacts must remain local and untracked.

## Synthetic verification

```bash
python -m pip install -e .
python -m pytest
```

The examples in `cfg/task/*.example.json` are fabricated and may be used for smoke tests only.

## Local classification

Supply every production asset by explicit path:

```bash
python -m script.verl.sft.evaluate_true_e2e \
  --model-path <local-model-dir> \
  --data <local-test.parquet> \
  --registry <local-registry.json> \
  --corpus <local-corpus.json> \
  --task-config <local-task.json> \
  --report <local-output-report.json>
```

Nothing under those runtime paths should be added to Git. See `docs/classification-runtime.md`.
