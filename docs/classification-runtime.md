# Runtime-local classification assets

## Interface

`agent.task.ClassificationAssets.from_files(...)` is the asset-loading seam. Callers provide:

- a leaf registry JSON path;
- a task configuration JSON path;
- optionally, a corpus JSON path for Stage 2 descriptions and examples.

The loader validates schemas, duplicate corpus IDs, and corpus-to-registry membership. It never searches repository directories or selects a named production dataset implicitly.

The two-stage evaluator requires explicit model, data, registry, corpus, task, and report paths. `--metadata-fields` may replace `--task-config` for callers that construct task configuration at invocation time.

## Repository versus runtime

Tracked:

- source and launcher code;
- synthetic `cfg/task/*.example.json` files;
- synthetic tests;
- approved `data/knowledge/**/*.json` files.

Runtime-local only:

- datasets and canonical records;
- production registries and corpora;
- parquet exports;
- checkpoints, models, generated reports, and analysis artifacts.

## Synthetic smoke

```bash
python -m pytest tests/evaluation/test_synthetic_classification_smoke.py
```

The smoke uses an injected deterministic generator and fabricated assets. A real run uses the same classification interface with local model generation.
