# Runtime-local classification assets

## Interface

`agent.task.ClassificationAssets.from_files(...)` is the asset-loading seam. Callers provide:

- a leaf registry JSON path;
- a task configuration JSON path;
- optionally, a corpus JSON path for Stage 2 descriptions and examples.

The loader validates schemas, duplicate corpus IDs, and corpus-to-registry membership. It never searches repository directories or selects a named production dataset implicitly.

The two-stage evaluator requires explicit model, data, registry, corpus, task, and report paths. `--metadata-fields` may replace `--task-config` for callers that construct task configuration at invocation time.

## Canonical pipeline (raw → reference checkpoint)

The training data flow is fully runtime-configured; no dataset knowledge is
built into source code. Every stage is deterministic and re-runs to
byte-identical outputs given identical inputs.

```text
tabular raw (xlsx/csv) ──script.preprocessing.cli preprocess──▶ processed/<ds>/all.json
classification standards ──(runtime-local standard conversion; generic
  builder NOT rebuilt yet, see note below)──▶ corpus/registry assets (runtime-local)
processed + config + registry ──script.canonical.build──▶ canonical/<ds>/all.json (schema v2)
canonical ──script.canonical.split──▶ embedded splits + train/val/test views + report
canonical ──script.verl.sft.export──▶ VeRL messages parquet (label-gap gate)
parquet + cfg/sft/reference.env ──script.verl.sft.run.sh──▶ reference SFT checkpoint
checkpoint ──script.verl.sft.record_checkpoint──▶ provenance json (tree sha256)
```

Note on corpus/registry assets: this repository CONSUMES pre-built registry
and corpus JSON through explicit paths. The generic standards→corpus→registry
builder was removed during the data-sanitation pass and is not rebuilt here
yet; until then, converting a classification standard into those assets is a
careful runtime-local step performed outside the repository (original logic is
archived for reference).

Key contracts:

- **Dataset configuration**: a JSON file loaded via
  `agent.task.load_dataset_configs` declares each dataset's leaf field,
  id strategy (`code`/`path`), identity fields, placeholder labels,
  projection exclusions, and registry derivation
  (`standard` / `shared-standard` / `dataset-universe`). See
  `cfg/task/datasets.example.json`. No built-in dataset table exists.
- **Canonical schema v2** is a superset of v1: original record fields stay
  untouched; `path_mask`, `leaf`, and a full `resolution` block (status,
  attempted category_id, raw-text reason) are added, plus `split`
  assignments written back by the canonical splitter. Empty levels remain
  visible in `classification` and are excluded from identity only when the
  dataset config says so.
- **Determinism**: sample ids derive from `uuid5(dataset|db|table|field)` —
  renaming an input file never changes identities; splits sort by stable id
  before assignment (row-order insensitive) and record seed/ratios/order in
  their reports.
- **Label semantics**: trailing bracket-style codes are detected and kept by
  default (`label_notes`); stripping requires `--strip-trailing-codes`.
  Historical hand-edits become auditable rewrite rules (`rewritten_from`).
- **Export gate**: labels appearing only in val/test block SFT export until
  explicitly waived (`--allow-label-gaps`); waivers are recorded. Export
  reports carry per-split parquet SHA-256 as lineage anchors.
- **Reference checkpoint**: one frozen recipe (`cfg/sft/reference.env.example`)
  produces THE shared RL starting checkpoint. Checkpoints from before the
  canonical pipeline are not comparable and must be labelled `legacy/`.

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
