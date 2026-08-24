# Stage 1 BGE-M3 Label-Description Retrieval Design

## Objective

Establish a frozen, training-free Stage 1 dense-retrieval baseline for the
Shougang `classification.level_4` task. The baseline receives only
`metadata.field_name`, ranks all 193 registry labels by similarity to audited
label text, and measures whether the gold label appears in the top 1, 3, or 5.

This experiment isolates candidate retrieval. It does not change SFT, DPO,
prompts, or the Stage 2 classifier, and it does not report an end-to-end
two-stage result.

## Scope and Data Boundary

- Query input is exactly `metadata.field_name`.
- Candidate documents come only from the existing leaf registry.
- A candidate document contains `category_id` and its registry `description`.
- If a registry description is empty, the document uses the label name alone
  and records provenance `label_name_fallback`.
- Registry descriptions must not be inferred from train, val, or test records.
- The full evaluation reads only `val.json`.
- The implementation must not enumerate, open, parse, upload, or read the real
  test split.
- Gold `classification.level_4` values are used only after retrieval to compute
  metrics; they never affect query encoding, candidate encoding, or ranking.
- Every report records `requested_splits=["val"]`,
  `real_test_split_read=false`, and `metadata_fields=["field_name"]`.

## Retrieval Methods

### Primary baseline: BGE-M3 dense retrieval

Use `BAAI/bge-m3` in its official dense retrieval mode:

- frozen encoder;
- normalized `[CLS]` hidden state as the 1024-dimensional dense embedding;
- inner product between normalized query and candidate embeddings;
- no sparse weights, ColBERT vectors, fine-tuning, or label-dependent query
  instructions;
- stable tie-breaking by original registry order.

The corpus encoder runs once for the 193 registry documents. Query encoding is
batched. The run records the exact model path, model identity or revision,
encoder configuration, embedding dimension, dtype, batch size, and checksums of
the registry and serialized embeddings.

### Control baseline: character n-gram retrieval

Run the existing deterministic character n-gram scorer over the same query and
the same registry documents. This is a lexical control, not the main vector
baseline. It must use the same stable tie-breaking and top-k evaluation code as
BGE-M3 so that differences reflect the scorer rather than metric drift.

## Corpus Representation

Each candidate document is rendered deterministically as:

```text
标签名称：{category_id}
标签描述：{description}
```

When `description` is empty, only the first line is emitted. Whitespace is
normalized without translating, expanding, or rewriting registry content.
`corpus_audit.json` records, for every category, its document text, provenance,
description presence, and content checksum. Aggregate coverage distinguishes
independent registry descriptions from label-name fallbacks.

## Architecture

Create a focused package under `src/method/retrieval/`:

- `corpus.py`: render and audit deterministic registry documents;
- `ranking.py`: validate score matrices and perform stable top-k ranking;
- `metrics.py`: compute micro and macro retrieval metrics and coverage;
- `bge_m3.py`: lazily load the frozen encoder and produce normalized dense
  vectors without making the optional GPU dependency necessary for unit tests;
- `evaluation.py`: enforce val-only loading, resumable prediction writing, and
  report assembly;
- `script/evaluate_stage1.py`: command-line entry point;
- `script/run_stage1_bge_m3.sh`: recoverable remote launcher.

Pure ranking, metrics, corpus, and audit code remains independent of PyTorch and
FlagEmbedding. The encoder is injected behind a small interface so tests use
real deterministic arrays rather than loading a multi-gigabyte model.

## Data Flow

1. Load and validate the 193-category leaf registry.
2. Render the audited candidate corpus and record its checksum.
3. Encode and cache normalized BGE-M3 candidate vectors.
4. Load only `val.json`, validate unique `source_id` values, and expose only
   `field_name` to the retriever.
5. Encode query batches and compute the complete query-by-label inner-product
   matrix.
6. Rank all labels with registry-order tie-breaking and write one prediction
   row per `source_id`.
7. Compute BGE-M3 metrics and the character n-gram control metrics using the
   same evaluator.
8. Cross-check prediction counts, unique IDs, OOV/duplicate candidates,
   aggregate metrics, checksums, and data-boundary fields.
9. Write `COMPLETE` only after all consistency checks pass. On failure, write a
   structured `FAILURE` marker and preserve completed artifacts for diagnosis.

## Prediction Contract

Each `predictions.jsonl` row contains:

- `source_id`;
- `field_name`;
- `golden_level_4` for evaluation only;
- ordered `top5` labels;
- aligned similarity scores;
- gold rank among all 193 labels;
- `hit_at_1`, `hit_at_3`, and `hit_at_5`;
- duplicate, OOV, and validity flags;
- retrieval method and corpus checksum.

No other record metadata is written into the retrieval input or prediction
contract.

## Metrics

For each retrieval method, report:

- Recall@1, Recall@3, and Recall@5 over all 2,028 validation rows;
- macro Recall@1, macro Recall@3, and macro Recall@5 over labels with positive
  validation support;
- mean reciprocal rank (MRR);
- mean and median gold rank;
- number and fraction of all 193 labels appearing in at least one Top-5;
- number of unique Top-5 tuples;
- per-class support and Recall@1/3/5;
- frequency buckets derived from train counts, with val used only for
  evaluation;
- duplicate candidates, OOV candidates, invalid rows, and missing gold ranks;
- model load, corpus encoding, query encoding, ranking, and total elapsed time;
- GPU allocated and peak memory where available.

Macro metrics exclude zero-support registry labels and explicitly record the
number of included classes. Frequency-bucket boundaries are computed from
train counts only and written to the run configuration before val scoring.

## Artifacts

The remote run root is:

```text
/root/autodl-tmp/artifacts/shougang/stage1-bge-m3-v1
```

It contains:

- `status.json`;
- `run_config.json`;
- `data_audit.json`;
- `corpus_audit.json`;
- `registry_embeddings.npz` plus checksum;
- `bge_m3/predictions.jsonl`;
- `bge_m3/per_class_metrics.json`;
- `char_ngram/predictions.jsonl`;
- `char_ngram/per_class_metrics.json`;
- `comparison_to_char_ngram.json`;
- `evaluation_report.json`;
- `runtime_report.json`;
- logs and `COMPLETE` or `FAILURE` markers.

The cached embedding file is reused only when the model identity, corpus
checksum, embedding configuration, dimension, and dtype all match.

## Testing Strategy

Implementation follows red-green-refactor. Tests are written and observed
failing before production code for:

- deterministic corpus rendering and description fallback provenance;
- normalized dense vectors and matrix shape validation;
- descending score order with stable registry-order tie-breaking;
- exact Recall@1/3/5, macro recall, MRR, rank, diversity, and coverage values;
- zero-support labels excluded from macro metrics;
- duplicate/OOV rejection and non-finite score rejection;
- unique `source_id` enforcement and deterministic resume behavior;
- val-only loading and explicit refusal of any split other than val;
- report consistency and `real_test_split_read=false`;
- shell launcher arguments and completion-marker behavior.

The remote smoke uses a small val prefix only to validate the real BGE-M3
encoder, 1024-dimensional normalized vectors, GPU execution, prediction schema,
and artifact writing. The full run then evaluates all 2,028 unique validation
IDs. A final verification command reruns the full relevant test suite and
cross-checks every report against the JSONL files before completion is claimed.

## Success Interpretation

This experiment establishes a baseline rather than guaranteeing improvement.
BGE-M3 is considered promising for the next end-to-end experiment when:

- its Recall@5 exceeds the character n-gram control on the same 2,028 rows;
- its macro Recall@5 does not reveal a large degradation hidden by frequent
  classes;
- it produces no invalid, duplicate, or OOV candidates; and
- candidate coverage and Top-5 diversity are materially broader than the old
  collapsed Stage 1 generator.

If these conditions hold, a separate follow-up experiment may feed the actual
BGE-M3 Top-5 into the completed DPO Stage 2 adapter. That integration is outside
the scope of this baseline and must not be inferred from Stage 1 recall alone.
