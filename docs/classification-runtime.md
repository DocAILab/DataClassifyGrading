# Runtime-local classification and training

This repository contains the task contracts and launchers. Production data and
model assets are supplied at invocation time; no command below searches the
checkout for a dataset, registry, corpus, grading standard, parquet release,
checkpoint, or report.

## Fresh checkout

The CPU test suite needs the test extra (which supplies `pytest` and
`pyarrow`):

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The VeRL server environment is separate. On the authorized server, install
CUDA 12.8 PyTorch first and then `requirements/verl.txt` using the exact
interpreter that will launch VeRL. The compatibility files deliberately do not
invent a vLLM pin; formal RLOO requires an exact version from the local VeRL
0.8 compatibility lock (`--vllm-version`). See [server-runbook.md](server-runbook.md).

Tracked examples are synthetic (`cfg/task/*.example.json`); the approved
knowledge maps under `data/knowledge/**/*.json` are not production datasets.
Keep production raw files, standards, registries, corpora, canonical records,
parquet releases, checkpoints, reports, and environment manifests outside the
checkout (or in ignored local directories). See [data-policy.md](data-policy.md).

## Runtime contract

| Concern | Required production contract |
| --- | --- |
| Prompt metadata | `metadata_fields` is exactly `field_name` for the formal finance+shougang release. Do not expose `table_name` implicitly. |
| Classification standard | A runtime-local standard is converted outside this repository into the registry/corpus JSON consumed by `--registry` and `--corpus`; those paths are explicit. |
| Canonical target | The resolved leaf is the canonical `category_id` (`target.category_id`, represented by the `leaf` resolution block). |
| Joint target | Stage 2 predicts the leaf and `data_level`; grading JSON uses `gt_field: "data_level"`. |
| Grading standards | Formal finance+shougang runs use a manifest with exactly one hashed grading JSON for each dataset. |
| Release lineage | Every SFT release writes `export_report.json` and per-split parquet SHA-256 values. The report is mandatory reference-provenance evidence. |

A task config may carry other task metadata, but the formal cascade validator
rejects anything other than `metadata_fields: ["field_name"]`. The `leaf` and
`data_level` labels are never inferred from a filename or a repository-local
default.

## Exact CLI inventory

Use the following commands with runtime-local paths (shown as shell variables
only to keep production paths out of Git).

### Normalize and resolve records

```bash
python -m script.preprocessing.cli preprocess \
  --input "$RAW/$DATASET.xlsx" \
  --mapping "$MAP/$DATASET.mapping.json" \
  --output "$RUN/processed/$DATASET/all.json" \
  --dataset "$DATASET"

python -m script.canonical.build \
  --processed-dir "$RUN/processed" \
  --canonical-dir "$RUN/canonical" \
  --config-file "$CFG/datasets.json" \
  --registry-dir "$ASSETS/registries" \
  --corpus-dir "$ASSETS/corpora" \
  --dataset finance \
  --dataset shougang

python -m script.canonical.split \
  --canonical-dir "$RUN/canonical" \
  --dataset finance \
  --dataset shougang \
  --split-type random \
  --seed 42
```

`canonical.build` resolves each configured leaf against an explicit standard
registry (and corpus for code IDs). `canonical.split` writes schema-v2
`all.json`, compatibility split views, and a deterministic split report;
unresolved records are excluded rather than silently trained.

### Audit field-only prompts

Run the identifiability gate in bundle mode. It audits each dataset against
its exact classification and grading standard, then writes one redacted
finance+shougang bundle for reference provenance:

```bash
python -m script.analysis.audit_prompt_conflicts --bundle \
  --canonical finance="$RUN/canonical/finance/all.json" \
  --canonical shougang="$RUN/canonical/shougang/all.json" \
  --classification-standard finance="$STANDARDS/finance.classification.json" \
  --classification-standard shougang="$STANDARDS/shougang.classification.json" \
  --grading-standard finance="$STANDARDS/finance.grading.json" \
  --grading-standard shougang="$STANDARDS/shougang.grading.json" \
  --split train \
  --level-field data_level \
  --report "$RUN/audits/finance-shougang.prompt.json"
```

The report stores only aggregate counts and standard hashes. A non-zero
`conflict_keys` is a blocking failure: normalized `field_name` plus the two
standard hashes must identify one `(leaf, data_level)` pair in each dataset.
The bundle must be passed to `record_checkpoint`; retain its per-dataset
entries for review.

### Export and validate one SFT release

Export each dataset independently. Formal joint runs use `field_name` and a
grading config whose ground-truth field is `data_level`:

```bash
python -m script.verl.sft.export \
  --canonical "$RUN/canonical/finance/all.json" \
  --output-dir "$RUN/releases/sft/finance" \
  --registry "$ASSETS/registries/finance.registry.json" \
  --corpus "$ASSETS/corpora/finance.corpus.json" \
  --metadata-fields field_name \
  --task-config "$CFG/finance.task.json" \
  --grading-config "$STANDARDS/finance.grading.json"

python -m script.verl.sft.validate \
  --dataset-dir "$RUN/releases/sft/finance" \
  --registry "$ASSETS/registries/finance.registry.json" \
  --corpus "$ASSETS/corpora/finance.corpus.json" \
  --metadata-fields field_name \
  --task-config "$CFG/finance.task.json" \
  --grading-config "$STANDARDS/finance.grading.json" \
  --report "$RUN/releases/sft/finance/validation.json"
```

Repeat both commands for `shougang`. `export` publishes a new directory only
after validation and writes `train.parquet`, `val.parquet`, `test.parquet`,
and **always** `export_report.json`. A non-empty output directory is never
overwritten. Label and `data_level` gaps must be passed or explicitly
reviewed/waived; the waiver is recorded in that report.

Build the approved sqrt-weighted SFT mixture only from those passed releases:

```bash
python -m script.verl.common.build_mixture \
  --family sft \
  --input finance="$RUN/releases/sft/finance" \
  --input shougang="$RUN/releases/sft/shougang" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --registry "$ASSETS/registries/shared.registry.json" \
  --corpus "$ASSETS/corpora/shared.corpus.json" \
  --task-config "$CFG/cascade.task.json" \
  --metadata-fields field_name \
  --output-dir "$RUN/releases/sft/finance-shougang"
```

The train mixture uses `p(dataset) ∝ sqrt(source_count)`. Its `val.parquet`
and `test.parquet` are trainer inputs/diagnostics; official metrics remain
per-dataset.

### Joint evaluation metrics

`evaluate_true_e2e` requires an explicit report path and can score a joint
`leaf + data_level` answer when `--grading-config` is supplied:

```bash
python -m script.verl.sft.evaluate_true_e2e \
  --model-path "$MODEL" \
  --data "$RUN/releases/sft/finance/test.parquet" \
  --registry "$ASSETS/registries/finance.registry.json" \
  --corpus "$ASSETS/corpora/finance.corpus.json" \
  --task-config "$CFG/finance.task.json" \
  --grading-config "$STANDARDS/finance.grading.json" \
  --report "$RUN/metrics/finance.true-e2e.json"
```

The report exposes strict joint EM (`strict_joint_em`, also `joint_em`) and
per-head diagnostics. `composite_macro_f1` treats `(canonical leaf,
data_level)` as one class; it is **not** the average of leaf and level F1.
Aggregate finance and shougang reports with equal dataset weight:

```bash
python -m script.analysis.aggregate_joint_metrics \
  --input finance="$RUN/metrics/finance.true-e2e.json" \
  --input shougang="$RUN/metrics/shougang.true-e2e.json" \
  --output "$RUN/metrics/finance-shougang.json"
```

## Reference-to-RLOO chain

The required order is:

```text
passed per-dataset SFT exports
  -> --family sft sqrt mixture
  -> frozen reference SFT run
  -> verified single-rank LoRA merge
  -> record_environment
  -> record_checkpoint (sha256-tree-v2)
  -> per-dataset RL export
  -> --family rl-cascade sqrt mixture
  -> validate_cascade
  -> formal RLOO preflight and VeRL launch
```

The complete server commands, including the per-dataset grading manifest,
are in [server-runbook.md](server-runbook.md). In brief:

1. Source a runtime copy of `cfg/sft/reference.env.example` and run
   `script/verl/sft/run.sh` with its `HYDRA_OVERRIDES` array.
2. Merge the resulting `world_size=1, rank=0` LoRA checkpoint with
   `script.verl.sft.merge_lora_checkpoint` into a new HF directory.
3. Run `script.verl.common.record_environment` with the selected interpreter
   and `--gpu-target "RTX PRO 6000 96GB"`.
4. Run `script.verl.sft.record_checkpoint` with the prompt-audit bundle and
   grading manifest. Its `sha256-tree-v2` digest, the passed
   `export_report.json`, effective Hydra config, environment manifest, base
   model, prompt audit, commit, and global step form the only accepted RL
   reference. RLOO must verify this provenance against the exact model path;
   every reference must carry current, complete provenance.
5. Export each canonical dataset with `script.verl.rl.export`, then build a
   second mixture with `--family rl-cascade`. Validate it with
   `script.verl.rl.validate_cascade`, passing a manifest that covers exactly
   `finance` and `shougang` grading standards. The formal release is Stage-1
   only; AgentLoop performs the dynamic Stage-2 turn during rollout.
6. Launch `script.verl.rl.rloo_experiment` with
   `--dataset finance+shougang`, the merged reference, the RL mixture,
   `--grading-manifest`, `--reference-provenance`, and an exact
   `--vllm-version`. `--dry-run` is the local inspection mode; the real launch
   is server-only.

## GPU gate

SFT merge and formal RLOO are authorized only on the designated
**RTX PRO 6000 96GB** server (CUDA 12.8, PyTorch 2.8.0, VeRL 0.8.0, one
visible GPU). Run `record_environment` immediately before releasing the
reference and retain its secret-free manifest. Confirm `torch_cuda.cuda_available`,
`device_name`, `device_count`, and approximately 96 GiB of VRAM before
starting VeRL. A CPU checkout may run unit tests and `--dry-run`; it is not a
training result. Run formal training only after the designated host gate and
current reference-provenance checks pass.
