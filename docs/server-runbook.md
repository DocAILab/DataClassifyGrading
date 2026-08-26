# Authorized server runbook

This is the operational sequence for the formal finance+shougang release. It
assumes an authorized **RTX PRO 6000 96GB** host with one visible GPU, CUDA
12.8, Python 3.12, PyTorch 2.8.0, VeRL 0.8.0, and a locally frozen exact
vLLM version. Replace every path below with a runtime-local path; do not copy
production paths, reports, or checkpoints into Git.

## 0. Set the runtime root and verify the host

```bash
export ROOT="$PWD"
export RUN="/srv/dataclassify/runs/<run-id>"
export RAW="/srv/dataclassify/raw"
export ASSETS="/srv/dataclassify/assets"
export STANDARDS="/srv/dataclassify/standards"
export CFG="/srv/dataclassify/config"
export PYTHON_BIN="/srv/dataclassify/venv/bin/python"
export VLLM_VERSION="<exact-version-from-the-approved-verl-0.8-lock>"
mkdir -p "$RUN" "$RUN/audits" "$RUN/metrics" "$RUN/releases"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON_BIN" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
"$PYTHON_BIN" -c 'import importlib.metadata as m; print(m.version("verl"), m.version("vllm"))'
```

The host gate must show an RTX PRO 6000 with roughly 96 GiB VRAM, CUDA 12.8,
and one visible device. A CPU interpreter is suitable only for tests and
`--dry-run`; formal training is restricted to the designated host.

Install the server stack once in this exact interpreter:

```bash
"$PYTHON_BIN" -m pip install -r "$ROOT/requirements/verl.txt"
"$PYTHON_BIN" -m pip install -e "$ROOT" --no-deps
"$PYTHON_BIN" -m pip check
```

`requirements/verl*.txt` intentionally leave vLLM unpinned. The approved
VeRL 0.8 lock supplies the exact version, which is passed to RLOO as
`--vllm-version`; never infer or silently upgrade it.

## 1. Build and audit canonical data

For each dataset, normalize the runtime raw source, resolve it against the
runtime standard registry/corpus, and split the canonical records:

```bash
for DATASET in finance shougang; do
  python -m script.preprocessing.cli preprocess \
    --input "$RAW/$DATASET.xlsx" \
    --mapping "$ASSETS/mappings/$DATASET.mapping.json" \
    --output "$RUN/processed/$DATASET/all.json" \
    --dataset "$DATASET"
done

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

The formal task config must expose exactly `field_name` and `table_name`:

```json
{"metadata_fields": ["field_name", "table_name"]}
```

Contract change 2026-08-25 (owner decision): field-only prompts are ambiguous
on real finance+shougang data (audit: 1552 conflict keys, ~54% train rows);
+table_name resolves to 15 keys / 30 rows (shougang) and 0 (finance).

The classification standard is represented by the explicit registry/corpus
JSON paths. There is no repository-local production standard lookup.

Audit both field-only prompt contracts and write one redacted bundle. The
key combines normalized `field_name`, classification-standard SHA-256, and
grading-standard SHA-256, and must identify one `(leaf, data_level)` pair per
dataset:

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

Stop if the bundle is not `status: passed` with `conflict_keys: 0`. It contains
redacted per-dataset entries and is the report supplied to
`record_checkpoint --prompt-audit-bundle`.

## 2. Publish and validate per-dataset SFT releases

Export and validate the two source releases independently. `field_name` and
`table_name` are required explicitly; `data_level` is supplied by the grading
config.

```bash
for DATASET in finance shougang; do
  python -m script.verl.sft.export \
    --canonical "$RUN/canonical/$DATASET/all.json" \
    --output-dir "$RUN/releases/sft/$DATASET" \
    --registry "$ASSETS/registries/$DATASET.registry.json" \
    --corpus "$ASSETS/corpora/$DATASET.corpus.json" \
    --metadata-fields field_name table_name \
    --task-config "$CFG/$DATASET.task.json" \
    --grading-config "$STANDARDS/$DATASET.grading.json"

  python -m script.verl.sft.validate \
    --dataset-dir "$RUN/releases/sft/$DATASET" \
    --registry "$ASSETS/registries/$DATASET.registry.json" \
    --corpus "$ASSETS/corpora/$DATASET.corpus.json" \
    --metadata-fields field_name table_name \
    --task-config "$CFG/$DATASET.task.json" \
    --grading-config "$STANDARDS/$DATASET.grading.json" \
    --report "$RUN/releases/sft/$DATASET/validation.json"
done
```

Each successful export creates `train.parquet`, `val.parquet`, `test.parquet`,
and `export_report.json`. The report is mandatory release evidence: it
contains the label/level gap gate and actual per-split parquet SHA-256 values.
Use a new empty output
directory; publication refuses to overwrite an existing non-empty release.

Build the SFT training mixture only from the two passed releases:

```bash
python -m script.verl.common.build_mixture \
  --family sft \
  --input finance="$RUN/releases/sft/finance" \
  --input shougang="$RUN/releases/sft/shougang" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --registry "$ASSETS/registries/shared.registry.json" \
  --corpus "$ASSETS/corpora/shared.corpus.json" \
  --task-config "$CFG/cascade.task.json" \
  --metadata-fields field_name table_name \
  --output-dir "$RUN/releases/sft/finance-shougang"
```

The train source counts follow `p(dataset) ∝ sqrt(source_count)`. Pooled
validation/test files are diagnostics only; official metrics are computed per
dataset and then averaged equally.

## 3. Run the frozen reference SFT

Copy `cfg/sft/reference.env.example` to a runtime-local file and fill in the
model, mixture, output, and interpreter paths. Keep its Hydra overrides as a
Bash array:

```bash
cp "$ROOT/cfg/sft/reference.env.example" "$RUN/reference.env"
# Edit DATASET_NAME, MODEL_PATH, DATA_DIR, OUTPUT_DIR, PYTHON_BIN, and any
# runtime-local values in $RUN/reference.env.
set -a
source "$RUN/reference.env"
set +a
PYTHON_BIN="$PYTHON_BIN" NUM_GPUS=1 \
  bash "$ROOT/script/verl/sft/run.sh" "${HYDRA_OVERRIDES[@]}"
```

Do not turn the array into one quoted multiline string. VeRL receives each
Hydra override as one argument. Save the effective Hydra configuration and
the selected final `global_step_*` directory under `$RUN`.

## 4. Merge and record the reference

The supported merge layout is one VeRL rank (`world_size=1`, `rank=0`):

```bash
python -m script.verl.sft.merge_lora_checkpoint \
  --checkpoint "$OUTPUT_DIR/checkpoints/global_step_<N>" \
  --base-model "$MODEL_PATH" \
  --output "$RUN/models/reference-merged"
```

Record the selected interpreter and GPU facts without dumping environment
variables or credentials:

```bash
"$PYTHON_BIN" -m script.verl.common.record_environment \
  --output "$RUN/environment.json" \
  --gpu-target "RTX PRO 6000 96GB"
```

Inspect `environment.json` before continuing. It must show CUDA available,
one device, the server GPU name, and approximately 96 GiB total VRAM. The
collector is a secret-free fact record; the host authorization gate remains a
human/server check.

Run `record_checkpoint` for the merged HF model. Its `sha256-tree-v2` digest
is the reference identity used by RL:

```bash
python -m script.verl.sft.record_checkpoint \
  --checkpoint-dir "$RUN/models/reference-merged" \
  --export-report "$RUN/releases/sft/finance-shougang/export_report.json" \
  --effective-config "$RUN/reference-effective.yaml" \
  --base-model "$MODEL_PATH" \
  --environment-report "$RUN/environment.json" \
  --prompt-audit-bundle "$RUN/audits/finance-shougang.prompt.json" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --git-commit "$(git -C "$ROOT" rev-parse HEAD)" \
  --global-step <N> \
  --output "$RUN/reference.provenance.json"
```

`--export-report`, `--prompt-audit-bundle`, and `--grading-manifest` bind the
joint reference to passed, hashed finance+shougang artifacts. The checkpoint
must have complete current provenance before it can be used by RL. Since
2026-08-26 the recorded `checkpoint_dir` is advisory: relocating the model
directory only warns; the sha256-tree digest remains the binding anchor.

## 5. Build and validate the RL cascade release

First publish and validate one five-field RL source release per dataset;
these rows preserve both stages for contract validation:

```bash
for DATASET in finance shougang; do
  python -m script.verl.rl.export \
    --canonical "$RUN/canonical/$DATASET/all.json" \
    --output-dir "$RUN/releases/rl/source/$DATASET" \
    --dataset "$DATASET" \
    --registry "$ASSETS/registries/$DATASET.registry.json" \
    --corpus "$ASSETS/corpora/$DATASET.corpus.json" \
    --metadata-fields field_name table_name \
    --task-config "$CFG/$DATASET.task.json" \
    --grading-config "$STANDARDS/$DATASET.grading.json"
done
```

Create one manifest that verifies the exact grading standard used by each
dataset. Paths may be relative to the manifest and the SHA-256 must match the
file bytes. Fill the digests from the exact files used above (for example,
`sha256sum "$STANDARDS/finance.grading.json" "$STANDARDS/shougang.grading.json"`).

```json
{
  "datasets": {
    "finance": {
      "path": "finance.grading.json",
      "sha256": "<64 lowercase hex characters>"
    },
    "shougang": {
      "path": "shougang.grading.json",
      "sha256": "<64 lowercase hex characters>"
    }
  }
}
```

The manifest is strict: its dataset set is exactly `finance` and `shougang`;
each grading config has level descriptions and `gt_field: "data_level"`.
Build the formal Stage-1-only cascade mixture from those passed RL releases:

```bash
python -m script.verl.common.build_mixture \
  --family rl-cascade \
  --input finance="$RUN/releases/rl/source/finance" \
  --input shougang="$RUN/releases/rl/source/shougang" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --registry "$ASSETS/registries/shared.registry.json" \
  --corpus "$ASSETS/corpora/shared.corpus.json" \
  --task-config "$CFG/cascade.task.json" \
  --metadata-fields field_name table_name \
  --output-dir "$RUN/releases/rl/finance-shougang"
```

Validate the cascade before any RLOO process starts:

```bash
python -m script.verl.rl.validate_cascade \
  --dataset-dir "$RUN/releases/rl/finance-shougang" \
  --registry "$ASSETS/registries/shared.registry.json" \
  --corpus "$ASSETS/corpora/shared.corpus.json" \
  --task-config "$CFG/cascade.task.json" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --report "$RUN/rl-cascade-validation.json"
```

The validator requires `metadata_fields: ["field_name", "table_name"]`,
passed/published release metadata, the approved sqrt policy, no source-id
overlap, both datasets in train, and no Stage-2 rows. A non-zero exit is a
release stop.

## 6. Formal RLOO preflight and launch

Use the merged reference and its provenance, the RL mixture, the same registry,
corpus, task config, and grading manifest:

```bash
"$PYTHON_BIN" -m script.verl.rl.rloo_experiment \
  --dataset finance+shougang \
  --model "$RUN/models/reference-merged" \
  --data-dir "$RUN/releases/rl/finance-shougang" \
  --registry "$ASSETS/registries/shared.registry.json" \
  --corpus "$ASSETS/corpora/shared.corpus.json" \
  --task-config "$CFG/cascade.task.json" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --reference-provenance "$RUN/reference.provenance.json" \
  --output-dir "$RUN/rloo" \
  --python-bin "$PYTHON_BIN" \
  --vllm-version "$VLLM_VERSION"
```

Formal policy fixes the joint release to `finance+shougang`, the cascade
rollout count to `CASCADE_N` or `2 * CASCADE_N` (4 or 8, recorded truthfully
in the run manifest), actor KL settings, and the exact reference digest.
Optional launcher knobs: `--experiment-name` for a stable run name and
`--max-ckpt-keep` for checkpoint rotation. RLOO
verifies provenance against the model directory before launching VeRL; it
does not infer a starting checkpoint or a vLLM version. Use `--dry-run` on a
CPU checkout to inspect commands, but do not call the real launch there.

## 7. Joint metrics and retention

Run true end-to-end evaluation separately on each dataset's test parquet with
`--grading-config`, then aggregate only the two reports:

```bash
python -m script.analysis.aggregate_joint_metrics \
  --input finance="$RUN/metrics/finance.true-e2e.json" \
  --input shougang="$RUN/metrics/shougang.true-e2e.json" \
  --output "$RUN/metrics/finance-shougang.json"
```

Retain the per-dataset reports, the equal-macro aggregate, both prompt audits,
the per-dataset grading manifest and referenced files, SFT and RL
`export_report.json` files, environment manifest, effective config, commit,
reference provenance, and validation logs under the runtime run directory.
Only publish redacted aggregate evidence where policy permits.
