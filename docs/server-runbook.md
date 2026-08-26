# Authorized server runbook

This is the operational sequence for the formal **shougang-only** release. It
assumes an authorized **RTX PRO 6000 96GB** host with one visible GPU and the
dedicated Qwen3.5 environment: Python 3.12, VeRL 0.9.0, and its locally
validated exact CUDA/PyTorch/vLLM lock. Replace every path below with a runtime-local path; do not copy
production paths, reports, or checkpoints into Git.

## 0. Set the runtime root and verify the host

```bash
export ROOT="$PWD"
export RUN="/srv/dataclassify/runs/<run-id>"
export RAW="/srv/dataclassify/raw"
export ASSETS="/srv/dataclassify/assets"
export STANDARDS="/srv/dataclassify/standards"
export CFG="/srv/dataclassify/config"
export PYTHON_BIN="/root/autodl-tmp/envs/verl-qwen35/bin/python"
export VLLM_VERSION="<exact-version-from-the-approved-verl-0.9-lock>"
mkdir -p "$RUN" "$RUN/audits" "$RUN/metrics" "$RUN/releases"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON_BIN" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
"$PYTHON_BIN" -c 'import importlib.metadata as m; print(m.version("verl"), m.version("vllm"))'
```

The host gate must show an RTX PRO 6000 with roughly 96 GiB VRAM and one
visible device; the reported CUDA/PyTorch/vLLM versions must match the approved
VeRL 0.9 lock. A CPU interpreter is suitable only for tests and
`--dry-run`; formal training is restricted to the designated host.

Use the already isolated `verl-qwen35` environment; its validated core is
recorded in `requirements/verl-qwen35-cu130.txt` (torch 2.13.0+cu130,
Transformers 5.10.4, vLLM 0.27.1, VeRL 0.9.0). Do not upgrade it in place and
do not reuse the archived `verl-sft` 0.8 environment:

```bash
"$PYTHON_BIN" -m pip install -e "$ROOT" --no-deps
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -c 'import verl; from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop; print(verl.__version__)'
```

The approved VeRL 0.9 lock supplies the exact vLLM version passed as
`--vllm-version`; never infer or silently upgrade it.

## 1. Build and audit the shougang canonical data

Normalize the runtime raw source, resolve it against the runtime standard
registry/corpus, and split the canonical records:

```bash
python -m script.preprocessing.cli preprocess \
  --input "$RAW/shougang.xlsx" \
  --mapping "$ASSETS/mappings/shougang.mapping.json" \
  --output "$RUN/processed/shougang/all.json" \
  --dataset shougang

python -m script.canonical.build \
  --processed-dir "$RUN/processed" \
  --canonical-dir "$RUN/canonical" \
  --config-file "$CFG/datasets.json" \
  --registry-dir "$ASSETS/registries" \
  --corpus-dir "$ASSETS/corpora" \
  --dataset shougang

python -m script.canonical.split \
  --canonical-dir "$RUN/canonical" \
  --dataset shougang \
  --split-type random \
  --seed 42
```

The formal task config must expose exactly `field_name` and `table_name`:

```json
{"metadata_fields": ["field_name", "table_name"]}
```

Rationale from the historical audit of the former finance+shougang release:
field-only prompts produced 1552 conflict keys (about 54% of train rows).
After adding `table_name`, the residual was 15 keys / 30 rows in shougang and
0 in finance. The formal scope is now shougang-only; keep both fields in every
formal prompt. The historical residual is not a waiver: the current release
audit below must pass with `conflict_keys: 0`.

The classification standard is represented by the explicit registry/corpus
JSON paths. There is no repository-local production standard lookup.

Audit the one shougang source and write one redacted prompt-audit bundle. The
bundle shape is retained for provenance, but it must contain exactly one
`shougang` entry. The key combines normalized `field_name`, `table_name`,
classification-standard SHA-256, and grading-standard SHA-256, and must
identify one `(leaf, data_level)` pair:

```bash
python -m script.analysis.audit_prompt_conflicts --bundle \
  --canonical shougang="$RUN/canonical/shougang/all.json" \
  --classification-standard shougang="$STANDARDS/shougang.classification.json" \
  --grading-standard shougang="$STANDARDS/shougang.grading.json" \
  --split train \
  --level-field data_level \
  --report "$RUN/audits/shougang.prompt.json"
```

Stop if the report is not `status: passed` with `conflict_keys: 0`. It contains
redacted shougang entries and is the report supplied to
`record_checkpoint --prompt-audit-bundle`.

## 2. Publish and validate the shougang SFT release

Export and validate the one shougang source independently. `field_name` and
`table_name` are required explicitly; `data_level` is supplied by the grading
config:

```bash
python -m script.verl.sft.export \
  --canonical "$RUN/canonical/shougang/all.json" \
  --output-dir "$RUN/releases/sft/source/shougang" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --metadata-fields field_name table_name \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json"

python -m script.verl.sft.validate \
  --dataset-dir "$RUN/releases/sft/source/shougang" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --metadata-fields field_name table_name \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json" \
  --report "$RUN/releases/sft/source/shougang/validation.json"
```

Each successful export creates `train.parquet`, `val.parquet`, `test.parquet`,
and `export_report.json`. The report is mandatory release evidence: it
contains the label/level gap gate and actual per-split parquet SHA-256 values.
The formal standardized release requires a passed `label_gap_gate` with no
blocking or waived label/`data_level` gaps. Use a new empty output directory;
publication refuses to overwrite an existing non-empty release.

Create the one-entry grading manifest before building the standardized release.
Copy the exact runtime grading standard next to the manifest (or use its
absolute path); its SHA-256 must match the file bytes. For a relative manifest
path:

```bash
cp "$STANDARDS/shougang.grading.json" "$RUN/shougang.grading.json"
sha256sum "$RUN/shougang.grading.json"
```

Fill the digest from that exact file:

```json
{
  "datasets": {
    "shougang": {
      "path": "shougang.grading.json",
      "sha256": "<64 lowercase hex characters>"
    }
  }
}
```

Build the standardized shougang SFT release from the passed source:

```bash
python -m script.verl.common.build_mixture \
  --family sft \
  --input shougang="$RUN/releases/sft/source/shougang" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --task-config "$CFG/shougang.task.json" \
  --metadata-fields field_name table_name \
  --output-dir "$RUN/releases/sft/shougang"
```

The one `--input shougang=...` is intentional: the approved singleton policy
is a passthrough with weight 1.0 and does not replicate rows.
The resulting `export_report.json` is the report bound to the reference.

## 3. Re-measure the Qwen3.5 token budget, then run the reference SFT

Copy `cfg/sft/reference.env.example` to a runtime-local file and fill in the
model, standardized shougang release, output, interpreter, and measured token
limit:

```bash
cp "$ROOT/cfg/sft/reference.env.example" "$RUN/reference.env"
# Edit MODEL_PATH, DATA_DIR, OUTPUT_DIR, PYTHON_BIN, QWEN35_MAX_TOKENS, and
# other runtime-local values in $RUN/reference.env.
set -a
source "$RUN/reference.env"
set +a
```

The example model is `Qwen3.5-9B`; a local senior SFT checkpoint may be used
through a runtime override. **All token limits from old Qwen2.5 runs are
invalid.** Before starting SFT, remeasure the selected Qwen3.5 model's
chat-template lengths and choose the limit from that report. Do not copy an
old number or invent a new numeric limit:

```bash
"$PYTHON_BIN" -m script.verl.sft.prompt_stats \
  --dataset-dir "$DATA_DIR" \
  --model "$MODEL_PATH" \
  --report "$RUN/metrics/shougang.token-stats.json"

# Set QWEN35_MAX_TOKENS from the measured report, then run the gate:
"$PYTHON_BIN" -m script.verl.sft.check_token_budget \
  --dataset-dir "$DATA_DIR" \
  --model "$MODEL_PATH" \
  --max-length "$QWEN35_MAX_TOKENS" \
  --report "$RUN/metrics/shougang.token-budget.json"
```

A failed token-budget gate stops training. Re-run it whenever the model, chat
template, or exported prompts change. Then launch the frozen reference with
its Hydra overrides array:

```bash
PYTHON_BIN="$PYTHON_BIN" NUM_GPUS=1 \
  bash "$ROOT/script/verl/sft/run.sh" "${HYDRA_OVERRIDES[@]}"
```

Do not turn the array into one quoted multiline string. VeRL receives each
Hydra override as one argument. Save the effective Hydra configuration and
the selected final `global_step_*` directory under `$RUN`.

## 4. Merge and record the one reference

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
is the identity used by RL:

```bash
python -m script.verl.sft.record_checkpoint \
  --checkpoint-dir "$RUN/models/reference-merged" \
  --export-report "$RUN/releases/sft/shougang/export_report.json" \
  --effective-config "$RUN/reference-effective.yaml" \
  --base-model "$MODEL_PATH" \
  --environment-report "$RUN/environment.json" \
  --prompt-audit-bundle "$RUN/audits/shougang.prompt.json" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --git-commit "$(git -C "$ROOT" rev-parse HEAD)" \
  --global-step <N> \
  --output "$RUN/reference.provenance.json"
```

Every provenance input binds the single shougang release: the passed
`export_report.json`, prompt-audit bundle, one-entry grading manifest,
effective config, environment, base model, commit, and global step. The
checkpoint must have complete current provenance before RL. The recorded
`checkpoint_dir` is advisory after relocation; the sha256-tree/count/bytes
remain the binding anti-tamper anchor.

## 5. Build and validate the shougang native-tool RL release

Publish and validate one five-field RL source release; these rows preserve
both stages for contract validation:

```bash
python -m script.verl.rl.export \
  --canonical "$RUN/canonical/shougang/all.json" \
  --output-dir "$RUN/releases/rl/source/shougang" \
  --dataset shougang \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --metadata-fields field_name table_name \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json"
```

Reuse the one-entry manifest created in Section 2. Its dataset set is
strictly `shougang`, and the grading config has level descriptions and
`gt_field: "data_level"`. Build the formal one-episode-per-source native-tool release from the passed
RL source:

```bash
python -m script.verl.common.build_mixture \
  --family rl-cascade \
  --input shougang="$RUN/releases/rl/source/shougang" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --task-config "$CFG/shougang.task.json" \
  --metadata-fields field_name table_name \
  --output-dir "$RUN/releases/rl/shougang"
```

Validate the native-tool release before any RLOO process starts:

```bash
python -m script.verl.rl.validate_cascade \
  --dataset-dir "$RUN/releases/rl/shougang" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --task-config "$CFG/shougang.task.json" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --report "$RUN/rl-cascade-validation.json"
```

The validator requires `metadata_fields: ["field_name", "table_name"]`,
passed/published release metadata, singleton passthrough policy, no source-id
overlap, shougang rows in train, no Stage-2 runtime rows, and `trajectory_format:
qwen3.5-native-tools-v1`. A non-zero exit is a release stop.

At rollout time VeRL 0.9's official ToolAgentLoop exposes exactly three
functions: `search_categories`, `get_category_details`, and
`get_category_examples`. Search is deterministic lexical retrieval over the
runtime registry/corpus JSON—there is no embedding model or vector database.
A trajectory may make no call, one call, or up to three sequential calls.
Assistant tool-call and terminal tokens have policy mask 1; tool observations
have mask 0. The terminal JSON uses a global opaque `choice_id` and the
approved level code.

## 6. Formal RLOO preflight and launch

Use the merged reference and its provenance, the standardized shougang RL
release, the same registry, corpus, task config, and grading manifest:

```bash
"$PYTHON_BIN" -m script.verl.rl.rloo_experiment \
  --dataset shougang \
  --model "$RUN/models/reference-merged" \
  --data-dir "$RUN/releases/rl/shougang" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --task-config "$CFG/shougang.task.json" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --reference-provenance "$RUN/reference.provenance.json" \
  --output-dir "$RUN/rloo" \
  --python-bin "$PYTHON_BIN" \
  --vllm-version "$VLLM_VERSION"
```

Formal policy fixes the release to `shougang`, the RLOO sibling count to
`CASCADE_N` or `2 * CASCADE_N` (4 or 8, recorded truthfully in the run
manifest), `qwen3_coder` tool parsing, at most three sequential calls, one
strict category+level exact-match reward, actor KL settings, and the exact
reference digest. Optional
launcher knobs are `--experiment-name` for a stable run name and
`--max-ckpt-keep` for checkpoint rotation. RLOO verifies provenance against
the model directory before launching VeRL; it does not infer a starting
checkpoint or a vLLM version. Use `--dry-run` on a CPU checkout to inspect
commands, but do not call the real launch there.

## 7. Evaluate shougang directly and retain evidence

Run true end-to-end evaluation on the shougang test parquet and retain the
single report as the official metric:

```bash
"$PYTHON_BIN" -m script.verl.sft.evaluate_true_e2e \
  --model-path "$RUN/models/reference-merged" \
  --data "$RUN/releases/sft/shougang/test.parquet" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json" \
  --report "$RUN/metrics/shougang.true-e2e.json"
```

Retain the shougang report, prompt audit, one-entry grading manifest and
referenced standard, SFT and RL `export_report.json` files, token-budget
reports, environment manifest, effective config, commit, reference provenance,
and validation logs under the runtime run directory. Publish only redacted
evidence where policy permits; keep the metric source-local.
