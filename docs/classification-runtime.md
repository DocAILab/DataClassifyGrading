# Runtime-local classification and training

This repository contains the task contracts and launchers. Production data and
model assets are supplied at invocation time; no command below searches the
checkout for a dataset, registry, corpus, grading standard, parquet release,
checkpoint, or report.

The formal release scope is **shougang only**. Every release, reference, RL
run, and metric report below names that source explicitly; all scores remain
source-local.

## Fresh checkout

The CPU test suite needs the test extra (which supplies `pytest` and
`pyarrow`):

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The VeRL server environment is separate. Formal native-tool RLOO requires
VeRL 0.9.0 in the dedicated Qwen3.5 environment. Its validated core is
recorded in `requirements/verl-qwen35-cu130.txt`: PyTorch 2.13.0+cu130,
Transformers 5.10.4 and vLLM 0.27.1. Use the same interpreter for preflight
and training; never reuse the archived VeRL 0.8 environment. RLOO still requires
its exact local vLLM version via `--vllm-version`. See
[server-runbook.md](server-runbook.md).

Tracked examples are synthetic (`cfg/task/*.example.json`); the approved
knowledge maps under `data/knowledge/**/*.json` are not production datasets.
Keep production raw files, standards, registries, corpora, canonical records,
parquet releases, checkpoints, reports, and environment manifests outside the
checkout (or in ignored local directories). See [data-policy.md](data-policy.md).

## Runtime contract

| Concern | Required production contract |
| --- | --- |
| Formal scope | The formal source is exactly `shougang`; release reports and manifests contain no other source. |
| Prompt metadata | `metadata_fields` is exactly `("field_name", "table_name", "field_description", "table_description")` (four native fields, user decision 2026-08-27). In the historical finance+shougang field-only audit, 1552 conflict keys (about 54% of train rows) were found; after adding `table_name`, the residual was 15 keys / 30 rows in shougang and 0 in finance. The formal scope is now shougang-only; keep all four fields and require a current clean audit. |
| Classification standard | A runtime-local standard is converted outside this repository into the registry/corpus JSON consumed by `--registry` and `--corpus`; those paths are explicit. |
| Canonical target | The resolved leaf is the canonical `category_id` (`target.category_id`, represented by the `leaf` resolution block). |
| Native-tool target | The terminal assistant JSON predicts one opaque global `choice_id` and `data_level`; grading JSON uses `gt_field: "data_level"`. The reward decodes the choice internally and is 1 only when both heads match. |
| Grading standard | The shougang run uses a manifest containing exactly one hashed grading JSON for `shougang`. |
| Release lineage | Every SFT and RL release writes `export_report.json` and per-split parquet SHA-256 values. The report is mandatory reference-provenance evidence. |

A task config may carry other task metadata, but the formal native-tool
validator rejects anything other than `metadata_fields: ["field_name",
"table_name", "field_description", "table_description"]`.
The `leaf` and `data_level` labels are never inferred from a filename or a
repository-local default. Any non-zero `conflict_keys` in the current shougang
audit is a blocking failure; the historical residual is not a waiver.

## Exact CLI inventory

Use the following commands with runtime-local paths (shown as shell variables
only to keep production paths out of Git).

### Normalize and resolve records

```bash
python -m script.preprocessing.cli preprocess \
  --input "$RAW/shougang.xlsx" \
  --mapping "$MAP/shougang.mapping.json" \
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

`canonical.build` resolves each configured leaf against an explicit standard
registry (and corpus for code IDs). `canonical.split` writes schema-v2
`all.json`, compatibility split views, and a deterministic split report;
unresolved records are excluded rather than silently trained.

### Audit field-and-table prompts

Run the identifiability gate for the one shougang source. Bundle mode keeps the
redacted prompt-audit shape required by reference provenance, but the bundle
must contain exactly one `shougang` entry:

```bash
python -m script.analysis.audit_prompt_conflicts --bundle \
  --canonical shougang="$RUN/canonical/shougang/all.json" \
  --classification-standard shougang="$STANDARDS/shougang.classification.json" \
  --grading-standard shougang="$STANDARDS/shougang.grading.json" \
  --split train \
  --level-field data_level \
  --report "$RUN/audits/shougang.prompt.json"
```

The report stores only summary counts and standard hashes. A non-zero
`conflict_keys` is a blocking failure: normalized `field_name` plus
`table_name` and the two standard hashes must identify one `(leaf, data_level)`
pair. The field_name+table_name audit key records the shougang conflict result
above; the formal prompt metadata contract is the four native fields (user
decision 2026-08-27), and no conflict waiver is permitted. Pass this exact redacted report to
`record_checkpoint`.

### Export and validate one SFT release

Export and validate the shougang source independently. `field_name` and
`table_name` are required explicitly; `data_level` is supplied by the grading
config:

```bash
python -m script.verl.sft.export \
  --canonical "$RUN/canonical/shougang/all.json" \
  --output-dir "$RUN/releases/sft/source/shougang" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --metadata-fields field_name table_name field_description table_description \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json"

python -m script.verl.sft.validate \
  --dataset-dir "$RUN/releases/sft/source/shougang" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --metadata-fields field_name table_name field_description table_description \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json" \
  --report "$RUN/releases/sft/source/shougang/validation.json"
```

`export` publishes a new directory only after validation and writes
`train.parquet`, `val.parquet`, `test.parquet`, and **always**
`export_report.json`. A non-empty output directory is never overwritten. The
formal standardized release requires a passed `label_gap_gate` with no
blocking or waived label/`data_level` gaps; the gate and hashes are recorded in
that report.

Before building the release, create the one-entry grading manifest. Its path
may be relative to the manifest and its SHA-256 must match the exact runtime
standard bytes:

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

Build the standardized shougang SFT release from that passed source. The
single `--input` is intentional: singleton materialization is a passthrough,
and does not replicate rows:

```bash
python -m script.verl.common.build_mixture \
  --family sft \
  --input shougang="$RUN/releases/sft/source/shougang" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --task-config "$CFG/shougang.task.json" \
  --metadata-fields field_name table_name field_description table_description \
  --output-dir "$RUN/releases/sft/shougang"
```

The output report records `shougang` as its sole input and a sampling weight of
1.0. Use the resulting standardized release for the frozen reference run.

### Re-measure the Qwen3.5 token budget

The example reference model is `Qwen3.5-9B` (a local senior SFT checkpoint may
be supplied via a runtime override). **All token limits from old Qwen2.5 runs
are invalid.** Before training, remeasure the selected Qwen3.5 checkpoint's
chat-template lengths with the repository tools and set a measured limit; do
not copy an old numeric limit or invent one:

```bash
python -m script.verl.sft.prompt_stats \
  --dataset-dir "$RUN/releases/sft/shougang" \
  --model "$MODEL_PATH" \
  --report "$RUN/metrics/shougang.token-stats.json"

# Set QWEN35_MAX_TOKENS from the measured report, then run the gate:
python -m script.verl.sft.check_token_budget \
  --dataset-dir "$RUN/releases/sft/shougang" \
  --model "$MODEL_PATH" \
  --max-length "$QWEN35_MAX_TOKENS" \
  --report "$RUN/metrics/shougang.token-budget.json"
```

A failed gate stops training. Re-run it whenever the model, chat template,
or exported prompts change.

### Evaluate shougang directly

`evaluate_true_e2e` requires an explicit report path and scores the shougang
test parquet directly:

```bash
python -m script.verl.sft.evaluate_true_e2e \
  --model-path "$MODEL_PATH" \
  --data "$RUN/releases/sft/shougang/test.parquet" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json" \
  --report "$RUN/metrics/shougang.true-e2e.json"
```

The report exposes strict leaf/level exact match and per-head diagnostics.
Retain this shougang report as the official evaluation; keep the score
source-local.

## Reference-to-RLOO chain

The required order is:

```text
passed shougang SFT source export
  -> --family sft singleton passthrough
  -> frozen shougang reference SFT run
  -> verified single-rank LoRA merge
  -> record_environment
  -> record_checkpoint
  -> shougang RL source export
  -> --family rl-cascade singleton passthrough
  -> qwen3.5-native-tools-v1 validation
  -> VeRL 0.9 ToolAgentLoop (no-tool / single-call / multi-call)
  -> strict terminal category+level exact-match RLOO
```

The complete server commands are in [server-runbook.md](server-runbook.md). In
brief:

1. Source a runtime copy of `cfg/sft/reference.env.example`, remeasure the
   Qwen3.5 chat-template budget, set its measured limit, and run
   `script/verl/sft/run.sh` with the `HYDRA_OVERRIDES` array.
2. Merge the resulting `world_size=1, rank=0` LoRA checkpoint with
   `script.verl.sft.merge_lora_checkpoint` into a new HF directory.
3. Run `script.verl.common.record_environment` with the selected interpreter and
   `--gpu-target "RTX PRO 6000 96GB"`.
4. Run `script.verl.sft.record_checkpoint` with the shougang prompt-audit
   bundle and one-entry grading manifest. Its `sha256-tree-v2` digest, the
   passed `export_report.json`, effective Hydra config, environment manifest,
   base model, prompt audit, commit, and global step form the only accepted RL
   reference. RLOO must verify this provenance against the model directory;
   the recorded `checkpoint_dir` is advisory after relocation, while the
   sha256-tree/count/bytes remain the binding anti-tamper anchor.
5. Export the canonical shougang data with `script.verl.rl.export`, then build
   the standardized release with one `--input shougang=...`. Validate it with
   `script.verl.rl.validate_cascade` and the one-entry shougang grading
   manifest. The formal release contains one native-tool episode row per
   source. VeRL's official ToolAgentLoop renders Qwen3.5 tool schemas and tool
   observations; there is no synthetic Stage-2 user bridge.
6. Launch `script.verl.rl.rloo_experiment` with `--dataset shougang`, the merged
   reference, the standardized shougang RL release, the grading manifest,
   reference provenance, and an exact `--vllm-version`. `--dry-run` is the local
   inspection mode; the real launch is server-only.

## GPU gate

SFT merge and formal RLOO are authorized only on the designated
**RTX PRO 6000 96GB** server using the dedicated, validated VeRL 0.9.0
Qwen3.5 environment (one visible GPU). Run `record_environment` immediately before releasing the
reference and retain its secret-free manifest. Confirm
`torch_cuda.cuda_available`, `device_name`, `device_count`, and approximately
96 GiB of VRAM before starting VeRL. A CPU checkout may run unit tests and
`--dry-run`; it is not a training result. Run formal training only after the
designated host gate, current token-budget gate, and current
reference-provenance checks pass.
