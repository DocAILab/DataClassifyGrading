# Runtime-local classification and data-release chain

This document records the **implemented** data contracts and the boundaries that
are not implemented yet. Production inputs and model assets are supplied at
runtime; no command below searches this checkout for a production dataset,
registry, corpus, standard, parquet release, checkpoint, or report.

The formal source is **shougang only**. Keep production files and every derived
artifact outside Git (or in ignored runtime directories). All paths in the
examples are placeholders. This update was a CPU/static documentation check;
no GPU, SSH, or training run is evidence for any step below.

## Runtime setup (CPU data checks only)

Replace the placeholders before running commands:

```bash
export ROOT="<CHECKOUT>"
export RUN="<RUN_DIR>"
export RAW="<RAW_DIR>"
export ASSETS="<ASSETS_DIR>"
export STANDARDS="<STANDARDS_DIR>"
export CFG="<CONFIG_DIR>"
export PYTHON_BIN="<PYTHON_WITH_TEST_DEPS>"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
```

The test extra supplies `pytest` and `pyarrow`:

```bash
"$PYTHON_BIN" -m pip install -e "${ROOT}[test]"
"$PYTHON_BIN" -m pytest
```

The VeRL environment is separate and is not needed for the CPU data checks.
Formal RLOO has its own locked environment (`requirements/verl-qwen35-cu130.txt`)
and designated host gate; neither was run for this documentation change.

## Entrypoint truth table

The two SFT CLIs are different data lines. Do not merge their names or claims:

| Entrypoint | Implemented output and validation | What it is **not** |
| --- | --- | --- |
| `script.verl.sft.export` | Legacy SFT parquet: for each source record, one `stage1` row and one `stage2` row. Every row has exactly `messages = [system, user, assistant]` (three messages), plus the stage-specific answer. It writes `train.parquet`, `val.parquet`, `test.parquet`, and `export_report.json`. | It is a three-message compatibility line; no native tools are represented. |
| `script.verl.sft.validate` | Validates the legacy parquet above, including exactly three messages and one stage1/stage2 pair per `source_id`. | It cannot validate `stage=tool_trajectory` rows. Running it on the new trajectory parquet is expected to fail. |
| `script.verl.sft.export_tool_trajectories` | New shougang-only trajectory line. `--mode collect` writes think-free JSONL context shards; export mode reads `mock` or `file:<path>` think and publishes variable-length tool trajectories plus `export_report.json`. Export mode runs `validate_tool_trajectory_dataset` before publication. | It does not currently plug into the formal SFT mixture or launcher (see below). |
| `script.verl.common.build_mixture --family sft` | Current formal SFT materializer for legacy stage1/stage2 releases only. It requires each input source to contain exactly one row of each stage and reseeds legacy prompts. | It cannot consume the new trajectory parquet. A mixture step is not necessarily required in principle, but this implementation is not an adapter. |
| `script/verl/sft/run.sh` | Thin VeRL 0.8 `verl.trainer.sft_trainer` launcher that forwards Hydra arguments. The official trainer path can read a parquet `messages` column through `create_sft_dataset`/`MultiTurnSFTDataset`, but this launcher has no VeRL-version or patch gate. | It is not a supported Qwen3.5 trajectory trainer: the 0.9 prefix-diff/answer-mask patches and full tool-role path are unverified here. |

The trajectory exporter uses `stage: "tool_trajectory"` and
`trajectory_format: "qwen3.5-native-tools-v2"`; the
`NATIVE_TOOL_TRAJECTORY_FORMAT` constant in
`src/agent/training/rl/sample.py` and the RL/SFT fixtures are authoritative.
Its four deterministic classes
are `direct` (no tool call), `single_tool` (one), `multi_tool` (three), and
`no_result` (two). Assistant/tool turns and terminal assistant turns are
variable-length chat messages; every assistant turn carries `reasoning_content`,
and tool results are re-executed against the runtime registry/corpus during
validation.

There is no standalone `script.verl.sft.validate`-style trajectory CLI today.
Use the exporter (which validates before publishing) or the CPU validation
snippet below for an already-published trajectory directory.

Its training role is **agentic SFT / trajectory bootstrapping**, not a
classification-first line: the deterministic trajectories teach the
native-tool trajectory capability (assistant/tool turn-taking plus the
terminal answer), and classification supervision is a separate, later
addition guarded by the post-SFT rollout gate below. That role describes the
data line's intent; it is not a training claim — the mixture/launcher seam
below is still an open blocker and no validated trainer consumes this parquet
in this checkout.

## Runtime contract

- The formal dataset name is exactly `shougang`.
- The current native prompt contract is the four fields
  `field_name`, `table_name`, `field_description`, and `table_description`.
  Pass them explicitly with `--metadata-fields`; do not infer fields from a
  filename or a local default.
- Canonical labels come only from a resolved record's `target.category_id`.
  The grading config supplies the `data_level` field and approved levels.
- Native terminal answers use an opaque global `choice_id` and a level; the
  canonical category id must not be copied into the prompt.
- A current prompt-conflict audit with `conflict_keys: 0` is a release blocker.
  Standards and their SHA-256 values are runtime-local.

## Implemented CPU data path

### 1. Normalize, resolve, split, and audit

These steps are independent of either SFT data line:

```bash
"$PYTHON_BIN" -m script.preprocessing.cli preprocess \
  --input "$RAW/shougang.xlsx" \
  --mapping "$ASSETS/mappings/shougang.mapping.json" \
  --output "$RUN/processed/shougang/all.json" \
  --dataset shougang

"$PYTHON_BIN" -m script.canonical.build \
  --processed-dir "$RUN/processed" \
  --canonical-dir "$RUN/canonical" \
  --config-file "$CFG/datasets.json" \
  --registry-dir "$ASSETS/registries" \
  --corpus-dir "$ASSETS/corpora" \
  --dataset shougang

"$PYTHON_BIN" -m script.canonical.split \
  --canonical-dir "$RUN/canonical" \
  --dataset shougang \
  --split-type random \
  --seed 42

"$PYTHON_BIN" -m script.analysis.audit_prompt_conflicts --bundle \
  --canonical shougang="$RUN/canonical/shougang/all.json" \
  --classification-standard shougang="$STANDARDS/shougang.classification.json" \
  --grading-standard shougang="$STANDARDS/shougang.grading.json" \
  --split train \
  --level-field data_level \
  --report "$RUN/audits/shougang.prompt.json"
```

Stop on unresolved records, a missing embedded split, or an audit that is not
`status: passed` with `conflict_keys: 0`.

### 2. Legacy stage1/stage2 export (implemented compatibility path)

Use this path only when a three-message stage1/stage2 SFT baseline is wanted:

```bash
"$PYTHON_BIN" -m script.verl.sft.export \
  --canonical "$RUN/canonical/shougang/all.json" \
  --output-dir "$RUN/releases/sft-legacy/shougang" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --metadata-fields field_name table_name field_description table_description \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json"

"$PYTHON_BIN" -m script.verl.sft.validate \
  --dataset-dir "$RUN/releases/sft-legacy/shougang" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --metadata-fields field_name table_name field_description table_description \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json" \
  --report "$RUN/releases/sft-legacy/shougang/validation.json"
```

The report's `stage` and `messages` checks are the authority: this exporter is
legacy stage1/stage2, not a tool-trajectory/native-tool release.

### 3. New trajectory `collect`

`collect` is CPU-only and emits one or more `think-<split>-NNN.jsonl` shards
plus `collect_report.json`. It does not emit parquet:

```bash
export CANONICAL="$RUN/canonical/shougang/all.json"
export REGISTRY="$ASSETS/registries/shougang.registry.json"
export CORPUS="$ASSETS/corpora/shougang.corpus.json"
export TASK_CONFIG="$CFG/shougang.task.json"
export GRADING_CONFIG="$STANDARDS/shougang.grading.json"
export THINK_COLLECT="$RUN/trajectory/think-collect"

"$PYTHON_BIN" -m script.verl.sft.export_tool_trajectories \
  --mode collect \
  --dataset shougang \
  --canonical "$CANONICAL" \
  --registry "$REGISTRY" \
  --corpus "$CORPUS" \
  --metadata-fields field_name table_name field_description table_description \
  --task-config "$TASK_CONFIG" \
  --grading-config "$GRADING_CONFIG" \
  --collect-dir "$THINK_COLLECT" \
  --shard-size 64
```

The repository has no command that generates label-aware `think` text. A
separate, reviewed generator must fill every shard line's terminal `think` and
its ordered `tool_think` list (one entry per tool-call assistant turn). Keep
those filled JSONL files in a runtime-local directory and preserve their
`sample_id` values. Do not invent a provider command or claim that collection
alone is a training release.

### 4. New trajectory `file` -> `assemble`/`export`

There is no separate `--mode file` or `--mode assemble`. `file:<path>` selects
the filled JSONL source, and normal export assembles contexts and thoughts into
parquet. The command below is the implemented assemble/publish path; it also
runs the trajectory validator before `os.replace` publication:

```bash
export THINK_FILLED="<FILLED_THINK_JSONL_OR_SHARD_DIR>"
export TRAJECTORY_RELEASE="$RUN/releases/sft-trajectory/shougang"

"$PYTHON_BIN" -m script.verl.sft.export_tool_trajectories \
  --mode export \
  --dataset shougang \
  --canonical "$CANONICAL" \
  --registry "$REGISTRY" \
  --corpus "$CORPUS" \
  --metadata-fields field_name table_name field_description table_description \
  --task-config "$TASK_CONFIG" \
  --grading-config "$GRADING_CONFIG" \
  --think-source "file:$THINK_FILLED" \
  --output-dir "$TRAJECTORY_RELEASE" \
  --failed-audit "$RUN/trajectory/assemble.failed.json"
```

For a credential-free CPU smoke (not a formal release), the same command can
use `--think-source mock`; the default `mock` generator is deterministic and
local. A `file:` export is the reviewed path when external reasoning text is
required.

### 5. Validate an existing trajectory release (CPU)

Export mode already validates. For an independent check, the public Python
function is callable from a shell heredoc; this is the current executable
validation seam because no dedicated trajectory validator module exists:

```bash
"$PYTHON_BIN" - "$TRAJECTORY_RELEASE" "$REGISTRY" "$CORPUS" "$TASK_CONFIG" "$GRADING_CONFIG" "$RUN/trajectory/validation.json" <<'PY'
import json
import sys
from pathlib import Path

from agent.task import GradingConfig, LeafRegistry, TaskConfig
from agent.task.assets import load_corpus_categories
from agent.training.sft import validate_tool_trajectory_dataset

release, registry_path, corpus_path, task_path, grading_path, report_path = sys.argv[1:]
registry = LeafRegistry.from_path(registry_path)
corpus = {item.category_id: item for item in load_corpus_categories(corpus_path)}
task = TaskConfig.from_path(task_path)
grading = GradingConfig.from_path(grading_path)
report = validate_tool_trajectory_dataset(
    release,
    registry,
    corpus=corpus,
    task_config=task,
    grading=grading,
    dataset="shougang",
)
Path(report_path).parent.mkdir(parents=True, exist_ok=True)
Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
raise SystemExit(0 if report.get("valid") else 1)
PY
```

Do **not** substitute `script.verl.sft.validate` in that snippet: that CLI is
the legacy three-message validator and is intentionally incompatible with the
new trajectory schema.

### 6. Optional trajectory stats (CPU/static)

Character statistics and (when a tokenizer is supplied) chat-template token
statistics are available without training:

```bash
"$PYTHON_BIN" -m script.verl.sft.tool_trajectory_stats \
  --dataset-dir "$TRAJECTORY_RELEASE" \
  --report "$RUN/metrics/shougang.trajectory-stats.json"
```

A Qwen3.5 tokenizer may be supplied with `--model <MODEL_DIR>` on a CPU host;
choose the actual training limit from the resulting report. Do not apply the
legacy `prompt_stats` stage1/stage2 assumptions to trajectory rows.

## Mixture and launcher seam (current blocker)

The new trajectory parquet is a valid, independently validated **data
artifact**, but it is not a validated training input in this checkout:
A mixture step is not necessarily required in principle: the official trainer
chain can read parquet `messages` directly. The current direct launcher path is
still unsupported and unverified (details below).

1. `build_mixture.py` reads an SFT input report and requires every SFT row's
   `stage` to be `stage1` or `stage2`; its grouping code requires exactly one
   row of each stage per source. A trajectory row has `stage: "tool_trajectory"`
   and variable-length messages, so `--family sft` rejects it before
   materialization. Its output validator also calls the legacy
   `validate_sft_dataset` contract. **The existing mixture does not support
   trajectory parquet.**
2. A direct parquet route is theoretically possible because the official
   trainer chain (`run.sh` -> `sft_trainer` -> `create_sft_dataset` ->
   `MultiTurnSFTDataset`) reads the `messages` column without requiring a
   mixture. For Qwen3.5, however, the repository's required VeRL 0.9
   prefix-diff/answer-mask patch bundle (including
   `script/verl/common/patches/verl-0.9.0/multiturn_sft_dataset-prefix-diff-answer-mask.patch`)
   must be applied and checked. The current
   `run.sh` is labelled VeRL 0.8, has no version/patch gate, and has not been
   validated with full tool-role messages, `reasoning_content`, or tool-result
   masking. Direct parquet -> patched VeRL 0.9 is therefore a **potential
   minimal path**, not a supported or accepted launcher.
3. `--family rl-cascade` is a different seam: it consumes the five-field RL
   parquet emitted by `script.verl.rl.export`, not SFT trajectory parquet. It
   cannot be used as an adapter by inference.

**Blocking TODO before any trajectory SFT training claim:** choose and
implement the direct 0.9 patched trainer route (or a trajectory-aware mixture
adapter), add explicit VeRL-version/required-patch checks, and validate the
full message/mask path with a CPU fixture through the actual launcher. Until
that work is complete, do not pass `$TRAJECTORY_RELEASE` to the existing
`build_mixture --family sft` or `run.sh`, and do not describe it as a
trained/reference checkpoint.

The legacy path remains executable and separate. Its singleton SFT mixture is
valid only for the old stage1/stage2 release. Before running that materializer,
copy the exact grading bytes and create the required one-entry manifest:

```bash
cp "$GRADING_CONFIG" "$RUN/grading.json"
GRADING_SHA256="$(sha256sum "$RUN/grading.json" | awk '{print $1}')"
cat > "$RUN/grading-manifest.json" <<JSON
{
  "datasets": {
    "shougang": {
      "path": "grading.json",
      "sha256": "$GRADING_SHA256"
    }
  }
}
JSON
```

Then run the materializer:

```bash
"$PYTHON_BIN" -m script.verl.common.build_mixture \
  --family sft \
  --input shougang="$RUN/releases/sft-legacy/shougang" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --registry "$REGISTRY" \
  --corpus "$CORPUS" \
  --task-config "$TASK_CONFIG" \
  --metadata-fields field_name table_name field_description table_description \
  --output-dir "$RUN/releases/sft-legacy/standardized"
```

## Mandatory post-SFT rollout gate

The decision gate remains mandatory before formal RLOO: a bounded native-tool
rollout must have `nonzero_reward_rate` in **5%–30% inclusive**. A failed or
missing gate blocks RLOO; if approximately 100% of rewards are zero, inspect
the trajectory, agent loop, reward shaping, and rollout parser before adding
classification data.

At present there is **no executable evaluator CLI** in this repository that
runs that bounded post-SFT native-tool rollout from a merged checkpoint and
writes the required rate. `evaluate_true_e2e` is a model/evaluation path, not a
replacement for this rollout gate, and the RL preflight path only plans a
launch. Therefore the gate is explicitly a **BLOCKING ACCEPTANCE PROCEDURE /
TODO**, not a command to pretend was run. Do not add a made-up evaluator
command or record a pass without a report containing the rollout denominator,
nonzero-reward numerator/rate, model/reference identity, release, parser, and
agent-loop configuration.

## Independent RL data path

The RL source/export seam is separate from trajectory SFT and is CPU-checkable
when its runtime assets are present:

```bash
"$PYTHON_BIN" -m script.verl.rl.export \
  --canonical "$CANONICAL" \
  --output-dir "$RUN/releases/rl/source/shougang" \
  --dataset shougang \
  --registry "$REGISTRY" \
  --corpus "$CORPUS" \
  --metadata-fields field_name table_name field_description table_description \
  --task-config "$TASK_CONFIG" \
  --grading-config "$GRADING_CONFIG"

"$PYTHON_BIN" -m script.verl.common.build_mixture \
  --family rl-cascade \
  --input shougang="$RUN/releases/rl/source/shougang" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --registry "$REGISTRY" \
  --corpus "$CORPUS" \
  --task-config "$TASK_CONFIG" \
  --metadata-fields field_name table_name field_description table_description \
  --output-dir "$RUN/releases/rl/shougang"

"$PYTHON_BIN" -m script.verl.rl.validate_cascade \
  --dataset-dir "$RUN/releases/rl/shougang" \
  --registry "$REGISTRY" \
  --corpus "$CORPUS" \
  --task-config "$TASK_CONFIG" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --report "$RUN/releases/rl/shougang/validation.json"
```

That RL materializer projects each validated stage1/stage2 source pair to one
native-tool stage1 episode; it does not consume or convert the SFT trajectory
parquet. Formal RLOO still requires the post-SFT gate, reference provenance,
and the unresolved trajectory mixture/launcher seam.

See [server-runbook.md](server-runbook.md) for the same sequence in operational
order. GPU/SSH execution is deliberately outside this CPU/static task.
