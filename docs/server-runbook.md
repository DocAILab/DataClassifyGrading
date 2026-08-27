# Authorized server runbook and current blockers

This is the operational order for the formal **shougang-only** release. It
also records which seams are currently blocked. Every path below is a
placeholder and must be replaced with a runtime-local path; do not copy
production data, standards, reports, or checkpoints into Git.

This documentation pass performed only text and CPU/static checks. No GPU,
SSH, VeRL training, rollout, or evaluator run was performed.

## 0. Runtime variables and host gate

Use a runtime-local checkout and asset directories:

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
mkdir -p "$RUN" "$RUN/audits" "$RUN/metrics" "$RUN/releases"
```

The following are **future host checks**, not commands run for this task:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON_BIN" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
"$PYTHON_BIN" -c 'import importlib.metadata as m; print(m.version("verl"), m.version("vllm"))'
```

Formal RLOO requires the locked Qwen3.5/VeRL environment in
`requirements/verl-qwen35-cu130.txt` and the designated one-GPU host. A CPU
interpreter with `pyarrow` is sufficient for all data commands in Sections 1–3
and for the static validators; it is not a training result.

## 1. Build and audit canonical records (CPU)

Normalize the runtime source, resolve it against explicit runtime standards,
and assign embedded schema-v2 splits:

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

The current native metadata contract is exactly
`field_name`, `table_name`, `field_description`, and `table_description`; pass
all four explicitly. Stop if a resolved record has no split, or the audit is
not `status: passed` with `conflict_keys: 0`.

## 2. Legacy SFT compatibility line (CPU; three messages only)

`script.verl.sft.export` and `script.verl.sft.validate` are the old
stage1/stage2 line. They are implemented and executable with runtime assets,
but they do not produce tool trajectories. The exporter
writes one stage1 and one stage2 row per source; each row has exactly
`[system, user, assistant]` (three messages).

```bash
"$PYTHON_BIN" -m script.verl.sft.export \
  --canonical "$RUN/canonical/shougang/all.json" \
  --output-dir "$RUN/releases/sft-legacy/source/shougang" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --metadata-fields field_name table_name field_description table_description \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json"

"$PYTHON_BIN" -m script.verl.sft.validate \
  --dataset-dir "$RUN/releases/sft-legacy/source/shougang" \
  --registry "$ASSETS/registries/shougang.registry.json" \
  --corpus "$ASSETS/corpora/shougang.corpus.json" \
  --metadata-fields field_name table_name field_description table_description \
  --task-config "$CFG/shougang.task.json" \
  --grading-config "$STANDARDS/shougang.grading.json" \
  --report "$RUN/releases/sft-legacy/source/shougang/validation.json"
```

The standalone validator intentionally rejects a row whose `stage` is not
`stage1`/`stage2` or whose messages are not exactly three messages. Never point
this validator at the trajectory release in Section 3.

## 3. New trajectory SFT data line (CPU)

This is the **agentic SFT / trajectory bootstrapping** line: its training
role is to teach the native-tool trajectory capability (tool-call
turn-taking plus the terminal answer), not classification-first behavior.
Classification data is a separate, later addition guarded by the Section 6
rollout gate. Consistent with the blockers in Section 4, that role is the
data line's intent, not a training claim: no validated trainer consumes this
parquet in this checkout.

The implemented entrypoint is
`script.verl.sft.export_tool_trajectories`. Its modes are deliberately
separate from the legacy CLI:

### 3.1 `collect`

`collect` derives prompts, deterministic tool calls/results, terminal labels,
and split assignments, then writes think-free JSONL shards. It does not write
parquet and does not call a model:

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

A reviewed external generator must fill each line's terminal `think` and the
ordered `tool_think` list (one non-empty entry for every tool-call assistant
turn). There is no generator CLI in this repository; do not invent one or
claim that empty collect shards are training data.

### 3.2 `file` -> `assemble`/`export`

There is no `--mode file` or `--mode assemble`. The `file:<path>` value is the
`--think-source` selector, and normal export performs the assemble/publish
step. The command below re-derives the contexts, injects file thoughts,
validates every row, hashes all three split parquet files, and publishes only
after validation:

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

For a deterministic CPU smoke only, replace `file:$THINK_FILLED` with
`mock`; the mock generator is not a substitute for a reviewed external think
source in a formal release. Export mode invokes
`validate_tool_trajectory_dataset` before publication and records the result in
`export_report.json`.

### 3.3 `validate`

There is no separate trajectory validator CLI. The export command above is the
public assemble+validate path. To validate an already-published directory, use
the current public Python function from a shell heredoc:

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

Do not replace this with `script.verl.sft.validate`: that command is the
legacy three-message validator. `script.verl.sft.tool_trajectory_stats` is an
additional CPU/static check for character lengths and optional tokenizer
rendering:

```bash
"$PYTHON_BIN" -m script.verl.sft.tool_trajectory_stats \
  --dataset-dir "$TRAJECTORY_RELEASE" \
  --report "$RUN/metrics/shougang.trajectory-stats.json"
```

## 4. Mixture and SFT launcher seam (blocking)

The new trajectory parquet is currently an independently validated data
artifact, not an approved SFT training input. A mixture step is not inherently
required: the official trainer chain can read a parquet `messages` column
directly. The currently shipped route is nevertheless unsupported and
unverified:

- `script.verl.common.build_mixture --family sft` reads the legacy release
  contract. It requires `stage` to be `stage1` or `stage2`, exactly one row of
  each stage per source, and runs the legacy SFT validator on its output. New
  rows have `stage: "tool_trajectory"` and variable-length messages, so this
  seam rejects them. **The existing mixture does not support trajectory
  parquet.**
- The potential minimal route is direct parquet -> the official 0.9 trainer
  chain (`run.sh` -> `sft_trainer` -> `create_sft_dataset` ->
  `MultiTurnSFTDataset`). Qwen3.5 additionally requires the tracked
  prefix-diff/answer-mask patch bundle, including
  `script/verl/common/patches/verl-0.9.0/multiturn_sft_dataset-prefix-diff-answer-mask.patch`.
  The current `script/verl/sft/run.sh` is
  labelled VeRL 0.8, has no version/patch gate, and has not been validated
  with full tool-role messages, `reasoning_content`, or tool-result masking.
  Therefore it is not a supported/accepted trajectory launcher, even though
  the underlying trainer can read `messages` directly.
- `--family rl-cascade` is not an adapter. It consumes the five-field RL
  parquet from `script.verl.rl.export`, not the SFT trajectory parquet.

**Blocking TODO:** choose and implement the direct 0.9 patched trainer route
(or a trajectory-aware mixture adapter), add explicit VeRL-version/required-
patch checks, and add a CPU fixture test through the dataset and actual
launcher seam. Only after that validation may a GPU training command be added.
Until then, do not pass the trajectory release to the existing
`build_mixture --family sft` or `script/verl/sft/run.sh`, and do not describe it
as a trained/reference checkpoint.

The old singleton SFT mixture remains valid only for the old release and is
not a trajectory workaround. Before invoking it (and the independent RL
materializer below), create the one-entry manifest from the exact grading
bytes:

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

Then run the legacy materializer:

```bash
"$PYTHON_BIN" -m script.verl.common.build_mixture \
  --family sft \
  --input shougang="$RUN/releases/sft-legacy/source/shougang" \
  --grading-manifest "$RUN/grading-manifest.json" \
  --registry "$REGISTRY" \
  --corpus "$CORPUS" \
  --task-config "$TASK_CONFIG" \
  --metadata-fields field_name table_name field_description table_description \
  --output-dir "$RUN/releases/sft-legacy/standardized"
```

## 5. Independent RL data seam (CPU)

The RL exporter/materializer has a separate, currently implemented path. It
starts from `script.verl.rl.export` (a stage1/stage2 RL pair), and
`--family rl-cascade` projects each pair to one native-tool stage1 episode. It
does not consume the SFT trajectory parquet:

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

The native trajectory format used by current code/tests and this validator is
**`qwen3.5-native-tools-v2`**. The
`NATIVE_TOOL_TRAJECTORY_FORMAT` constant in
`src/agent/training/rl/sample.py` and the RL/SFT fixtures are authoritative;
keep this value consistent in reports and release expectations.

## 6. Mandatory post-SFT rollout gate (blocking acceptance TODO)

The decision rule remains mandatory before formal RLOO: run a bounded
native-tool rollout using the same shougang release, registry/corpus, parser,
and agent loop, and accept only `nonzero_reward_rate` in **5%–30% inclusive**.
A failed or missing gate blocks RLOO. If approximately 100% of rewards are
zero, inspect the trajectory, agent loop, reward shaping, and rollout parser
before adding classification data.

There is currently **no executable evaluator CLI** in this repository that
performs this bounded post-SFT rollout from a merged checkpoint and writes the
required report. `evaluate_true_e2e` is not this rollout evaluator, and the
RL preflight path only inspects/plans a command. Therefore this is a
**BLOCKING ACCEPTANCE PROCEDURE / TODO**, not a command to fabricate. Do not
add a fake evaluator invocation or record a gate pass without a report carrying
the numerator, denominator, rate, model/reference identity, release, parser,
and agent-loop settings.

## 7. Formal launch status

No formal SFT or RLOO launch is authorized from the trajectory line until the
mixture/launcher TODO in Section 4 and the rollout gate in Section 6 are
closed. Once those acceptance procedures are implemented and evidenced, the
server owner may separately perform the host gate, token-budget check,
reference provenance recording, and formal RLOO launch. Those future GPU steps
are intentionally not represented here as completed commands.
